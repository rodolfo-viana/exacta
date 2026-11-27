"""Overlap, softmax and cross-entropy losses at saturated logits, against a high-precision reference.

Three things lost digits where the dominant softmax probability rounds to one (a logit gap of 37 in
binary64, 17 in binary32), or where an overlap ratio does:

* the value of a Dice or Tversky loss, ``1 - ratio``, is ``0`` or has only absolute accuracy when the ratio
  is within roundoff of one, although the loss itself (``1e-34``) is representable;
* the gradient of the squared Dice, ``2 (p N / D - g) / D``, subtracts two almost equal terms where ``p = g = 1``;
* ``torch.softmax`` differentiates as ``p (d - sum p d)``, the cross-entropy ``-log p_t`` is ``-log(1)`` and its
  gradient ``p - onehot`` is zero for the dominant class.

The references below evaluate each loss from its textbook definition (``1 - ratio``, ``-log p``, ``(1 - p)**gamma``,
with the softmax in mpmath) at 160 working digits, from the exact values of the float inputs, and differentiate it
by a central difference of step ``1e-30`` in the same arithmetic. They share no formula with the implementation: no
``q = 1 - p``, no ratio of non-negative sums. The 160 digits keep the squared Dice at a logit gap of 80 (a loss of
``1e-68``, a gradient of ``1e-72``) 50 digits above the resolution of the difference.

Errors are relative, in units of the roundoff ``u`` of the dtype: of the value and of every gradient element
(which spans many orders of magnitude within one batch), at saturated, mixed and moderate logits.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from exacta import _triton_losses
from exacta.losses import (
    DiceCELoss, DiceFocalLoss, DiceLoss, FocalLoss, FocalTverskyLoss, GeneralizedDiceCELoss, GeneralizedDiceLoss,
    GeneralizedWassersteinDiceLoss, SkeletonRecallLoss, TopKCELoss, TverskyLoss,
)

mpmath = pytest.importorskip("mpmath")
mp = mpmath.mp

DIGITS = 160
STEP = 1e-30
U = {torch.float64: 2.0**-53, torch.float32: 2.0**-24}
TINY = {torch.float64: 1e-290, torch.float32: 1e-36}
GAPS = (10.0, 20.0, 30.0, 40.0, 80.0)
GPU = torch.cuda.is_available() and _triton_losses.available()
BACKENDS = ["torch", pytest.param("triton", marks=pytest.mark.skipif(not GPU, reason="needs CUDA and Triton"))]
DTYPES = [torch.float64, torch.float32]
SHAPE = (2, 3, 8)  # cases, classes, voxels


# -- reference -----------------------------------------------------------------------------------------


def _probabilities(zs, shape):
    batch, classes, voxels = shape
    p = [None] * len(zs)
    for b, v in itertools.product(range(batch), range(voxels)):
        idx = [(b * classes + c) * voxels + v for c in range(classes)]
        top = max(zs[i] for i in idx)
        e = [mp.exp(zs[i] - top) for i in idx]
        total = mp.fsum(e)
        for i, term in zip(idx, e):
            p[i] = term / total
    return p


def _group_sums(p, gs, idx):
    inter = mp.fsum(p[i] * gs[i] for i in idx)
    psum, gsum = mp.fsum(p[i] for i in idx), mp.fsum(gs[i] for i in idx)
    return {
        "inter": inter, "p": psum, "g": gsum,
        "p2": mp.fsum(p[i] ** 2 for i in idx), "g2": mp.fsum(gs[i] ** 2 for i in idx),
        "fp": mp.fsum(p[i] * (1 - gs[i]) for i in idx), "fn": mp.fsum((1 - p[i]) * gs[i] for i in idx),
        "l1": mp.fsum(abs(p[i] - gs[i]) for i in idx),
    }


def _value_overlap(zs, gs, shape, *, tversky=None, gamma=1, smooth=1e-5, squared=False, smooth_numerator=True,
                   micro=False, batch_dice=False, include_background=True, semimetric=False):
    """Mean over (case, class) of ``(1 - numerator / denominator) ** gamma``, from the definition."""
    batch, classes, voxels = shape
    p = _probabilities(zs, shape)
    s, gamma = mp.mpf(smooth), mp.mpf(gamma)
    first = 0 if include_background else 1
    cases = [list(range(batch))] if batch_dice else [[b] for b in range(batch)]
    losses = []
    for case in cases:
        per_class = [_group_sums(p, gs, [(b * classes + c) * voxels + v for b in case for v in range(voxels)])
                     for c in range(first, classes)]
        if micro:
            per_class = [{k: mp.fsum(d[k] for d in per_class) for k in per_class[0]}]
        for d in per_class:
            if semimetric:
                inter = (d["p"] + d["g"] - d["l1"]) / 2
                fp, fn = d["p"] - inter, d["g"] - inter
            else:
                inter, fp, fn = d["inter"], d["fp"], d["fn"]
            if tversky is None:
                numerator = 2 * inter
                denominator = (d["p2"] + d["g2"] if squared else d["p"] + d["g"]) + s
            else:
                numerator = inter
                denominator = inter + mp.mpf(tversky[0]) * fp + mp.mpf(tversky[1]) * fn + s
            if smooth_numerator:
                numerator = numerator + s
            error = 1 - numerator / denominator
            losses.append(mp.mpf(0) if error == 0 else error**gamma)
    return mp.fsum(losses) / len(losses)


def _value_generalized(zs, gs, shape, *, smooth=1e-5, semimetric=False):
    """Generalized Dice with inverse squared class volumes: one ratio per case, mean over cases."""
    batch, classes, voxels = shape
    p = _probabilities(zs, shape)
    s = mp.mpf(smooth)
    losses = []
    for b in range(batch):
        d = [_group_sums(p, gs, [(b * classes + c) * voxels + v for v in range(voxels)]) for c in range(classes)]
        weight = [1 / x["g"] ** 2 for x in d]
        inter = [(x["p"] + x["g"] - x["l1"]) / 2 if semimetric else x["inter"] for x in d]
        numerator = 2 * mp.fsum(w * i for w, i in zip(weight, inter)) + s
        denominator = mp.fsum(w * (x["p"] + x["g"]) for w, x in zip(weight, d)) + s
        losses.append(1 - numerator / denominator)
    return mp.fsum(losses) / batch


def _value_ce(zs, gs, shape, *, weights=None, index=True):
    """Mean of ``-sum_c w_c g_c log p_c`` over voxels; normalized by the target weights for index targets."""
    batch, classes, voxels = shape
    p = _probabilities(zs, shape)
    w = [mp.mpf(1)] * classes if weights is None else [mp.mpf(x) for x in weights]
    total, norm = [], []
    for b, v in itertools.product(range(batch), range(voxels)):
        idx = [(b * classes + c) * voxels + v for c in range(classes)]
        total.append(-mp.fsum(w[c] * gs[i] * mp.log(p[i]) for c, i in enumerate(idx) if gs[i] != 0))
        norm.append(mp.fsum(w[c] * gs[i] for c, i in enumerate(idx)))
    return mp.fsum(total) / (mp.fsum(norm) if weights is not None and index else batch * voxels)


def _value_focal(zs, gs, shape, *, gamma, alpha):
    """Mean over voxels of ``-sum_c w_c g_c (1 - p_c)**gamma log p_c``; background weight ``1 - alpha``."""
    batch, classes, voxels = shape
    p = _probabilities(zs, shape)
    gamma = mp.mpf(gamma)
    terms = []
    for b, v in itertools.product(range(batch), range(voxels)):
        for c in range(classes):
            i = (b * classes + c) * voxels + v
            w = 1 if alpha is None else (1 - mp.mpf(alpha) if c == 0 else mp.mpf(alpha))
            if gs[i] != 0:
                terms.append(-w * gs[i] * (1 - p[i]) ** gamma * mp.log(p[i]))
    return mp.fsum(terms) / (batch * voxels)


VALUES = {"overlap": _value_overlap, "generalized": _value_generalized, "ce": _value_ce, "focal": _value_focal}


@functools.lru_cache(maxsize=None)
def _reference(terms: tuple, z_bytes: bytes, g_bytes: bytes, shape: tuple, dtype: torch.dtype):
    """Value and per-element gradient (central difference), from the exact float inputs."""
    z = torch.frombuffer(bytearray(z_bytes), dtype=dtype).reshape(shape)
    g = torch.frombuffer(bytearray(g_bytes), dtype=dtype).reshape(shape)
    with mpmath.workdps(DIGITS):
        zs = [mp.mpf(v) for v in z.double().flatten().tolist()]
        gs = [mp.mpf(v) for v in g.double().flatten().tolist()]
        step = mp.mpf(STEP)

        def value(x):
            return mp.fsum(weight * VALUES[kind](x, gs, shape, **dict(options)) for weight, kind, options in terms)

        grad = []
        for i in range(len(zs)):
            up, down = list(zs), list(zs)
            up[i] += step
            down[i] -= step
            grad.append((value(up) - value(down)) / (2 * step))
        return value(zs), grad


def reference(terms: tuple, z: torch.Tensor, g: torch.Tensor):
    z, g = z.detach().contiguous(), g.detach().contiguous()
    key = (terms, z.numpy().tobytes(), g.numpy().tobytes(), tuple(z.shape), z.dtype)
    return _reference(*key)


def _relative(got: float, ref) -> float:
    with mpmath.workdps(DIGITS):
        if ref == 0:
            return 0.0 if got == 0 else float("inf")
        return float(abs(mp.mpf(got) - ref) / abs(ref))


# -- the losses under test -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    """A loss: its constructor (given the backend), the reference terms it sums, and its input conventions."""

    make: Callable[[str], torch.nn.Module]
    terms: tuple  # (weight, kind, options as a sorted tuple of pairs)
    top32: float = 40.0  # largest logit gap in binary32 (squared Dice: q**2, focal: q**3 must stay normal)
    soft: bool = False  # one-hot targets with soft labels instead of class indices
    torch_only: bool = False  # not implemented by the fused kernels


def _terms(*items):
    return tuple((w, kind, tuple(sorted(options.items()))) for w, kind, options in items)


def _spec(make, *items, **kw):
    return Spec(make, _terms(*items), **kw)


_TV = {"tversky": (0.3, 0.7)}
SPECS = {
    "dice": _spec(lambda b: DiceLoss(backend=b), (1.0, "overlap", {})),
    "dice_squared": _spec(lambda b: DiceLoss(squared_pred=True, backend=b), (1.0, "overlap", {"squared": True}),
                          top32=20.0),
    "dice_batch_dice": _spec(lambda b: DiceLoss(batch_dice=True, backend=b), (1.0, "overlap", {"batch_dice": True})),
    "dice_no_background": _spec(lambda b: DiceLoss(include_background=False, backend=b),
                                (1.0, "overlap", {"include_background": False})),
    "tversky": _spec(lambda b: TverskyLoss(alpha=0.3, beta=0.7, backend=b), (1.0, "overlap", _TV)),
    "tversky_unbalanced": _spec(lambda b: TverskyLoss(alpha=0.5, beta=0.7, backend=b),
                                (1.0, "overlap", {"tversky": (0.5, 0.7)})),
    "focal_tversky": _spec(lambda b: FocalTverskyLoss(gamma=0.75, alpha=0.3, beta=0.7, backend=b),
                           (1.0, "overlap", {**_TV, "gamma": 0.75})),
    "generalized_dice": _spec(lambda b: GeneralizedDiceLoss(backend=b), (1.0, "generalized", {})),
    "ce": _spec(lambda b: DiceCELoss(lambda_dice=0.0, backend=b), (1.0, "ce", {})),
    "ce_weighted": _spec(lambda b: DiceCELoss(lambda_dice=0.0, class_weight=[0.5, 1.0, 2.0], backend=b),
                         (1.0, "ce", {"weights": (0.5, 1.0, 2.0)})),
    "focal": _spec(lambda b: FocalLoss(gamma=2.0, alpha=0.25, backend=b),
                   (1.0, "focal", {"gamma": 2.0, "alpha": 0.25}), top32=20.0),
    "focal_gamma_half": _spec(lambda b: FocalLoss(gamma=0.5, backend=b), (1.0, "focal", {"gamma": 0.5, "alpha": None}),
                              top32=20.0),
    "dice_ce": _spec(lambda b: DiceCELoss(lambda_dice=0.7, lambda_ce=1.3, backend=b),
                     (0.7, "overlap", {}), (1.3, "ce", {})),
    "dice_squared_ce": _spec(lambda b: DiceCELoss(squared_pred=True, lambda_dice=0.7, lambda_ce=1.3, backend=b),
                             (0.7, "overlap", {"squared": True}), (1.3, "ce", {}), top32=20.0),
    "dice_focal": _spec(lambda b: DiceFocalLoss(gamma=2.0, alpha=0.25, lambda_dice=0.7, lambda_focal=1.3, backend=b),
                        (0.7, "overlap", {}), (1.3, "focal", {"gamma": 2.0, "alpha": 0.25}), top32=20.0),
    "generalized_dice_ce": _spec(lambda b: GeneralizedDiceCELoss(lambda_dice=0.7, lambda_ce=1.3, backend=b),
                                 (0.7, "generalized", {}), (1.3, "ce", {})),
    "dice_soft_targets": _spec(lambda b: DiceLoss(target_format="one-hot", backend=b), (1.0, "overlap", {}),
                               soft=True),
    "dice_squared_soft_targets": _spec(
        lambda b: DiceLoss(target_format="one-hot", squared_pred=True, backend=b),
        (1.0, "overlap", {"squared": True}), soft=True, top32=20.0),
    "ce_soft_targets": _spec(lambda b: DiceCELoss(target_format="one-hot", lambda_dice=0.0, backend=b),
                             (1.0, "ce", {"index": False}), soft=True),
    "dice_micro": _spec(lambda b: DiceLoss(average="micro", backend=b), (1.0, "overlap", {"micro": True}),
                        torch_only=True),
    "dice_unsmoothed_numerator": _spec(lambda b: DiceLoss(smooth_numerator=False, backend=b),
                                       (1.0, "overlap", {"smooth_numerator": False}), torch_only=True),
    "tversky_unsmoothed_numerator": _spec(
        lambda b: TverskyLoss(alpha=0.3, beta=0.7, smooth_numerator=False, backend=b),
        (1.0, "overlap", {**_TV, "smooth_numerator": False}), torch_only=True),
    "dice_semimetric": _spec(lambda b: DiceLoss(target_format="one-hot", soft_label=True, backend=b),
                             (1.0, "overlap", {"semimetric": True}), soft=True, torch_only=True),
    "tversky_semimetric": _spec(
        lambda b: TverskyLoss(alpha=0.3, beta=0.7, target_format="one-hot", soft_label=True, backend=b),
        (1.0, "overlap", {**_TV, "semimetric": True}), soft=True, torch_only=True),
    "generalized_dice_semimetric": _spec(
        lambda b: GeneralizedDiceLoss(target_format="one-hot", soft_label=True, backend=b),
        (1.0, "generalized", {"semimetric": True}), soft=True, torch_only=True),
}


def measure(name: str, backend: str, dtype: torch.dtype, z: torch.Tensor, g: torch.Tensor):
    """Relative errors in units of the roundoff of ``dtype``: value; worst gradient element relative to itself;
    worst gradient element relative to the largest element of its voxel; and |value|.

    The softmax Jacobian ``p (I - 1 p')`` makes the gradient of a moderate voxel a difference of terms of the size
    of its neighbours, so that an element can be small by cancellation whatever the arithmetic does: the second
    measure is the one that applies there. In a saturated voxel the elements are sums of terms of one sign and
    the first measure holds too.
    """
    spec = SPECS[name]
    device = "cuda" if backend == "triton" else "cpu"
    x = z.to(dtype).to(device).clone().requires_grad_()
    target = g.to(device) if spec.soft else g.argmax(1).to(device)
    loss = spec.make(backend)(x, target.to(dtype) if spec.soft else target)
    (grad,) = torch.autograd.grad(loss, x)
    assert grad.dtype == dtype and loss.dtype == dtype
    value, reference_grad = reference(spec.terms, z.to(dtype).cpu(), g.to(dtype).cpu())
    batch, classes, voxels = z.shape
    got = grad.double().flatten().tolist()
    with mpmath.workdps(DIGITS):
        absolute = [abs(mp.mpf(a) - r) for a, r in zip(got, reference_grad)]
        magnitude = [abs(r) for r in reference_grad]
        # Elements below the normal range of the dtype (1e-38 in binary32) are not representable relatively.
        normal = [m > TINY[dtype] * 1e4 for m in magnitude]
        element = [float(a / m) for a, m, ok in zip(absolute, magnitude, normal) if ok]
        voxel = []
        for i, ok in enumerate(normal):
            if ok:
                b, v = i // (classes * voxels), i % voxels
                scale = max(magnitude[(b * classes + c) * voxels + v] for c in range(classes))
                voxel.append(float(absolute[i] / scale))
    assert len(element) >= len(got) // 2, "too many reference gradients outside the range of the dtype"
    u = U[dtype]
    return _relative(loss.item(), value) / u, max(element) / u, max(voxel) / u, abs(float(value))


# -- data ----------------------------------------------------------------------------------------------


def _top(name: str, dtype: torch.dtype) -> float:
    return 80.0 if dtype == torch.float64 else SPECS[name].top32


def _batch(name: str, kind: str, dtype: torch.dtype, seed: int, shape=SHAPE) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits (exact in binary32) and targets (one-hot, soft for ``soft`` specs).

    ``saturated``: every voxel has one class above the others by a gap of 10 to 80 (capped by the dtype) and
    the target is that class, so that the losses are tiny; ``mixed``: half of the voxels saturated and a fifth
    of the saturated ones confidently wrong, the others moderate; ``moderate``: no saturation. Every
    (case, class) holds at least one positive and one negative voxel.
    """
    batch, classes, voxels = shape
    gen = torch.Generator().manual_seed(seed)
    grid = torch.tensor([m for m in GAPS if m <= _top(name, dtype)], dtype=torch.float64)
    target = torch.randint(classes, (batch, voxels), generator=gen)
    target[:, :classes] = torch.arange(classes)
    noise = torch.randn(batch, classes, voxels, generator=gen, dtype=torch.float64)
    dominant = target.clone()
    gap = grid[torch.randint(len(grid), (batch, voxels), generator=gen)]
    if kind == "mixed":
        wrong = torch.rand(batch, voxels, generator=gen) < 0.2
        wrong[:, :classes] = False
        dominant = torch.where(wrong, (target + 1 + torch.randint(classes - 1, (batch, voxels), generator=gen)) % classes,
                               target)
        gap = torch.where(torch.rand(batch, voxels, generator=gen) < 0.5, gap, torch.zeros_like(gap))
    if kind == "moderate":
        gap = torch.zeros_like(gap)
    z = 3.0 * noise if kind != "saturated" else noise
    z = z + gap[:, None, :] * F.one_hot(dominant, classes).movedim(-1, 1)
    g = F.one_hot(target, classes).movedim(-1, 1).to(torch.float64)
    if SPECS[name].soft:
        soft = torch.rand(batch, classes, voxels, generator=gen, dtype=torch.float64) * 0.2
        g = 0.8 * g + soft / soft.sum(1, keepdim=True) * 0.2
        g = g.float().double()
    return z.float().double(), g


# Bounds, in units of the roundoff u of the dtype, on the value and on every gradient element of a saturated
# batch; the errors measured are at most 3 u in the value and 9 u in the gradient in binary64 on both backends,
# 7 u and 40 u in binary32 on PyTorch and 3 u and 75 u in the fused binary32 kernels. In binary32 the shifted logit
# z - max is itself rounded (up to |gap| u / 2 relative in every probability, 20 u at a gap of 40), and the fused
# kernels use the fast exponential, whose argument rounding costs about |gap| u; focal terms cube it. The
# reference sees the exact difference. Before the fix the same quantities were off by 100% (1e16 u in binary64).
BOUND = {("torch", torch.float64): 16.0, ("triton", torch.float64): 16.0,
         ("torch", torch.float32): 96.0, ("triton", torch.float32): 160.0}
# Mixed and moderate batches contain voxels where the softmax Jacobian p (I - 1 p') makes an element a small
# difference of terms of the size of its neighbours; relative to the largest element of its voxel the error is
# at most a few hundred u (a condition number of the problem, the same before the fix, up to 169 u measured).
BOUND_VOXEL = 512.0


# -- tests ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("kind", ["saturated", "mixed", "moderate"])
@pytest.mark.parametrize("dtype", DTYPES, ids=["binary64", "binary32"])
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", list(SPECS))
def test_value_and_gradient_are_relatively_accurate(name, backend, dtype, kind, seed):
    if backend == "triton" and SPECS[name].torch_only:
        pytest.skip("runs on the PyTorch backend only")
    z, g = _batch(name, kind, dtype, seed)
    value_error, element_error, voxel_error, magnitude = measure(name, backend, dtype, z, g)
    assert voxel_error <= BOUND_VOXEL, f"gradient, relative to its voxel: {voxel_error:.3g} u"
    if kind == "saturated":
        assert element_error <= BOUND[backend, dtype], f"gradient, element by element: {element_error:.3g} u"
    assert value_error <= BOUND[backend, dtype], f"value: {value_error:.3g} u (|value| = {magnitude:.3g})"


@pytest.mark.parametrize("dtype", DTYPES, ids=["binary64", "binary32"])
@pytest.mark.parametrize("name", ["dice", "tversky", "generalized_dice", "dice_squared", "ce"])
def test_saturated_batches_have_small_losses(name, dtype):
    # Guards the data above: the losses compared are of the order exp(-gap) (at most 1e-3 here, 1e-34 at a gap
    # of 80), where 1 - ratio has an absolute error of about u: a relative error of 1e-13 or more.
    z, g = _batch(name, "saturated", dtype, 0)
    value, _ = reference(SPECS[name].terms, z.to(dtype), g.to(dtype))
    assert 0 < abs(float(value)) < 1e-3


@pytest.mark.parametrize("dtype", DTYPES, ids=["binary64", "binary32"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_cross_entropy_of_a_dominant_class_keeps_its_value_and_gradient(backend, dtype):
    # -log p_t = log1p(exp(-40) + exp(-80)) = 4.2e-18 for the dominant target, and the gradient p - onehot is
    # (-4.2e-18, 4.2e-18, 4.2e-34), where torch.softmax and F.cross_entropy give exactly 0.
    device = "cuda" if backend == "triton" else "cpu"
    z = torch.tensor([[[40.0], [0.0], [-40.0]]], dtype=dtype, device=device, requires_grad=True)
    target = torch.zeros(1, 1, dtype=torch.long, device=device)
    loss = DiceCELoss(lambda_dice=0.0, backend=backend)(z, target)
    (grad,) = torch.autograd.grad(loss, z)
    with mpmath.workdps(DIGITS):
        e = [mp.mpf(1), mp.exp(mp.mpf(-40)), mp.exp(mp.mpf(-80))]
        total, p = mp.fsum(e), [x / mp.fsum(e) for x in e]
        expected_value, expected_grad = mp.log(total), [mp.fsum(p[1:]) * -1, p[1], p[2]]
    assert _relative(loss.item(), expected_value) <= 8 * U[dtype] * (1 if backend == "torch" else 16)
    for got, ref in zip(grad.flatten().tolist(), expected_grad):
        assert _relative(got, ref) <= 8 * U[dtype] * (1 if backend == "torch" else 16)


@pytest.mark.parametrize("backend", BACKENDS)
def test_squared_dice_gradient_is_not_a_difference_of_equal_terms(backend):
    # Perfectly predicted voxels with a gap of 40: the gradient is of the order 1e-35 relative to the (p N / D - g)
    # of its two terms, which agreed to 15 digits and were subtracted.
    device = "cuda" if backend == "triton" else "cpu"
    z = torch.tensor([[[40.0, -40.0, -40.0, 40.0], [-40.0, 40.0, 40.0, -40.0]]], dtype=torch.float64, device=device,
                     requires_grad=True)
    target = torch.tensor([[0, 1, 1, 0]], device=device)
    (grad,) = torch.autograd.grad(DiceLoss(squared_pred=True, backend=backend)(z, target), z)
    terms = _terms((1.0, "overlap", {"squared": True}))
    _, ref = reference(terms, z.detach().cpu().reshape(1, 2, 4), F.one_hot(target.cpu(), 2).movedim(-1, 1).double())
    for got, expected in zip(grad.flatten().tolist(), ref):
        assert _relative(got, expected) <= 32 * U[torch.float64]


# -- the reference itself, and moderate logits ------------------------------------------------------------


@pytest.mark.parametrize("name", ["dice", "dice_squared", "tversky", "focal_tversky", "generalized_dice", "dice_ce",
                                  "dice_focal", "ce_weighted", "dice_micro", "dice_semimetric"])
def test_reference_gradient_matches_autograd_of_the_plain_composition(name):
    # Validates the numerical derivatives above at moderate logits, against PyTorch's own autograd of softmax
    # and the textbook formula (binary64): agreement to 1e-9 shows the step and the digits are adequate.
    spec = SPECS[name]
    z, g = _batch(name, "moderate", torch.float64, 3)
    value, grad = reference(spec.terms, z, g)
    x = z.clone().requires_grad_()
    target = g if spec.soft else g.argmax(1)
    loss = spec.make("torch")(x, target)
    (got,) = torch.autograd.grad(loss, x)
    assert abs(loss.item() - float(value)) <= 1e-13 * abs(float(value))
    reference_grad = torch.tensor([float(v) for v in grad], dtype=torch.float64).reshape(z.shape)
    assert (got - reference_grad).abs().max() <= 1e-12 * reference_grad.abs().max()


def _textbook(z, target, *, kind="dice", squared=False, alpha=0.5, beta=0.5, gamma=1.0, smooth=1e-5,
              batch_dice=False, include_background=True, micro=False, smooth_numerator=True):
    """The documented formulas, with ``1 - ratio``, on torch.softmax: the implementation before this change."""
    p = z.softmax(1)
    g = F.one_hot(target, z.shape[1]).movedim(-1, 1).to(z.dtype)
    if not include_background:
        p, g = p[:, 1:], g[:, 1:]
    dims = (0, *range(2, p.ndim)) if batch_dice else tuple(range(2, p.ndim))
    inter, fp, fn = (p * g).sum(dims), (p * (1 - g)).sum(dims), ((1 - p) * g).sum(dims)
    psum, gsum = (p.square() if squared else p).sum(dims), (g.square() if squared else g).sum(dims)
    if micro:
        inter, fp, fn, psum, gsum = (x.sum(-1, keepdim=True) for x in (inter, fp, fn, psum, gsum))
    if kind == "dice":
        numerator, denominator = 2 * inter, psum + gsum + smooth
    else:
        numerator, denominator = inter, inter + alpha * fp + beta * fn + smooth
    if smooth_numerator:
        numerator = numerator + smooth
    error = 1 - numerator / denominator
    return (error if gamma == 1 else torch.where(error == 0, 0, error.clamp_min(1e-300) ** gamma)).mean()


TEXTBOOK = {
    "dice": (lambda b: DiceLoss(backend=b), {}),
    "dice_squared": (lambda b: DiceLoss(squared_pred=True, backend=b), {"squared": True}),
    "dice_batch_dice": (lambda b: DiceLoss(batch_dice=True, backend=b), {"batch_dice": True}),
    "dice_no_background": (lambda b: DiceLoss(include_background=False, backend=b), {"include_background": False}),
    "tversky": (lambda b: TverskyLoss(alpha=0.3, beta=0.7, backend=b), {"kind": "tversky", "alpha": 0.3, "beta": 0.7}),
    "tversky_batch_dice_no_background": (
        lambda b: TverskyLoss(alpha=0.2, beta=0.9, batch_dice=True, include_background=False, backend=b),
        {"kind": "tversky", "alpha": 0.2, "beta": 0.9, "batch_dice": True, "include_background": False}),
    "focal_tversky": (lambda b: FocalTverskyLoss(gamma=0.75, alpha=0.3, beta=0.7, backend=b),
                      {"kind": "tversky", "alpha": 0.3, "beta": 0.7, "gamma": 0.75}),
    "dice_micro": (lambda b: DiceLoss(average="micro", backend=b), {"micro": True}),
    "dice_unsmoothed_numerator": (lambda b: DiceLoss(smooth_numerator=False, backend=b),
                                  {"smooth_numerator": False}),
    "dice_smooth_zero": (lambda b: DiceLoss(smooth=0.0, backend=b), {"smooth": 0.0}),
    "tversky_micro_unsmoothed": (
        lambda b: TverskyLoss(alpha=0.3, beta=0.7, average="micro", smooth_numerator=False, backend=b),
        {"kind": "tversky", "alpha": 0.3, "beta": 0.7, "micro": True, "smooth_numerator": False}),
}
TORCH_ONLY = {"dice_micro", "dice_unsmoothed_numerator", "tversky_micro_unsmoothed"}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", list(TEXTBOOK))
def test_moderate_logits_agree_with_the_textbook_formulas(name, backend):
    # Where nothing saturates, values and gradients stay those of 1 - ratio, to a few roundoffs: the ratio of
    # non-negative sums is the same function.
    if backend == "triton" and name in TORCH_ONLY:
        pytest.skip("runs on the PyTorch backend only")
    make, options = TEXTBOOK[name]
    gen = torch.Generator().manual_seed(5)
    z = 2.0 * torch.randn(3, 4, 6, 5, dtype=torch.float64, generator=gen)
    target = torch.randint(4, (3, 6, 5), generator=gen)
    results = []
    device = "cuda" if backend == "triton" else "cpu"
    for loss_of in (make(backend), lambda x, t: _textbook(x, t, **options)):
        x = z.to(device).clone().requires_grad_()
        loss = loss_of(x, target.to(device))
        (grad,) = torch.autograd.grad(loss, x)
        results.append((loss.item(), grad.cpu()))
    (new, new_grad), (old, old_grad) = results
    assert abs(new - old) <= 16 * U[torch.float64] * abs(old)
    assert (new_grad - old_grad).abs().max() <= 1e-13 * old_grad.abs().max()


def test_empty_class_conventions_are_unchanged():
    # skip_absent, zero_absent and an empty target without either, at moderate logits: the closed forms of the
    # textbook definition. Class 2 is absent from both cases.
    gen = torch.Generator().manual_seed(8)
    z = torch.randn(2, 3, 20, dtype=torch.float64, generator=gen).requires_grad_()
    target = torch.randint(2, (2, 20), generator=gen)
    p = z.softmax(1)
    g = F.one_hot(target, 3).movedim(-1, 1).double()
    ratio = (2 * (p * g).sum(2) + 1e-5) / (p.sum(2) + g.sum(2) + 1e-5)  # (case, class)
    plain = 1 - ratio.mean()
    skipped = (1 - ratio[:, :2]).mean()
    zeroed = (1 - ratio[:, :2]).sum() / 6
    for options, expected in ((dict(), plain), (dict(skip_absent=True), skipped), (dict(zero_absent=True), zeroed)):
        loss = DiceLoss(backend="torch", **options)(z, target)
        assert abs(loss.item() - expected.item()) <= 16 * U[torch.float64]
    # zero_absent gives the absent class no gradient, skip_absent as well
    (grad,) = torch.autograd.grad(DiceLoss(zero_absent=True, backend="torch")(z, target), z)
    (reference_grad,) = torch.autograd.grad(zeroed, z)
    torch.testing.assert_close(grad, reference_grad, rtol=1e-12, atol=1e-16)


@pytest.mark.parametrize("name", ["dice", "dice_ce", "dice_squared", "tversky", "generalized_dice"])
def test_masked_voxels_leave_a_saturated_loss_unchanged(name):
    # One case: the masked loss equals the loss of the voxels that remain, at saturated logits, in value and gradient.
    z, g = _batch(name, "saturated", torch.float64, 4, shape=(1, 3, 10))
    target = g.argmax(1)
    keep = torch.ones(1, 10, dtype=torch.bool)
    keep[0, [2, 5, 7]] = False
    # the masked voxels get nonfinite logits: they must not leak into the loss or the gradient
    z_masked = z.clone()
    z_masked[:, :, ~keep[0]] = float("nan")
    module = SPECS[name].make("torch")
    x = z_masked.requires_grad_()
    masked = module(x, target, mask=keep)
    (masked_grad,) = torch.autograd.grad(masked, x)
    y = z[:, :, keep[0]].clone().requires_grad_()
    compact = module(y, target[:, keep[0]])
    (compact_grad,) = torch.autograd.grad(compact, y)
    assert abs(masked.item() - compact.item()) <= 8 * U[torch.float64] * abs(compact.item())
    assert torch.isfinite(masked_grad).all() and (masked_grad[:, :, ~keep[0]] == 0).all()
    torch.testing.assert_close(masked_grad[:, :, keep[0]], compact_grad, rtol=1e-14, atol=0)


@pytest.mark.skipif(not GPU, reason="needs CUDA and Triton")
@pytest.mark.parametrize("name", ["dice", "dice_ce", "dice_squared", "tversky", "generalized_dice", "focal",
                                  "dice_focal"])
def test_fused_and_torch_backends_agree_with_a_mask_at_saturated_logits(name):
    z, g = _batch(name, "mixed", torch.float64, 6, shape=(2, 3, 12))
    target = g.argmax(1)
    mask = torch.rand(2, 12, generator=torch.Generator().manual_seed(1)) < 0.75
    mask[:, :3] = True
    outs = []
    for backend, device in (("torch", "cpu"), ("triton", "cuda")):
        x = z.to(device).requires_grad_()
        loss = SPECS[name].make(backend)(x, target.to(device), mask=mask.to(device))
        (grad,) = torch.autograd.grad(loss, x)
        outs.append((loss.item(), grad.cpu()))
    (a, ga), (b, gb) = outs
    assert abs(a - b) <= 16 * U[torch.float64] * abs(a)
    assert ((ga - gb).abs() <= 1e-13 * ga.abs().clamp_min(1e-300) * 1e3 + 32 * U[torch.float64] * ga.abs()).all()


# -- derivatives of the new building blocks, double backward and soft targets ---------------------------


def test_softmax_pair_and_stats_have_correct_first_and_second_derivatives():
    from exacta import _softmax

    gen = torch.Generator().manual_seed(1)
    z = (2.0 * torch.randn(2, 4, 3, dtype=torch.float64, generator=gen)).requires_grad_()
    for function in (_softmax.softmax, lambda t: _softmax.softmax_pair(t)[1], _softmax.neg_log_softmax,
                     lambda t: torch.stack(_softmax.softmax_stats(t))):
        assert torch.autograd.gradcheck(function, (z,))
        assert torch.autograd.gradgradcheck(function, (z,))


@pytest.mark.parametrize("squared", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_index_overlap_statistics_have_correct_first_and_second_derivatives(squared, masked):
    from exacta import _softmax

    gen = torch.Generator().manual_seed(4)
    z = (2.0 * torch.randn(2, 3, 5, dtype=torch.float64, generator=gen)).requires_grad_()
    target = torch.randint(3, (2, 5), generator=gen)
    valid = torch.rand(2, 5, generator=gen) < 0.7 if masked else None

    def function(t):
        inter, first, third, _, nll = _softmax.index_overlap_statistics(t, target, valid, squared=squared, with_nll=True)
        return inter, first, third, nll

    assert torch.autograd.gradcheck(function, (z,))
    assert torch.autograd.gradgradcheck(function, (z,))


def test_cross_entropy_index_has_correct_first_and_second_derivatives():
    from exacta import _softmax

    gen = torch.Generator().manual_seed(6)
    z = (3.0 * torch.randn(2, 4, 3, 2, dtype=torch.float64, generator=gen)).requires_grad_()
    target = torch.randint(4, (2, 3, 2), generator=gen)
    assert torch.autograd.gradcheck(lambda t: _softmax.cross_entropy_index(t, target), (z,))
    assert torch.autograd.gradgradcheck(lambda t: _softmax.cross_entropy_index(t, target), (z,))


@pytest.mark.parametrize("dtype", DTYPES, ids=["binary64", "binary32"])
@pytest.mark.parametrize("squared", [False, True])
@pytest.mark.parametrize("scale", [2.0, 10.0], ids=["moderate", "saturated"])
def test_index_statistics_agree_with_the_one_hot_composition(dtype, squared, scale):
    # The sums and their gradient from the label indices equal those of the one-hot composition (softmax pair and
    # products), to a few roundoffs relative to their size, at moderate and at saturated logits.
    from exacta import _softmax
    from exacta.losses.segmentation import _overlap_statistics

    gen = torch.Generator().manual_seed(9)
    target = torch.randint(3, (2, 7, 6), generator=gen)
    z = (scale * torch.randn(2, 3, 7, 6, generator=gen) + 2 * scale * F.one_hot(target, 3).movedim(-1, 1)).to(dtype)
    valid = torch.rand(2, 7, 6, generator=gen) < 0.8
    weights = torch.randn(4, 2, 3, generator=gen).to(dtype)

    def combine(stats, nll):
        return sum((w * s).sum() for w, s in zip(weights, stats)) + (nll * 0.5).sum()

    outs = []
    for fast in (True, False):
        x = z.clone().requires_grad_()
        if fast:
            *stats, nll = _softmax.index_overlap_statistics(x, target, valid, squared=squared, with_nll=True)
        else:
            clean = torch.where(valid[:, None], x, 0)
            probs, complement = _softmax.softmax_pair(clean)
            onehot = torch.where(valid[:, None], F.one_hot(target, 3).movedim(-1, 1).to(dtype), 0)
            stats = _overlap_statistics(torch.where(valid[:, None], probs, 0), torch.where(valid[:, None], complement, 0),
                                        onehot, (2, 3), semimetric=False, squared=squared)
            nll = _softmax.neg_log_softmax(x).gather(1, target[:, None])[:, 0]
        (grad,) = torch.autograd.grad(combine(stats, nll), x)
        outs.append((stats, grad))
    (fast_stats, fast_grad), (stats, grad) = outs
    u = U[dtype]
    for a, b in zip(fast_stats, stats):
        assert ((a - b).abs() <= 8 * u * b.abs() + 1e-300).all()
    scale_of_grad = grad.abs().amax(1, keepdim=True).clamp_min(1e-300)
    assert ((fast_grad - grad).abs() <= 200 * u * scale_of_grad).all()


def test_labels_outside_the_classes_raise_on_the_index_paths():
    z = torch.randn(1, 3, 4, dtype=torch.float64)
    for target in (torch.full((1, 4), 3), torch.full((1, 4), -1)):
        with pytest.raises(RuntimeError, match="Class values"):
            DiceLoss(backend="torch")(z, target)
        with pytest.raises(RuntimeError, match="Class values"):
            DiceCELoss(lambda_dice=0.0, backend="torch")(z, target)


def test_softmax_stats_are_relatively_accurate_at_saturation():
    from exacta import _softmax

    z = torch.tensor([[40.0, 0.0, -40.0], [-3.0, 50.0, 49.0]], dtype=torch.float64).reshape(2, 3, 1)
    p, q, nll = _softmax.softmax_stats(z)
    with mpmath.workdps(DIGITS):
        for b in range(2):
            top = max(float(v) for v in z[b, :, 0])
            e = [mp.exp(mp.mpf(float(v)) - top) for v in z[b, :, 0]]
            total = mp.fsum(e)
            for c in range(3):
                assert _relative(p[b, c, 0].item(), e[c] / total) <= 4 * U[torch.float64]
                assert _relative(q[b, c, 0].item(), mp.fsum(e[k] for k in range(3) if k != c) / total) <= 4 * U[torch.float64]
                assert _relative(nll[b, c, 0].item(), -mp.log(e[c] / total)) <= 4 * U[torch.float64]


def test_error_ratio_has_correct_first_and_second_derivatives():
    from exacta._loss_reductions import error_ratio

    gen = torch.Generator().manual_seed(2)
    error = torch.rand(5, dtype=torch.float64, generator=gen).requires_grad_()
    numerator = (1 + torch.rand(5, dtype=torch.float64, generator=gen)).requires_grad_()
    assert torch.autograd.gradcheck(error_ratio, (error, numerator))
    assert torch.autograd.gradgradcheck(error_ratio, (error, numerator))


def test_softmax_losses_support_double_backward_on_the_torch_backend():
    gen = torch.Generator().manual_seed(7)
    z = torch.randn(2, 3, 20, dtype=torch.float64, generator=gen)
    target = torch.randint(3, (2, 20), generator=gen)
    v = torch.randn(2, 3, 20, dtype=torch.float64, generator=gen)

    def hessian_vector(loss_of):
        x = z.clone().requires_grad_()
        (first,) = torch.autograd.grad(loss_of(x), x, create_graph=True)
        (second,) = torch.autograd.grad((first * v).sum(), x)
        return second

    def plain(x, squared):
        p = x.softmax(1)
        g = F.one_hot(target, 3).movedim(-1, 1).double()
        denominator = (p.square().sum(2) + g.square().sum(2) if squared else p.sum(2) + g.sum(2)) + 1e-5
        error = 1 - (2 * (p * g).sum(2) + 1e-5) / denominator
        return error.mean() + F.cross_entropy(x, target)

    for squared in (False, True):
        fixed = hessian_vector(lambda x: DiceCELoss(squared_pred=squared, backend="torch")(x, target))
        expected = hessian_vector(lambda x: plain(x, squared))
        torch.testing.assert_close(fixed, expected, rtol=1e-9, atol=1e-13)


def test_dice_and_cross_entropy_propagate_gradients_into_a_soft_target_on_the_torch_backend():
    gen = torch.Generator().manual_seed(3)
    z = torch.randn(1, 3, 7, dtype=torch.float64, generator=gen)
    soft = torch.rand(1, 3, 7, dtype=torch.float64, generator=gen)
    soft = soft / soft.sum(1, keepdim=True)

    def dice_ce(t):
        return DiceCELoss(target_format="one-hot", backend="torch")(z, t)

    t = soft.clone().requires_grad_()
    (got,) = torch.autograd.grad(dice_ce(t), t)
    r = soft.clone().requires_grad_()
    p = z.softmax(1)
    error = 1 - (2 * (p * r).sum(2) + 1e-5) / (p.sum(2) + r.sum(2) + 1e-5)
    plain = error.mean() + F.cross_entropy(z, r)
    (expected,) = torch.autograd.grad(plain, r)
    torch.testing.assert_close(got, expected, rtol=1e-11, atol=1e-14)


@pytest.mark.parametrize("backend", BACKENDS)
def test_losses_and_gradients_stay_finite_beyond_the_range_of_the_exponential(backend):
    # exp underflows for a gap above 745 (binary64) or 104 (binary32); nothing may become NaN or inf, for every
    # softmax loss, in particular the focal powers q ** gamma for 0 < gamma < 1.
    device = "cuda" if backend == "triton" else "cpu"
    for dtype, magnitude in ((torch.float64, 800.0), (torch.float32, 150.0)):
        gen = torch.Generator().manual_seed(0)
        target = torch.randint(3, (2, 16), generator=gen)
        target[:, :3] = torch.arange(3)
        z = torch.randn(2, 3, 16, generator=gen) + magnitude * F.one_hot(target, 3).movedim(-1, 1)
        modules = [DiceLoss(backend=backend), DiceLoss(squared_pred=True, backend=backend),
                   TverskyLoss(alpha=0.3, beta=0.7, backend=backend), GeneralizedDiceLoss(backend=backend),
                   DiceCELoss(backend=backend), FocalLoss(gamma=0.5, backend=backend),
                   FocalLoss(gamma=2.0, backend=backend), FocalTverskyLoss(gamma=0.75, backend=backend),
                   DiceFocalLoss(gamma=0.5, backend=backend)]
        for module in modules:
            x = z.to(dtype).to(device).requires_grad_()
            loss = module(x, target.to(device))
            (grad,) = torch.autograd.grad(loss, x)
            assert torch.isfinite(loss) and torch.isfinite(grad).all(), type(module).__name__


# -- the other losses that use the softmax -------------------------------------------------------------------


def test_topk_cross_entropy_and_lovasz_softmax_gradients_survive_a_dominant_class():
    # TopKCELoss(fraction=1) is the mean of -log p_t. At a logit gap of 40 its gradient is of the order 1e-18 per
    # logit and was exactly 0.
    z = torch.tensor([[[40.0, -40.0], [0.0, 0.0], [-40.0, 40.0]]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[0, 2]])
    loss = TopKCELoss(fraction=1.0)(z, target)
    (grad,) = torch.autograd.grad(loss, z)
    with mpmath.workdps(DIGITS):
        e = [mp.mpf(1), mp.exp(mp.mpf(-40)), mp.exp(mp.mpf(-80))]  # exp(z - max) for (40, 0, -40)
        total = mp.fsum(e)
        value = mp.log(total)
        # dz_c = (p_c - onehot_c) / 2 for each of the two voxels; the second is the mirror image of the first
        first = [-mp.fsum(e[1:]) / total / 2, e[1] / total / 2, e[2] / total / 2]
    assert _relative(loss.item(), value) <= 8 * U[torch.float64]
    for c in range(3):
        assert _relative(grad[0, c, 0].item(), first[c]) <= 8 * U[torch.float64]
        assert _relative(grad[0, 2 - c, 1].item(), first[c]) <= 8 * U[torch.float64]


def test_skeleton_recall_value_and_gradient_at_saturated_logits():
    # 1 - (covered + smooth) / (skeleton + smooth) with covered = skeleton to within 1e-34 (a logit gap of 80).
    from exacta.losses.topology import skeleton_recall_target

    target = torch.zeros(1, 7, 7, dtype=torch.long)
    target[:, 3, 1:6] = 1
    skeleton = skeleton_recall_target(target)
    gen = torch.Generator().manual_seed(0)
    sign = 2.0 * target.double() - 1.0  # +1 on the structure, -1 elsewhere
    z = torch.randn(1, 2, 7, 7, dtype=torch.float64, generator=gen) + 40.0 * torch.stack((-sign, sign), 1)
    z = z.float().double().requires_grad_()
    loss = SkeletonRecallLoss(smooth=1e-5)(z, target, target_skeleton=skeleton)
    (grad,) = torch.autograd.grad(loss, z)
    with mpmath.workdps(DIGITS):
        zs = [mp.mpf(v) for v in z.detach().flatten().tolist()]
        covered_mask = skeleton[0].eq(1).flatten().tolist()

        def value(x):
            cov = mp.mpf(0)
            for v in range(49):
                top = max(x[v], x[49 + v])
                e0, e1 = mp.exp(x[v] - top), mp.exp(x[49 + v] - top)
                if covered_mask[v]:
                    cov += e1 / (e0 + e1)
            count = sum(covered_mask)
            return 1 - (cov + mp.mpf("1e-5")) / (count + mp.mpf("1e-5"))

        expected = value(zs)
        step = mp.mpf(STEP)
        index = [49 + v for v in range(49) if covered_mask[v]][:3] + [v for v in range(49) if covered_mask[v]][:2]
        reference_grad = {}
        for i in index:
            up, down = list(zs), list(zs)
            up[i] += step
            down[i] -= step
            reference_grad[i] = (value(up) - value(down)) / (2 * step)
    assert 0 < loss.item() < 1e-30
    assert _relative(loss.item(), expected) <= 32 * U[torch.float64]
    flat = grad.flatten().tolist()
    for i, ref in reference_grad.items():
        assert _relative(flat[i], ref) <= 32 * U[torch.float64]


def test_wasserstein_dice_value_at_saturated_logits_is_relatively_accurate():
    # The loss is all_error / (2 TP + all_error + smooth), a ratio of non-negative sums, and 1e-34 at a gap of 80.
    import math

    matrix = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.5], [1.0, 0.5, 0.0]], dtype=torch.float64)
    gen = torch.Generator().manual_seed(0)
    target = torch.randint(3, (2, 12), generator=gen)
    z = (torch.randn(2, 3, 12, dtype=torch.float64, generator=gen) + 80.0 * F.one_hot(target, 3).movedim(-1, 1))
    z = z.float().double().requires_grad_()
    loss = GeneralizedWassersteinDiceLoss(matrix)(z, target)
    with mpmath.workdps(DIGITS):
        p = _probabilities([mp.mpf(v) for v in z.detach().flatten().tolist()], (2, 3, 12))
        m = [[mp.mpf(float(x)) for x in row] for row in matrix]
        total_error, tp, losses = [], [], []
        for b in range(2):
            error = [mp.fsum(p[(b * 3 + c) * 12 + v] * m[c][int(target[b, v])] for c in range(3)) for v in range(12)]
            weight = [1 if int(target[b, v]) != 0 else 0 for v in range(12)]
            numerator = 2 * mp.fsum(w * (1 - e) for w, e in zip(weight, error)) + mp.mpf("1e-5")
            denominator = numerator + mp.fsum(error)
            losses.append(1 - numerator / denominator)
        expected = mp.fsum(losses) / 2
    assert math.isfinite(loss.item()) and 0 < loss.item() < 1e-30
    assert _relative(loss.item(), expected) <= 32 * U[torch.float64]
