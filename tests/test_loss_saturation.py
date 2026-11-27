"""Sigmoid-based losses at saturated logits, against a high-precision reference.

With |z| of 30 or more, ``sigma(z)`` rounds to 1 in binary64 and ``sigma(z) - g``, ``1 - sigma(z)``
and ``sigma(z) * (1 - sigma(z))`` lose every digit, although the loss and its gradient are
perfectly representable (``4e-18`` at ``z = 40``). The references below evaluate the losses and
their analytic gradients with mpmath at 120 working digits (above the 50 that the comparison needs,
so that cancellation at ``|z| = 80`` cannot reach the digits compared), from the exact values of the
float inputs.

The errors are relative: per element for the gradient, which spans many orders of magnitude within
one batch, and for the value of losses that are sums of non-negative terms (BCE, focal). The value of
a Dice or Tversky loss is ``1 - ratio``; it is compared relative to itself too, but only on batches
where the loss is of order one (a loss of ``1e-18`` next to a ratio of one is below the resolution of
that subtraction, whatever the sigmoid does).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from exacta import _logistic, _triton_losses
from exacta.losses import (
    DiceCELoss, DiceFocalLoss, DiceLoss, FocalLoss, FocalTverskyLoss, GeneralizedDiceLoss, TverskyLoss,
)

mpmath = pytest.importorskip("mpmath")
mp = mpmath.mp

DIGITS = 120
U = {torch.float64: 2.0**-53, torch.float32: 2.0**-24}
MAGNITUDES = (10.0, 20.0, 30.0, 40.0, 80.0)
GPU = torch.cuda.is_available() and _triton_losses.available()
BACKENDS = ["torch", pytest.param("triton", marks=pytest.mark.skipif(not GPU, reason="needs CUDA and Triton"))]
DTYPES = [torch.float64, torch.float32]


# -- reference ---------------------------------------------------------------------------------------


def _logistic_parts(zs):
    """sigma(z), sigma(-z) and sigma(z) * sigma(-z), each correctly rounded to the working precision."""
    p = [1 / (1 + mp.exp(-z)) for z in zs]
    q = [1 / (1 + mp.exp(z)) for z in zs]
    return p, q, [a * b for a, b in zip(p, q)]


def _softplus(z):
    return mp.log1p(mp.exp(-abs(z))) + max(z, 0)


def _ref_bce(zs, gs, shape):
    """Mean of softplus(z) - g z over all elements; the gradient is (1 - g) sigma(z) - g sigma(-z)."""
    p, q, _ = _logistic_parts(zs)
    n = len(zs)
    value = mp.fsum(_softplus(z) - g * z for z, g in zip(zs, gs)) / n
    return value, [((1 - g) * a - g * b) / n for g, a, b in zip(gs, p, q)]


def _ref_focal(zs, gs, shape, *, gamma, alpha):
    """Mean of -a g q^gamma log p - b (1 - g) p^gamma log q, with p = sigma(z) and q = sigma(-z)."""
    p, q, _ = _logistic_parts(zs)
    a, b = (1, 1) if alpha is None else (mp.mpf(alpha), 1 - mp.mpf(alpha))
    gamma = mp.mpf(gamma)
    n = len(zs)
    value, grad = [], []
    for z, g, pi, qi in zip(zs, gs, p, q):
        pos_ce, neg_ce = _softplus(-z), _softplus(z)  # -log p and -log q
        value.append(a * g * qi**gamma * pos_ce + b * (1 - g) * pi**gamma * neg_ce)
        grad.append((-a * g * qi**gamma * (qi + gamma * pi * pos_ce)
                     + b * (1 - g) * pi**gamma * (pi + gamma * qi * neg_ce)) / n)
    return mp.fsum(value) / n, grad


def _ref_overlap(zs, gs, shape, *, tversky=None, gamma=1, smooth=1e-5):
    """Mean over (case, class) of (1 - ratio)^gamma, Dice by default and Tversky with (alpha, beta)."""
    batch, classes, voxels = shape
    p, q, dz = _logistic_parts(zs)
    s, gamma = mp.mpf(smooth), mp.mpf(gamma)
    groups = batch * classes
    value, grad = [], [None] * len(zs)
    for k in range(groups):
        idx = range(k * voxels, (k + 1) * voxels)
        inter = mp.fsum(p[i] * gs[i] for i in idx)
        if tversky is None:
            num, den = 2 * inter + s, mp.fsum(p[i] + gs[i] for i in idx) + s
            d_num = {i: 2 * gs[i] for i in idx}
            d_den = {i: mp.mpf(1) for i in idx}
        else:
            alpha, beta = mp.mpf(tversky[0]), mp.mpf(tversky[1])
            fp = mp.fsum(p[i] * (1 - gs[i]) for i in idx)
            fn = mp.fsum(q[i] * gs[i] for i in idx)
            num, den = inter + s, inter + alpha * fp + beta * fn + s
            d_num = {i: gs[i] for i in idx}
            d_den = {i: gs[i] + alpha * (1 - gs[i]) - beta * gs[i] for i in idx}
        error = 1 - num / den
        value.append(error**gamma)
        slope = -gamma * error ** (gamma - 1)
        for i in idx:
            ratio_slope = (d_num[i] * den - num * d_den[i]) / den**2
            grad[i] = slope * ratio_slope / groups * dz[i]
    return mp.fsum(value) / groups, grad


def _ref_generalized(zs, gs, shape, *, smooth=1e-5):
    """Generalized Dice with inverse squared class volumes, one ratio per case."""
    batch, classes, voxels = shape
    p, q, dz = _logistic_parts(zs)
    s = mp.mpf(smooth)
    value, grad = [], [None] * len(zs)
    for b in range(batch):
        groups = [range((b * classes + c) * voxels, (b * classes + c + 1) * voxels) for c in range(classes)]
        volume = [mp.fsum(gs[i] for i in idx) for idx in groups]
        weight = [1 / v**2 for v in volume]
        num = 2 * mp.fsum(w * mp.fsum(p[i] * gs[i] for i in idx) for w, idx in zip(weight, groups)) + s
        den = mp.fsum(w * (mp.fsum(p[i] for i in idx) + v)
                      for w, v, idx in zip(weight, volume, groups)) + s
        value.append(1 - num / den)
        for w, idx in zip(weight, groups):
            for i in idx:
                grad[i] = -(2 * gs[i] * w * den - num * w) / den**2 / batch * dz[i]
    return mp.fsum(value) / batch, grad


@dataclass(frozen=True)
class Spec:
    """A loss under test: its constructor, and the reference terms (weight, function, options) it sums."""

    make: Callable[[str], torch.nn.Module]
    terms: tuple
    floor: float = 0.0  # smallest |value| for which the value is compared relatively (see the module note)
    top32: float = 40.0  # largest |z| in binary32 (focal terms are about exp(-3|z|) there: 1e-26 at 20)


def _bce(backend):
    return DiceCELoss(activation="sigmoid", target_format="one-hot", lambda_dice=0.0, lambda_ce=1.0, backend=backend)


SPECS = {
    "bce": Spec(_bce, ((1.0, _ref_bce, {}),)),
    "focal_gamma0": Spec(
        lambda b: FocalLoss(activation="sigmoid", target_format="one-hot", gamma=0.0, backend=b),
        ((1.0, _ref_bce, {}),)),
    "focal": Spec(
        lambda b: FocalLoss(activation="sigmoid", target_format="one-hot", gamma=2.0, alpha=0.25, backend=b),
        ((1.0, _ref_focal, {"gamma": 2.0, "alpha": 0.25}),), top32=20.0),
    "dice": Spec(
        lambda b: DiceLoss(activation="sigmoid", target_format="one-hot", backend=b),
        ((1.0, _ref_overlap, {}),), floor=0.05),
    "tversky": Spec(
        lambda b: TverskyLoss(alpha=0.3, beta=0.7, activation="sigmoid", target_format="one-hot", backend=b),
        ((1.0, _ref_overlap, {"tversky": (0.3, 0.7)}),), floor=0.05),
    "focal_tversky": Spec(
        lambda b: FocalTverskyLoss(gamma=0.75, alpha=0.3, beta=0.7, activation="sigmoid",
                                   target_format="one-hot", backend=b),
        ((1.0, _ref_overlap, {"tversky": (0.3, 0.7), "gamma": 0.75}),), floor=0.05),
    "generalized_dice": Spec(
        lambda b: GeneralizedDiceLoss(activation="sigmoid", target_format="one-hot", backend=b),
        ((1.0, _ref_generalized, {}),), floor=0.05),
    "dice_bce": Spec(
        lambda b: DiceCELoss(activation="sigmoid", target_format="one-hot", lambda_dice=0.7, lambda_ce=1.3,
                             backend=b),
        ((0.7, _ref_overlap, {}), (1.3, _ref_bce, {})), floor=0.05),
    "dice_focal": Spec(
        lambda b: DiceFocalLoss(activation="sigmoid", target_format="one-hot", gamma=2.0, alpha=0.25,
                                lambda_dice=0.7, lambda_focal=1.3, backend=b),
        ((0.7, _ref_overlap, {}), (1.3, _ref_focal, {"gamma": 2.0, "alpha": 0.25})), floor=0.05),
}
POINTWISE = ("bce", "focal_gamma0", "focal")
OVERLAP = tuple(name for name in SPECS if name not in POINTWISE)


def _reference(spec: Spec, z: torch.Tensor, g: torch.Tensor):
    """Value (mpf) and per-element gradient (list of mpf) from the exact float inputs."""
    with mpmath.workdps(DIGITS):
        zs = [mp.mpf(v) for v in z.double().flatten().tolist()]
        gs = [mp.mpf(v) for v in g.double().flatten().tolist()]
        value, grad = mp.mpf(0), [mp.mpf(0)] * len(zs)
        for weight, function, options in spec.terms:
            v, d = function(zs, gs, tuple(z.shape), **options)
            value += weight * v
            grad = [a + weight * b for a, b in zip(grad, d)]
        return value, grad


def _relative(got: float, reference) -> float:
    with mpmath.workdps(DIGITS):
        return float(abs(mp.mpf(got) - reference) / abs(reference))


def measure(name: str, backend: str, dtype: torch.dtype, z: torch.Tensor, g: torch.Tensor) -> tuple[float, float, float]:
    """Relative errors, in units of the roundoff of ``dtype``: value, worst gradient element, |value|."""
    spec = SPECS[name]
    device = "cuda" if backend == "triton" else "cpu"
    x = z.to(dtype).to(device).requires_grad_()
    loss = spec.make(backend)(x, g.to(dtype).to(device))
    (grad,) = torch.autograd.grad(loss, x)
    assert grad.dtype == dtype and loss.dtype == dtype
    value, reference_grad = _reference(spec, z.to(dtype), g.to(dtype))
    u = U[dtype]
    worst = max(_relative(got, ref) for got, ref in zip(grad.double().flatten().tolist(), reference_grad))
    return _relative(loss.item(), value) / u, worst / u, abs(float(value))


# -- data ----------------------------------------------------------------------------------------------


def _top(name: str, dtype: torch.dtype) -> float:
    return 80.0 if dtype == torch.float64 else SPECS[name].top32


def _batch(name: str, kind: str, dtype: torch.dtype, seed: int,
           shape=(2, 3, 16)) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits and 0/1 targets; every (case, class) holds at least one positive and one negative."""
    gen = torch.Generator().manual_seed(seed)
    top = _top(name, dtype)
    sign = torch.where(torch.rand(shape, generator=gen) < 0.5, -1.0, 1.0)
    if kind == "saturated":
        grid = torch.tensor([m for m in MAGNITUDES if m <= top], dtype=torch.float64)
        z = sign * grid[torch.randint(len(grid), shape, generator=gen)]
    else:
        saturated = torch.rand(shape, generator=gen) < 0.5
        low = min(30.0, top / 2)
        z = torch.where(saturated, sign * (low + (top - low) * torch.rand(shape, generator=gen)),
                        3.0 * torch.randn(shape, generator=gen))
    g = (torch.rand(shape, generator=gen) < 0.5).to(torch.float64)
    g[..., 0], g[..., 1] = 1.0, 0.0
    return z.to(dtype), g.to(dtype)


# Bounds, in units of the roundoff u of the dtype, on the value and on the gradient of every element.
# They cover the rounding of the sums behind the Dice coefficients (16 voxels per class here) and the
# 1 to 2 ulp of the exponentials; the errors measured are at most 7u in binary64 on both backends and in
# binary32 on PyTorch. The fused binary32 kernels use the fast exponential, whose argument rounding costs
# about |z| u (43u at |z| = 40), hence their looser bound. Before the fix the same quantities were off by
# 100% (about 1e16 u in binary64) wherever a probability had rounded to one.
BOUND = {("torch", torch.float64): 16.0, ("triton", torch.float64): 16.0,
         ("torch", torch.float32): 16.0, ("triton", torch.float32): 96.0}


# -- tests ---------------------------------------------------------------------------------------------


def _pointwise_cases(name: str, dtype: torch.dtype):
    """(logit, target) pairs; soft targets only for BCE."""
    soft = (2.0**-20, 1.0 - 2.0**-20) if name == "bce" else ()
    for magnitude, sign, target in itertools.product(MAGNITUDES, (-1.0, 1.0), (0.0, 1.0, *soft)):
        if magnitude <= _top(name, dtype):
            yield sign * magnitude, target


@pytest.mark.parametrize("dtype", DTYPES, ids=["binary64", "binary32"])
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", POINTWISE)
def test_pointwise_loss_and_gradient_are_relatively_accurate_at_every_logit(name, backend, dtype):
    bound = BOUND[backend, dtype]
    failures = []
    for z, g in _pointwise_cases(name, dtype):
        value_error, grad_error, _ = measure(name, backend, dtype, torch.full((1, 1, 1), z), torch.full((1, 1, 1), g))
        if value_error > bound or grad_error > bound:
            failures.append(f"z={z:+g} g={g:g}: value {value_error:.3g} u, gradient {grad_error:.3g} u")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", ["saturated", "mixed"])
@pytest.mark.parametrize("dtype", DTYPES, ids=["binary64", "binary32"])
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", list(SPECS))
def test_batches_with_saturated_logits_are_relatively_accurate(name, backend, dtype, kind, seed):
    z, g = _batch(name, kind, dtype, seed)
    value_error, grad_error, magnitude = measure(name, backend, dtype, z, g)
    assert grad_error <= BOUND[backend, dtype], f"gradient: {grad_error:.3g} u"
    if magnitude >= SPECS[name].floor:  # 1 - ratio is not relative-accurate for tiny losses
        assert value_error <= BOUND[backend, dtype], f"value: {value_error:.3g} u"


def test_overlap_values_are_compared_only_where_one_minus_ratio_has_digits_to_spare():
    # Guards the 'floor' above: the batches used must not be vacuous for the value check.
    for name in OVERLAP:
        for kind, seed in itertools.product(("saturated", "mixed"), (0, 1, 2)):
            assert measure(name, "torch", torch.float64, *_batch(name, kind, torch.float64, seed))[2] >= SPECS[name].floor


@pytest.mark.parametrize("backend", BACKENDS)
def test_perfectly_saturated_overlap_gradient_is_not_lost(backend):
    # The value is 1 - ratio with ratio rounded to one, but the gradient is a product of
    # sigma(z) * sigma(-z) and must not vanish: it is about 4e-18 here, and was exactly 0.
    device = "cuda" if backend == "triton" else "cpu"
    z = torch.tensor([[[40.0, -40.0, 40.0, -40.0]]], dtype=torch.float64, device=device, requires_grad=True)
    g = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]], dtype=torch.float64, device=device)
    (grad,) = torch.autograd.grad(DiceLoss(activation="sigmoid", target_format="one-hot", backend=backend)(z, g), z)
    reference = _reference(SPECS["dice"], z.detach().cpu(), g.cpu())[1]
    for got, ref in zip(grad.flatten().tolist(), reference):
        assert _relative(got, ref) <= 16 * U[torch.float64]


@pytest.mark.parametrize("dtype", DTYPES, ids=["binary64", "binary32"])
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", list(SPECS))
def test_losses_and_gradients_stay_finite_beyond_the_range_of_the_exponential(name, backend, dtype):
    # sigma underflows to 0 for z below -745 (binary64) or -104 (binary32); nothing may become NaN or inf,
    # in particular not the focal powers p ** gamma for 0 < gamma < 1.
    device = "cuda" if backend == "triton" else "cpu"
    gen = torch.Generator().manual_seed(0)
    magnitude = torch.tensor([30.0, 100.0, 150.0, 800.0])[torch.randint(4, (2, 3, 16), generator=gen)]
    z = (torch.where(torch.rand(2, 3, 16, generator=gen) < 0.5, -1.0, 1.0) * magnitude).to(dtype).to(device)
    g = (torch.rand(2, 3, 16, generator=gen) < 0.5).to(dtype).to(device)
    g[..., 0], g[..., 1] = 1.0, 0.0
    modules = [SPECS[name].make(backend)]
    if name == "focal":
        modules += [FocalLoss(activation="sigmoid", target_format="one-hot", gamma=gamma, alpha=0.25, backend=backend)
                    for gamma in (0.0, 0.5)]
    for module in modules:
        x = z.clone().requires_grad_()
        loss = module(x, g)
        (grad,) = torch.autograd.grad(loss, x)
        assert torch.isfinite(loss) and torch.isfinite(grad).all()


@pytest.mark.parametrize("name", list(SPECS))
def test_reference_gradient_matches_a_numerical_derivative(name):
    # Validates the analytic gradients above, at moderate logits and with a 1e-30 step at 120 digits.
    z, g = _batch("bce", "mixed", torch.float64, 5, shape=(1, 2, 4))
    z = 2.0 * z.clamp(-3, 3)
    spec = SPECS[name]
    _, grad = _reference(spec, z, g)
    with mpmath.workdps(DIGITS):
        step = mp.mpf(10) ** -30
        base = [mp.mpf(v) for v in z.flatten().tolist()]
        gs = [mp.mpf(v) for v in g.flatten().tolist()]

        def value(zs):
            return mp.fsum(weight * function(zs, gs, tuple(z.shape), **options)[0]
                           for weight, function, options in spec.terms)

        for i, expected in enumerate(grad):
            up, down = list(base), list(base)
            up[i] += step
            down[i] -= step
            assert abs((value(up) - value(down)) / (2 * step) - expected) < mp.mpf(10) ** -25 * abs(expected)


def _plain_dice(module):
    return lambda z, g: module(z.sigmoid(), g)


def _plain_bce(z, g):
    """The behavior before the fix: PyTorch's own binary cross-entropy with logits."""
    return F.binary_cross_entropy_with_logits(z, g)


def _plain_focal(z, g, gamma=2.0, alpha=0.25):
    ce = F.binary_cross_entropy_with_logits(z, g, reduction="none")
    p = z.sigmoid()
    modulation = g * (1 - p) ** gamma + (1 - g) * p**gamma
    return ((g * alpha + (1 - g) * (1 - alpha)) * modulation * ce).mean()


# The same losses with torch.sigmoid applied outside and activation="none", which is the previous forward.
PLAIN = {
    "bce": _plain_bce,
    "focal": _plain_focal,
    "dice": _plain_dice(DiceLoss(activation="none", target_format="one-hot")),
    "tversky": _plain_dice(TverskyLoss(alpha=0.3, beta=0.7, activation="none", target_format="one-hot")),
    "focal_tversky": _plain_dice(
        FocalTverskyLoss(gamma=0.75, alpha=0.3, beta=0.7, activation="none", target_format="one-hot")),
    "generalized_dice": _plain_dice(GeneralizedDiceLoss(activation="none", target_format="one-hot")),
    "dice_bce": lambda z, g: (0.7 * _plain_dice(DiceLoss(activation="none", target_format="one-hot"))(z, g)
                              + 1.3 * _plain_bce(z, g)),
    "dice_focal": lambda z, g: (0.7 * _plain_dice(DiceLoss(activation="none", target_format="one-hot"))(z, g)
                                + 1.3 * _plain_focal(z, g)),
}


@pytest.mark.parametrize("name", list(PLAIN))
def test_moderate_logits_agree_with_the_plain_composition(name):
    # Where nothing saturates, values and gradients must stay those of the composition with
    # torch.sigmoid and F.binary_cross_entropy_with_logits, to binary64 rounding.
    z, g = _batch(name, "mixed", torch.float64, 3, shape=(2, 3, 32))
    z = z.clamp(-6, 6)
    results = []
    for loss_of in (SPECS[name].make("torch"), PLAIN[name]):
        x = z.clone().requires_grad_()
        loss = loss_of(x, g)
        (grad,) = torch.autograd.grad(loss, x)
        results.append((loss.item(), grad))
    (fixed, fixed_grad), (plain, plain_grad) = results
    assert abs(fixed - plain) <= 8 * U[torch.float64] * abs(plain)
    assert (fixed_grad - plain_grad).abs().max() <= 1e-13 * plain_grad.abs().max()


def test_the_internal_sigmoid_and_bce_have_correct_first_and_second_derivatives():
    z = torch.tensor([-5.0, -0.3, 0.0, 0.7, 4.0], dtype=torch.float64, requires_grad=True)
    g = torch.tensor([0.0, 1.0, 0.25, 0.5, 1.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(_logistic.sigmoid, (z,))
    assert torch.autograd.gradgradcheck(_logistic.sigmoid, (z,))
    assert torch.autograd.gradcheck(_logistic.bce_with_logits, (z, g))  # also with respect to the target
    assert torch.autograd.gradgradcheck(_logistic.bce_with_logits, (z, g))


def test_sigmoid_losses_support_double_backward_on_the_torch_backend():
    gen = torch.Generator().manual_seed(7)
    z = torch.randn(2, 3, 20, dtype=torch.float64, generator=gen)
    g = (torch.rand(2, 3, 20, dtype=torch.float64, generator=gen) < 0.5).to(torch.float64)
    v = torch.randn(2, 3, 20, dtype=torch.float64, generator=gen)

    def hessian_vector(loss_of):
        x = z.clone().requires_grad_()
        (first,) = torch.autograd.grad(loss_of(x), x, create_graph=True)
        (second,) = torch.autograd.grad((first * v).sum(), x)
        return second

    fixed = hessian_vector(lambda x: DiceCELoss(activation="sigmoid", target_format="one-hot", backend="torch")(x, g))
    plain = hessian_vector(lambda x: DiceLoss(activation="none", target_format="one-hot")(x.sigmoid(), g)
                           + F.binary_cross_entropy_with_logits(x, g))
    torch.testing.assert_close(fixed, plain, rtol=1e-10, atol=1e-13)


def test_bce_propagates_gradients_into_a_soft_target_on_the_torch_backend():
    z = torch.randn(1, 2, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    t = torch.rand(1, 2, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(3)).requires_grad_()
    loss = _bce("torch")(z, t)
    (got,) = torch.autograd.grad(loss, t)
    reference = t.detach().clone().requires_grad_()
    (expected,) = torch.autograd.grad(F.binary_cross_entropy_with_logits(z, reference), reference)
    torch.testing.assert_close(got, expected, rtol=1e-13, atol=0)


def test_masked_bce_with_class_weights_keeps_its_value_and_gradient():
    gen = torch.Generator().manual_seed(4)
    z = (3.0 * torch.randn(2, 3, 12, dtype=torch.float64, generator=gen)).requires_grad_()
    target = torch.randint(0, 3, (2, 12), generator=gen)
    mask = torch.rand(2, 12, generator=gen) < 0.7
    weight = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    loss = DiceCELoss(activation="sigmoid", lambda_dice=0.0, class_weight=weight, backend="torch")(z, target, mask=mask)
    onehot = F.one_hot(target, 3).movedim(-1, 1).double()
    terms = F.binary_cross_entropy_with_logits(
        torch.where(mask[:, None], z, 0.0), torch.where(mask[:, None], onehot, 0.0), reduction="none"
    ) * weight[None, :, None]
    expected = torch.where(mask[:, None], terms, 0.0).sum() / (mask.sum() * 3)
    assert abs(loss.item() - expected.item()) <= 8 * U[torch.float64] * abs(expected.item())
