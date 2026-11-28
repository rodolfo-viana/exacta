"""Hausdorff, Lovasz and boundary losses at saturated logits, against a high-precision reference.

* ``HausdorffDTLoss`` differentiates ``(p - y)**2``, which is ``0`` where ``p`` rounds to ``y`` (a logit gap of
  37 in binary64, 17 in binary32) although ``p - y = -q`` with ``q = 1 - p = 4e-18``.
* ``LovaszSoftmaxLoss`` sorts the errors ``|g - p|``, which round to ``0`` there: the order of the saturated voxels
  (and so the increments of the Jaccard sequence assigned to them, that is, the gradient) was the index order, and
  the value ``0``.
* ``LovaszHingeLoss`` works on the logits (``1 - z (2t - 1)``, exact by Sterbenz's lemma near the margin and
  clipped to zero beyond it) and ``BoundaryLoss`` is the signed sum ``mean(p * phi)``: both are checked here to
  show that they never lost digits at saturation.

The references evaluate the textbook definitions in mpmath at 160 working digits from the exact float inputs
(softmax in mpmath, errors and weights from the definition, Jaccard increments in exact rational arithmetic, the
sort by the exact errors), and chain the analytic gradient through the softmax or sigmoid in mpmath. The distance
maps are data (taken from ``distance_maps``, of the exact masks of the reference). Errors are relative, in units of
the roundoff ``u`` of the dtype.
"""

from __future__ import annotations

import functools
import itertools
from fractions import Fraction

import pytest
import torch
import torch.nn.functional as F

from exacta import _triton_losses
from exacta.losses import BoundaryLoss, HausdorffDTLoss, LovaszHingeLoss, LovaszSoftmaxLoss, distance_maps

mpmath = pytest.importorskip("mpmath")
mp = mpmath.mp

DIGITS = 160
U = {torch.float64: 2.0**-53, torch.float32: 2.0**-24}
GAPS = (10.0, 20.0, 30.0, 40.0, 80.0)
CUDA = torch.cuda.is_available() and _triton_losses.available()
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not CUDA, reason="needs CUDA and Triton"))]
DTYPES = [torch.float64, torch.float32]
KINDS = ["saturated", "mixed", "moderate"]


# -- data ----------------------------------------------------------------------------------------------


def _images(kind: str, dtype: torch.dtype, seed: int, *, top: float, shape=(2, 3, 6, 6)):
    """Logits (exact in binary32) and class labels of small images.

    ``saturated``: every voxel has one class above the others by a gap of 10 to ``top``, and it is the label;
    ``mixed``: half of the voxels saturated (a fifth of those confidently wrong), the others moderate;
    ``moderate``: no saturation. Every class is present in every image.
    """
    batch, classes = shape[:2]
    gen = torch.Generator().manual_seed(seed)
    labels = torch.randint(classes, (batch, *shape[2:]), generator=gen)
    flat = labels.flatten(1)
    flat[:, :classes] = torch.arange(classes)
    labels = flat.reshape(labels.shape)
    grid = torch.tensor([g for g in GAPS if g <= top], dtype=torch.float64)
    gap = grid[torch.randint(len(grid), labels.shape, generator=gen)]
    dominant = labels.clone()
    noise = torch.randn(*shape, generator=gen, dtype=torch.float64)
    if kind == "mixed":
        wrong = torch.rand(labels.shape, generator=gen) < 0.2
        dominant = torch.where(wrong, (labels + 1 + torch.randint(classes - 1, labels.shape, generator=gen)) % classes,
                               labels)
        gap = torch.where(torch.rand(labels.shape, generator=gen) < 0.5, gap, torch.zeros_like(gap))
    if kind == "moderate":
        gap = torch.zeros_like(gap)
    z = (noise if kind == "saturated" else 3.0 * noise) + gap[:, None] * F.one_hot(dominant, classes).movedim(-1, 1)
    return z.float().double(), labels


# -- mpmath references ---------------------------------------------------------------------------------


def _softmax_columns(z):
    """p[b][c][v] from logits z[b][c][v] (nested lists of mpf), per voxel."""
    batch, classes, voxels = len(z), len(z[0]), len(z[0][0])
    p = [[[None] * voxels for _ in range(classes)] for _ in range(batch)]
    for b, v in itertools.product(range(batch), range(voxels)):
        top = max(z[b][c][v] for c in range(classes))
        e = [mp.exp(z[b][c][v] - top) for c in range(classes)]
        total = mp.fsum(e)
        for c in range(classes):
            p[b][c][v] = e[c] / total
    return p


def _nested(t: torch.Tensor):
    batch, classes = t.shape[:2]
    flat = t.reshape(batch, classes, -1).double().tolist()
    return [[[mp.mpf(x) for x in row] for row in case] for case in flat]


def _chain_softmax(p, dp):
    """dL/dz[b][k][v] = p_k (dp_k - sum_c p_c dp_c)."""
    batch, classes, voxels = len(p), len(p[0]), len(p[0][0])
    out = [[[None] * voxels for _ in range(classes)] for _ in range(batch)]
    for b, v in itertools.product(range(batch), range(voxels)):
        mean = mp.fsum(p[b][c][v] * dp[b][c][v] for c in range(classes))
        for k in range(classes):
            out[b][k][v] = p[b][k][v] * (dp[b][k][v] - mean)
    return out


def _chain_sigmoid(z, dp):
    out = [[[None] * len(z[0][0]) for _ in z[0]] for _ in z]
    for b, c, v in itertools.product(range(len(z)), range(len(z[0])), range(len(z[0][0]))):
        e = mp.exp(-abs(z[b][c][v]))
        small = e / (1 + e)
        out[b][c][v] = small / (1 + e) * dp[b][c][v]  # sigma (1 - sigma) = e / (1 + e)^2
    return out


def _sigmoid_columns(z):
    batch, classes, voxels = len(z), len(z[0]), len(z[0][0])
    return [[[1 / (1 + mp.exp(-z[b][c][v])) for v in range(voxels)] for c in range(classes)] for b in range(batch)]


def _activation(z, activation):
    return _softmax_columns(z) if activation == "softmax" else _sigmoid_columns(z)


def _chain(z, p, dp, activation):
    return _chain_softmax(p, dp) if activation == "softmax" else _chain_sigmoid(z, dp)


def _flatten(nested):
    return [x for case in nested for row in case for x in row]


@functools.lru_cache(maxsize=None)
def _hausdorff_reference(z_bytes: bytes, labels_bytes: bytes, shape: tuple, dtype: torch.dtype, activation: str):
    """Value and gradient of ``mean((p - y)**2 (d_p**2 + d_y**2))`` over the foreground channels."""
    z = torch.frombuffer(bytearray(z_bytes), dtype=dtype).reshape(shape)
    labels = torch.frombuffer(bytearray(labels_bytes), dtype=torch.int64).reshape(shape[0], *shape[2:])
    with mpmath.workdps(DIGITS):
        zs = _nested(z)
        p = _activation(zs, activation)
        truth = F.one_hot(labels, shape[1]).movedim(-1, 1).double()
        target_dt = distance_maps(truth[:, 1:], spacing=1.0)
        mask = torch.tensor([[[p[b][c][v] > mp.mpf(1) / 2 for v in range(len(p[0][0]))] for c in range(1, shape[1])]
                             for b in range(shape[0])]).reshape(shape[0], shape[1] - 1, *shape[2:])
        pred_dt = distance_maps(mask, spacing=1.0)
        weight = (pred_dt.pow(2.0) + target_dt.pow(2.0)).reshape(shape[0], shape[1] - 1, -1).tolist()
        y = truth.reshape(shape[0], shape[1], -1).tolist()
        count = shape[0] * (shape[1] - 1) * len(p[0][0])
        value = mp.mpf(0)
        dp = [[[mp.mpf(0)] * len(p[0][0]) for _ in range(shape[1])] for _ in range(shape[0])]
        for b, c, v in itertools.product(range(shape[0]), range(1, shape[1]), range(len(p[0][0]))):
            gap = p[b][c][v] - int(y[b][c][v])
            w = mp.mpf(weight[b][c - 1][v])
            value += gap**2 * w
            dp[b][c][v] = 2 * gap * w / count
        return value / count, _flatten(_chain(zs, p, dp, activation))


@functools.lru_cache(maxsize=None)
def _boundary_reference(z_bytes: bytes, labels_bytes: bytes, shape: tuple, dtype: torch.dtype, activation: str):
    """Value, gradient and the mean of ``|p * phi|`` of ``mean(p * phi)`` over the foreground channels, phi the
    signed distance map."""
    z = torch.frombuffer(bytearray(z_bytes), dtype=dtype).reshape(shape)
    labels = torch.frombuffer(bytearray(labels_bytes), dtype=torch.int64).reshape(shape[0], *shape[2:])
    with mpmath.workdps(DIGITS):
        zs = _nested(z)
        p = _activation(zs, activation)
        truth = F.one_hot(labels, shape[1]).movedim(-1, 1).double()
        phi = distance_maps(truth[:, 1:], spacing=1.0, signed=True).reshape(shape[0], shape[1] - 1, -1).tolist()
        voxels = len(p[0][0])
        count = shape[0] * (shape[1] - 1) * voxels
        dp = [[[mp.mpf(0)] * voxels for _ in range(shape[1])] for _ in range(shape[0])]
        terms = []
        for b, c, v in itertools.product(range(shape[0]), range(1, shape[1]), range(voxels)):
            terms.append(p[b][c][v] * mp.mpf(phi[b][c - 1][v]))
            dp[b][c][v] = mp.mpf(phi[b][c - 1][v]) / count
        return mp.fsum(terms) / count, _flatten(_chain(zs, p, dp, activation)), mp.fsum(abs(t) for t in terms) / count


def exact_increments(fg_sorted: list[int]) -> list[Fraction]:
    """Delta J_k of the Jaccard sequence of one sorted vector of valid positions, in exact arithmetic."""
    total = sum(fg_sorted)
    out, previous, cum_fg, cum_bg = [], Fraction(0), 0, 0
    for f in fg_sorted:
        cum_fg += f
        cum_bg += 1 - f
        jaccard = 1 - Fraction(total - cum_fg, total + cum_bg)
        out.append(jaccard - previous)
        previous = jaccard
    return out


def _lovasz_pooled(errors, foreground):
    """Lovasz value of one class and the derivative with respect to each error: sorted by the exact errors."""
    order = sorted(range(len(errors)), key=lambda i: (-errors[i], i))
    increments = exact_increments([int(foreground[i]) for i in order])
    value = mp.fsum(errors[i] * mp.mpf(inc.numerator) / mp.mpf(inc.denominator) for i, inc in zip(order, increments))
    derivative = [None] * len(errors)
    for i, inc in zip(order, increments):
        derivative[i] = mp.mpf(inc.numerator) / mp.mpf(inc.denominator)
    return value, derivative


@functools.lru_cache(maxsize=None)
def _lovasz_softmax_reference(z_bytes: bytes, labels_bytes: bytes, shape: tuple, dtype: torch.dtype):
    """``LovaszSoftmaxLoss()``: classes present, pooled over the batch, mean over the classes."""
    z = torch.frombuffer(bytearray(z_bytes), dtype=dtype).reshape(shape)
    labels = torch.frombuffer(bytearray(labels_bytes), dtype=torch.int64).reshape(shape[0], *shape[2:])
    with mpmath.workdps(DIGITS):
        zs = _nested(z)
        p = _softmax_columns(zs)
        batch, classes, voxels = shape[0], shape[1], len(p[0][0])
        flat_labels = labels.reshape(batch, -1).tolist()
        dp = [[[mp.mpf(0)] * voxels for _ in range(classes)] for _ in range(batch)]
        values, tightest = [], 1.0
        for c in range(classes):
            index = [(b, v) for b in range(batch) for v in range(voxels)]
            foreground = [flat_labels[b][v] == c for b, v in index]
            if not any(foreground):
                continue
            errors = [abs(int(f) - p[b][c][v]) for (b, v), f in zip(index, foreground)]
            ranked = sorted(errors, reverse=True)
            tightest = min(tightest, min((float((a - b) / a) for a, b in zip(ranked, ranked[1:]) if a > 0),
                                         default=1.0))
            value, derivative = _lovasz_pooled(errors, foreground)
            values.append(value)
            for (b, v), f, d in zip(index, foreground, derivative):
                dp[b][c][v] = (-d if f else d)  # d |g - p| / dp = -1 for a foreground voxel, +1 for a background one
        n = len(values)
        dp = [[[x / n for x in row] for row in case] for case in dp]
        return mp.fsum(values) / n, _flatten(_chain_softmax(p, dp)), tightest


@functools.lru_cache(maxsize=None)
def _lovasz_hinge_reference(z_bytes: bytes, labels_bytes: bytes, shape: tuple, dtype: torch.dtype):
    """``LovaszHingeLoss(multilabel=True, per_image=False)`` on (B, C, V) logits with 0/1 targets (1 if the label is c)."""
    z = torch.frombuffer(bytearray(z_bytes), dtype=dtype).reshape(shape)
    labels = torch.frombuffer(bytearray(labels_bytes), dtype=torch.int64).reshape(shape[0], *shape[2:])
    with mpmath.workdps(DIGITS):
        batch, classes = shape[:2]
        flat = z.reshape(batch, classes, -1).double().tolist()
        flat_labels = labels.reshape(batch, -1).tolist()
        voxels = len(flat[0][0])
        grad = [[[mp.mpf(0)] * voxels for _ in range(classes)] for _ in range(batch)]
        values = []
        for c in range(classes):
            index = [(b, v) for b in range(batch) for v in range(voxels)]
            foreground = [flat_labels[b][v] == c for b, v in index]
            if not any(foreground):
                continue
            signs = [1 if f else -1 for f in foreground]
            raw = [1 - mp.mpf(flat[b][c][v]) * s for (b, v), s in zip(index, signs)]
            errors = [max(e, mp.mpf(0)) for e in raw]
            value, derivative = _lovasz_pooled(errors, foreground)
            values.append(value)
            for (b, v), s, e, d in zip(index, signs, raw, derivative):
                grad[b][c][v] = -s * d if e > 0 else mp.mpf(0)
        n = len(values)
        return mp.fsum(values) / n, [x / n for x in _flatten(grad)]


def _relative(got: float, reference) -> float:
    with mpmath.workdps(DIGITS):
        if reference == 0:
            return 0.0 if got == 0 else float("inf")
        return float(abs(mp.mpf(got) - reference) / abs(reference))


def _cached(function, z: torch.Tensor, labels: torch.Tensor, *extra):
    z = z.detach().contiguous()
    return function(z.numpy().tobytes(), labels.contiguous().numpy().tobytes(), tuple(z.shape), z.dtype, *extra)


# -- measurement ---------------------------------------------------------------------------------------


def _errors(loss_of, reference, z: torch.Tensor, labels: torch.Tensor, dtype, device):
    """Relative errors in units of u: value, worst gradient element, worst gradient element relative to the
    largest of its voxel; and |value|."""
    x = z.to(dtype).to(device).clone().requires_grad_()
    loss = loss_of(x, labels.to(device))
    (grad,) = torch.autograd.grad(loss, x)
    assert grad.dtype == dtype and loss.dtype == dtype
    value, expected = reference
    tiny = 1e-290 if dtype == torch.float64 else 1e-36
    batch, classes = z.shape[:2]
    got = grad.double().flatten().tolist()
    with mpmath.workdps(DIGITS):
        magnitude = [abs(r) for r in expected]
        normal = [m > tiny * 1e4 for m in magnitude]
        absolute = [abs(mp.mpf(a) - r) for a, r in zip(got, expected)]
        element = [float(a / m) for a, m, ok in zip(absolute, magnitude, normal) if ok]
        voxels = len(expected) // (batch * classes)
        voxel = []
        for i, ok in enumerate(normal):
            if ok:
                b, v = i // (classes * voxels), i % voxels
                scale = max(magnitude[(b * classes + c) * voxels + v] for c in range(classes))
                voxel.append(float(absolute[i] / scale))
    assert len(element) >= len(got) // 2, "too many reference gradients outside the range of the dtype"
    u = U[dtype]
    return _relative(loss.item(), value) / u, max(element) / u, max(voxel) / u, abs(float(value))


# Bounds in units of u: value and, on saturated batches, every gradient element; and, relative to the largest element
# of the voxel (the softmax Jacobian makes a moderate voxel's elements differences of their neighbours), the gradient
# of mixed and moderate batches.
BOUND = {torch.float64: 32.0, torch.float32: 96.0}
BOUND_VOXEL = 512.0


def _top(dtype: torch.dtype) -> float:
    return 80.0 if dtype == torch.float64 else 20.0


# -- tests ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["binary64", "binary32"])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("activation", ["softmax", "sigmoid"])
def test_hausdorff_value_and_gradient_are_relatively_accurate(activation, device, dtype, kind, seed):
    z, labels = _images(kind, dtype, seed, top=_top(dtype))
    reference = _cached(_hausdorff_reference, z.to(dtype), labels, activation)
    loss = HausdorffDTLoss(activation=activation)
    value_error, element_error, voxel_error, _ = _errors(
        lambda x, t: loss(x, t, spacing=1.0), reference, z, labels, dtype, device)
    assert value_error <= BOUND[dtype], f"value: {value_error:.3g} u"
    assert voxel_error <= (BOUND[dtype] if kind == "saturated" else BOUND_VOXEL), f"gradient: {voxel_error:.3g} u"
    if kind == "saturated" and activation == "sigmoid":
        assert element_error <= BOUND[dtype], f"gradient element: {element_error:.3g} u"


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["binary64", "binary32"])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("activation", ["softmax", "sigmoid"])
def test_boundary_value_is_as_accurate_as_a_signed_sum_can_be(activation, device, dtype, kind, seed):
    # mean(p * phi) is a sum of terms of both signs (phi < 0 inside the target): its error is a few roundoffs of the
    # sum of the magnitudes mean(|p * phi|), which is the conditioning of the sum itself (a relative error of that
    # size is all a signed sum can have, whatever its formula, where the terms cancel); where it does not cancel,
    # as for a softmax prediction that is right everywhere, the relative error is a few roundoffs too.
    z, labels = _images(kind, dtype, seed, top=_top(dtype))
    value, expected, scale = _cached(_boundary_reference, z.to(dtype), labels, activation)
    loss = BoundaryLoss(activation=activation)
    value_error, _, voxel_error, magnitude = _errors(
        lambda x, t: loss(x, t, spacing=1.0), (value, expected), z, labels, dtype, device)
    assert value_error * magnitude <= BOUND[dtype] * float(scale), f"value: {value_error:.3g} u of {magnitude:.3g}"
    if kind == "saturated" and activation == "softmax":
        assert value_error <= BOUND[dtype], f"value: {value_error:.3g} u"
    assert voxel_error <= (BOUND[dtype] if kind == "saturated" else BOUND_VOXEL), f"gradient: {voxel_error:.3g} u"


def _lovasz_images(kind, dtype, seed):
    return _images(kind, dtype, seed, top=_top(dtype), shape=(1, 3, 4, 6))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["binary64", "binary32"])
@pytest.mark.parametrize("device", DEVICES)
def test_lovasz_softmax_value_and_gradient_are_relatively_accurate(device, dtype, kind, seed):
    z, labels = _lovasz_images(kind, dtype, seed)
    value, expected, tightest = _cached(_lovasz_softmax_reference, z.to(dtype), labels)
    if tightest < 512 * U[dtype]:  # two errors within rounding of each other: their order is not defined by the data
        pytest.skip(f"two errors differ by a relative {tightest:.2g}, below the rounding of the run")
    value_error, element_error, voxel_error, _ = _errors(
        lambda x, t: LovaszSoftmaxLoss()(x, t), (value, expected), z, labels, dtype, device)
    assert value_error <= BOUND[dtype], f"value: {value_error:.3g} u"
    assert voxel_error <= (BOUND[dtype] if kind == "saturated" else BOUND_VOXEL), f"gradient: {voxel_error:.3g} u"


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["binary64", "binary32"])
@pytest.mark.parametrize("device", DEVICES)
def test_lovasz_hinge_value_and_gradient_never_lost_digits(device, dtype, kind, seed):
    # The errors are 1 - z (2t - 1) on the logits: exact near the margin, clipped to zero beyond it.
    z, labels = _lovasz_images(kind, dtype, seed)
    reference = _cached(_lovasz_hinge_reference, z.to(dtype), labels)
    classes = z.shape[1]
    target = F.one_hot(labels, classes).movedim(-1, 1)

    def hinge(x, t):
        return LovaszHingeLoss(multilabel=True, per_image=False)(x, F.one_hot(t, classes).movedim(-1, 1))

    value_error, element_error, _, magnitude = _errors(hinge, reference, z, labels, dtype, device)
    assert target.shape == z.shape
    assert value_error <= BOUND[dtype] or magnitude == 0, f"value: {value_error:.3g} u"
    assert element_error <= BOUND[dtype], f"gradient element: {element_error:.3g} u"


def test_lovasz_softmax_saturated_voxels_are_ranked_by_their_exact_errors():
    # Errors of 1e-18 to 1e-22, all different: where p rounds to one the error 1 - p of a foreground voxel was 0, and
    # the stable sort ranked those voxels by index; they are now ranked by their exact errors, so that the Jaccard
    # increments (the gradient) are assigned to the voxels in the exact order.
    z = torch.zeros(1, 2, 1, 4, dtype=torch.float64)
    z[0, 0, 0] = torch.tensor([-40.0, -30.0, 45.0, 50.0])  # class 0 logit; class 1 is 0: gaps of 40, 30, -45, -50
    labels = torch.tensor([[[1, 1, 0, 0]]])
    x = z.clone().requires_grad_()
    loss = LovaszSoftmaxLoss()(x, labels)
    (grad,) = torch.autograd.grad(loss, x)
    reference = _lovasz_softmax_reference(z.numpy().tobytes(), labels.numpy().tobytes(), tuple(z.shape), z.dtype)
    assert reference[2] > 1e-3  # no ties
    assert _relative(loss.item(), reference[0]) <= 32 * U[torch.float64]
    for got, ref in zip(grad.flatten().tolist(), reference[1]):
        assert _relative(got, ref) <= 32 * U[torch.float64]
