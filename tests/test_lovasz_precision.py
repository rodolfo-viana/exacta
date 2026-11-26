"""Lovasz gradients against an exact rational reference.

The gradient of a Lovasz loss is the increment Delta J_k of the Jaccard sequence J_k = 1 - I_k / U_k at
each sorted position. Differencing rounded values of J_k loses about log2(n) bits; the increments are
now formed from integer counts with one division, so every gradient component must be correct to a few
units of the last place of the largest one, at any n. The sort order is taken from the run (it is part
of the definition of the subgradient); everything after it is recomputed with ``fractions.Fraction``.
"""
from fractions import Fraction

import pytest
import torch

from exacta.losses import LovaszHingeLoss, LovaszSoftmaxLoss, ranking


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


def ranked_gradient(errors: torch.Tensor, fg: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Delta J_k at the original position of every valid element (zero elsewhere), as float64.

    ``errors`` are in the dtype of the run, on its device, so the order is the one Exacta uses."""
    order = torch.sort(torch.where(valid, errors, -torch.inf), descending=True, stable=True).indices
    order = order[: int(valid.sum())].cpu()
    increments = exact_increments([int(f) for f in fg.cpu()[order]])
    out = torch.zeros(errors.shape, dtype=torch.float64)
    out[order] = torch.tensor([float(d) for d in increments], dtype=torch.float64)
    return out


def softmax_reference(probs, y, valid, *, classes, per_image, first=0):
    """Gradient of ``LovaszSoftmaxLoss(from_logits=False)`` with respect to its probabilities."""
    batch, channels = probs.shape[:2]

    def pooled(p, t, v):  # p: (C, n); t, v: (n,)
        grad = torch.zeros(p.shape, dtype=torch.float64)
        selected = 0
        for c in range(first, p.shape[0]):
            foreground = (t == c) & v
            if not (foreground.any() if classes == "present" else v.any()):
                continue
            selected += 1
            errors = (foreground.to(p.dtype) - p[c]).abs()
            slope = (p[c].double() - foreground.double()).sign().cpu()  # derivative of |fg - p|
            grad[c] = ranked_gradient(errors, foreground, v) * slope
        return grad / max(selected, 1)

    if per_image:
        cases = [pooled(probs[b].flatten(1), y[b].flatten(), valid[b].flatten()) for b in range(batch)]
        active = [bool(valid[b].any()) for b in range(batch)]
        grad = torch.stack(cases) * torch.tensor(active, dtype=torch.float64).reshape(-1, 1, 1) / max(sum(active), 1)
        return grad.reshape(probs.shape)
    flat = probs.movedim(1, 0).reshape(channels, -1)
    grad = pooled(flat, y.flatten(), valid.flatten())
    return grad.reshape(channels, batch, *probs.shape[2:]).movedim(0, 1)


def hinge_reference(x, y, valid, *, per_image):
    """Gradient of ``LovaszHingeLoss`` (target and valid shaped like ``x``) with respect to the logits."""
    batch, channels = x.shape[:2]
    if per_image:
        rows = [(b, c) for b in range(batch) for c in range(channels)]
        take = lambda a, row: a[row[0], row[1]].flatten()  # noqa: E731
    else:
        rows = [(c,) for c in range(channels)]
        take = lambda a, row: a[:, row[0]].flatten()  # noqa: E731
    grad = torch.zeros(x.shape, dtype=torch.float64)
    active = 0
    results = []
    for row in rows:
        z, t, v = take(x, row), take(y, row), take(valid, row)
        if not v.any():
            continue
        active += 1
        sign = 2 * t.to(z.dtype) - 1
        errors = 1 - z * sign
        slope = (-sign.double() * (errors > 0).double()).cpu()  # derivative of max(1 - z (2t - 1), 0)
        results.append((row, ranked_gradient(errors, t.bool() & v, v) * slope))
    for row, g in results:
        if per_image:
            grad[row[0], row[1]] = g.reshape(x.shape[2:]) / active
        else:
            grad[:, row[0]] = g.reshape(x.shape[0], *x.shape[2:]) / active
    return grad


def assert_few_ulps(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype, ulps: float = 4.0) -> None:
    """Every component within ``ulps`` units of the last place of the largest component."""
    expected = expected.to(torch.float64)
    bound = ulps * torch.finfo(dtype).eps * float(expected.abs().max())
    worst = float((actual.detach().double().cpu() - expected).abs().max())
    assert worst <= bound, f"max error {worst:.3e} exceeds {ulps} ulps of the largest component ({bound:.3e})"


def quantize(a: torch.Tensor, steps: int) -> torch.Tensor:
    return torch.round(a * steps) / steps


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("per_image", [False, True])
@pytest.mark.parametrize("classes", ["present", "all"])
@pytest.mark.parametrize("variant", ["plain", "masked", "ties"])
def test_softmax_gradient_matches_exact_reference(device, dtype, per_image, classes, variant):
    torch.manual_seed(2024)
    shape = (2, 3, 6, 9, 10)  # 540 voxels per case, 1080 pooled
    probs = torch.rand(shape, dtype=torch.float64).clamp(0.01, 0.99)
    if variant == "ties":  # many equal errors, and errors of exactly 0 and 1
        probs = quantize(torch.rand(shape, dtype=torch.float64), 4)
    probs = probs.to(dtype).to(device).requires_grad_()
    y = torch.randint(0, 3, (shape[0], *shape[2:]), device=device)
    mask = torch.rand(y.shape, device=device) > 0.3 if variant == "masked" else None
    valid = torch.ones_like(y, dtype=torch.bool) if mask is None else mask
    loss = LovaszSoftmaxLoss(from_logits=False, classes=classes, per_image=per_image)(probs, y, mask=mask)
    (grad,) = torch.autograd.grad(loss, probs)
    assert grad.dtype == dtype
    expected = softmax_reference(probs.detach(), y, valid, classes=classes, per_image=per_image)
    assert_few_ulps(grad, expected, dtype)
    assert (grad[~valid[:, None].expand_as(grad)] == 0).all()


def test_softmax_gradient_without_background(device):
    torch.manual_seed(5)
    probs = torch.rand((2, 3, 7, 8), dtype=torch.float32, device=device).clamp(0.02, 0.98).requires_grad_()
    y = torch.randint(0, 3, (2, 7, 8), device=device)
    valid = torch.ones_like(y, dtype=torch.bool)
    loss = LovaszSoftmaxLoss(from_logits=False, include_background=False)(probs, y)
    (grad,) = torch.autograd.grad(loss, probs)
    expected = softmax_reference(probs.detach(), y, valid, classes="present", per_image=False, first=1)
    assert_few_ulps(grad, expected, torch.float32)
    assert (grad[:, 0] == 0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("per_image", [False, True])
@pytest.mark.parametrize("multilabel", [False, True])
@pytest.mark.parametrize("variant", ["plain", "masked", "ties"])
def test_hinge_gradient_matches_exact_reference(device, dtype, per_image, multilabel, variant):
    torch.manual_seed(2025)
    channels = 3 if multilabel else 1
    shape = (2, channels, 6, 9, 10)
    z = torch.randn(shape, dtype=torch.float64)
    if variant == "ties":
        z = quantize(z, 2)
    z = z.to(dtype).to(device).requires_grad_()
    y = torch.randint(0, 2, shape, device=device)
    mask = torch.rand(shape, device=device) > 0.3 if variant == "masked" else None
    valid = torch.ones_like(y, dtype=torch.bool) if mask is None else mask
    target = y if multilabel else y[:, 0]
    loss = LovaszHingeLoss(multilabel=multilabel, per_image=per_image)(z, target, mask=mask)
    (grad,) = torch.autograd.grad(loss, z)
    assert grad.dtype == dtype
    expected = hinge_reference(z.detach(), y, valid, per_image=per_image)
    assert_few_ulps(grad, expected, dtype)
    assert (grad[~valid] == 0).all()


def test_gradient_error_does_not_grow_with_the_number_of_voxels(device):
    """The old error was about 0.45 n u: 2.7e-2 of the largest component at n = 1e6 in binary32."""
    torch.manual_seed(0)
    worst = {}
    for dtype in (torch.float32, torch.float64):
        for n in (500, 5000, 50000):
            p1 = torch.rand(1, 1, n, dtype=dtype, device=device)
            probs = torch.cat((1 - p1, p1), 1).requires_grad_()
            y = (torch.rand(1, n, device=device) < 0.3).long()
            (grad,) = torch.autograd.grad(LovaszSoftmaxLoss(from_logits=False)(probs, y), probs)
            fg = (y[0] == 1)
            order = torch.sort((fg.to(dtype) - p1[0, 0]).abs(), descending=True, stable=True).indices.cpu()
            f = fg.cpu()[order].long()
            total, cum = int(f.sum()), f.cumsum(0)
            inter, union = total - cum, total + torch.arange(1, n + 1) - cum
            prev_inter = torch.cat((torch.tensor([total]), inter[:-1]))
            prev_union = torch.cat((torch.tensor([1]), union[:-1]))
            # Independent integer form: (I_{k-1} U_k - I_k U_{k-1}) / (U_{k-1} U_k), exact numerator in int64.
            numerator = prev_inter * union - inter * prev_union
            numerator[0] = union[0] - inter[0]
            denominator = prev_union * union
            expected = torch.zeros(n, dtype=torch.float64)
            expected[order] = numerator.double() / denominator.double() * torch.where(f > 0, -1.0, 1.0).double()
            worst[(dtype, n)] = float((grad[0, 1].double().cpu() - expected / 2).abs().max() / (expected / 2).abs().max())
    for (dtype, n), err in worst.items():
        assert err <= 4 * torch.finfo(dtype).eps, (dtype, n, err)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("limit", [None, 4])
def test_increments_equal_exact_fractions(dtype, limit, monkeypatch):
    """The helper itself, with invalid tails, no foreground, all foreground, and the two division paths."""
    if limit is not None:  # n > limit divides twice instead of forming the int64 denominator
        monkeypatch.setattr(ranking, "_JACCARD_DENOMINATOR_LIMIT", limit)
    generator = torch.Generator().manual_seed(11)
    for density in (0.0, 0.05, 0.5, 0.95, 1.0):
        for n, valid_count in ((1, 1), (2, 2), (17, 17), (300, 300), (300, 211), (40, 0)):
            fg = torch.rand(n, generator=generator) < density
            keep = torch.arange(n) < valid_count
            fg &= keep
            got = ranking._jaccard_increments(fg, keep, fg.sum(), dtype)
            assert got.dtype == dtype and torch.isfinite(got).all()
            expected = exact_increments([int(f) for f in fg[:valid_count]])
            for k, ref in enumerate(expected):
                if limit is None and dtype == torch.float64:
                    assert float(got[k]) == float(ref)  # one rounding: the correctly rounded value
                else:
                    tolerance = (3 * 2.0**-53 if dtype == torch.float64 else torch.finfo(dtype).eps) * abs(float(ref))
                    assert abs(float(got[k]) - float(ref)) <= tolerance
            assert (got[valid_count:] == 0).all()


def test_increments_batched_rows_share_no_state():
    generator = torch.Generator().manual_seed(3)
    fg = torch.rand(4, 50, generator=generator) < 0.4
    keep = torch.ones(4, 50, dtype=torch.bool)
    keep[2, 30:] = False
    fg &= keep
    got = ranking._jaccard_increments(fg, keep, fg.sum(1, keepdim=True), torch.float64)
    for r in range(4):
        valid_count = int(keep[r].sum())
        alone = ranking._jaccard_increments(fg[r], keep[r], fg[r].sum(), torch.float64)
        assert torch.equal(got[r], alone)
        assert [float(d) for d in exact_increments([int(f) for f in fg[r, :valid_count]])] == got[r, :valid_count].tolist()


def test_increments_beyond_the_exact_integers_of_binary32():
    """Counts above 2^24 are not representable in binary32; the old cumulative sums stopped at 16777216."""
    n = 2**24 + 4096
    generator = torch.Generator().manual_seed(7)
    fg = torch.rand(n, generator=generator) < 0.3
    keep = torch.ones(n, dtype=torch.bool)
    got = ranking._jaccard_increments(fg, keep, fg.sum(), torch.float32)
    cum = fg.cumsum(0)  # int64
    total = int(cum[-1])
    positions = [0, 1, 2, 2**24 - 1, 2**24, 2**24 + 1, n - 2, n - 1] + torch.randint(2, n, (200,), generator=generator).tolist()
    for k in positions:
        def jaccard(i: int) -> Fraction:  # position i, counted from 0, has i + 1 elements up to it
            f = int(cum[i])
            return 1 - Fraction(total - f, total + (i + 1) - f)
        ref = jaccard(k) - (jaccard(k - 1) if k else 0)
        assert abs(float(got[k]) - float(ref)) <= torch.finfo(torch.float32).eps * abs(float(ref))
