"""Independent Bernoulli scores, binning, reductions and streaming/distributed semantics."""
from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist
from _process import destroy_process_group
import torch.multiprocessing as mp

from exacta import _native, _triton_probability, functional as F, metrics as M


@pytest.fixture(params=["auto", "torch"])
def probability_backend(request, monkeypatch):
    if request.param == "torch":
        monkeypatch.setattr(_native, "available", lambda: False)
        monkeypatch.setattr(_triton_probability, "available", lambda: False)
    return request.param


def oracle(x, y, valid, logits, bins=7):
    exp = np.exp(-np.abs(x))
    p = np.where(x >= 0, 1/(1+exp), exp/(1+exp)) if logits else x
    brier = np.full(x.shape[:2], np.nan)
    nll = brier.copy()
    sums = np.zeros((*x.shape[:2], 3, bins))
    counts = np.zeros(x.shape[:2])
    for b in range(x.shape[0]):
        for c in range(x.shape[1]):
            keep = valid[b, c]
            conf, truth, raw = p[b, c][keep], y[b, c][keep], x[b, c][keep]
            counts[b, c] = len(conf)
            if len(conf):
                brier[b, c] = np.mean((conf-truth)**2)
                with np.errstate(divide="ignore"):
                    nll[b, c] = (np.logaddexp(0, np.where(truth, -raw, raw)) if logits else
                                 np.where(truth, -np.log(conf), -np.log1p(-conf))).mean()
            for value, label in zip(conf, truth):
                k = min(int(value*bins), bins-1)
                sums[b, c, :, k] += [1, value, label]
    return brier, nll, sums, counts


def reduce_score(scores, counts, average):
    if average == "none":
        return scores
    result = np.full((scores.shape[0], 1), np.nan)
    for b in range(scores.shape[0]):
        keep = counts[b] > 0
        if keep.any():
            result[b, 0] = (scores[b, keep].mean() if average == "macro" else
                            np.average(scores[b, keep], weights=counts[b, keep]))
    return result


def reference_ece(sums, average):
    if average == "micro":
        sums = sums.sum(1, keepdims=True)
    count, conf, positives = sums[:, :, 0], sums[:, :, 1], sums[:, :, 2]
    total = count.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.abs(positives-conf).sum(-1)/total
    return reduce_score(result, total, "macro") if average == "macro" else result


@pytest.mark.parametrize("shape", [(3, 3, 7, 9), (3, 3, 4, 5, 7)])
@pytest.mark.parametrize("logits", [False, True])
@pytest.mark.parametrize("mask_kind", ["none", "shared", "per-label"])
def test_scores_and_bins_match_numpy(device, probability_backend, shape, logits, mask_kind):
    rng = np.random.default_rng(307)
    x = rng.normal(size=shape) if logits else rng.random(shape)
    y = rng.integers(0, 2, size=shape).astype(float)
    mask = None
    valid = np.ones(shape, dtype=bool)
    if mask_kind != "none":
        mask_shape = shape if mask_kind == "per-label" else (shape[0], 1, *shape[2:])
        mask = rng.random(mask_shape) > .35
        mask[2] = False
        if mask_kind == "per-label":
            mask[0, 1] = False
        valid = np.broadcast_to(mask, shape)
        x[~valid], y[~valid] = np.nan, .123
    # Noncontiguous input/mask views must not mix channels or cases.
    pred, target = [torch.tensor(a, device=device).transpose(-1, -2) for a in (x, y)]
    m = None if mask is None else torch.tensor(mask, device=device).transpose(-1, -2)
    brier, nll, sums, counts = oracle(x, y, valid, logits)
    kw = dict(multilabel=True, from_logits=logits, mask=m)
    for average in ("none", "macro", "micro"):
        for fn, expected in ((F.brier_score, brier), (F.negative_log_likelihood, nll)):
            got = fn(pred, target, average=average, **kw)
            np.testing.assert_allclose(got.cpu(), reduce_score(expected, counts, average), atol=2e-14, rtol=2e-13)
        got = F.expected_calibration_error(pred, target, bins=7, average=average, **kw)
        np.testing.assert_allclose(got.cpu(), reference_ece(sums, average), atol=2e-14, rtol=2e-13)
    diagram = F.reliability_diagram(pred, target, bins=7, **kw)
    np.testing.assert_array_equal(diagram.counts.cpu(), sums[:, :, 0])
    with np.errstate(invalid="ignore", divide="ignore"):
        np.testing.assert_allclose(diagram.confidence.cpu(), sums[:, :, 1]/sums[:, :, 0], atol=2e-14)
        np.testing.assert_allclose(diagram.accuracy.cpu(), sums[:, :, 2]/sums[:, :, 0], atol=2e-14)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("target_dtype", [torch.bool, torch.uint8, torch.int16, torch.int64, torch.float32, torch.float64])
def test_dtype_contract(device, probability_backend, dtype, target_dtype):
    p = torch.tensor([[[[.1, .25, .5, .75, .9]]]], dtype=dtype, device=device).repeat(1, 2, 1, 1)
    target = torch.tensor([[[[0, 1, 0, 1, 1]]]], dtype=target_dtype, device=device).repeat(1, 2, 1, 1)
    brier, nll, _, _ = oracle(p.double().cpu().numpy(), target.double().cpu().numpy(),
                              np.ones(p.shape, bool), False)
    for fn, expected in ((F.brier_score, brier), (F.negative_log_likelihood, nll)):
        got = fn(p, target, multilabel=True)
        assert got.dtype == torch.float64 and got.device == p.device
        np.testing.assert_allclose(got.cpu(), expected, rtol=2e-13, atol=2e-14)


def test_bin_edges_and_micro_error_cancellation(device, probability_backend):
    p = torch.tensor([[[[0., .25, .5, .75, 1.]], [[0., .25, .5, .75, 1.]]]], device=device, dtype=torch.float64)
    y = torch.tensor([[[[0, 1, 0, 1, 1]], [[1, 0, 1, 0, 0]]]], device=device)
    diag = F.reliability_diagram(p, y, multilabel=True, bins=4)
    torch.testing.assert_close(diag.counts.cpu(), torch.tensor([[[1, 1, 1, 2], [1, 1, 1, 2]]]))
    p = torch.full((1, 2, 1, 4), .5, device=device)
    y = torch.tensor([[[[0, 0, 0, 0]], [[1, 1, 1, 1]]]], device=device)
    assert F.expected_calibration_error(p, y, multilabel=True, average="macro").item() == .5
    assert F.expected_calibration_error(p, y, multilabel=True, average="micro").item() == 0


def test_endpoint_probabilities_and_extreme_logits(device, probability_backend):
    p = torch.tensor([[[[0., 1., 0., 1.]]]], dtype=torch.float64, device=device)
    target = torch.tensor([[[[0, 1, 1, 0]]]], device=device)
    assert F.negative_log_likelihood(p, target, multilabel=True).isinf().all()
    mask = torch.tensor([[[True, True, False, False]]], device=device)
    assert F.negative_log_likelihood(p, target, mask=mask, multilabel=True).item() == 0
    x = torch.tensor([[[[-1000., 1000., -100., 100.]]]], dtype=torch.float64, device=device)
    y = torch.tensor([[[[1, 0, 0, 1]]]], device=device)
    assert F.negative_log_likelihood(x, y, multilabel=True, from_logits=True).item() == 500
    tiny = F.negative_log_likelihood(x[..., 2:], y[..., 2:], multilabel=True, from_logits=True)
    assert tiny.item() == pytest.approx(np.logaddexp(0., -100.), rel=1e-12, abs=0)


@pytest.mark.parametrize("shape", [(0, 2, 3, 5), (2, 3, 0, 5), (2, 3, 4, 0, 5)])
def test_empty_layouts(device, probability_backend, shape):
    p = torch.empty(shape, device=device)
    target = torch.empty(shape, dtype=torch.bool, device=device)
    mask = torch.empty((shape[0], 1, *shape[2:]), dtype=torch.bool, device=device)
    for fn in (F.brier_score, F.negative_log_likelihood, F.expected_calibration_error):
        for average in ("none", "macro", "micro"):
            got = fn(p, target, multilabel=True, mask=mask, average=average)
            assert got.shape == (shape[0], shape[1] if average == "none" else 1)
            assert got.isnan().all()


@pytest.mark.parametrize("bad", [1 + 1e-9, 1 - 1e-9, 1e-100, float("nan"), float("inf")])
def test_target_validation_before_narrowing(device, probability_backend, bad):
    p = torch.full((1, 2, 1, 2), .5, device=device, dtype=torch.float32)
    target = torch.ones(p.shape, dtype=torch.float64, device=device)
    target[..., 0] = bad
    mask = torch.ones_like(target, dtype=torch.bool)
    with pytest.raises(ValueError, match="binary"):
        F.brier_score(p, target, multilabel=True, mask=mask)
    mask[..., 0] = False
    torch.testing.assert_close(F.brier_score(p, target, multilabel=True, mask=mask),
                               torch.full((1, 2), .25, device=device, dtype=torch.float64))


def test_invalid_options_and_shapes(device):
    p = torch.full((1, 2, 3, 4), .5, device=device)
    y = torch.ones_like(p)
    for fn in (F.brier_score, F.negative_log_likelihood, F.expected_calibration_error):
        for kwargs in ({"ignore_index": -1}, {"average": "bad"}, {"mask": torch.ones_like(p)},
                       {"mask": torch.ones((2, 3, 4), device=device, dtype=torch.bool)}):
            with pytest.raises((ValueError, TypeError)):
                fn(p, y, multilabel=True, **kwargs)
        with pytest.raises(ValueError, match="same.*shape"):
            fn(p, y[:, 0], multilabel=True)
    with pytest.raises(ValueError, match="top-label"):
        F.reliability_diagram(p, y, multilabel=True, mode="top-label")
    with pytest.raises(ValueError, match="top-label"):
        M.Calibration(multilabel=True, mode="top-label")
    for bad in (-.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite|probabilities"):
            F.brier_score(torch.full_like(p, bad), y, multilabel=True)
    if device.type == "cuda":
        with pytest.raises(ValueError, match="same device"):
            F.brier_score(p, y.cpu(), multilabel=True)


def test_numpy_autograd_and_large_bin_fallback(device):
    p = torch.full((2, 3, 2, 5), .6, dtype=torch.float64, device=device, requires_grad=True)
    y = torch.zeros_like(p)
    mask = torch.ones_like(p, dtype=torch.bool)
    scores = F.brier_score(p, y, multilabel=True)
    assert not scores.requires_grad
    expected = F.reliability_diagram(p, y, multilabel=True, mask=mask, bins=65)
    actual = F.reliability_diagram(p.detach().cpu().numpy()[..., ::-1], y.cpu().numpy()[..., ::-1],
                                   mask=mask.cpu().numpy()[..., ::-1], multilabel=True, bins=65)
    np.testing.assert_array_equal(actual.counts, expected.counts.cpu().numpy())
    np.testing.assert_allclose(actual.ece, expected.ece.cpu().numpy())


def test_stateful_case_average_and_pooled_bins(device):
    p = torch.full((2, 2, 2, 3), .6, device=device, dtype=torch.float64)
    y = torch.zeros_like(p)
    y[1] = 1
    mask = torch.ones_like(p, dtype=torch.bool)
    mask[0, 1] = False
    for average in ("none", "macro", "micro"):
        for cls, fn in ((M.BrierScore, F.brier_score), (M.NegativeLogLikelihood, F.negative_log_likelihood),
                        (M.ExpectedCalibrationError, F.expected_calibration_error)):
            metric = cls(multilabel=True, average=average)
            for b in range(2):
                metric.update(p[b:b+1], y[b:b+1], mask=mask[b:b+1])
            torch.testing.assert_close(metric.compute().per_case,
                                       fn(p, y, mask=mask, multilabel=True, average=average), equal_nan=True)
    pooled = M.Calibration(multilabel=True, bins=2)
    pooled.update(p, y, mask=mask)
    torch.testing.assert_close(pooled.compute().ece, torch.tensor([[.1, .4]], device=device, dtype=torch.float64))
    before = pooled.compute().counts.clone()
    mask.zero_()
    torch.testing.assert_close(pooled.compute().counts, before)
    pooled.reset()
    with pytest.raises(RuntimeError, match="before update"):
        pooled.compute()


def _distributed(rank, path, empty):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{path}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        p = torch.full((1, 2, 2, 3), .6, dtype=torch.float64)
        y = torch.full_like(p, rank)
        mask = torch.ones_like(p, dtype=torch.bool)
        mask[:, 1] = False
        cal = M.Calibration(multilabel=True, bins=2)
        metric = M.BrierScore(multilabel=True, average="micro")
        if not empty or rank == 1:
            cal.update(p, y, mask=mask)
            metric.update(p, y, mask=mask)
        expected_ece = torch.tensor([[.4 if empty else .1, float("nan")]], dtype=torch.float64)
        torch.testing.assert_close(cal.compute().ece, expected_ece, equal_nan=True)
        assert metric.compute().mean.item() == pytest.approx(.16 if empty else .26)
    finally:
        destroy_process_group()


@pytest.mark.parametrize("empty", [False, True])
def test_distributed_multilabel_empty_rank(tmp_path, empty):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    mp.spawn(_distributed, args=(tmp_path / "init", empty), nprocs=2)
