"""Instance label maps: matching, PQ, AJI, AJI+ and SEG against independent dense references."""

import math
from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import reference
from _process import destroy_process_group
from exacta import instances

pytest.importorskip("scipy")
from scipy import ndimage  # noqa: E402


def _touching(shape, n, rng):
    """Touching Voronoi cells in a ragged foreground, with random non-contiguous values."""
    grid = np.indices(shape).reshape(len(shape), -1).T
    seeds = rng.integers(0, shape, (n, len(shape)))
    labels = ((grid[:, None, :] - seeds[None]) ** 2).sum(-1).argmin(1).reshape(shape) + 1
    foreground = ndimage.binary_opening(rng.random(shape) < 0.75)
    values = rng.permutation(np.arange(1, 10 * n + 10))[: n + 1]
    values[0] = 0
    return values[np.where(foreground, labels, 0)]


def _pairs(count, seed, shape=(28, 32)):
    rng = np.random.default_rng(seed)
    for trial in range(count):
        target = _touching(shape, int(rng.integers(1, 20)), rng)
        if trial % 3 == 0:
            pred = np.roll(target, int(rng.integers(-2, 3)), axis=0)
        else:
            pred = _touching(shape, int(rng.integers(1, 20)), rng)
        if trial % 5 == 0:
            pred = np.where(rng.random(shape) < 0.1, 0, pred)
        yield pred, target


def test_scores_match_dense_references(device):
    for pred, target in _pairs(80, 1):
        got = instances.evaluate(torch.from_numpy(pred)[None].to(device), torch.from_numpy(target)[None].to(device))
        assert got.aji.device.type == device.type
        assert got.aji.item() == pytest.approx(reference.aji(target, pred), abs=1e-12)
        assert got.aji_plus.item() == pytest.approx(reference.aji_plus(target, pred), abs=1e-12)
        assert got.seg.item() == pytest.approx(reference.ctc_seg(target, pred), abs=1e-12)
        tp, fp, fn, iou = reference.panoptic(target, pred)
        assert (got.tp.item(), got.fp.item(), got.fn.item()) == (tp, fp, fn)
        detections = tp + (fp + fn) / 2
        assert got.pq.item() == pytest.approx(iou / detections if detections else math.nan, abs=1e-12, nan_ok=True)
        assert got.accuracy.item() == pytest.approx(tp / (tp + fp + fn) if tp else 0.0, abs=1e-12)


def test_touching_instances_stay_apart_and_values_are_relabelled():
    target = np.zeros((1, 4, 6), np.int64)
    target[0, :, :3], target[0, :, 3:] = 7, 1000  # two touching instances
    pred = target.copy()
    pred[0, :, 3:] = 5
    got = instances.evaluate(pred, target)
    assert (got.tp, got.fp, got.fn) == (2, 0, 0) and got.pq == 1 and got.aji == 1 and got.seg == 1
    huge = np.where(target == 1000, 2**40, target)  # sparse values take the sorting path
    np.testing.assert_equal(instances.evaluate(pred, huge).pq, 1.0)
    merged = np.where(pred > 0, 1, 0)  # one prediction across both
    got = instances.evaluate(merged, target)
    assert got.tp == 0 and got.fp == 1 and got.fn == 2 and got.pq == 0
    # AJI: both targets take the merged prediction; its area counts in each union.
    assert got.aji == pytest.approx((12 + 12) / (24 + 24))
    assert got.aji_plus == pytest.approx(12 / (24 + 12))


def test_threshold_ties_and_sequences():
    target = np.array([[[1, 1, 0, 2]]])
    pred = np.array([[[1, 0, 0, 2]]])  # IoU 1/2 and 1
    strict = instances.evaluate(pred, target)
    loose = instances.evaluate(pred, target, strict=False)
    assert (strict.tp, loose.tp) == (1, 2)
    several = instances.evaluate(pred, target, threshold=[0.25, 0.5, 0.75], strict=True)
    np.testing.assert_array_equal(several.tp, [[2, 1, 1]])
    assert several.aji.shape == (1,)
    with pytest.raises(ValueError, match="thresholds"):
        instances.evaluate(pred, target, threshold=1.5)
    with pytest.raises(ValueError, match="nonnegative"):
        instances.evaluate(pred - 2, target)
    with pytest.raises(TypeError, match="integer"):
        instances.evaluate(pred.astype(float), target)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.uint16, torch.uint32, torch.uint64, torch.int16, torch.int32])
def test_unsigned_and_narrow_label_dtypes(device, dtype):
    pred, target = next(_pairs(1, 6))
    pred, target = np.minimum(pred, 200), np.minimum(target, 200)  # uint8 range
    expected = instances.evaluate(pred[None], target[None])
    got = instances.evaluate(torch.from_numpy(pred).to(dtype).to(device)[None], torch.from_numpy(target).to(dtype).to(device)[None])
    for name in ("pq", "aji", "aji_plus", "seg"):
        assert getattr(got, name).item() == pytest.approx(getattr(expected, name).item(), abs=1e-12)
    if dtype == torch.uint64:
        huge = torch.full((1, 2, 2), 2**63, dtype=torch.uint64)
        with pytest.raises(ValueError, match="below 2\\*\\*63"):
            instances.evaluate(huge, huge)


def test_hungarian_maximises_summed_iou_below_half():
    # One prediction overlaps two targets below 0.5; a second prediction overlaps only the first.
    target = np.array([[[1, 1, 1, 2, 2, 2, 0, 0]]])
    pred = np.array([[[1, 1, 3, 3, 3, 0, 0, 0]]])
    greedy = instances.evaluate(pred, target, threshold=0.3, matching="greedy")
    hungarian = instances.evaluate(pred, target, threshold=0.3, matching="hungarian")
    assert greedy.tp == 2 and hungarian.tp == 2
    assert hungarian.pq * (hungarian.tp + (hungarian.fp + hungarian.fn) / 2) >= greedy.pq * 2 - 1e-12


def test_empty_cases():
    empty = np.zeros((1, 5, 5), np.int64)
    some = empty.copy()
    some[0, 1:3, 1:3] = 4
    none = instances.evaluate(empty, empty)
    assert all(math.isnan(getattr(none, name).item()) for name in ("pq", "aji", "seg", "precision", "recall"))
    assert instances.evaluate(empty, empty, both_empty=1.0).pq == 1
    only_pred = instances.evaluate(some, empty)
    assert only_pred.pq == 0 and only_pred.aji == 0 and math.isnan(only_pred.seg.item()) and math.isnan(only_pred.recall.item())
    only_target = instances.evaluate(empty, some)
    assert only_target.pq == 0 and only_target.seg == 0 and math.isnan(only_target.precision.item())


def test_three_dimensional_instances(device):
    rng = np.random.default_rng(3)
    target = _touching((8, 10, 12), 6, rng)
    pred = np.roll(target, 1, axis=1)
    got = instances.evaluate(torch.from_numpy(pred)[None].to(device), torch.from_numpy(target)[None].to(device))
    assert got.aji.item() == pytest.approx(reference.aji(target, pred), abs=1e-12)
    assert got.seg.item() == pytest.approx(reference.ctc_seg(target, pred), abs=1e-12)


def test_dsb2018_and_dataset_aggregation():
    pairs = list(_pairs(6, 4))
    pred = np.stack([p for p, _ in pairs])
    target = np.stack([t for _, t in pairs])
    thresholds = [k / 20 for k in range(10, 20)]
    expected = []
    for p, t in pairs:
        per = [reference.panoptic(t, p, value) for value in thresholds]
        expected.append(np.mean([tp / (tp + fp + fn) if tp else 0.0 for tp, fp, fn, _ in per]))
    np.testing.assert_allclose(instances.dsb2018(pred, target), expected, atol=1e-12)

    cases, pooled = instances.InstanceMetrics(), instances.InstanceMetrics(aggregate="pooled")
    for metric in (cases, pooled):
        metric.update(pred[:2], target[:2])
        metric.update(pred[2:], target[2:])
    per_case = instances.evaluate(pred, target)
    got = cases.compute()
    assert got.pq.item() == pytest.approx(np.nanmean(per_case.pq))
    assert got.tp.item() == per_case.tp.sum()
    stats = [reference.panoptic(t, p) for p, t in pairs]
    tp, fp, fn, iou = (sum(s[k] for s in stats) for k in range(4))
    result = pooled.compute()
    assert result.pq.item() == pytest.approx(iou / (tp + (fp + fn) / 2), abs=1e-12)
    targets = sum(len(np.unique(t)) - 1 for _, t in pairs)
    seg_sum = sum(reference.ctc_seg(t, p) * (len(np.unique(t)) - 1) for p, t in pairs)
    assert result.seg.item() == pytest.approx(seg_sum / targets, abs=1e-12)
    cases.reset()
    with pytest.raises(RuntimeError, match="before update"):
        cases.compute()


def _distributed_instances(rank, init_file):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        pairs = list(_pairs(3, 5))
        pred, target = np.stack([p for p, _ in pairs]), np.stack([t for _, t in pairs])
        pooled = instances.InstanceMetrics(aggregate="pooled")
        cases = instances.InstanceMetrics()
        if rank == 0:  # rank 1 has no cases
            pooled.update(pred, target)
            cases.update(pred, target)
        stats = [reference.panoptic(t, p) for p, t in pairs]
        tp, fp, fn, iou = (sum(s[k] for s in stats) for k in range(4))
        assert pooled.compute().pq.item() == pytest.approx(iou / (tp + (fp + fn) / 2), abs=1e-12)
        assert cases.compute().pq.item() == pytest.approx(np.nanmean(instances.evaluate(pred, target).pq), abs=1e-12)
    finally:
        destroy_process_group()


def test_distributed_pooled_instances_include_empty_ranks(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    mp.spawn(_distributed_instances, args=(tmp_path / "instances-init",), nprocs=2)
