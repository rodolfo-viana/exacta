"""F-beta, public confusion counts and dataset-pooled overlap aggregation."""

from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import exacta.functional as EF
import exacta.metrics as M
import reference
from _data import pair
from _process import destroy_process_group

sklearn_metrics = pytest.importorskip("sklearn.metrics")

POOLED = {
    "dice": (M.Dice, lambda t, p, labels: sklearn_metrics.f1_score(t, p, labels=labels, average=None, zero_division=np.nan)),
    "iou": (M.IoU, lambda t, p, labels: sklearn_metrics.jaccard_score(t, p, labels=labels, average=None)),
    "sensitivity": (M.Sensitivity, lambda t, p, labels: sklearn_metrics.recall_score(t, p, labels=labels, average=None, zero_division=np.nan)),
    "precision": (M.Precision, lambda t, p, labels: sklearn_metrics.precision_score(t, p, labels=labels, average=None, zero_division=np.nan)),
}


@pytest.mark.parametrize("beta", [0.25, 1.0, 2.0, 1e6])
@pytest.mark.parametrize("shape", [(32, 40), (16, 18, 14)])
def test_fbeta_matches_reference(device, beta, shape):
    pred, target = pair(shape, 4, seed=5)
    expected = reference.per_case(reference.fbeta, pred, target, 4, beta=beta)
    got = EF.fbeta(torch.from_numpy(pred).to(device), torch.from_numpy(target).to(device), num_classes=4, beta=beta)
    np.testing.assert_allclose(got.cpu().numpy(), expected, rtol=1e-12)


def test_fbeta_one_is_dice_and_handles_empty_classes(device):
    pred, target = (torch.from_numpy(a).to(device) for a in pair((20, 20, 20), 3, seed=6))
    torch.testing.assert_close(EF.fbeta(pred, target, num_classes=3), EF.dice(pred, target, num_classes=3))
    empty = torch.zeros((1, 4, 4), dtype=torch.long, device=device)
    assert EF.fbeta(empty, empty, num_classes=2, beta=2.0)[0, 1].isnan()
    assert EF.fbeta(empty, empty, num_classes=2, beta=2.0, both_empty=1.0)[0, 1] == 1.0
    for beta in (0, -1.0, float("inf"), float("nan"), True):
        with pytest.raises(ValueError, match="beta"):
            EF.fbeta(empty, empty, num_classes=2, beta=beta)


def test_confusion_counts_are_exhaustive_and_keep_array_type():
    pred, target = pair((12, 10, 8), 3, seed=7)
    counts = EF.confusion_counts(pred, target, num_classes=3)
    assert isinstance(counts, EF.ConfusionCounts) and isinstance(counts.tp, np.ndarray)
    assert counts.tp.dtype == np.int64 and counts.tp.shape == (2, 3)
    for b in range(2):
        for c in range(3):
            p, t = pred[b] == c, target[b] == c
            assert (counts.tp[b, c], counts.fp[b, c], counts.fn[b, c], counts.tn[b, c]) == (
                (p & t).sum(), (p & ~t).sum(), (~p & t).sum(), (~p & ~t).sum())
    mask = np.zeros(pred.shape, bool)
    mask[:, :6] = True
    masked = EF.confusion_counts(pred, target, num_classes=3, mask=mask)
    assert (masked.tp + masked.fp + masked.fn + masked.tn == mask[:, None].reshape(2, 1, -1).sum(-1)).all()


@pytest.mark.parametrize("name", list(POOLED))
@pytest.mark.parametrize("include_background", [True, False])
def test_pooled_scores_match_flattened_voxels(device, name, include_background):
    pred, target = pair((14, 16, 12), 4, seed=8, batch=3)
    cls, ref = POOLED[name]
    metric = cls(num_classes=4, include_background=include_background, aggregate="pooled")
    for sl in (slice(0, 1), slice(1, 3)):
        metric.update(torch.from_numpy(pred[sl]).to(device), torch.from_numpy(target[sl]).to(device))
    result = metric.compute()
    labels = list(range(0 if include_background else 1, 4))
    expected = ref(target.ravel(), pred.ravel(), labels)
    assert result.per_case is None and result.per_class.device.type == device.type
    np.testing.assert_allclose(result.per_class.cpu().numpy(), expected, rtol=1e-12)
    np.testing.assert_allclose(result.mean.item(), np.nanmean(expected), rtol=1e-12)


def test_pooled_masks_and_ignored_labels_select_voxels(device):
    pred, target = pair((14, 16, 12), 3, seed=9)
    target[0, :3] = -1
    keep = np.ones(pred.shape, bool)
    keep[1, :, :4] = False
    metric = M.IoU(num_classes=3, ignore_index=-1, aggregate="pooled")
    metric.update(torch.from_numpy(pred).to(device), torch.from_numpy(target).to(device), mask=torch.from_numpy(keep).to(device))
    observed = keep & (target != -1)
    expected = sklearn_metrics.jaccard_score(target[observed], pred[observed], labels=[0, 1, 2], average=None)
    np.testing.assert_allclose(metric.compute().per_class.cpu().numpy(), expected, rtol=1e-12)


def test_pooled_undefined_classes_are_left_out_of_the_mean():
    pred = torch.tensor([[[0, 1], [1, 0]]])
    target = torch.tensor([[[0, 0], [1, 0]]])
    metric = M.Dice(num_classes=4, aggregate="pooled")  # class 2 never occurs; class 3 neither
    metric.update(pred, target)
    pred[0, 0, 0] = 3  # class 3 only predicted in the second batch
    metric.update(pred, target)
    result = metric.compute()
    assert result.per_class[2].isnan() and result.per_class[3] == 0
    torch.testing.assert_close(result.mean, result.per_class[[0, 1, 3]].mean())
    metric.reset()
    with pytest.raises(RuntimeError, match="before update"):
        metric.compute()


def test_pooled_collection_matches_metrics_and_rejects_distances(device):
    pred, target = (torch.from_numpy(a).to(device) for a in pair((14, 16, 12), 4, seed=10, batch=3))
    selection = {"dice": {}, "fbeta": {"beta": 2.0}, "specificity": {}, "relative_volume_difference": {"absolute": True}}
    collection = M.MetricCollection(metrics=selection, num_classes=4, include_background=False, aggregate="pooled")
    returned = collection.update(pred, target)
    torch.testing.assert_close(returned["dice"], EF.dice(pred, target, num_classes=4, include_background=False))
    pooled = collection.compute()
    single = {
        "dice": M.Dice(num_classes=4, include_background=False, aggregate="pooled"),
        "fbeta": M.FBeta(beta=2.0, num_classes=4, include_background=False, aggregate="pooled"),
        "specificity": M.Specificity(num_classes=4, include_background=False, aggregate="pooled"),
        "relative_volume_difference": M.RelativeVolumeDifference(absolute=True, num_classes=4,
                                                                 include_background=False, aggregate="pooled"),
    }
    for name, metric in single.items():
        metric.update(pred, target)
        torch.testing.assert_close(pooled[name].per_class, metric.compute().per_class)
    for selection in (["dice", "hd95"], ["cldice"], ["predicted_volume"]):
        with pytest.raises(ValueError, match="pooled"):
            M.MetricCollection(metrics=selection, num_classes=4, aggregate="pooled")
    with pytest.raises(ValueError, match="aggregate"):
        M.Dice(num_classes=4, aggregate="mean")


def _distributed_pooled(rank, init_file):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=3,
                            timeout=timedelta(seconds=30))
    try:
        pred, target = pair((10, 12, 8), 3, seed=11, batch=2)
        metric = M.Dice(num_classes=3, aggregate="pooled")
        collection = M.MetricCollection(metrics=["iou"], num_classes=3, aggregate="pooled")
        if rank < 2:  # rank 2 has no cases
            metric.update(torch.from_numpy(pred[rank:rank + 1]), torch.from_numpy(target[rank:rank + 1]))
            collection.update(torch.from_numpy(pred[rank:rank + 1]), torch.from_numpy(target[rank:rank + 1]))
        expected = sklearn_metrics.f1_score(target.ravel(), pred.ravel(), average=None)
        np.testing.assert_allclose(metric.compute().per_class.numpy(), expected, rtol=1e-12)
        expected = sklearn_metrics.jaccard_score(target.ravel(), pred.ravel(), average=None)
        np.testing.assert_allclose(collection.compute()["iou"].per_class.numpy(), expected, rtol=1e-12)
        mismatched = M.Dice(num_classes=3, aggregate="pooled", include_background=rank != 0)
        mismatched.update(torch.from_numpy(pred[:1]), torch.from_numpy(target[:1]))
        with pytest.raises(ValueError, match="agree"):
            mismatched.compute()
    finally:
        destroy_process_group()


def test_distributed_pooled_counts_include_empty_ranks(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    mp.spawn(_distributed_pooled, args=(tmp_path / "pooled-init",), nprocs=3)
