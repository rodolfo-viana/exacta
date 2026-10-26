"""Ranking oracles enumerate thresholds and positive-negative pairs independently."""
from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist
from _process import destroy_process_group
import torch.multiprocessing as mp

from exacta import functional as f, metrics

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]


def oracle(score, target):
    positives, negatives = score[target], score[~target]
    if not len(positives):
        return np.nan, np.nan
    previous, ap = 0., 0.
    for threshold in np.unique(score)[::-1]:
        selected = score >= threshold
        recall = target[selected].sum()/len(positives)
        ap += (recall-previous)*target[selected].mean()
        previous = recall
    auc = (np.mean((positives[:, None] > negatives) + .5*(positives[:, None] == negatives))
           if len(negatives) else np.nan)
    return ap, auc


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("kind", ["binary", "multiclass", "multilabel"])
@pytest.mark.parametrize("bins", [None, 2, 11])
@pytest.mark.parametrize("from_logits", [False, True])
def test_ranking_oracle(device, dim, kind, bins, from_logits):
    rng = np.random.default_rng(831)
    shape = (2, 1 if kind == "binary" else 3, *([3]*dim))
    # Continuous categorical logits avoid accidental near-ties whose ordering can
    # differ by one rounding unit between NumPy and Torch's softmax reductions.
    logits = rng.normal(size=shape) if kind == "multiclass" else rng.integers(-4, 5, shape).astype(float)
    if kind == "multiclass":
        prob = np.exp(logits-logits.max(1, keepdims=True))
        prob /= prob.sum(1, keepdims=True)
    else:
        prob = 1/(1+np.exp(-logits))
    y = rng.integers(0, 2, shape) if kind == "multilabel" else rng.integers(0, max(2, shape[1]), (2, *shape[2:]))
    mask = rng.random(y.shape) > .3
    p = torch.tensor(logits if from_logits else prob, device=device, dtype=torch.float64)
    t, m = torch.tensor(y, device=device), torch.tensor(mask, device=device)
    kwargs = dict(from_logits=from_logits, multilabel=kind == "multilabel", bins=bins, mask=m)
    result = f.voxel_ranking(p, t, **kwargs)
    assert result.bins == bins
    for b in range(shape[0]):
        for c in range(shape[1]):
            keep = mask[b, c] if kind == "multilabel" else mask[b]
            target = y[b, c] == 1 if kind == "multilabel" else y[b] == (c if kind == "multiclass" else 1)
            score = prob[b, c][keep]
            if bins is not None:
                score = np.minimum(np.floor(score*bins), bins-1)
            expected = oracle(score, target[keep])
            np.testing.assert_allclose([result.average_precision[b, c].cpu(), result.auroc[b, c].cpu()],
                                       expected, rtol=1e-12, equal_nan=True)
            curve = result.curves[b][c]
            assert curve.thresholds[0] == torch.inf
            assert (curve.thresholds[1:] <= curve.thresholds[:-1]).all()
            if bins is not None:
                assert curve.thresholds.numel() == bins+1
    torch.testing.assert_close(f.average_precision(p, t, **kwargs), result.average_precision, equal_nan=True)
    torch.testing.assert_close(f.auroc(p, t, **kwargs), result.auroc, equal_nan=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bins", [None, 5, 100])
def test_pooled_state_case_means_and_mutation(device, bins):
    p = torch.tensor([.2, .3, .3, .6, .6, .9, .8, .1], dtype=torch.float64, device=device).reshape(2, 1, 2, 2)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 1, 0], device=device).reshape(2, 2, 2)
    reference = f.voxel_ranking(p.movedim(0, -1).reshape(1, 1, 2, 4),
                               y.movedim(0, -1).reshape(1, 2, 4), bins=bins)
    pooled = metrics.VoxelRanking(bins=bins)
    for b in range(2):
        pooled.update(p[b:b+1], y[b:b+1])
    for cls, field in [(metrics.AveragePrecision, "average_precision"), (metrics.AUROC, "auroc")]:
        metric = cls(bins=bins)
        metric.update(p, y)
        torch.testing.assert_close(metric.compute().per_case, getattr(f.voxel_ranking(p, y, bins=bins), field))
    p.fill_(torch.nan)
    y.fill_(99)
    for _ in range(2):
        result = pooled.compute()
        torch.testing.assert_close(result.average_precision, reference.average_precision)
        torch.testing.assert_close(result.auroc, reference.auroc)
    if bins is not None:
        assert pooled._hist.numel() == 2*bins
        assert pooled._chunks == [[]]
    pooled.reset()
    with pytest.raises(RuntimeError, match="before update"):
        pooled.compute()


def test_ranking_ties_extreme_logits_endpoints_and_empty():
    p = np.array([0., .5, .5, 1.]).reshape(1, 1, 2, 2)
    y = np.array([0, 1, 0, 1]).reshape(1, 2, 2)
    r = f.voxel_ranking(p, y)
    assert r.auroc[0, 0] == .875
    assert r.average_precision[0, 0] == pytest.approx(5/6)
    np.testing.assert_array_equal(r.curves[0][0].thresholds, [np.inf, 1, .5, 0])
    p = torch.tensor([1000., 1001.]).reshape(1, 1, 1, 2)
    y = torch.tensor([0, 1]).reshape(1, 1, 2)
    assert f.auroc(p, y, from_logits=True) == 1  # No saturating sigmoid in exact mode.
    for positive in (0, 1):
        t = torch.full_like(y, positive)
        assert f.auroc(p, t, from_logits=True).isnan().all()
        ap = f.average_precision(p, t, from_logits=True)
        assert ap == 1 if positive else ap.isnan().all()
    mask = torch.zeros_like(y, dtype=torch.bool)
    r = f.voxel_ranking(p*torch.nan, y*99, mask=mask)
    assert r.average_precision.isnan().all() and r.auroc.isnan().all()
    assert r.curves[0][0].thresholds.numel() == 1
    p = torch.empty(0, 1, 2, 3)
    assert f.auroc(p, torch.empty(0, 2, 3, dtype=torch.long)).shape == (0, 1)


def test_mask_shapes_ignore_and_validation():
    torch.manual_seed(832)
    p = torch.rand(2, 3, 4, 5, dtype=torch.float64).softmax(1)
    t = torch.randint(3, (2, 4, 5))
    t[:, 0] = -1
    p[:, :, 0] = torch.nan
    r = f.voxel_ranking(p, t, ignore_index=-1, include_background=False)
    assert r.auroc.shape == (2, 2)
    with pytest.raises(ValueError, match="targets"):
        f.voxel_ranking(p, t)
    with pytest.raises(ValueError, match="bins"):
        f.auroc(p, t, bins=True)
    p = torch.rand(1, 2, 2, 3, dtype=torch.float64)
    y = torch.zeros_like(p)
    for mask in (torch.ones(1, 2, 3, dtype=torch.bool), torch.ones(1, 1, 2, 3, dtype=torch.bool),
                 torch.ones_like(p, dtype=torch.bool)):
        assert f.auroc(p, y, multilabel=True, mask=mask).isnan().all()
    with pytest.raises(ValueError, match="targets"):
        f.auroc(p, y+1e-12, multilabel=True)
    with pytest.raises(ValueError, match="sum to one"):
        f.auroc(p*0, torch.zeros(1, 2, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="ignore_index"):
        f.auroc(p, y, multilabel=True, ignore_index=-1)
    with pytest.raises(ValueError, match="two channels"):
        f.auroc(p[:, :1], y[:, 0].long(), include_background=False)


def test_multiclass_logit_translation_preserves_ties():
    p = torch.tensor([[[[1., 101.]], [[2., 102.]], [[3., 103.]]]], dtype=torch.float64)
    t = torch.tensor([[[0, 1]]])
    result = f.voxel_ranking(p, t, from_logits=True)
    assert result.auroc[0, 0] == .5
    assert result.auroc[0, 1] == .5
    assert result.curves[0][0].thresholds.numel() == 2


def _distributed_worker(rank, path):
    dist.init_process_group("gloo", init_method=f"file://{path}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=40))
    torch.set_num_threads(1)
    try:
        for bins in (None, 4):
            for empty_rank in (False, True):
                p = torch.tensor([.1, .5, .5, .9, .7, .2, .4, .8]).reshape(2, 1, 2, 2)
                t = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]).reshape(2, 2, 2)
                metric = metrics.VoxelRanking(bins=bins)
                if not empty_rank:
                    metric.update(p[rank:rank+1], t[rank:rank+1])
                elif rank == 1:
                    metric.update(p, t)
                result = metric.compute()
                reference = f.voxel_ranking(p.reshape(1, 1, 2, 4), t.reshape(1, 2, 4), bins=bins)
                torch.testing.assert_close(result.average_precision, reference.average_precision)
                torch.testing.assert_close(result.auroc, reference.auroc)
    finally:
        destroy_process_group()


def test_ranking_distributed_empty_rank(tmp_path):
    if not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    mp.spawn(_distributed_worker, args=(str(tmp_path/"ranking"),), nprocs=2, join=True)
