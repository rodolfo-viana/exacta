"""Masked evaluation against independent voxel selection and hand-computed ratios."""

from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from exacta import _native, _triton_counts, functional as F, metrics as M
from exacta.functional import overlap


@pytest.fixture(params=["auto", "torch"])
def count_backend(request, monkeypatch):
    if request.param == "torch":
        monkeypatch.setattr(_native, "available", lambda: False)
        monkeypatch.setattr(_triton_counts, "available", lambda: False)
    return request.param


NAMES = ("dice", "iou", "sensitivity", "precision", "specificity", "volume_similarity",
         "relative_volume_difference")
CLASSES = (M.Dice, M.IoU, M.Sensitivity, M.Precision, M.Specificity, M.VolumeSimilarity,
           M.RelativeVolumeDifference)


def _oracle(pred, target, keep, classes, one_hot=False):
    counts = np.zeros((4, pred.shape[0], classes), dtype=np.int64)
    for b in range(pred.shape[0]):
        for c in range(classes):
            p = (pred[b, c] == 1 if one_hot else pred[b] == c)[keep[b]]
            t = (target[b, c] == 1 if one_hot else target[b] == c)[keep[b]]
            counts[:, b, c] = [np.count_nonzero(p & t), np.count_nonzero(p & ~t),
                               np.count_nonzero(~p & t), np.count_nonzero(~p & ~t)]
    tp, fp, fn, tn = counts.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = [2*tp/(2*tp+fp+fn), tp/(tp+fp+fn), tp/(tp+fn), tp/(tp+fp), tn/(tn+fp),
                  1-np.abs(fp-fn)/(2*tp+fp+fn), (fp-fn)/(tp+fn)]
    return counts, dict(zip(NAMES, scores))


@pytest.mark.parametrize("one_hot,mode", [(False, "mask"), (False, "ignore"), (False, "both"), (True, "mask")])
@pytest.mark.parametrize("shape", [(3, 7, 11), (3, 5, 7, 9)])
def test_counts_and_scores_match_selected_voxels(device, count_backend, one_hot, shape, mode, monkeypatch):
    rng = np.random.default_rng(103)
    p, t = [rng.integers(0, 3, size=shape) for _ in range(2)]
    mask = rng.random(shape) > .3
    mask[1] = False  # An entirely excluded case must not add true negatives.
    ignored = rng.random(shape) < .2 if mode != "mask" else np.zeros(shape, bool)
    if mode == "ignore":
        ignored[1] = True
    keep = (~ignored) & (mask if mode != "ignore" else True)
    if one_hot:
        p, t = [np.moveaxis(np.eye(3)[x], -1, 1) for x in (p, t)]
        p[np.broadcast_to(~keep[:, None], p.shape)] = np.nan
        t[np.broadcast_to(~keep[:, None], t.shape)] = np.inf
    else:
        p[~keep] = 2**40
        if mode != "ignore":
            t[~mask] = -999
        t[ignored] = -100
    expected, scores = _oracle(p, t, keep, 3, one_hot)
    # Negative-stride NumPy input followed by noncontiguous Torch views.
    pred, target = [torch.tensor(x, device=device).transpose(-1, -2) for x in (p, t)]
    supplied_mask = torch.tensor(mask, device=device).transpose(-1, -2)[:, None] if mode != "ignore" else None
    kwargs = dict(input_format="one-hot" if one_hot else "index", num_classes=3,
                  ignore_index=-100 if mode != "mask" else None, mask=supplied_mask)
    monkeypatch.setattr(overlap, "_CHUNK", 31)
    counts = overlap._confusion_counts(pred, target, kwargs["input_format"], 3, False,
                                       ignore_index=kwargs["ignore_index"], mask=supplied_mask)
    for got, gold in zip(counts, expected[:, :, 1:]):
        np.testing.assert_array_equal(got.cpu().numpy(), gold)
    shared = F.evaluate(pred, target, metrics=NAMES, include_background=False, **kwargs)
    for name in NAMES:
        actual = getattr(F, name)(pred, target, include_background=False, **kwargs)
        np.testing.assert_allclose(actual.cpu(), scores[name][:, 1:], rtol=1e-14, equal_nan=True)
        torch.testing.assert_close(actual, shared[name], equal_nan=True)


@pytest.mark.parametrize("ignore", [0, 255, -(2**63), 2**63-1])
def test_ignore_target_label_excludes_prediction_before_validation(device, count_backend, ignore):
    p = torch.tensor([[[1, 0], [1, 2**40]]], device=device)
    t = torch.tensor([[[1, 1], [0, ignore]]], device=device)
    keep = t != ignore
    expected = F.dice(p, t, num_classes=2, mask=keep)
    torch.testing.assert_close(F.dice(p, t, num_classes=2, ignore_index=ignore), expected, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32])
def test_ignore_does_not_wrap_to_label_dtype(device, count_backend, dtype):
    x = torch.tensor([[[0, 1], [1, 0]]], dtype=dtype, device=device)
    ignore = 2 if dtype == torch.bool else 2**torch.iinfo(dtype).bits
    torch.testing.assert_close(F.dice(x, x, num_classes=2, ignore_index=ignore),
                               torch.ones((1, 2), device=device, dtype=torch.float64))


@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.int16, torch.int64,
                                  torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_binary_dtypes_and_shared_channel_mask(device, count_backend, dtype):
    p = torch.tensor([[[[1, 0, 1]], [[0, 1, 1]]]], dtype=dtype, device=device)
    t = torch.tensor([[[[1, 1, 0]], [[1, 0, 0]]]], dtype=dtype, device=device)
    mask = torch.tensor([[[True, False, True]]], device=device)
    actual = F.dice(p, t, input_format="one-hot", mask=mask)
    torch.testing.assert_close(actual, torch.tensor([[2/3, 0]], dtype=torch.float64, device=device))


@pytest.mark.parametrize("one_hot", [False, True])
def test_included_invalid_values_still_raise(device, count_backend, one_hot):
    shape = (1, 2, 2, 2) if one_hot else (1, 2, 2)
    p = torch.zeros(shape, device=device, dtype=torch.float64 if one_hot else torch.long)
    t = p.clone()
    mask = torch.ones((1, 2, 2), device=device, dtype=torch.bool)
    p[..., 0, 0] = float("nan") if one_hot else 999
    with pytest.raises(ValueError, match="binary|class indices"):
        F.dice(p, t, input_format="one-hot" if one_hot else "index", num_classes=2, mask=mask)
    mask[..., 0, 0] = False
    F.dice(p, t, input_format="one-hot" if one_hot else "index", num_classes=2, mask=mask)


def test_many_classes_masked_fallback(device):
    assert not _triton_counts.fits(40)
    p = torch.arange(40, device=device).reshape(1, 5, 8)
    t = p.roll(1, -1)
    mask = p % 3 != 0
    counts, _ = _oracle(p.cpu().numpy(), t.cpu().numpy(), mask.cpu().numpy(), 40)
    actual = overlap._confusion_counts(p, t, "index", 40, True, mask=mask)
    np.testing.assert_array_equal(torch.stack(actual).cpu().numpy(), counts)


def test_unsigned_overflow_is_not_a_negative_ignore_label(device, count_backend):
    target = torch.tensor([[[2**64-1, 0]]], dtype=torch.uint64, device=device)
    pred = torch.zeros((1, 1, 2), dtype=torch.long, device=device)
    with pytest.raises(ValueError, match="class indices"):
        F.dice(pred, target, num_classes=2, ignore_index=-1)


def test_complex_one_hot_values_are_checked_without_losing_imaginary_part(device, count_backend):
    p = torch.tensor([[[[1+1j, 1]]]], device=device)
    t = torch.ones_like(p)
    mask = torch.tensor([[[False, True]]], device=device)
    torch.testing.assert_close(F.dice(p, t, input_format="one-hot", mask=mask),
                               torch.ones((1, 1), dtype=torch.float64, device=device))
    with pytest.raises(ValueError, match="binary"):
        F.dice(p, t, input_format="one-hot", mask=torch.ones_like(mask))


def test_native_masked_bindings_reject_incompatible_buffers():
    p = torch.zeros(4, dtype=torch.long)
    out = torch.empty((2, 2), dtype=torch.long)
    with pytest.raises(ValueError, match="do not agree"):
        _native._C.masked_confusion(p, p, torch.ones(3, dtype=torch.uint8), True, -1, out)
    p = p.reshape(2, 2)
    counts = [torch.empty(2, dtype=torch.long) for _ in range(4)]
    for channels, mask in [(0, torch.ones((1, 2), dtype=torch.uint8)),
                           (2, torch.ones((2, 2), dtype=torch.uint8))]:
        with pytest.raises(ValueError, match="disagree"):
            _native._C.masked_channel_counts(p, p, mask, channels, *counts)


@pytest.mark.parametrize("shape", [(0, 4, 5), (2, 0, 5), (2, 4, 0)])
@pytest.mark.parametrize("one_hot", [False, True])
def test_empty_dimensions(device, count_backend, shape, one_hot):
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    p = torch.empty((shape[0], 3, *shape[1:]) if one_hot else shape, dtype=torch.long, device=device)
    got = F.dice(p, p, mask=mask, num_classes=3, input_format="one-hot" if one_hot else "index")
    assert got.shape == (shape[0], 3)
    assert got.isnan().all()


def test_empty_policy_and_numpy_arrays():
    p = np.full((2, 4, 5), 999)[..., ::-1]
    mask = np.zeros(p.shape, dtype=bool)[..., ::-1]
    for name in NAMES:
        options = {"both_empty" if name in ("dice", "iou") else "zero_division": .25}
        got = getattr(F, name)(p, p, num_classes=2, mask=mask, **options)
        assert isinstance(got, np.ndarray)
        np.testing.assert_array_equal(got, np.full((2, 2), .25))


def test_invalid_mask_and_ignore_contract(device):
    p = torch.zeros((1, 2, 3), dtype=torch.long, device=device)
    for mask, error in [(torch.ones_like(p), TypeError),
                        (torch.ones((1, 2, 2, 3), dtype=torch.bool, device=device), ValueError),
                        (torch.tensor(True, device=device), ValueError)]:
        with pytest.raises(error, match="mask"):
            F.dice(p, p, num_classes=2, mask=mask)
    if device.type == "cuda":
        with pytest.raises(ValueError, match="same device"):
            F.dice(p, p, num_classes=2, mask=torch.ones(p.shape, dtype=torch.bool))
    for ignore in (True, 1.5, "none", 2**63, -(2**63)-1):
        with pytest.raises((TypeError, ValueError), match="ignore_index"):
            F.dice(p, p, num_classes=2, ignore_index=ignore)
    with pytest.raises(ValueError, match="one-hot"):
        F.dice(p[:, None], p[:, None], input_format="one-hot", ignore_index=-1)


@pytest.mark.parametrize("name", ["hd95", "assd", "nsd", "hausdorff_distance", "cldice"])
@pytest.mark.parametrize("option", ["mask", "ignore_index"])
def test_collections_reject_undefined_geometry_masking(name, option):
    p = torch.zeros((1, 3, 4), dtype=torch.long)
    opts = {"mask": torch.ones_like(p, dtype=torch.bool)} if option == "mask" else {"ignore_index": -1}
    for selection in ([name], ["dice", name]):
        with pytest.raises(ValueError, match="complete segmentations"):
            F.evaluate(p, p, metrics=selection, num_classes=2, spacing=1, **opts)


def test_stateful_metrics_and_collection_do_not_retain_masks(device):
    p = torch.tensor([[[1, 0], [1, 99]], [[99, 99], [99, 99]]], device=device)
    t = torch.tensor([[[1, 1], [0, -1]], [[-1, -1], [-1, -1]]], device=device)
    mask = torch.ones_like(p, dtype=torch.bool)
    collection = M.MetricCollection(metrics=NAMES, num_classes=2, ignore_index=-1)
    collection.update(p, t, mask=mask)
    for name, cls in zip(NAMES, CLASSES):
        metric = cls(num_classes=2, ignore_index=-1)
        got = metric.update(p, t, mask=mask)
        torch.testing.assert_close(got, collection.compute()[name].per_case, equal_nan=True)
        metric.update(p[:1], t[:1], mask=torch.zeros_like(mask[:1]))
        torch.testing.assert_close(metric.compute().per_class, got[0], equal_nan=True)
    expected = collection.compute()["dice"].per_case.clone()
    mask.zero_()
    torch.testing.assert_close(collection.compute()["dice"].per_case, expected, equal_nan=True)


def _distributed_masked(rank, init_file):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        p = torch.tensor([[[1, 0], [1, 1]]])
        t = torch.tensor([[[1, 1], [0, 1]]]) if rank else torch.full_like(p, -1)
        metric = M.Dice(num_classes=2, ignore_index=-1)
        collection = M.MetricCollection(metrics=["dice", "specificity"], num_classes=2, ignore_index=-1)
        metric.update(p, t)
        collection.update(p, t)
        actual = collection.compute()
        torch.testing.assert_close(metric.compute().per_case, actual["dice"].per_case, equal_nan=True)
        assert actual["dice"].per_case[0].isnan().all()
        torch.testing.assert_close(actual["dice"].per_class, torch.tensor([0., 2/3], dtype=torch.float64))
        torch.testing.assert_close(actual["specificity"].per_class, torch.tensor([2/3, 0.], dtype=torch.float64))
    finally:
        dist.destroy_process_group()


def test_distributed_masked_metrics(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    mp.spawn(_distributed_masked, args=(tmp_path / "masked-init",), nprocs=2)
