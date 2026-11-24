"""Skeleton Recall loss against the authors' transform and loss, written with scikit-image."""

import numpy as np
import pytest
import torch

from exacta.losses import SkeletonRecallLoss, skeleton_recall_target

morphology = pytest.importorskip("skimage.morphology")
ndimage = pytest.importorskip("scipy.ndimage")


def _authors_target(seg: np.ndarray) -> np.ndarray:
    """SkeletonTransform._add_skeleton of MIC-DKFZ/Skeleton-Recall, for one label map."""
    bin_seg = seg > 0
    if not bin_seg.any():
        return np.zeros_like(seg)
    skel = morphology.skeletonize(bin_seg).astype(np.int16)
    footprint = ndimage.iterate_structure(ndimage.generate_binary_structure(skel.ndim, 1), 2)
    skel = morphology.dilation(skel, footprint=footprint)
    return skel * seg.astype(np.int16)


def _authors_loss(logits: torch.Tensor, skeleton: torch.Tensor, batch_dice: bool, smooth: float) -> torch.Tensor:
    """SoftSkeletonRecallLoss with softmax and do_bg=False."""
    x = logits.softmax(1)[:, 1:]
    y = torch.nn.functional.one_hot(skeleton.long(), logits.shape[1]).movedim(-1, 1)[:, 1:].to(x.dtype)
    axes = tuple(range(2, x.ndim))
    sum_gt, inter = y.sum(axes), (x * y).sum(axes)
    if batch_dice:
        sum_gt, inter = sum_gt.sum(0), inter.sum(0)
    return -((inter + smooth) / torch.clip(sum_gt + smooth, 1e-8)).mean()


def _vessels(shape, seed, classes=3):
    rng = np.random.default_rng(seed)
    seg = np.zeros(shape, np.int64)
    for c in range(1, classes):
        lines = rng.random(shape) < 0.02
        seg[ndimage.binary_dilation(lines, iterations=2)] = c
    return seg


@pytest.mark.parametrize("shape", [(40, 44), (14, 16, 18)])
def test_target_matches_authors_transform(shape):
    seg = np.stack([_vessels(shape, seed) for seed in (1, 2)])
    seg[1, ...] = 0 if len(shape) == 3 else seg[1]  # an empty case in 3-D
    expected = np.stack([_authors_target(s) for s in seg])
    got = skeleton_recall_target(torch.from_numpy(seg))
    np.testing.assert_array_equal(got.numpy(), expected)


def test_target_options_and_ignored_labels():
    seg = torch.from_numpy(_vessels((40, 44), 3))[None]
    per_class = skeleton_recall_target(seg, union=False)
    assert set(per_class.unique().tolist()) <= {0, 1, 2} and bool((per_class == 0)[seg == 0].all())
    plain = skeleton_recall_target(seg, tube=0)
    assert int((plain > 0).sum()) < int((skeleton_recall_target(seg) > 0).sum())
    lee = skeleton_recall_target(seg, method="lee", tube=0)
    np.testing.assert_array_equal(lee[0].numpy() > 0, morphology.skeletonize(seg[0].numpy() > 0, method="lee"))
    ignored = seg.clone()
    ignored[0, :5] = 255
    cleaned = torch.where(ignored == 255, 0, ignored)
    torch.testing.assert_close(skeleton_recall_target(ignored, ignore_index=255), skeleton_recall_target(cleaned))
    one_hot = torch.nn.functional.one_hot(seg, 3).movedim(-1, 1)
    channels = skeleton_recall_target(one_hot, target_format="one-hot", union=True)
    torch.testing.assert_close(channels, torch.nn.functional.one_hot(skeleton_recall_target(seg), 3).movedim(-1, 1).bool()
                               & ~torch.tensor([True, False, False])[None, :, None, None])
    with pytest.raises(ValueError, match="tube"):
        skeleton_recall_target(seg, tube=-1)
    with pytest.raises(ValueError, match="nonnegative"):
        skeleton_recall_target(seg - 1)


@pytest.mark.parametrize("batch_dice", [False, True])
@pytest.mark.parametrize("shape", [(40, 44), (12, 14, 16)])
def test_loss_matches_authors_loss(device, batch_dice, shape):
    seg = torch.from_numpy(np.stack([_vessels(shape, seed) for seed in (4, 5)])).to(device)
    logits = torch.randn((2, 3, *shape), generator=torch.Generator().manual_seed(6), dtype=torch.float64).to(device)
    skeleton = skeleton_recall_target(seg)
    expected = 1 + _authors_loss(logits, skeleton, batch_dice, 1e-5)
    loss = SkeletonRecallLoss(batch_dice=batch_dice)
    torch.testing.assert_close(loss(logits, seg), expected)
    torch.testing.assert_close(loss(logits, seg, target_skeleton=skeleton), expected)


def test_gradients_masks_and_absent_skeletons():
    seg = torch.from_numpy(_vessels((20, 22), 7))[None]
    logits = torch.randn((1, 3, 20, 22), dtype=torch.float64, requires_grad=True)
    loss = SkeletonRecallLoss()
    skeleton = skeleton_recall_target(seg)
    assert torch.autograd.gradcheck(lambda x: loss(x, seg, target_skeleton=skeleton), (logits,))
    keep = torch.ones((1, 20, 22), dtype=torch.bool)
    keep[:, :, :11] = False
    masked = loss(logits, seg, target_skeleton=skeleton, mask=keep)
    cropped = SkeletonRecallLoss()(logits[..., 11:], seg[..., 11:], target_skeleton=skeleton[..., 11:])
    torch.testing.assert_close(masked, cropped)
    empty = torch.zeros_like(seg)
    assert loss(logits, empty).item() == 0  # no skeleton anywhere: recall 1 for every class
    ignored = SkeletonRecallLoss(ignore_index=-1)(logits, torch.where(keep, seg, -1))
    expected = SkeletonRecallLoss()(logits, seg, target_skeleton=skeleton_recall_target(torch.where(keep, seg, 0)), mask=keep)
    torch.testing.assert_close(ignored, expected)
    with pytest.raises(ValueError, match="smooth"):
        SkeletonRecallLoss(smooth=0)
    with pytest.raises(ValueError, match="batch_dice"):
        SkeletonRecallLoss(distributed=True)


def test_sigmoid_channels():
    seg = torch.from_numpy(_vessels((30, 32), 8))[None]
    channels = torch.stack([seg > 0, seg == 2], 1).to(torch.uint8)  # overlapping labels
    logits = torch.randn((1, 2, 30, 32), dtype=torch.float64)
    loss = SkeletonRecallLoss(activation="sigmoid", target_format="one-hot", include_background=True)
    skeleton = skeleton_recall_target(channels, target_format="one-hot")
    p = logits.sigmoid()
    recall = ((p * skeleton).sum((2, 3)) + 1e-5) / (skeleton.sum((2, 3)) + 1e-5)
    torch.testing.assert_close(loss(logits, channels), 1 - recall.mean())
