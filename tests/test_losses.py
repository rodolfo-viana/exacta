import warnings

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from exacta.losses import DiceCELoss, DiceLoss


def _inputs(device, shape=(2, 3, 6, 7, 8), seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(shape, generator=g, dtype=dtype)
    target = torch.randint(0, shape[1], (shape[0], *shape[2:]), generator=g)
    return logits.to(device), target.to(device)


def _dice_loss_ref(probs, onehot, smooth, batch_dice):
    axes = (0, *range(2, probs.ndim)) if batch_dice else tuple(range(2, probs.ndim))
    inter = (probs * onehot).sum(axes)
    den = probs.sum(axes) + onehot.sum(axes)
    return 1 - ((2 * inter + smooth) / (den + smooth)).mean()


@pytest.mark.parametrize("batch_dice", [False, True])
@pytest.mark.parametrize("include_background", [True, False])
def test_dice_loss_matches_formula(device, batch_dice, include_background):
    logits, target = _inputs(device)
    loss = DiceLoss(include_background=include_background, batch_dice=batch_dice)(logits, target)
    probs = torch.softmax(logits.double(), 1).cpu().numpy()
    onehot = F.one_hot(target, 3).movedim(-1, 1).double().cpu().numpy()
    if not include_background:
        probs, onehot = probs[:, 1:], onehot[:, 1:]
    expected = _dice_loss_ref(probs, onehot, 1e-5, batch_dice)
    np.testing.assert_allclose(loss.item(), expected, rtol=1e-5)


def test_one_hot_target_equals_index(device):
    logits, target = _inputs(device)
    onehot = F.one_hot(target, 3).movedim(-1, 1)
    for loss_cls in (DiceLoss, DiceCELoss):
        torch.testing.assert_close(
            loss_cls(target_format="one-hot")(logits, onehot), loss_cls()(logits, target)
        )


def test_dice_ce_is_weighted_sum(device):
    logits, target = _inputs(device)
    loss = DiceCELoss(lambda_dice=0.3, lambda_ce=2.0)(logits, target)
    expected = 0.3 * DiceLoss()(logits, target) + 2.0 * F.cross_entropy(logits, target)
    torch.testing.assert_close(loss, expected)


def test_sigmoid_uses_bce(device):
    logits, target = _inputs(device)
    onehot = F.one_hot(target, 3).movedim(-1, 1).float()
    loss = DiceCELoss(activation="sigmoid")(logits, target)
    expected = DiceLoss(activation="sigmoid")(logits, target) + F.binary_cross_entropy_with_logits(
        logits, onehot
    )
    torch.testing.assert_close(loss, expected)


def test_perfect_prediction_is_near_zero(device):
    _, target = _inputs(device)
    logits = F.one_hot(target, 3).movedim(-1, 1).float() * 50
    assert DiceLoss()(logits, target).item() < 1e-6
    assert DiceCELoss()(logits, target).item() < 1e-6


@pytest.mark.parametrize("loss_cls", [DiceLoss, DiceCELoss])
@pytest.mark.parametrize("batch_dice", [False, True])
def test_gradcheck(loss_cls, batch_dice):
    logits, target = _inputs("cpu", shape=(2, 3, 3, 4, 2), dtype=torch.float64)
    logits.requires_grad_(True)
    loss_fn = loss_cls(include_background=False, batch_dice=batch_dice)
    assert torch.autograd.gradcheck(lambda x: loss_fn(x, target), (logits,))


def test_half_precision_is_computed_in_float32():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    logits, target = _inputs("cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        loss = DiceCELoss()(logits.half(), target)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    torch.testing.assert_close(loss, DiceCELoss()(logits.half().float(), target))


@pytest.mark.parametrize("batch_dice", [False, True])
def test_matches_monai(batch_dice):
    monai_losses = pytest.importorskip("monai.losses")
    logits, target = _inputs("cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        monai_dice = monai_losses.DiceLoss(softmax=True, to_onehot_y=True, batch=batch_dice)
        monai_dice_ce = monai_losses.DiceCELoss(softmax=True, to_onehot_y=True, batch=batch_dice)
        expected_dice = monai_dice(logits, target[:, None])
        expected_dice_ce = monai_dice_ce(logits, target[:, None])
    torch.testing.assert_close(DiceLoss(batch_dice=batch_dice)(logits, target), expected_dice)
    torch.testing.assert_close(DiceCELoss(batch_dice=batch_dice)(logits, target), expected_dice_ce)
