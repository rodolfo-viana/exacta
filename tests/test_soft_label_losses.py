"""Dice semimetric losses (soft_label=True) against MONAI and their defining properties."""

import pytest
import torch

from exacta.losses import (
    DiceCELoss, DiceFocalLoss, DiceLoss, FocalTverskyLoss, GeneralizedDiceCELoss, GeneralizedDiceLoss, TverskyLoss,
)

monai_losses = pytest.importorskip("monai.losses")


def _soft(shape, seed, device):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(shape, generator=g, dtype=torch.float64)
    target = torch.rand(shape, generator=g, dtype=torch.float64).softmax(1)
    return logits.to(device), target.to(device)


@pytest.mark.parametrize("batch_dice", [False, True])
@pytest.mark.parametrize("include_background", [True, False])
@pytest.mark.parametrize("activation", ["softmax", "sigmoid"])
def test_dice_matches_monai_soft_label(device, batch_dice, include_background, activation):
    logits, target = _soft((2, 3, 9, 10, 7), 1, device)
    ours = DiceLoss(soft_label=True, target_format="one-hot", activation=activation, batch_dice=batch_dice,
                    include_background=include_background)(logits, target)
    reference = monai_losses.DiceLoss(soft_label=True, softmax=activation == "softmax",
                                      sigmoid=activation == "sigmoid", batch=batch_dice,
                                      include_background=include_background, smooth_nr=1e-5, smooth_dr=1e-5)
    torch.testing.assert_close(ours, reference(logits, target))


@pytest.mark.parametrize("alpha,beta", [(0.3, 0.7), (1.0, 1.0)])
def test_tversky_matches_monai_soft_label(device, alpha, beta):
    logits, target = _soft((2, 3, 9, 10), 2, device)
    ours = TverskyLoss(alpha=alpha, beta=beta, soft_label=True, target_format="one-hot")(logits, target)
    reference = monai_losses.TverskyLoss(alpha=alpha, beta=beta, soft_label=True, softmax=True,
                                         smooth_nr=1e-5, smooth_dr=1e-5)
    torch.testing.assert_close(ours, reference(logits, target))


LOSSES = [
    lambda **kw: DiceLoss(**kw),
    lambda **kw: TverskyLoss(alpha=0.3, beta=0.7, **kw),
    lambda **kw: FocalTverskyLoss(**kw),
    lambda **kw: GeneralizedDiceLoss(**kw),
    lambda **kw: DiceCELoss(**kw),
    lambda **kw: DiceFocalLoss(**kw),
    lambda **kw: GeneralizedDiceCELoss(**kw),
]


@pytest.mark.parametrize("make", LOSSES)
def test_identical_for_hard_labels(device, make):
    g = torch.Generator().manual_seed(3)
    logits = torch.randn((2, 3, 8, 9), generator=g, dtype=torch.float64).to(device).requires_grad_()
    labels = torch.randint(0, 3, (2, 8, 9), generator=g).to(device)
    hard = torch.nn.functional.one_hot(labels, 3).movedim(-1, 1).double()
    keep = torch.rand((2, 8, 9), generator=g).to(device) > 0.2
    plain = make(target_format="one-hot", backend="torch")(logits, hard, mask=keep)
    (grad_plain,) = torch.autograd.grad(plain, logits)
    semimetric = make(target_format="one-hot", soft_label=True)(logits, hard, mask=keep)
    (grad_semimetric,) = torch.autograd.grad(semimetric, logits)
    torch.testing.assert_close(semimetric, plain)
    torch.testing.assert_close(grad_semimetric, grad_plain)
    # Index targets are hard: soft_label keeps every backend, including the fused one.
    torch.testing.assert_close(make(soft_label=True)(logits, labels, mask=keep), plain)


@pytest.mark.parametrize("make", LOSSES[:4])
def test_minimum_is_the_soft_target(make):
    """The semimetric loss is zero exactly at p = g and positive elsewhere; the product form is not
    minimal at a soft target, and gradient descent from p = g lowers it (Wang et al., 2023)."""
    g = torch.Generator().manual_seed(4)
    target = torch.rand((1, 2, 64), generator=g, dtype=torch.float64)
    target = target / target.sum(1, keepdim=True)
    semimetric = make(activation="none", target_format="one-hot", soft_label=True, smooth=0.0)
    assert semimetric(target, target).abs().item() < 1e-12
    for _ in range(20):
        other = (target + 0.05 * torch.randn(target.shape, generator=g, dtype=torch.float64)).clamp(0, 1)
        assert semimetric(other, target).item() > 1e-6
    product = make(activation="none", target_format="one-hot", smooth=0.0)
    start = product(target, target).item()
    p = target.clone().requires_grad_()
    opt = torch.optim.Adam([p], lr=0.01)
    for _ in range(300):
        opt.zero_grad()
        product(p.clamp(0, 1), target).backward()
        opt.step()
    assert product(p.detach().clamp(0, 1), target).item() < start - 1e-3


def test_options_contract():
    with pytest.raises(ValueError, match="squared_pred"):
        DiceLoss(soft_label=True, squared_pred=True)
    with pytest.raises(NotImplementedError, match="soft_label"):
        DiceLoss(soft_label=True, target_format="one-hot", backend="triton")
    with pytest.raises(TypeError, match="soft_label"):
        DiceLoss(soft_label=1)
