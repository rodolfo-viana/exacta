import pytest
import torch

from exacta.losses import DeepSupervisionLoss, DiceCELoss


def test_weighted_resolutions_and_gradients(device):
    torch.manual_seed(301)
    xs = [torch.randn(2, 3, n, n, n, device=device, dtype=torch.float64, requires_grad=True)
          for n in (8, 4, 2)]
    ys = [torch.randint(3, (2, n, n, n), device=device) for n in (8, 4, 2)]
    masks = [torch.rand(y.shape, device=device) > 0.3 for y in ys]
    weights = [2.0, 0.5, 0.25]  # Deliberately do not sum to one.
    loss = DiceCELoss(ignore_index=-100)
    actual = DeepSupervisionLoss(loss, weights)(xs, ys, masks=masks)
    expected = sum(w * loss(x, y, mask=m) for w, x, y, m in zip(weights, xs, ys, masks))
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual, xs)
    ge = torch.autograd.grad(expected, xs)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e)


def test_zero_weight_is_not_evaluated_and_masks_are_optional():
    x = torch.randn(1, 2, 3, requires_grad=True)
    unused = torch.full_like(x, float("nan"), requires_grad=True)
    y = torch.zeros(1, 3, dtype=torch.long)
    loss = DeepSupervisionLoss(DiceCELoss(), [1, 0])
    loss([x, unused], [y, y]).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert unused.grad is None


@pytest.mark.parametrize("weights", [[], [0, 0], [-1, 2], [float("nan")], [float("inf")]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError, match="weights"):
        DeepSupervisionLoss(DiceCELoss(), weights)


def test_lengths_and_scalar_contract():
    x = torch.randn(1, 2, 3)
    y = torch.zeros(1, 3, dtype=torch.long)
    loss = DeepSupervisionLoss(DiceCELoss(), [1])
    with pytest.raises(TypeError, match="sequences"):
        loss(x, y)
    with pytest.raises(ValueError, match="same length"):
        loss([x, x], [y])
    with pytest.raises(ValueError, match="masks"):
        loss([x], [y], masks=[])
    with pytest.raises(ValueError, match="scalar"):
        DeepSupervisionLoss(torch.nn.MSELoss(reduction="none"), [1])([x], [x])


def test_compile():
    torch.manual_seed(303)
    xs = [torch.randn(2, 3, 4, 4, requires_grad=True), torch.randn(2, 3, 2, 2, requires_grad=True)]
    ys = [torch.randint(3, (2, 4, 4)), torch.randint(3, (2, 2, 2))]
    fn = DeepSupervisionLoss(DiceCELoss(backend="torch"), [1, 0.5])
    expected = fn(xs, ys)
    actual = torch.compile(fn, fullgraph=True)(xs, ys)
    torch.testing.assert_close(actual, expected)
    for a, e in zip(torch.autograd.grad(actual, xs), torch.autograd.grad(expected, xs)):
        torch.testing.assert_close(a, e)
