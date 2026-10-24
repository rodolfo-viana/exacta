"""Check fused compositions against independently composed PyTorch objectives."""

import pytest
import torch
import torch.nn.functional as F

from exacta.losses import DiceCELoss, DiceFocalLoss, DiceLoss, FocalLoss, GeneralizedDiceCELoss, GeneralizedDiceLoss


@pytest.mark.parametrize("kind", ["dice_ce", "gdl_ce", "dice_focal"])
@pytest.mark.parametrize("activation", ["softmax", "sigmoid"])
@pytest.mark.parametrize("soft", [False, True])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_composed_value_and_gradient(device, kind, activation, soft, batch, weighted):
    if kind == "dice_focal" and weighted:
        pytest.skip("focal uses alpha, not CE class weights")
    torch.manual_seed(340)
    x = torch.randn(2, 3, 4, 5, 3, device=device, dtype=torch.float64).transpose(-1, -2).requires_grad_()
    labels = torch.randint(3, (2, 4, 3, 5), device=device)
    y = torch.randn_like(x).softmax(1) if soft else labels
    valid = torch.rand(labels.shape, device=device) > 0.3
    valid[0] = False
    kwargs = dict(activation=activation, target_format="one-hot" if soft else "index",
                  batch_dice=batch, include_background=False, smooth=0.1)
    weight = x.new_tensor([0.0, 0.002, 0.005]) if weighted else None
    overlap = (GeneralizedDiceLoss if kind == "gdl_ce" else DiceLoss)(backend="torch", **kwargs)(x, y, mask=valid)
    if kind == "dice_focal":
        fn = DiceFocalLoss(**kwargs, gamma=0.5, alpha=0.2, lambda_dice=0.7, lambda_focal=1.3)
        pointwise = FocalLoss(activation=activation, target_format=kwargs["target_format"],
                              gamma=0.5, alpha=0.2, backend="torch")(x, y, mask=valid)
    else:
        fn = (GeneralizedDiceCELoss if kind == "gdl_ce" else DiceCELoss)(
            **kwargs, class_weight=weight, lambda_dice=0.7, lambda_ce=1.3)
        # Gather only valid voxels, then use PyTorch's native weighted reduction.
        z = x.movedim(1, -1)[valid]
        truth = y.movedim(1, -1)[valid] if soft else F.one_hot(y[valid], 3).to(x.dtype)
        if activation == "sigmoid":
            pointwise = F.binary_cross_entropy_with_logits(z, truth, weight=weight)
        else:
            pointwise = F.cross_entropy(z, truth if soft else y[valid], weight=weight)
    expected = 0.7 * overlap + 1.3 * pointwise
    actual = fn(x, y, mask=valid)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)
    ga, = torch.autograd.grad(actual, x)
    ge, = torch.autograd.grad(expected, x)
    torch.testing.assert_close(ga, ge, rtol=1e-10, atol=1e-13)
    assert torch.count_nonzero(ga.movedim(1, -1)[~valid]) == 0


@pytest.mark.parametrize("cls", [DiceCELoss, GeneralizedDiceCELoss, DiceFocalLoss])
def test_disabled_overlap_is_not_nan(device, cls):
    x = torch.full((1, 2, 3), -1000.0, device=device, requires_grad=True)
    x.data[:, 0] = 1000
    y = torch.zeros(1, 3, device=device, dtype=torch.long)
    actual = cls(lambda_dice=0, include_background=False, smooth=0)(x, y)
    actual.backward()
    assert actual.item() == 0
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("cls", [DiceCELoss, GeneralizedDiceCELoss])
def test_all_zero_class_weights_and_buffer(device, cls):
    x = torch.randn(2, 3, 7, device=device, requires_grad=True)
    y = torch.zeros(2, 7, device=device, dtype=torch.long)
    fn = cls(lambda_dice=0, class_weight=[0, 0, 0]).to(device)
    assert "class_weight" in fn.state_dict()
    actual = fn(x, y)
    actual.backward()
    assert actual.item() == 0
    assert torch.count_nonzero(x.grad) == 0


@pytest.mark.parametrize("cls", [DiceCELoss, GeneralizedDiceCELoss, DiceFocalLoss])
def test_all_ignored_nonfinite_logits(device, cls):
    x = torch.full((2, 3, 7), float("nan"), device=device, requires_grad=True)
    y = torch.full((2, 7), -1, device=device, dtype=torch.long)
    actual = cls(ignore_index=-1, smooth=0)(x, y)
    actual.backward()
    assert actual.item() == 0 and torch.count_nonzero(x.grad) == 0


@pytest.mark.parametrize("bad", [[], [[1, 2]], [1, -1], [float("nan")], [float("inf")]])
def test_weight_validation(bad):
    with pytest.raises(ValueError, match="class_weight"):
        DiceCELoss(class_weight=bad)


def test_coefficients_and_channels():
    for options in ({"lambda_dice": -1}, {"lambda_ce": float("nan")}, {"lambda_dice": 0, "lambda_ce": 0}):
        with pytest.raises(ValueError):
            DiceCELoss(**options)
    with pytest.raises(ValueError, match="one entry"):
        DiceCELoss(class_weight=[1])(torch.zeros(1, 2, 3), torch.zeros(1, 3, dtype=torch.long))


@pytest.mark.parametrize("cls", [DiceCELoss, GeneralizedDiceCELoss, DiceFocalLoss])
def test_soft_target_gradcheck(cls):
    torch.manual_seed(341)
    x = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
    y = torch.rand_like(x, requires_grad=True)
    options = {} if cls == DiceFocalLoss else {"class_weight": [0.2, 0.5, 0.8]}
    fn = cls(target_format="one-hot", backend="torch", **options)
    assert torch.autograd.gradcheck(fn, (x, y))
    assert torch.autograd.gradgradcheck(fn, (x, y))


@pytest.mark.parametrize("cls", [DiceCELoss, GeneralizedDiceCELoss, DiceFocalLoss])
def test_compile_composition(device, cls):
    torch.manual_seed(342)
    x = torch.randn(2, 3, 4, 5, device=device, requires_grad=True)
    y = torch.randint(3, (2, 4, 5), device=device)
    options = {} if cls == DiceFocalLoss else {"class_weight": [0.2, 0.5, 0.8]}
    fn = cls(**options)
    expected = fn(x, y)
    actual = torch.compile(fn, fullgraph=True)(x, y)
    torch.testing.assert_close(actual, expected)
    ga, = torch.autograd.grad(actual, x)
    ge, = torch.autograd.grad(expected, x)
    torch.testing.assert_close(ga, ge)


@pytest.mark.parametrize("cls", [DiceCELoss, GeneralizedDiceCELoss, DiceFocalLoss])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_composed_mixed_precision(device, cls, dtype):
    torch.manual_seed(343)
    x = torch.randn(2, 3, 5, 6, 7, dtype=dtype, device=device, requires_grad=True)
    y = torch.randint(3, (2, 5, 6, 7), device=device)
    mask = torch.rand(y.shape, device=device) > 0.25
    options = {} if cls == DiceFocalLoss else {"class_weight": [0.2, 0.5, 0.8]}
    expected = cls(backend="torch", **options)(x, y, mask=mask)
    actual = cls(**options)(x, y, mask=mask)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
    ga, = torch.autograd.grad(actual, x)
    ge, = torch.autograd.grad(expected, x)
    torch.testing.assert_close(ga, ge, rtol=0.01, atol=1e-6)
