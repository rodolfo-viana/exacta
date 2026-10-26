"""Independent objectives and gradients for squared Dice and Lovasz-hinge."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from exacta.losses import DiceLoss, DiceCELoss, DiceFocalLoss, DiceTopKCELoss, LovaszHingeLoss

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("activation", ["softmax", "sigmoid", "none"])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("soft", [False, True])
def test_squared_dice_objective_gradient(device, dim, activation, batch, soft):
    torch.manual_seed(771)
    x = torch.randn((2, 3, *([3]*dim)), device=device, dtype=torch.float64, requires_grad=True)
    if activation == "none":
        x = x.sigmoid().detach().requires_grad_()
    labels = torch.randint(3, (2, *([3]*dim)), device=device)
    target = torch.rand_like(x) if soft else labels
    mask = torch.rand(labels.shape, device=device) > .3
    mask[0] = False
    kwargs = dict(squared_pred=True, activation=activation, target_format="one-hot" if soft else "index",
                  batch_dice=batch, include_background=False, smooth=.13)
    p = x.softmax(1) if activation == "softmax" else x.sigmoid() if activation == "sigmoid" else x
    g = target if soft else F.one_hot(target, 3).movedim(-1, 1).double()
    p, g = (torch.where(mask[:, None], a, 0)[:, 1:] for a in (p, g))
    dims = (0, *range(2, x.ndim)) if batch else tuple(range(2, x.ndim))
    num = 2*(p*g).sum(dims)+.13
    den = (p*p+g*g).sum(dims)+.13
    ref = (1-num/den).mean() if batch else (1-num/den)[1].mean()
    rg = torch.autograd.grad(ref, x)[0]
    for backend in (["torch", "triton"] if device == "cuda" else ["torch"]):
        value = DiceLoss(backend=backend, **kwargs)(x, target, mask=mask)
        grad = torch.autograd.grad(value, x)[0]
        torch.testing.assert_close(value, ref, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(grad, rg, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cls", [DiceCELoss, DiceFocalLoss, DiceTopKCELoss])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_squared_dice_compositions(device, cls, dtype):
    torch.manual_seed(773)
    x = torch.randn((2, 3, 4, 5), device=device, dtype=dtype, requires_grad=True)
    y = torch.randint(3, (2, 4, 5), device=device)
    y[0, 0] = -1
    kw = dict(squared_pred=True, ignore_index=-1)
    ref = cls(backend="torch", **kw)(x, y)
    expected = torch.autograd.grad(ref, x)[0]
    actual = cls(**kw)(x, y)
    observed = torch.autograd.grad(actual, x)[0]
    torch.testing.assert_close(actual, ref, rtol=.02 if dtype == torch.bfloat16 else .003, atol=1e-5)
    torch.testing.assert_close(observed, expected, rtol=.02, atol=1e-4)


def test_squared_soft_target_gradcheck_and_zero_mask():
    torch.manual_seed(774)
    x = torch.randn(1, 2, 2, 3, dtype=torch.float64, requires_grad=True)
    y = torch.rand_like(x, requires_grad=True)
    loss = DiceLoss(squared_pred=True, target_format="one-hot", backend="torch")
    assert torch.autograd.gradcheck(loss, (x, y))
    assert torch.autograd.gradgradcheck(loss, (x, y))
    loss = DiceLoss(squared_pred=True, target_format="one-hot", smooth=0)
    value = loss(x*torch.nan, y, mask=torch.zeros((1, 2, 3), dtype=torch.bool))
    assert value == 0


def hinge_reference(x, y):
    """Scalar NumPy definition, sorting errors and accumulating Jaccard differences."""
    if not len(x):
        return 0.
    errors = 1-x*(2*y-1)
    order = np.argsort(-errors, kind="stable")
    total, seen_positive, seen_negative = y.sum(), 0, 0
    previous, value = 0., 0.
    for i in order:
        seen_positive += y[i]
        seen_negative += 1-y[i]
        jaccard = 1-(total-seen_positive)/(total+seen_negative)
        value += max(errors[i], 0)*(jaccard-previous)
        previous = jaccard
    return value


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("multilabel", [False, True])
@pytest.mark.parametrize("per_image", [False, True])
def test_hinge_independent_definition(device, dim, multilabel, per_image):
    rng = np.random.default_rng(775)
    shape = (2, 3 if multilabel else 1, *([3]*dim))
    x = rng.normal(size=shape)
    y = rng.integers(0, 2, shape)
    mask = rng.random(shape) > .3
    mask[0, 0] = False
    pairs = []
    for c in range(shape[1]):
        groups = range(shape[0]) if per_image else [slice(None)]
        for b in groups:
            p, t = x[b, c][mask[b, c]], y[b, c][mask[b, c]]
            if len(p):
                pairs.append(hinge_reference(p, t))
    inp = torch.tensor(x, dtype=torch.float64, device=device, requires_grad=True)
    target = torch.tensor(y if multilabel else y[:, 0], device=device)
    observed = torch.tensor(mask, device=device)
    loss = LovaszHingeLoss(multilabel=multilabel, per_image=per_image)
    value = loss(inp, target, mask=observed)
    np.testing.assert_allclose(value.detach().cpu(), np.mean(pairs), rtol=1e-12)
    grad = torch.autograd.grad(value, inp)[0]
    assert torch.isfinite(grad).all()
    assert (grad[~observed] == 0).all()


def test_hinge_gradcheck_endpoints_ties_empty_and_invalid():
    torch.manual_seed(776)
    x = torch.randn(1, 1, 2, 3, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[[1, 0, 1], [0, 0, 1]]])
    loss = LovaszHingeLoss()
    assert torch.autograd.gradcheck(lambda z: loss(z, y), (x,))
    tied = torch.zeros_like(x, requires_grad=True)
    a = loss(tied, y)
    assert a == 1
    torch.testing.assert_close(torch.autograd.grad(a, tied)[0], torch.autograd.grad(loss(tied, y), tied)[0])
    perfect = (2*y[:, None]-1).double().requires_grad_()
    assert loss(perfect, y) == 0
    assert torch.count_nonzero(torch.autograd.grad(loss(perfect, y), perfect)[0]) == 0
    ignored = torch.full_like(y, -1)
    bad = torch.full_like(x, torch.nan, requires_grad=True)
    zero = LovaszHingeLoss(ignore_index=-1)(bad, ignored)
    assert zero == 0
    assert (torch.autograd.grad(zero, bad)[0] == 0).all()
    with pytest.raises(ValueError, match="binary"):
        loss(x, y.double()+1e-9)
    with pytest.raises(ValueError, match="finite"):
        loss(bad, y)
    with pytest.raises(ValueError, match="mask"):
        LovaszHingeLoss(multilabel=True, ignore_index=-1)
    for shape in ((0, 1, 2, 3), (1, 1, 0, 3)):
        z = torch.empty(shape, requires_grad=True)
        t = torch.empty((shape[0], *shape[2:]))
        assert loss(z, t) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_completion_losses_compile(device):
    torch._dynamo.reset()
    try:
        torch.manual_seed(777)
        x = torch.randn(2, 2, 3, 4, device=device, requires_grad=True)
        t = torch.randint(2, x.shape, device=device)
        for loss in (LovaszHingeLoss(multilabel=True),
                     DiceLoss(squared_pred=True, target_format="one-hot", activation="sigmoid")):
            gold = loss(x, t)
            expected = torch.autograd.grad(gold, x)[0]
            actual = torch.compile(loss)(x, t)
            torch.testing.assert_close(actual, gold)
            torch.testing.assert_close(torch.autograd.grad(actual, x)[0], expected)
    finally:
        torch._dynamo.reset()
