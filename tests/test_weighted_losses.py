"""Independent definitions, reference libraries and derivative checks for weighted losses."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.special import expit, softmax

from exacta.losses import DiceCELoss, DiceLoss, FocalLoss, GeneralizedDiceLoss, TverskyLoss
from test_losses import _both_backends, _fused_inputs, _inputs, needs_gpu


@pytest.mark.parametrize("activation", ["softmax", "sigmoid", "none"])
@pytest.mark.parametrize("batch_dice", [False, True])
@pytest.mark.parametrize("include_background", [False, True])
def test_tversky_matches_independent_formula(device, activation, batch_dice, include_background):
    logits, target = _inputs(device, dtype=torch.float64)
    if activation == "none":
        logits = logits.sigmoid()
    x = logits.cpu().numpy()
    p = softmax(x, axis=1) if activation == "softmax" else expit(x) if activation == "sigmoid" else x
    g = np.eye(x.shape[1])[target.cpu().numpy()].transpose(0, 4, 1, 2, 3)
    if not include_background:
        p, g = p[:, 1:], g[:, 1:]
    dims = (0, 2, 3, 4) if batch_dice else (2, 3, 4)
    tp = (p * g).sum(dims)
    fp = (p * (1 - g)).sum(dims)
    fn = ((1 - p) * g).sum(dims)
    expected = 1 - ((tp + 0.1) / (tp + 0.2 * fp + 0.8 * fn + 0.1)).mean()
    got = TverskyLoss(alpha=0.2, beta=0.8, smooth=0.1, activation=activation,
                      batch_dice=batch_dice, include_background=include_background)(logits, target)
    np.testing.assert_allclose(got.item(), expected, rtol=1e-12)


@pytest.mark.parametrize("alpha,beta", [(0.0, 1.0), (1.0, 0.0), (0.2, 0.8), (0.7, 1.2)])
@pytest.mark.parametrize("activation", ["softmax", "sigmoid", "none"])
@pytest.mark.parametrize("batch_dice", [False, True])
@pytest.mark.parametrize("include_background", [False, True])
@needs_gpu
def test_tversky_fused_soft_targets(alpha, beta, activation, batch_dice, include_background):
    logits, _ = _fused_inputs(activation, "index", shape=(2, 3, 3, 4, 5), dtype=torch.float64)
    target = (logits * 0.7).softmax(1)
    results = _both_backends(lambda b: TverskyLoss(
        alpha=alpha, beta=beta, activation=activation, target_format="one-hot", batch_dice=batch_dice,
        include_background=include_background, backend=b,
    ), logits, target)
    for ref, got in zip(*results):
        torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-15)


def test_tversky_dice_relationship_includes_smoothing(device):
    logits, target = _inputs(device)
    torch.testing.assert_close(TverskyLoss(smooth=2.0)(logits, target),
                               DiceLoss(smooth=4.0)(logits, target))


def test_tversky_false_negative_penalty_direction(device):
    truth = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]], device=device)
    missing = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=device)
    extra = torch.tensor([[[1.0, 1.0, 1.0, 0.0]]], device=device)
    recall_loss = TverskyLoss(alpha=0.2, beta=0.8, activation="none", target_format="one-hot")
    precision_loss = TverskyLoss(alpha=0.8, beta=0.2, activation="none", target_format="one-hot")
    assert recall_loss(missing, truth) > precision_loss(missing, truth)
    assert recall_loss(extra, truth) < precision_loss(extra, truth)


@pytest.mark.parametrize("activation", ["softmax", "sigmoid"])
@pytest.mark.parametrize("alpha", [None, 0.3])
@pytest.mark.parametrize("gamma", [0.0, 0.5, 2.0])
def test_focal_matches_independent_formula(device, activation, alpha, gamma):
    logits, _ = _inputs(device, dtype=torch.float64)
    # Fractional targets exercise the documented linear extension of hard-label focal loss.
    truth = (logits * 0.7).softmax(1)
    x, g = logits.cpu().numpy(), truth.cpu().numpy()
    if activation == "softmax":
        p = softmax(x, axis=1)
        terms = -g * (1 - p) ** gamma * np.log(p)
        if alpha is not None:
            terms *= np.array([1 - alpha, alpha, alpha]).reshape(1, 3, 1, 1, 1)
        expected = terms.sum(1).mean()
    else:
        p = expit(x)
        a, b = (1, 1) if alpha is None else (alpha, 1 - alpha)
        expected = (-a * g * (1 - p) ** gamma * np.log(p)
                    - b * (1 - g) * p ** gamma * np.log1p(-p)).mean()
    got = FocalLoss(activation=activation, alpha=alpha, gamma=gamma,
                    target_format="one-hot")(logits, truth)
    np.testing.assert_allclose(got.item(), expected, rtol=1e-12)


@pytest.mark.parametrize("activation", ["softmax", "sigmoid"])
@pytest.mark.parametrize("target_format", ["index", "one-hot"])
def test_focal_gamma_zero_equals_cross_entropy(device, activation, target_format):
    logits, indices = _inputs(device, dtype=torch.float64)
    truth = F.one_hot(indices, logits.shape[1]).movedim(-1, 1).double()
    target = indices if target_format == "index" else truth
    logits.requires_grad_()
    expected = (F.cross_entropy(logits, indices) if activation == "softmax"
                else F.binary_cross_entropy_with_logits(logits, truth))
    got = FocalLoss(activation=activation, target_format=target_format, gamma=0)(logits, target)
    torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-15)
    torch.testing.assert_close(torch.autograd.grad(got, logits)[0],
                               torch.autograd.grad(expected, logits)[0], rtol=1e-12, atol=1e-15)


@needs_gpu
@pytest.mark.parametrize("activation", ["softmax", "sigmoid"])
@pytest.mark.parametrize("target_format", ["index", "one-hot"])
@pytest.mark.parametrize("alpha", [None, 0.0, 0.3, 1.0])
@pytest.mark.parametrize("gamma", [0.0, 0.5, 2.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_focal_fused_matches_autograd(activation, target_format, alpha, gamma, dtype):
    logits, target = _fused_inputs(activation, target_format, shape=(2, 3, 3, 4, 5), dtype=dtype)
    if target_format == "one-hot":
        target = (logits * 0.7).softmax(1)
    results = _both_backends(lambda b: FocalLoss(
        activation=activation, target_format=target_format, alpha=alpha, gamma=gamma, backend=b,
    ), logits, target)
    tol = dict(rtol=1e-12, atol=1e-15) if dtype == torch.float64 else dict(rtol=1e-5, atol=1e-8)
    for ref, got in zip(*results):
        torch.testing.assert_close(got, ref, **tol)


@pytest.mark.parametrize("loss_cls", [TverskyLoss, FocalLoss])
@pytest.mark.parametrize("activation", ["softmax", "sigmoid"])
def test_weighted_losses_gradcheck(device, loss_cls, activation):
    logits, target = _inputs(device, shape=(2, 3, 2, 2, 3), dtype=torch.float64)
    fn = loss_cls(activation=activation, alpha=0.3)
    assert torch.autograd.gradcheck(lambda x: fn(x, target), (logits.requires_grad_(),))


@pytest.mark.parametrize("loss_cls", [TverskyLoss, FocalLoss])
@pytest.mark.parametrize("activation", ["softmax", "sigmoid"])
def test_weighted_losses_torch_double_backward_and_target_grad(loss_cls, activation):
    logits, _ = _inputs("cpu", shape=(1, 3, 2, 2, 2), dtype=torch.float64)
    truth = logits.softmax(1).requires_grad_()
    fn = loss_cls(activation=activation, alpha=0.3, target_format="one-hot", backend="torch")
    inputs = (logits.requires_grad_(), truth)
    assert torch.autograd.gradcheck(fn, inputs)
    assert torch.autograd.gradgradcheck(fn, inputs)


@pytest.mark.parametrize("activation", ["softmax", "sigmoid"])
@pytest.mark.parametrize("alpha", [None, 0.3])
def test_focal_matches_monai_on_hard_labels(device, activation, alpha):
    losses = pytest.importorskip("monai.losses")
    logits, target = _inputs(device)
    reference = losses.FocalLoss(to_onehot_y=True, use_softmax=activation == "softmax",
                                 alpha=alpha, gamma=0.5, reduction="none")(logits, target[:, None])
    # MONAI's mean also divides by C in softmax mode. Our reduction agrees with CE at gamma=0.
    expected = reference.sum(1).mean() if activation == "softmax" else reference.mean()
    torch.testing.assert_close(FocalLoss(activation=activation, alpha=alpha, gamma=0.5)(logits, target),
                               expected)


@pytest.mark.parametrize("batch_dice", [False, True])
def test_tversky_matches_monai(device, batch_dice):
    losses = pytest.importorskip("monai.losses")
    logits, target = _inputs(device)
    expected = losses.TverskyLoss(to_onehot_y=True, softmax=True, include_background=False,
                                  batch=batch_dice, alpha=0.2, beta=0.8)(logits, target[:, None])
    torch.testing.assert_close(TverskyLoss(alpha=0.2, beta=0.8, include_background=False,
                                          batch_dice=batch_dice)(logits, target), expected)


@pytest.mark.parametrize("gamma", [0.0, 0.1, 0.5, 2.0])
@pytest.mark.parametrize("activation", ["softmax", "sigmoid"])
@pytest.mark.parametrize("backend", ["auto", "torch"])
def test_focal_confident_predictions_have_finite_gradients(device, gamma, activation, backend):
    logits = torch.tensor([[[-1e4, 50.0, 1e4], [1e4, -50.0, -1e4]]], device=device, requires_grad=True)
    target = torch.tensor([[1, 0, 0]], device=device)
    fn = FocalLoss(activation=activation, gamma=gamma, backend=backend)
    loss = fn(logits, target)
    grad = torch.autograd.grad(loss, logits)[0]
    assert torch.isfinite(loss) and torch.isfinite(grad).all()
    assert loss < 1e-10
    # Equally confident but wrong: finite, nonzero loss and gradients.
    wrong = fn(logits, 1 - target)
    assert torch.isfinite(wrong) and wrong > 1000
    grad = torch.autograd.grad(wrong, logits)[0]
    assert torch.isfinite(grad).all() and grad.abs().max() > 0.1


@needs_gpu
@pytest.mark.parametrize("loss_cls", [TverskyLoss, FocalLoss])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_weighted_losses_low_precision(loss_cls, dtype):
    logits, target = _fused_inputs("softmax", "index", dtype=dtype)
    results = _both_backends(lambda b: loss_cls(backend=b, alpha=0.3), logits, target)
    assert results[1][0].dtype == torch.float32 and results[1][1].dtype == dtype
    for ref, got in zip(*results):
        torch.testing.assert_close(got, ref)


@needs_gpu
@pytest.mark.parametrize("loss_cls", [TverskyLoss, FocalLoss])
def test_weighted_losses_compile_fullgraph(loss_cls):
    logits, target = _fused_inputs("softmax", "index", shape=(2, 3, 3, 4, 5))
    fn = loss_cls(alpha=0.3, backend="triton")
    eager = logits.clone().requires_grad_()
    compiled = logits.clone().requires_grad_()
    expected = fn(eager, target)
    got = torch.compile(fn, fullgraph=True)(compiled, target)
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(torch.autograd.grad(got, compiled)[0],
                               torch.autograd.grad(expected, eager)[0])
    invalid = target.clone()
    invalid[0, 0, 0, 0] = 2**32 + 1
    assert torch.compile(fn, fullgraph=True)(logits, invalid).isnan()
    with pytest.raises(ValueError, match="class indices"):
        fn(logits, invalid)


@needs_gpu
@pytest.mark.parametrize("loss_cls", [TverskyLoss, FocalLoss])
def test_weighted_losses_autograd_limitations_are_explicit(loss_cls):
    logits, _ = _fused_inputs("softmax", "index", shape=(2, 3, 3, 4, 5))
    logits.requires_grad_()
    target = logits.detach().softmax(1).requires_grad_()
    auto = loss_cls(target_format="one-hot")(logits, target)
    assert torch.autograd.grad(auto, target)[0].isfinite().all()
    fused = loss_cls(target_format="one-hot", backend="triton")
    with pytest.raises(NotImplementedError, match="gradients into the target"):
        fused(logits, target)
    with pytest.raises(RuntimeError, match="double backward"):
        torch.autograd.grad(fused(logits, target.detach()), logits, create_graph=True)


@pytest.mark.parametrize("loss_cls", [DiceLoss, DiceCELoss, TverskyLoss, FocalLoss, GeneralizedDiceLoss])
def test_losses_reject_noninteger_indices_and_invalid_input(device, loss_cls):
    logits, target = _inputs(device)
    with pytest.raises(TypeError, match="integer dtype"):
        loss_cls()(logits, target.float() + 0.1)
    with pytest.raises(TypeError, match="floating-point"):
        loss_cls()(logits.long(), target)
    with pytest.raises(ValueError, match="at least one channel"):
        loss_cls()(logits[:, :0], target)
    with pytest.raises(ValueError, match="target_format"):
        loss_cls(target_format="invalid")


@needs_gpu
@pytest.mark.parametrize("loss_cls", [DiceLoss, DiceCELoss, TverskyLoss, FocalLoss, GeneralizedDiceLoss])
@pytest.mark.parametrize("target_format", ["index", "one-hot"])
def test_losses_reject_device_mismatch(loss_cls, target_format):
    logits, target = _fused_inputs("softmax", target_format)
    with pytest.raises(ValueError, match="same device"):
        loss_cls(target_format=target_format)(logits, target.cpu())


@pytest.mark.parametrize("loss_cls,param", [(TverskyLoss, "alpha"), (TverskyLoss, "beta"),
                                          (TverskyLoss, "smooth"), (FocalLoss, "gamma"),
                                          (FocalLoss, "alpha")])
@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_weighted_losses_invalid_parameters(loss_cls, param, value):
    with pytest.raises(ValueError, match=param):
        loss_cls(**{param: value})


def test_focal_invalid_alpha_and_activation():
    with pytest.raises(ValueError, match="alpha"):
        FocalLoss(alpha=1.01)
    with pytest.raises(ValueError, match="activation"):
        FocalLoss(activation="none")


@needs_gpu
@pytest.mark.parametrize("loss_cls", [TverskyLoss, FocalLoss])
@pytest.mark.parametrize("classes", [1, 5, 117])
def test_weighted_losses_strided_inputs_and_class_counts(loss_cls, classes):
    activation = "sigmoid" if classes == 1 else "softmax"
    logits, target = _fused_inputs(activation, "index", shape=(2, classes, 4, 5, 6))
    logits = logits.to(memory_format=torch.channels_last_3d)
    target = F.one_hot(target, classes).movedim(-1, 1)
    results = _both_backends(lambda b: loss_cls(activation=activation, target_format="one-hot", backend=b),
                            logits, target)
    for ref, got in zip(*results):
        torch.testing.assert_close(got, ref)


@needs_gpu
@pytest.mark.parametrize("activation", ["softmax", "sigmoid"])
@pytest.mark.parametrize("offset", [1e4, 1e8, -1e8])
def test_focal_large_logit_offsets(activation, offset):
    logits, target = _fused_inputs(activation, "index")
    results = _both_backends(lambda b: FocalLoss(activation=activation, backend=b), logits + offset, target)
    for ref, got in zip(*results):
        torch.testing.assert_close(got, ref)


@pytest.mark.parametrize("loss_cls", [TverskyLoss, FocalLoss])
@pytest.mark.parametrize("shape", [(0, 3, 4, 5, 6), (2, 3, 0, 5, 6)])
def test_weighted_losses_empty_inputs_match_torch(device, loss_cls, shape):
    logits, target = _inputs(device, shape=shape)
    torch.testing.assert_close(loss_cls()(logits, target), loss_cls(backend="torch")(logits, target),
                               equal_nan=True)
