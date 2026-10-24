"""Generalized Dice: independent definitions, reference behavior and derivatives."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.special import expit, softmax

from exacta.losses import DiceLoss, GeneralizedDiceLoss
from test_losses import _both_backends, _fused_inputs, _inputs, needs_gpu


def _reference(logits, labels, *, activation, weight_type, empty_target, batch_dice,
               include_background, smooth):
    probabilities = (softmax(logits, axis=1) if activation == "softmax"
                     else expit(logits) if activation == "sigmoid" else logits)
    truth = np.moveaxis(np.eye(logits.shape[1])[labels], -1, 1)
    if not include_background:
        probabilities, truth = probabilities[:, 1:], truth[:, 1:]
    # Build each case/class statistic separately, without the production reduction helper.
    groups = [range(len(labels))] if batch_dice else [[i] for i in range(len(labels))]
    values = []
    for group in groups:
        volumes, predicted, intersections = [], [], []
        for channel in range(probabilities.shape[1]):
            p, g = probabilities[list(group), channel], truth[list(group), channel]
            volumes.append(float(g.sum()))
            predicted.append(float(p.sum()))
            intersections.append(float((p * g).sum()))
        if weight_type == "uniform":
            weights = [1.0] * len(volumes)
        else:
            power = 2 if weight_type == "square" else 1
            present_weights = [v ** -power for v in volumes if v > 0]
            replacement = max(present_weights, default=1.0 if empty_target == "uniform" else 0.0)
            weights = [v ** -power if v > 0 else replacement for v in volumes]
        if not any(weights):
            values.append(0.0)
        else:
            numerator = smooth + 2 * sum(w * i for w, i in zip(weights, intersections))
            denominator = smooth + sum(w * (p + g) for w, p, g in zip(weights, predicted, volumes))
            values.append(1 - numerator / denominator)
    return np.mean(values)


@pytest.mark.parametrize("activation", ["softmax", "sigmoid", "none"])
@pytest.mark.parametrize("weight_type", ["square", "simple", "uniform"])
@pytest.mark.parametrize("batch_dice", [False, True])
@pytest.mark.parametrize("include_background", [False, True])
@pytest.mark.parametrize("empty_target", ["zero", "uniform"])
def test_independent_formula(device, activation, weight_type, batch_dice, include_background, empty_target):
    logits, target = _inputs(device, shape=(2, 5, 2, 3, 4), dtype=torch.float64)
    target[0] = 0  # An entirely empty foreground case.
    target[1] %= 3  # Two classes absent, alongside present classes of different sizes.
    if activation == "none":
        logits = logits.sigmoid()
    options = dict(activation=activation, weight_type=weight_type, batch_dice=batch_dice,
                   include_background=include_background, empty_target=empty_target, smooth=0.01)
    expected = _reference(logits.cpu().numpy(), target.cpu().numpy(), **options)
    actual = GeneralizedDiceLoss(**options)(logits, target)
    assert actual.dtype == torch.float64
    np.testing.assert_allclose(actual.item(), expected, rtol=2e-12, atol=1e-15)


@pytest.mark.parametrize("weight_type", ["square", "simple", "uniform"])
@pytest.mark.parametrize("batch_dice", [False, True])
@pytest.mark.parametrize("include_background", [False, True])
def test_monai_value_and_gradient(device, weight_type, batch_dice, include_background):
    monai_losses = pytest.importorskip("monai.losses")
    logits, target = _inputs(device, shape=(2, 5, 3, 4, 5), dtype=torch.float64)
    target[0] = 0
    target[1] %= 3
    logits.requires_grad_()
    ours = GeneralizedDiceLoss(weight_type=weight_type, batch_dice=batch_dice,
                               include_background=include_background)(logits, target)
    reference = monai_losses.GeneralizedDiceLoss(
        w_type=weight_type, softmax=True, to_onehot_y=True, batch=batch_dice,
        include_background=include_background,
    )(logits, target[:, None])
    # MONAI computes its class weights in float32 even for float64 inputs.
    torch.testing.assert_close(ours, reference, rtol=1e-6, atol=1e-10)
    torch.testing.assert_close(torch.autograd.grad(ours, logits)[0],
                               torch.autograd.grad(reference, logits)[0], rtol=2e-6, atol=1e-10)


@needs_gpu
@pytest.mark.parametrize("weight_type", ["square", "simple", "uniform"])
@pytest.mark.parametrize("activation", ["softmax", "sigmoid", "none"])
@pytest.mark.parametrize("target_format", ["index", "one-hot"])
@pytest.mark.parametrize("batch_dice", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_value_and_gradient(weight_type, activation, target_format, batch_dice, dtype):
    logits, target = _fused_inputs(activation, target_format, shape=(2, 5, 7, 9, 19), dtype=dtype)
    if target_format == "one-hot":
        target = (logits * 0.7).softmax(1)  # Soft targets also exercise product-based overlap.
        target[:, -1] = 0
    else:
        target %= 4
    logits = logits.to(memory_format=torch.channels_last_3d)
    results = _both_backends(lambda b: GeneralizedDiceLoss(
        weight_type=weight_type, activation=activation, target_format=target_format,
        batch_dice=batch_dice, include_background=False, backend=b,
    ), logits, target)
    tolerance = dict(rtol=2e-12, atol=1e-15) if dtype == torch.float64 else dict(rtol=2e-5, atol=1e-9)
    for expected, actual in zip(*results):
        torch.testing.assert_close(actual, expected, **tolerance)


@pytest.mark.parametrize("weight_type", ["square", "simple"])
@pytest.mark.parametrize("empty_target", ["zero", "uniform"])
@pytest.mark.parametrize("smooth", [0.0, 0.1])
@pytest.mark.parametrize("backend", ["auto", "torch"])
def test_empty_target_policy(device, weight_type, empty_target, smooth, backend):
    logits = torch.zeros(2, 1, 2, 3, 4, dtype=torch.float64, device=device, requires_grad=True)
    target = torch.zeros_like(logits)
    fn = GeneralizedDiceLoss(activation="sigmoid", target_format="one-hot", weight_type=weight_type,
                             empty_target=empty_target, smooth=smooth, backend=backend)
    value = fn(logits, target)
    gradient = torch.autograd.grad(value, logits)[0]
    expected = 0.0 if empty_target == "zero" else 1 - smooth / (12 + smooth)
    assert value.item() == pytest.approx(expected)
    if empty_target == "zero" or smooth == 0:
        assert torch.count_nonzero(gradient) == 0
    else:
        assert (gradient > 0).all()


@pytest.mark.parametrize("batch_dice", [False, True])
@pytest.mark.parametrize("weight_type", ["square", "simple", "uniform"])
def test_finite_difference_input_gradient(device, batch_dice, weight_type):
    logits, target = _inputs(device, shape=(2, 3, 2, 2, 2), dtype=torch.float64)
    fn = GeneralizedDiceLoss(weight_type=weight_type, batch_dice=batch_dice, include_background=False)
    assert torch.autograd.gradcheck(lambda x: fn(x, target), (logits.requires_grad_(),))


@pytest.mark.parametrize("weight_type", ["square", "simple", "uniform"])
def test_soft_target_gradients_and_double_backward(weight_type):
    logits, _ = _inputs("cpu", shape=(1, 3, 2, 2, 2), dtype=torch.float64)
    target = logits.softmax(1).requires_grad_()
    fn = GeneralizedDiceLoss(weight_type=weight_type, target_format="one-hot", backend="torch")
    inputs = (logits.requires_grad_(), target)
    assert torch.autograd.gradcheck(fn, inputs)
    assert torch.autograd.gradgradcheck(fn, inputs)


@needs_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision(dtype):
    logits, target = _fused_inputs("softmax", "index", dtype=dtype)
    results = _both_backends(lambda b: GeneralizedDiceLoss(backend=b), logits, target)
    assert results[1][0].dtype == torch.float32 and results[1][1].dtype == dtype
    for expected, actual in zip(*results):
        torch.testing.assert_close(actual, expected)


@needs_gpu
@pytest.mark.parametrize("batch_dice", [False, True])
def test_compile_and_invalid_labels(batch_dice):
    logits, target = _fused_inputs("softmax", "index", shape=(2, 3, 3, 4, 5))
    fn = GeneralizedDiceLoss(batch_dice=batch_dice, backend="triton")
    eager = logits.clone().requires_grad_()
    compiled = logits.clone().requires_grad_()
    compiled_fn = torch.compile(fn, fullgraph=True)
    expected, actual = fn(eager, target), compiled_fn(compiled, target)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, compiled)[0],
                               torch.autograd.grad(expected, eager)[0])
    target[0, 0, 0, 0] = 2**32 + 1
    assert compiled_fn(logits, target).isnan()
    with pytest.raises(ValueError, match="class indices"):
        fn(logits, target)


@needs_gpu
def test_target_gradients_fall_back_and_double_backward_rejected():
    logits, _ = _fused_inputs("softmax", "index", shape=(1, 3, 2, 2, 2))
    logits.requires_grad_()
    target = logits.detach().softmax(1).requires_grad_()
    actual = GeneralizedDiceLoss(target_format="one-hot")(logits, target)
    assert torch.autograd.grad(actual, target)[0].isfinite().all()
    fused = GeneralizedDiceLoss(target_format="one-hot", backend="triton")
    with pytest.raises(NotImplementedError, match="gradients into the target"):
        fused(logits, target)
    with pytest.raises(RuntimeError, match="double backward"):
        torch.autograd.grad(fused(logits, target.detach()), logits, create_graph=True)


@pytest.mark.parametrize("weight_type", ["square", "simple", "uniform"])
def test_one_hot_and_index_agree(device, weight_type):
    logits, target = _inputs(device)
    onehot = F.one_hot(target, logits.shape[1]).movedim(-1, 1)
    torch.testing.assert_close(GeneralizedDiceLoss(weight_type=weight_type)(logits, target),
                               GeneralizedDiceLoss(weight_type=weight_type, target_format="one-hot")(
                                   logits, onehot))


def test_uniform_is_ratio_of_sums_not_mean_of_ratios():
    probabilities = torch.tensor([[[1.0, 0.7, 0.8], [0.0, 0.3, 0.2]]], dtype=torch.float64)
    labels = torch.tensor([[0, 0, 1]])
    fn = GeneralizedDiceLoss(activation="none", weight_type="uniform", smooth=0)
    assert fn(probabilities, labels).item() == pytest.approx(1 - 1.9 / 3)
    assert not torch.isclose(fn(probabilities, labels), DiceLoss(activation="none", smooth=0)(
        probabilities, labels))


@pytest.mark.parametrize("backend", ["auto", "torch"])
def test_tiny_soft_targets_keep_finite_weights_and_gradients(device, backend):
    probabilities = torch.full((1, 2, 3, 4, 5), 0.25, device=device, requires_grad=True)
    target = torch.full_like(probabilities, 1e-25)
    result = GeneralizedDiceLoss(activation="none", target_format="one-hot", backend=backend)(
        probabilities, target)
    assert result.isfinite()
    assert torch.autograd.grad(result, probabilities)[0].isfinite().all()


@pytest.mark.parametrize("kwargs,match", [({"weight_type": "bad"}, "weight_type"),
                                         ({"empty_target": "bad"}, "empty_target"),
                                         ({"smooth": -1}, "smooth"),
                                         ({"backend": "bad"}, "backend"),
                                         ({"activation": "bad"}, "activation")])
def test_invalid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GeneralizedDiceLoss(**kwargs)


@pytest.mark.parametrize("shape", [(0, 3, 4, 5, 6), (2, 3, 0, 5, 6)])
def test_empty_tensors_match_torch(device, shape):
    logits, target = _inputs(device, shape=shape)
    torch.testing.assert_close(GeneralizedDiceLoss()(logits, target),
                               GeneralizedDiceLoss(backend="torch")(logits, target), equal_nan=True)
