import warnings

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from exacta import _triton_losses
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


# --- Fused Triton kernels -----------------------------------------------------------------------

GPU = torch.cuda.is_available() and _triton_losses.available()
needs_gpu = pytest.mark.skipif(not GPU, reason="needs CUDA and Triton")

FUSED_CONFIGS = [
    (loss_cls, activation, target_format, include_background, batch_dice)
    for loss_cls in (DiceLoss, DiceCELoss)
    for activation in ("softmax", "sigmoid", "none")
    for target_format in ("index", "one-hot")
    for include_background in (True, False)
    for batch_dice in (False, True)
    if not (loss_cls is DiceCELoss and activation == "none")
]


def _both_backends(loss_fn_factory, logits, target):
    """Loss and input gradient from the PyTorch and the Triton backend."""
    out = []
    for backend in ("torch", "triton"):
        x = logits.detach().clone().requires_grad_()
        loss = loss_fn_factory(backend)(x, target)
        loss.backward()
        out.append((loss.detach(), x.grad))
    return out


def _fused_inputs(activation, target_format, shape=(2, 4, 9, 10, 11), dtype=torch.float32, seed=0):
    logits, target = _inputs("cuda", shape=shape, seed=seed, dtype=dtype)
    if activation == "none":
        logits = logits.softmax(1)
    if target_format == "one-hot":
        target = F.one_hot(target, shape[1]).movedim(-1, 1)
    return logits, target


CONFIG_NAMES = ("loss_cls", "activation", "target_format", "include_background", "batch_dice")


@needs_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(CONFIG_NAMES, FUSED_CONFIGS)
def test_fused_matches_torch(loss_cls, activation, target_format, include_background, batch_dice, dtype):
    logits, target = _fused_inputs(activation, target_format, dtype=dtype)
    kw = dict(
        activation=activation,
        target_format=target_format,
        include_background=include_background,
        batch_dice=batch_dice,
    )
    (loss_t, grad_t), (loss_f, grad_f) = _both_backends(lambda b: loss_cls(backend=b, **kw), logits, target)
    tol = dict(rtol=1e-12, atol=1e-15) if dtype == torch.float64 else dict(rtol=1e-5, atol=1e-8)
    assert loss_f.dtype == loss_t.dtype
    torch.testing.assert_close(loss_f, loss_t, **tol)
    torch.testing.assert_close(grad_f, grad_t, **tol)


@needs_gpu
@pytest.mark.parametrize("shape", [(3, 2, 17, 19), (1, 5, 1000), (2, 3, 4, 5, 6, 7)])
def test_fused_other_shapes(shape):
    logits, target = _fused_inputs("softmax", "index", shape=shape)
    (loss_t, grad_t), (loss_f, grad_f) = _both_backends(lambda b: DiceCELoss(backend=b), logits, target)
    torch.testing.assert_close(loss_f, loss_t)
    torch.testing.assert_close(grad_f, grad_t)


@needs_gpu
@pytest.mark.parametrize("classes", [1, 117])
def test_fused_class_counts(classes):
    # 1 class: binary sigmoid segmentation; 117: TotalSegmentator-sized, padded to 128 channels
    activation = "sigmoid" if classes == 1 else "softmax"
    logits, target = _fused_inputs(activation, "index", shape=(2, classes, 6, 7, 8))
    (loss_t, grad_t), (loss_f, grad_f) = _both_backends(
        lambda b: DiceCELoss(activation=activation, backend=b), logits, target
    )
    torch.testing.assert_close(loss_f, loss_t)
    torch.testing.assert_close(grad_f, grad_t)


@needs_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_half_precision(dtype):
    logits, target = _fused_inputs("softmax", "index", dtype=dtype)
    (loss_t, grad_t), (loss_f, grad_f) = _both_backends(lambda b: DiceCELoss(backend=b), logits, target)
    assert loss_f.dtype == torch.float32 and grad_f.dtype == dtype
    torch.testing.assert_close(loss_f, loss_t)  # both compute in float32 from the same values
    torch.testing.assert_close(grad_f, grad_t)


@needs_gpu
@pytest.mark.parametrize(CONFIG_NAMES, FUSED_CONFIGS[::3])
def test_fused_gradcheck(loss_cls, activation, target_format, include_background, batch_dice):
    logits, target = _fused_inputs(activation, target_format, shape=(2, 3, 3, 4, 2), dtype=torch.float64)
    loss_fn = loss_cls(
        activation=activation,
        target_format=target_format,
        include_background=include_background,
        batch_dice=batch_dice,
        backend="triton",
    )
    assert torch.autograd.gradcheck(lambda x: loss_fn(x, target), (logits.requires_grad_(),))


@needs_gpu
def test_fused_accepts_non_contiguous_input():
    logits, target = _fused_inputs("softmax", "index")
    strided = logits.to(memory_format=torch.channels_last_3d)
    assert not strided.is_contiguous()
    loss_fn = DiceCELoss(backend="triton")
    torch.testing.assert_close(loss_fn(strided, target), loss_fn(logits, target))


@needs_gpu
def test_fused_matches_monai():
    monai_losses = pytest.importorskip("monai.losses")
    logits, target = _inputs("cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = monai_losses.DiceCELoss(softmax=True, to_onehot_y=True)(logits, target[:, None])
    got = DiceCELoss(backend="triton")(logits.cuda(), target.cuda())
    torch.testing.assert_close(got.cpu(), expected)


@needs_gpu
def test_fused_rejects_out_of_range_classes():
    logits, target = _fused_inputs("softmax", "index")
    with pytest.raises(ValueError, match="class indices must lie in"):
        DiceLoss(backend="triton")(logits, target + 1)


def test_backend_validation():
    with pytest.raises(ValueError, match="backend must be"):
        DiceLoss(backend="cuda")
    logits, target = _inputs("cpu")
    with pytest.raises(NotImplementedError, match="needs a CUDA input"):
        DiceLoss(backend="triton")(logits, target)


@needs_gpu
def test_fused_compiles_without_graph_breaks():
    logits, target = _fused_inputs("softmax", "index", shape=(2, 4, 8, 9, 10))
    loss_fn = DiceCELoss(backend="triton")
    compiled = torch.compile(lambda x, t: loss_fn(x, t), fullgraph=True)  # fullgraph: fail on any break
    eager_x, compiled_x = logits.clone().requires_grad_(), logits.clone().requires_grad_()
    eager, got = loss_fn(eager_x, target), compiled(compiled_x, target)
    eager.backward()
    got.backward()
    torch.testing.assert_close(got, eager)
    torch.testing.assert_close(compiled_x.grad, eager_x.grad)


@needs_gpu
def test_fused_out_of_range_classes_give_nan_when_compiled():
    logits, target = _fused_inputs("softmax", "index", shape=(2, 4, 8, 9, 10))
    target[1, 2, 3, 4] = 4
    loss_fn = DiceLoss(backend="triton")
    assert torch.compile(lambda x, t: loss_fn(x, t), fullgraph=True)(logits, target).isnan()


# --- Regressions from code review ---------------------------------------------------------------


@needs_gpu
def test_fused_handles_classes_masked_with_neg_inf_logits():
    # Masking absent classes with -inf logits is common; 0 * inf must not reach the loss.
    logits, target = _fused_inputs("softmax", "index", shape=(2, 4, 8, 9, 10))
    logits[:, 3] = float("-inf")
    target = target % 3
    (loss_t, grad_t), (loss_f, grad_f) = _both_backends(lambda b: DiceCELoss(backend=b), logits, target)
    assert torch.isfinite(loss_f)
    torch.testing.assert_close(loss_f, loss_t)
    torch.testing.assert_close(grad_f, grad_t)
    one_hot = F.one_hot(target, 4).movedim(-1, 1)
    torch.testing.assert_close(DiceCELoss(target_format="one-hot", backend="triton")(logits, one_hot), loss_f)


@needs_gpu
@pytest.mark.parametrize("loss_cls", [DiceLoss, DiceCELoss])
def test_gradients_reach_a_soft_target(loss_cls):
    logits, target = _fused_inputs("softmax", "index", shape=(2, 3, 6, 7, 8))
    teacher = torch.randn_like(logits).softmax(1).requires_grad_()
    grads = []
    for backend in ("torch", "auto"):
        teacher.grad = None
        loss_cls(target_format="one-hot", backend=backend)(logits, teacher).backward()
        grads.append(teacher.grad)
    assert grads[1] is not None  # "auto" falls back to PyTorch rather than dropping the gradient
    torch.testing.assert_close(grads[1], grads[0])
    with pytest.raises(NotImplementedError, match="gradients into the target"):
        loss_cls(target_format="one-hot", backend="triton")(logits, teacher)


@needs_gpu
def test_fused_double_backward_raises():
    logits, target = _fused_inputs("softmax", "index", shape=(2, 3, 6, 7, 8))
    x = logits.requires_grad_()
    loss = DiceCELoss(backend="triton")(x, target)
    # A gradient penalty needs create_graph=True; the fused backward cannot provide it and must
    # say so instead of returning a gradient without a graph (the penalty would silently vanish).
    with pytest.raises(RuntimeError, match="does not support double backward"):
        torch.autograd.grad(loss, x, create_graph=True)
    assert torch.autograd.grad(DiceCELoss(backend="triton")(x, target), x)[0].isfinite().all()


@needs_gpu
@pytest.mark.parametrize("shape", [(0, 3, 4, 5, 6), (2, 3, 0, 5, 6)])
def test_fused_path_returns_nan_on_empty_inputs(shape):
    logits, target = _fused_inputs("softmax", "index", shape=shape)
    assert DiceCELoss(backend="auto")(logits, target).isnan()
    assert DiceCELoss(backend="torch")(logits, target).isnan()


@needs_gpu
def test_compiled_indices_beyond_int32_give_nan():
    logits, target = _fused_inputs("softmax", "index", shape=(2, 4, 8, 9, 10))
    target[0, 1, 2, 3] = 2**32 + 1  # would wrap to class 1 in 32 bits
    loss_fn = DiceLoss(backend="triton")
    assert torch.compile(lambda x, t: loss_fn(x, t), fullgraph=True)(logits, target).isnan()


@needs_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("offset", [1e4, 1e8, -1e8])
@pytest.mark.parametrize("target_format", ["index", "one-hot"])
def test_fused_ce_stable_under_logit_offsets(dtype, offset, target_format):
    logits, target = _fused_inputs("softmax", target_format, shape=(2, 4, 2, 3, 5), dtype=dtype)
    logits = logits + offset
    (loss_t, grad_t), (loss_f, grad_f) = _both_backends(
        lambda b: DiceCELoss(backend=b, lambda_dice=0.0, target_format=target_format), logits, target
    )
    tol = dict(rtol=1e-12, atol=1e-15) if dtype == torch.float64 else dict(rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(loss_f, loss_t, **tol)
    torch.testing.assert_close(grad_f, grad_t, **tol)
