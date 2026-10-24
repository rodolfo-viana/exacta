"""Ignored voxels must behave like removing those observations before computing loss."""

import pytest
import torch
import torch.nn.functional as F

from exacta.losses import DiceCELoss, DiceLoss, FocalLoss, GeneralizedDiceLoss, TverskyLoss
from test_losses import _inputs, needs_gpu

LOSSES = [DiceLoss, DiceCELoss, TverskyLoss, GeneralizedDiceLoss, FocalLoss]


def _options(cls, activation, target_format, batch_dice=False):
    common = dict(activation=activation, target_format=target_format)
    if cls is FocalLoss:
        return dict(**common, gamma=0.5, alpha=0.3)
    common.update(include_background=False, smooth=0.01, batch_dice=batch_dice)
    if cls is TverskyLoss:
        common.update(alpha=0.2, beta=0.8)
    return common


def _cropped_reference(cls, logits, target, valid, options):
    """Physically select observations; never zero or multiply full volumes by a mask."""
    inputs, targets = [], []
    one_hot = options['target_format'] == 'one-hot'
    for b in range(len(logits)):
        keep = valid[b].flatten()
        if keep.any():
            inputs.append(logits[b].flatten(1)[:, keep].unsqueeze(0))
            targets.append((target[b].flatten(1)[:, keep] if one_hot else target[b].flatten()[keep]).unsqueeze(0))
    if not inputs:
        return logits.flatten()[:0].sum()
    combined_input, combined_target = torch.cat(inputs, -1), torch.cat(targets, -1)
    if cls is FocalLoss:
        return cls(backend='torch', **options)(combined_input, combined_target)
    overlap_cls = DiceLoss if cls is DiceCELoss else cls
    fn = overlap_cls(backend='torch', **options)
    overlap = (fn(combined_input, combined_target) if options['batch_dice'] else
               torch.stack([fn(x, y) for x, y in zip(inputs, targets)]).mean())
    if cls is not DiceCELoss:
        return overlap
    if options['activation'] == 'softmax':
        ce = F.cross_entropy(combined_input, combined_target)
    else:
        truth = combined_target if one_hot else F.one_hot(combined_target, logits.shape[1]).movedim(-1, 1)
        ce = F.binary_cross_entropy_with_logits(combined_input, truth.to(logits.dtype))
    return overlap + ce


@pytest.mark.parametrize('cls', LOSSES)
@pytest.mark.parametrize('activation', ['softmax', 'sigmoid'])
@pytest.mark.parametrize('mode', ['mask', 'ignore', 'both', 'soft'])
@pytest.mark.parametrize('batch_dice', [False, True])
def test_matches_removing_observations(device, cls, activation, mode, batch_dice):
    logits, target = _inputs(device, shape=(3, 4, 2, 3, 4), dtype=torch.float64)
    logits.requires_grad_()
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[0] = False
    mask[1, :, :, ::2] = False
    mask[2, :, 0] = False
    supplied_mask = mask[:, None] if mode in ('mask', 'both', 'soft') else None
    valid = mask.clone()
    ignore_index = -100 if mode in ('ignore', 'both') else None
    if mode == 'soft':
        target = logits.detach().softmax(1)
        target = torch.where(mask[:, None], target, torch.nan)
    elif mode == 'ignore':
        target[~mask] = -100
    else:
        target[~mask] = 2**32 + 1  # Values outside the class range are allowed only when excluded.
        if mode == 'both':
            target[2, :, 2] = -100
            valid &= target != -100
    options = _options(cls, activation, 'one-hot' if mode == 'soft' else 'index', batch_dice)
    logits = torch.where(valid[:, None], logits.detach(), torch.nan).requires_grad_()
    actual = cls(ignore_index=ignore_index, **options)(logits, target, mask=supplied_mask)
    expected = _cropped_reference(cls, logits, target, valid, options)
    torch.testing.assert_close(actual, expected, rtol=3e-11, atol=1e-13)
    actual_grad, = torch.autograd.grad(actual, logits)
    expected_grad, = torch.autograd.grad(expected, logits)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-10, atol=1e-13)
    assert torch.count_nonzero(actual_grad.movedim(1, -1)[~valid]) == 0


@pytest.mark.parametrize('cls', LOSSES)
@pytest.mark.parametrize('backend', ['auto', 'torch'])
@pytest.mark.parametrize('activation', ['softmax', 'sigmoid'])
@pytest.mark.parametrize('mode', ['mask', 'ignore', 'soft'])
def test_fully_ignored_nonfinite_batch_returns_zero(device, cls, backend, activation, mode):
    logits = torch.full((2, 3, 2, 3, 4), torch.nan, device=device, requires_grad=True)
    target = torch.full((2, 2, 3, 4), -100, device=device, dtype=torch.long)
    mask = torch.zeros_like(target, dtype=torch.bool) if mode != 'ignore' else None
    options = _options(cls, activation, 'one-hot' if mode == 'soft' else 'index')
    if cls is not FocalLoss:
        options['smooth'] = 0
    if mode == 'soft':
        target = torch.full_like(logits, torch.nan, requires_grad=True)
    fn = cls(backend=backend, ignore_index=-100 if mode == 'ignore' else None, **options)
    result = fn(logits, target, mask=mask)
    inputs = (logits, target) if mode == 'soft' else (logits,)
    gradients = torch.autograd.grad(result, inputs)
    assert result.item() == 0
    for gradient in gradients:
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) == 0


@needs_gpu
@pytest.mark.parametrize('cls', LOSSES)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_fused_partial_tiles_and_strides(cls, dtype):
    logits, target = _inputs('cuda', shape=(2, 5, 7, 9, 19), dtype=dtype)
    logits = logits.to(memory_format=torch.channels_last_3d)
    mask = (target != 2).transpose(-1, -2).contiguous().transpose(-1, -2)
    target[:, 0] = -100
    values = []
    for backend in ('torch', 'triton'):
        x = logits.detach().clone().requires_grad_()
        loss = cls(backend=backend, ignore_index=-100)(x, target, mask=mask)
        values.append((loss, torch.autograd.grad(loss, x)[0]))
    for expected, actual in zip(*values):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('cls', LOSSES)
def test_finite_difference_gradients(device, cls):
    logits, target = _inputs(device, shape=(2, 3, 2, 2, 2), dtype=torch.float64)
    target[0, 0] = -100
    mask = target != 2
    fn = cls(ignore_index=-100)
    assert torch.autograd.gradcheck(lambda x: fn(x, target, mask=mask), (logits.requires_grad_(),))


@pytest.mark.parametrize('cls', LOSSES)
def test_soft_target_gradients_and_second_derivatives(cls):
    logits, _ = _inputs('cpu', shape=(1, 3, 2, 2, 2), dtype=torch.float64)
    target = logits.softmax(1).requires_grad_()
    mask = torch.ones((1, 2, 2, 2), dtype=torch.bool)
    mask[:, 0] = False
    fn = cls(target_format='one-hot', backend='torch')
    inputs = (logits.requires_grad_(), target)
    assert torch.autograd.gradcheck(lambda x, y: fn(x, y, mask=mask), inputs)
    assert torch.autograd.gradgradcheck(lambda x, y: fn(x, y, mask=mask), inputs)


@needs_gpu
@pytest.mark.parametrize('cls', LOSSES)
@pytest.mark.parametrize('target_format', ['index', 'one-hot'])
def test_compile_fullgraph(cls, target_format):
    torch._dynamo.reset()  # Each loss/target combination is an independent compilation test.
    logits, target = _inputs('cuda', shape=(2, 3, 2, 3, 4))
    mask = target != 2
    options = dict(target_format=target_format, backend='triton')
    if target_format == 'index':
        options['ignore_index'] = -100
        target[:, 0] = -100
    else:
        target = F.one_hot(target, 3).movedim(-1, 1).float()
    fn = cls(**options)
    compiled_fn = torch.compile(fn, fullgraph=True)
    eager = logits.clone().requires_grad_()
    compiled = logits.clone().requires_grad_()
    expected = fn(eager, target, mask=mask)
    actual = compiled_fn(compiled, target, mask=mask)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, compiled)[0], torch.autograd.grad(expected, eager)[0])
    if target_format == 'index':
        target[1, 1, 0, 0] = 2**32 + 1
        mask[1, 1, 0, 0] = True
        assert compiled_fn(logits, target, mask=mask).isnan()
        with pytest.raises(ValueError, match='class indices'):
            fn(logits, target, mask=mask)


@pytest.mark.parametrize('cls', LOSSES)
@pytest.mark.parametrize('ignore_index', [0, 255, -(2**63), 2**63 - 1])
def test_ignore_index_range(device, cls, ignore_index):
    logits, target = _inputs(device, shape=(2, 3, 2, 3, 4))
    target[:, 0] = ignore_index
    expected = cls()(logits, torch.where(target == ignore_index, 0, target), mask=target != ignore_index)
    actual = cls(ignore_index=ignore_index)(logits, target)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('cls', LOSSES)
def test_invalid_configuration_and_masks(device, cls):
    for ignore_index in (True, 1.5, 'ignore'):
        with pytest.raises(TypeError, match='ignore_index'):
            cls(ignore_index=ignore_index)
    with pytest.raises(ValueError, match='64-bit'):
        cls(ignore_index=2**63)
    with pytest.raises(ValueError, match='one-hot'):
        cls(ignore_index=-100, target_format='one-hot')
    logits, target = _inputs(device)
    with pytest.raises(TypeError, match='boolean'):
        cls()(logits, target, mask=torch.ones_like(target, dtype=torch.float32))
    with pytest.raises(ValueError, match='shape'):
        cls()(logits, target, mask=torch.ones_like(logits, dtype=torch.bool))
    if device.type == 'cuda':
        with pytest.raises(ValueError, match='same device'):
            cls()(logits, target, mask=torch.ones_like(target, device='cpu', dtype=torch.bool))


@pytest.mark.parametrize('cls', LOSSES)
@pytest.mark.parametrize('dtype', [torch.uint8, torch.int8, torch.int16, torch.int32])
def test_ignore_index_does_not_wrap_to_target_dtype(device, cls, dtype):
    logits, target = _inputs(device, shape=(2, 3, 2, 3, 4))
    target = target.to(dtype)
    # All fit in int64, but would wrap to a valid class (zero) in the target dtype.
    ignore_index = 2 ** torch.iinfo(dtype).bits
    actual = cls(ignore_index=ignore_index)(logits, target)
    expected = cls()(logits, target)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('cls', [DiceLoss, TverskyLoss, GeneralizedDiceLoss])
@pytest.mark.parametrize('activation', ['none', 'sigmoid'])
def test_single_channel_masked_overlap(device, cls, activation):
    logits, _ = _inputs(device, shape=(2, 1, 2, 3, 4), dtype=torch.float64)
    target = (logits > 0).to(logits.dtype)
    if activation == 'none':
        logits = logits.sigmoid()
    logits.requires_grad_()
    mask = torch.ones((2, 2, 3, 4), device=device, dtype=torch.bool)
    mask[:, :, 0] = False
    options = dict(activation=activation, target_format='one-hot', batch_dice=False)
    actual = cls(**options)(logits, target, mask=mask)
    expected = _cropped_reference(cls, logits, target, mask, options)
    torch.testing.assert_close(actual, expected, rtol=3e-11, atol=1e-13)
    torch.testing.assert_close(torch.autograd.grad(actual, logits)[0],
                               torch.autograd.grad(expected, logits)[0], rtol=3e-10, atol=1e-13)


@pytest.mark.parametrize('cls', LOSSES)
def test_cpu_compilation_with_partial_and_fully_ignored_batches(cls):
    torch._dynamo.reset()
    logits, target = _inputs('cpu', shape=(2, 3, 2, 3, 4))
    target[:, 0] = -100
    fn = cls(ignore_index=-100, backend='torch')
    compiled_fn = torch.compile(fn, fullgraph=True)
    for mask in (target != 2, torch.zeros_like(target, dtype=torch.bool)):
        eager = logits.clone().requires_grad_()
        compiled = logits.clone().requires_grad_()
        expected, actual = fn(eager, target, mask=mask), compiled_fn(compiled, target, mask=mask)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, compiled)[0],
                                   torch.autograd.grad(expected, eager)[0])
