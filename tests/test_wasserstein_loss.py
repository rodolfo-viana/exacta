"""Independent class-transport formula, reference comparisons and gradient checks."""
import numpy as np
import pytest
import torch

from exacta.losses import GeneralizedWassersteinDiceLoss as Loss

MATRIX = [[0., 1., 1.], [1., 0., .25], [1., .25, 0.]]


def reference(logits, target, valid, matrix, mode, smooth):
    """Scalar NumPy definition: expected ground cost against a crisp target."""
    output = []
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = matrix / matrix.max()
    for x, t, keep in zip(logits, target, valid):
        x, t = x[:, keep], t[keep]
        if not t.size:
            output.append(0.)
            continue
        p = np.exp(x - x.max(0))
        p /= p.sum(0)
        counts = np.bincount(t, minlength=matrix.shape[0])
        tp, error = 0., 0.
        for i, label in enumerate(t):
            w = sum(matrix[label, c] * p[c, i] for c in range(matrix.shape[0]))
            alpha = 1 / (counts[label] + 1) if mode == 'GDL' else float(label != 0)
            tp += alpha * (1 - w)
            error += (alpha if mode == 'GDL' else 1) * w
        denominator = 2 * tp + error + smooth
        output.append(1 - (2 * tp + smooth) / denominator if denominator else 0.)
    return np.asarray(output)


@pytest.mark.parametrize('mode', ['default', 'GDL'])
@pytest.mark.parametrize('shape', [(3, 3, 5, 7), (3, 3, 3, 4, 5)])
@pytest.mark.parametrize('smooth', [0., .13])
def test_wasserstein_definition_and_reductions(device, mode, shape, smooth):
    rng = np.random.default_rng(871)
    x = rng.normal(size=shape)
    t = rng.integers(0, 3, (shape[0], *shape[2:]))
    valid = rng.random(t.shape) > .25
    valid[0] = False
    expected = reference(x, t, valid, MATRIX, mode, smooth)
    x = torch.tensor(x, dtype=torch.float64, device=device).transpose(-1, -2).requires_grad_()
    t = torch.tensor(t, device=device).transpose(-1, -2)
    valid = torch.tensor(valid, device=device).transpose(-1, -2)
    # Exclusions must sanitize invalid labels and nonfinite predictions before softmax.
    x = torch.where(valid[:, None], x, torch.nan).detach().requires_grad_()
    t[~valid] = -5
    for reduction, value in [('none', expected), ('sum', expected.sum()), ('mean', expected[1:].mean())]:
        loss = Loss(MATRIX, weighting_mode=mode, smooth=smooth, reduction=reduction, ignore_index=-5)
        actual = loss(x, t, mask=valid[:, None])
        np.testing.assert_allclose(actual.detach().cpu(), value, rtol=1e-12, atol=1e-12)
        grad = torch.autograd.grad(actual.sum(), x)[0]
        assert torch.isfinite(grad).all()
        assert grad[~valid[:, None].expand_as(grad)].eq(0).all()


@pytest.mark.parametrize('mode', ['default', 'GDL'])
def test_wasserstein_matches_monai_values_and_gradients(device, mode):
    monai = pytest.importorskip('monai.losses')
    gen = torch.Generator().manual_seed(872)
    x = torch.randn((2, 3, 4, 5), generator=gen, dtype=torch.float64).to(device).requires_grad_()
    t = torch.randint(3, (2, 4, 5), generator=gen).to(device)
    matrix = torch.tensor(MATRIX, dtype=torch.float64, device=device)
    loss = Loss(matrix, weighting_mode=mode, smooth=.07)
    ref = monai.GeneralizedWassersteinDiceLoss(matrix, weighting_mode=mode, smooth_nr=.07, smooth_dr=.07)
    actual, expected = loss(x, t), ref(x, t)
    # MONAI computes GDL class weights in float32 even for float64 logits.
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-10)
    torch.testing.assert_close(torch.autograd.grad(actual, x)[0], torch.autograd.grad(expected, x)[0],
                               rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize('mode', ['default', 'GDL'])
def test_wasserstein_first_and_second_derivatives(mode):
    x = torch.randn((1, 3, 2, 3), generator=torch.Generator().manual_seed(873),
                    dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[[0, 1, 2], [2, -1, 1]]])
    loss = Loss(MATRIX, weighting_mode=mode, ignore_index=-1)
    assert torch.autograd.gradcheck(loss, (x, t))
    assert torch.autograd.gradgradcheck(loss, (x, t))


def test_binary_wasserstein_is_foreground_dice(device):
    x = torch.randn((2, 2, 5, 7), generator=torch.Generator().manual_seed(874)).to(device).requires_grad_()
    t = torch.randint(2, (2, 5, 7), generator=torch.Generator().manual_seed(875)).to(device)
    p = x.softmax(1)[:, 1]
    expected = (1 - (2 * (p * t).sum((1, 2)) + .1) / (p.sum((1, 2)) + t.sum((1, 2)) + .1)).mean()
    actual = Loss([[0, 1], [1, 0]], smooth=.1)(x, t)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, x)[0], torch.autograd.grad(expected, x)[0])


@pytest.mark.parametrize('mode', ['default', 'GDL'])
@pytest.mark.parametrize('shape', [(0, 3, 4, 5), (2, 3, 0, 5), (2, 3, 4, 5)])
def test_wasserstein_empty_inputs_and_zero_gradient(device, mode, shape):
    x = torch.full(shape, torch.nan, device=device, requires_grad=True)
    t = torch.full((shape[0], *shape[2:]), -1, device=device)
    for reduction in ['none', 'mean', 'sum']:
        value = Loss(MATRIX, weighting_mode=mode, smooth=0, reduction=reduction, ignore_index=-1)(x, t)
        assert value.shape == ((shape[0],) if reduction == 'none' else ())
        assert value.eq(0).all()
        assert torch.autograd.grad(value.sum(), x)[0].eq(0).all()


@pytest.mark.parametrize('mode', ['default', 'GDL'])
def test_wasserstein_perfect_confusions_and_missing_classes(device, mode):
    truth = torch.tensor([[[0, 1, 2]]], device=device)
    perfect = torch.nn.functional.one_hot(truth, 3).movedim(-1, 1).float() * 1000
    assert Loss(MATRIX, weighting_mode=mode, smooth=0)(perfect, truth) == 0
    foreground = torch.ones((1, 1, 1), dtype=torch.long, device=device)
    near = torch.tensor([[[[-1000.]], [[-1000.]], [[1000.]]]], device=device)
    far = torch.tensor([[[[1000.]], [[-1000.]], [[-1000.]]]], device=device)
    assert 0 < Loss(MATRIX, weighting_mode=mode)(near, foreground) < Loss(MATRIX, weighting_mode=mode)(far, foreground)
    background = torch.zeros_like(foreground)
    assert Loss(MATRIX, weighting_mode=mode, smooth=0)(far, background) == 0
    with_nan = perfect.clone()
    with_nan[0, 0, 0, 0] = torch.nan
    assert Loss(MATRIX, weighting_mode=mode)(with_nan, truth).isnan()


@pytest.mark.parametrize('matrix', [[], [0, 1], [[0]], [[0, 1, 1], [1, 0, 1]],
    [[0, 0], [0, 0]], [[0, -1], [-1, 0]], [[0, float('inf')], [float('inf'), 0]],
    [[0, float('nan')], [float('nan'), 0]], [[1, 1], [1, 0]], [[0, 1], [.5, 0]]])
def test_wasserstein_rejects_invalid_matrices(matrix):
    with pytest.raises(ValueError):
        Loss(matrix)


def test_wasserstein_matrix_normalization_copy_and_weights():
    matrix = torch.tensor(MATRIX, dtype=torch.float64)
    loss = Loss(matrix * 100)
    torch.testing.assert_close(loss.dist_matrix, matrix)
    matrix.fill_(0)
    assert loss.dist_matrix[0, 1] == 1
    torch.testing.assert_close(Loss([[0., 1e100], [1e100, 0.]]).dist_matrix,
                               torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64))
    with pytest.raises(ValueError, match='background'):
        Loss([[0, .5, 1], [.5, 0, .75], [1, .75, 0]])
    Loss([[0, .5, 1], [.5, 0, .75], [1, .75, 0]], weighting_mode='GDL')
    with pytest.raises(TypeError, match='real'):
        Loss(torch.tensor([[0, 1j], [1j, 0]]))
    for kwargs in [dict(smooth=-1), dict(smooth=float('inf')), dict(weighting_mode='other'),
                   dict(reduction='other'), dict(ignore_index=True)]:
        with pytest.raises((ValueError, TypeError)):
            Loss(MATRIX, **kwargs)
    restored = Loss(MATRIX)
    restored.load_state_dict(loss.state_dict())
    torch.testing.assert_close(restored.dist_matrix, loss.dist_matrix)


def test_wasserstein_rejects_invalid_inputs():
    loss = Loss(MATRIX)
    x = torch.zeros((1, 3, 2, 2))
    t = torch.zeros((1, 2, 2), dtype=torch.long)
    with pytest.raises(ValueError, match='channels'):
        loss(x[:, :2], t)
    with pytest.raises(TypeError, match='integer'):
        loss(x, t.float())
    with pytest.raises(ValueError, match='shape'):
        loss(x, t[:, None])
    with pytest.raises(ValueError, match='class indices'):
        loss(x, t - 1)
    with pytest.raises(ValueError, match='class indices'):
        loss(x, t + 3)
    with pytest.raises(TypeError, match='boolean'):
        loss(x, t, mask=t)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('mode', ['default', 'GDL'])
def test_wasserstein_precision_and_buffer_device(device, dtype, mode):
    gen = torch.Generator().manual_seed(876)
    x = torch.randn((2, 3, 7, 9), generator=gen).to(device=device, dtype=dtype).requires_grad_()
    t = torch.randint(3, (2, 7, 9), generator=gen).to(device)
    loss = Loss(MATRIX, weighting_mode=mode).to(device)
    assert loss.dist_matrix.device == x.device
    actual = loss(x, t)
    expected = loss(x.double(), t)
    assert actual.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-7)
    assert torch.isfinite(torch.autograd.grad(actual, x)[0]).all()


@pytest.mark.parametrize('mode', ['default', 'GDL'])
def test_wasserstein_compilation(device, mode):
    torch._dynamo.reset()
    try:
        x = torch.randn((2, 3, 5, 7), generator=torch.Generator().manual_seed(877)).to(device).requires_grad_()
        t = torch.randint(3, (2, 5, 7), generator=torch.Generator().manual_seed(878)).to(device)
        mask = torch.ones_like(t, dtype=torch.bool)
        mask[0] = False
        loss = Loss(MATRIX, weighting_mode=mode).to(device)
        expected = loss(x, t, mask=mask)
        expected_grad = torch.autograd.grad(expected, x)[0]
        compiled = torch.compile(loss, fullgraph=True)
        actual = compiled(x, t, mask=mask)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, x)[0], expected_grad)
        # Compiled invalid labels yield NaN without unsafe indexing or clamping away errors.
        assert compiled(x, t + 3, mask=mask).isnan()
        assert compiled(x, t - 3, mask=mask).isnan()
    finally:
        torch._dynamo.reset()


def test_gdl_allows_nonmaximal_background_costs_and_cost_free_labels(device):
    matrix = [[0, .5, 1], [.5, 0, .75], [1, .75, 0]]
    x = torch.randn((2, 3, 2, 3), generator=torch.Generator().manual_seed(881), dtype=torch.float64).to(device)
    t = torch.tensor([[[0, 1, 2], [1, 1, 0]], [[2, 2, 2], [1, 0, 0]]], device=device)
    expected = reference(x.cpu().numpy(), t.cpu().numpy(), np.ones(t.shape, dtype=bool), matrix, 'GDL', 1e-5)
    result = Loss(matrix, weighting_mode='GDL', reduction='none')(x, t)
    np.testing.assert_allclose(result.cpu(), expected, rtol=1e-12, atol=1e-12)
    zero_cost = [[0, 1, 1], [1, 0, 0], [1, 0, 0]]
    prediction = torch.tensor([[[[-1000.]], [[-1000.]], [[1000.]]]], device=device)
    target = torch.ones((1, 1, 1), dtype=torch.long, device=device)
    assert Loss(zero_cost, smooth=0)(prediction, target) == 0


def test_wasserstein_deep_supervision_masks_and_gradients(device):
    from exacta.losses import DeepSupervisionLoss

    x = torch.randn((2, 3, 4, 6), generator=torch.Generator().manual_seed(882)).to(device).requires_grad_()
    y = torch.randint(3, (2, 4, 6), generator=torch.Generator().manual_seed(883)).to(device)
    y[0] = -1
    base = Loss(MATRIX, ignore_index=-1)
    deep = DeepSupervisionLoss(base, weights=[.75, .25])
    coarse = x[:, :, ::2, ::2]
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[:, -1] = False
    coarse_target, coarse_mask = y[:, ::2, ::2], mask[:, ::2, ::2]
    expected = .75 * base(x, y, mask=mask) + .25 * base(coarse, coarse_target, mask=coarse_mask)
    actual = deep([x, coarse], [y, coarse_target], masks=[mask, coarse_mask])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, x)[0], torch.autograd.grad(expected, x)[0])
