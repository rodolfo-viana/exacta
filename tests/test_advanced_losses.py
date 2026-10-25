"""Independent formulas, reference parity and gradients for the additional objectives."""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from exacta import losses as L


@pytest.mark.parametrize('batch', [False, True])
@pytest.mark.parametrize('gamma', [0.75, 1., 2.])
def test_focal_tversky_formula_and_backends(device, batch, gamma):
    torch.manual_seed(814)
    x = torch.randn(2, 3, 4, 5, dtype=torch.float64, device=device, requires_grad=True)
    y = torch.randint(0, 3, (2, 4, 5), device=device)
    mask = torch.rand(y.shape, device=device) > .2
    probs = x.softmax(1) * mask[:, None]
    truth = F.one_hot(y, 3).movedim(-1, 1) * mask[:, None]
    dims = (0, 2, 3) if batch else (2, 3)
    inter = (probs * truth).sum(dims)
    fp = (probs * (1-truth)).sum(dims)
    fn = ((1-probs) * truth).sum(dims)
    expected = ((1-(inter+1e-5)/(inter+.3*fp+.7*fn+1e-5))**gamma).mean()
    eg, = torch.autograd.grad(expected, x, retain_graph=True)
    for backend in ('torch', 'auto'):
        actual = L.FocalTverskyLoss(batch_dice=batch, gamma=gamma, backend=backend)(x, y, mask=mask)
        ag, = torch.autograd.grad(actual, x)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(ag, eg)


@pytest.mark.parametrize('backend', ['torch', 'auto'])
def test_focal_tversky_perfect_and_ignored(device, backend):
    y = torch.ones(2, 1, 3, 4, dtype=torch.float64, device=device)
    x = y.clone().requires_grad_()
    loss = L.FocalTverskyLoss(activation='none', target_format='one-hot', smooth=0, backend=backend)
    value = loss(x, y)
    value.backward()
    assert value == 0 and (x.grad == 0).all()
    x = torch.full_like(x, torch.nan, requires_grad=True)
    value = loss(x, y, mask=torch.zeros_like(y, dtype=torch.bool))
    value.backward()
    assert value == 0 and (x.grad == 0).all()


@pytest.mark.parametrize('fraction', [.01, .5, 1.])
def test_topk_valid_voxels_and_gradient(device, fraction):
    torch.manual_seed(122)
    x = torch.randn(2, 3, 3, 4, dtype=torch.float64, device=device, requires_grad=True)
    y = torch.randint(0, 3, (2, 3, 4), device=device)
    y[0, 0] = -1
    valid = y != -1
    expected = F.cross_entropy(x, y, reduction='none', ignore_index=-1)[valid]
    expected = expected.sort(descending=True).values[:math.ceil(fraction*int(valid.sum()))].mean()
    eg, = torch.autograd.grad(expected, x, retain_graph=True)
    actual = L.TopKCELoss(fraction=fraction, ignore_index=-1)(x, y)
    ag, = torch.autograd.grad(actual, x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(ag, eg)


def test_ranking_mask_nan_and_empty(device):
    for loss in [L.TopKCELoss(), L.DiceTopKCELoss(smooth=0), L.LovaszSoftmaxLoss(),
                 L.LovaszSoftmaxLoss(classes='all', per_image=True)]:
        x = torch.full((2, 3, 4, 5), torch.nan, device=device, requires_grad=True)
        y = torch.full((2, 4, 5), -5, device=device)
        value = loss(x, y, mask=torch.zeros_like(y, dtype=torch.bool))
        value.backward()
        assert value == 0 and (x.grad == 0).all()


def lovasz_numpy(prob, target, classes):
    # Algorithm 1: evaluate Jaccard set loss on prefixes of sorted voxel errors.
    values = []
    for c in range(prob.shape[0]):
        truth = (target.ravel() == c)
        if classes == 'present' and not truth.any():
            continue
        error = np.abs(truth-prob[c].ravel())
        order = np.argsort(-error, kind='stable')
        gradient, previous = [], 0.
        for k in range(1, len(order)+1):
            changed = np.zeros_like(truth)
            changed[order[:k]] = True
            prediction = truth ^ changed
            union = (prediction | truth).sum()
            jaccard = 1 - (prediction & truth).sum()/union if union else 0.
            gradient.append(jaccard-previous)
            previous = jaccard
        values.append(np.dot(error[order], gradient))
    return np.mean(values) if values else 0.


@pytest.mark.parametrize('classes', ['present', 'all'])
@pytest.mark.parametrize('per_image', [False, True])
def test_lovasz_reference(device, classes, per_image):
    torch.manual_seed(412)
    x = torch.randn(2, 4, 3, 4, dtype=torch.float64, device=device, requires_grad=True)
    y = torch.randint(0, 3, (2, 3, 4), device=device)
    mask = torch.rand(y.shape, device=device) > .3
    p, t, valid = x.softmax(1).detach().cpu().numpy(), y.cpu().numpy(), mask.cpu().numpy()
    if per_image:
        expected = np.mean([lovasz_numpy(a[:, m], b[m], classes) for a,b,m in zip(p,t,valid)])
    else:
        expected = lovasz_numpy(p.transpose(1,0,2,3)[:,valid], t[valid], classes)
    actual = L.LovaszSoftmaxLoss(classes=classes, per_image=per_image)(x, y, mask=mask)
    assert actual.item() == pytest.approx(expected)
    assert torch.autograd.gradcheck(lambda a: L.LovaszSoftmaxLoss(classes=classes, per_image=per_image)(a,y,mask=mask), (x,))


@pytest.mark.parametrize('signed', [False, True])
def test_distance_maps_scipy_spacing_and_degenerate(device, signed):
    rng = np.random.default_rng(71)
    arr = rng.random((2, 3, 5, 6, 7)) > .6
    arr[0, 0] = False
    arr[1, 2] = True
    spacing = [(1.,2.,3.), (.5,1.5,2.5)]
    actual = L.distance_maps(torch.tensor(arr, device=device), spacing=spacing, signed=signed).cpu().numpy()
    for b in range(2):
        for c in range(3):
            m = arr[b,c]
            if signed:
                if not m.any():
                    expected = np.full(m.shape, np.linalg.norm(np.array(m.shape)*spacing[b]))
                else:
                    inside = distance_transform_edt(np.pad(m, 1), sampling=spacing[b])[1:-1,1:-1,1:-1]
                    expected = distance_transform_edt(~m, sampling=spacing[b]) - inside
            else:
                expected = (distance_transform_edt(m, sampling=spacing[b]) + distance_transform_edt(~m, sampling=spacing[b])
                            if m.any() and not m.all() else np.zeros(m.shape))
            np.testing.assert_allclose(actual[b,c], expected, rtol=1e-14)


@pytest.mark.parametrize('name', ['BoundaryLoss','HausdorffDTLoss'])
def test_geometry_formula_cache_and_gradients(device, name):
    torch.manual_seed(21)
    x = torch.randn(1, 2, 3, 4, dtype=torch.float64, device=device, requires_grad=True)
    y = torch.randint(0,2,(1,3,4), device=device)
    truth = F.one_hot(y,2).movedim(-1,1)[:,1:].double()
    probs = x.softmax(1)[:,1:]
    loss = getattr(L,name)()
    dt = L.distance_maps(truth, spacing=(1,2), signed=name=='BoundaryLoss')
    if name == 'BoundaryLoss':
        expected = (probs*dt).mean()
    else:
        pd = L.distance_maps(probs.detach()>.5,spacing=(1,2))
        expected = ((probs-truth)**2*(pd**2+dt**2)).mean()
    actual = loss(x,y,spacing=(1,2))
    cached = loss(x,y,spacing=(1,2),distance_map=dt)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(cached,expected)
    torch.testing.assert_close(torch.autograd.grad(actual,x)[0],torch.autograd.grad(expected,x)[0])
    assert torch.autograd.gradcheck(lambda a: loss(a,y,spacing=(1,2)), (x,))


def test_hd_monai_reference(device):
    monai = pytest.importorskip('monai.losses')
    torch.manual_seed(81)
    x = torch.randn(2,3,5,6,7,device=device,requires_grad=True)
    y = torch.randint(0,3,(2,5,6,7),device=device)
    # Reference computes CPU distance fields; compare value and gradient there.
    cpu = x.detach().cpu().requires_grad_()
    ref = monai.HausdorffDTLoss(include_background=False,to_onehot_y=True,softmax=True)(cpu,y.cpu()[:,None])
    actual = L.HausdorffDTLoss()(x,y,spacing=1)
    torch.testing.assert_close(actual.cpu(),ref)
    torch.testing.assert_close(torch.autograd.grad(actual,x)[0].cpu(),torch.autograd.grad(ref,cpu)[0])


def reference_soft_skeleton(x, iterations):
    pool = F.max_pool2d if x.ndim == 4 else F.max_pool3d
    dims = x.ndim-2
    def erosion(a):
        return torch.stack([-pool(-a,tuple(3 if j==i else 1 for j in range(dims)),1,
                                       tuple(1 if j==i else 0 for j in range(dims))) for i in range(dims)]).amin(0)
    def opening(a):
        return pool(erosion(a),3,1,1)
    result = (x-opening(x)).relu()
    for _ in range(iterations):
        x = erosion(x)
        delta = (x-opening(x)).relu()
        result = result + (delta-result*delta).relu()
    return result


@pytest.mark.parametrize('shape', [(2,3,5,6),(1,2,4,5,6)])
def test_soft_cldice_reference_and_cache(device, shape):
    torch.manual_seed(617)
    x = torch.randn(shape,dtype=torch.float64,device=device,requires_grad=True)
    y = torch.randint(0,shape[1],(shape[0],*shape[2:]),device=device)
    p = x.softmax(1)[:,1:]
    t = F.one_hot(y,shape[1]).movedim(-1,1)[:,1:].double()
    sp, st = reference_soft_skeleton(p,2), reference_soft_skeleton(t,2)
    dims = tuple(range(2,len(shape)))
    a = ((sp*t).sum(dims)+1)/(sp.sum(dims)+1)
    b = ((st*p).sum(dims)+1)/(st.sum(dims)+1)
    expected = (1-2*a*b/(a+b)).mean()
    actual = L.SoftClDiceLoss(iterations=2)(x,y)
    torch.testing.assert_close(actual,expected)
    # Minimum tie subgradients differ for stacked amin vs pairwise min; use gradcheck separately.
    cached = L.SoftClDiceLoss(iterations=2)(x,y,target_skeleton=st)
    torch.testing.assert_close(actual,cached)
    assert torch.isfinite(torch.autograd.grad(actual,x)[0]).all()


@pytest.mark.parametrize('cls,kwargs', [(L.FocalTverskyLoss,{'gamma':0}), (L.TopKCELoss,{'fraction':0}),
    (L.TopKCELoss,{'fraction':float('nan')}), (L.HausdorffDTLoss,{'alpha':-1}),
    (L.SoftClDiceLoss,{'iterations':1.5}), (L.SoftClDiceLoss,{'smooth':0}),
    (L.LovaszSoftmaxLoss,{'classes':'wrong'})])
def test_advanced_loss_validation(cls,kwargs):
    with pytest.raises((ValueError,TypeError)):
        cls(**kwargs)


def test_soft_cldice_gradcheck():
    torch.manual_seed(910)
    x=torch.randn(1,2,3,4,dtype=torch.float64,requires_grad=True)
    target=torch.randint(0,2,(1,3,4))
    assert torch.autograd.gradcheck(lambda a:L.SoftClDiceLoss(iterations=1)(a,target),(x,))


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_advanced_mixed_precision(device,dtype):
    torch.manual_seed(77)
    x=torch.randn(1,3,3,4,device=device,dtype=dtype,requires_grad=True)
    target=torch.randint(0,3,(1,3,4),device=device)
    for loss in [L.FocalTverskyLoss(),L.TopKCELoss(),L.LovaszSoftmaxLoss(),L.SoftClDiceLoss(iterations=1)]:
        value=loss(x,target)
        expected=loss(x.detach().float(),target)
        assert value.dtype==torch.float32
        torch.testing.assert_close(value,expected,atol=2e-5,rtol=2e-5)
        assert torch.autograd.grad(value,x)[0].isfinite().all()


def test_geometry_mask_keeps_target_geometry(device):
    x=torch.randn(1,2,3,4,device=device,requires_grad=True)
    target=torch.zeros(1,3,4,dtype=torch.long,device=device);target[0,1,1]=1
    mask=torch.ones_like(target,dtype=torch.bool);mask[0,1,1]=False
    dt=L.distance_maps((target==1)[:,None],spacing=1,signed=True)
    value=L.BoundaryLoss()(x,target,spacing=1,mask=mask)
    expected=(x.softmax(1)[:,1:]*dt)[mask[:,None]].mean().float()
    torch.testing.assert_close(value,expected)
    gradient=torch.autograd.grad(value,x)[0]
    assert (gradient[:,:,1,1]==0).all()


@pytest.mark.parametrize('factory',[lambda:L.FocalTverskyLoss(backend='torch'),L.TopKCELoss,L.LovaszSoftmaxLoss,
                                     lambda:L.SoftClDiceLoss(iterations=1)])
def test_compiled_advanced_losses(factory,device):
    torch.manual_seed(834)
    x=torch.randn(1,3,3,4,device=device,requires_grad=True)
    y=torch.randint(0,3,(1,3,4),device=device)
    loss=factory()
    expected=loss(x,y)
    eg=torch.autograd.grad(expected,x)[0]
    # aot_eager exercises capture/autograd on all supported Torch versions without
    # making CPU CI pay for machine-code compilation; benchmark script tests Inductor.
    # Keep backend/shape-specialization state isolated from the other compiler tests.
    # Mixing aot_eager captures with subsequent Triton/Inductor captures in one
    # process can trigger expensive symbolic shape generalization in PyTorch.
    torch._dynamo.reset()
    try:
        actual=torch.compile(loss,backend='aot_eager')(x,y)
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(torch.autograd.grad(actual,x)[0],eg)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize('shape',[(2,3,5,6),(2,2,4,5,6)])
@pytest.mark.parametrize('erosion',[False,True])
def test_native_soft_pool_value_gradient_and_second_derivative(shape,erosion):
    from exacta._soft_pool import pool
    torch.manual_seed(621)
    x=torch.randn(shape,dtype=torch.float64).transpose(-1,-2).requires_grad_()
    ndim=x.ndim-2
    pooling=F.max_pool2d if ndim==2 else F.max_pool3d
    if erosion:
        expected=torch.stack([-pooling(-x,tuple(3 if j==i else 1 for j in range(ndim)),1,
                                            tuple(1 if j==i else 0 for j in range(ndim))) for i in range(ndim)]).amin(0)
    else:
        expected=pooling(x,3,1,1)
    actual=pool(x,erosion)
    torch.testing.assert_close(actual,expected)
    weights=torch.randn_like(actual)
    torch.testing.assert_close(torch.autograd.grad(actual,x,weights)[0],torch.autograd.grad(expected,x,weights)[0])
    tiny=torch.randn((1,1,3,4),dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda a:pool(a,erosion),(tiny,))
    assert torch.autograd.gradgradcheck(lambda a:pool(a,erosion),(tiny,))


def test_soft_cldice_native_and_torch_cpu_gradients():
    torch.manual_seed(812)
    x=torch.randn(2,3,5,6,7,dtype=torch.float64,requires_grad=True)
    y=torch.randint(0,3,(2,5,6,7))
    actual=L.SoftClDiceLoss(iterations=3,backend='native')(x,y)
    expected=L.SoftClDiceLoss(iterations=3,backend='torch')(x,y)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual,x)[0],torch.autograd.grad(expected,x)[0])
