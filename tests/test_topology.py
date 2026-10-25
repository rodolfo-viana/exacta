import numpy as np
import pytest
import torch

from exacta import functional as F, metrics


@pytest.mark.parametrize('shape', [(11,13),(5,7,9),(1,9,11),(4,4,4),(9,9,9)])
def test_lee_reference(device, shape):
    reference = pytest.importorskip('skimage.morphology').skeletonize
    rng = np.random.default_rng(6271)
    for density in [0.,.2,.5,.9,1.]:
        for _ in range(4):
            mask = rng.random(shape) < density
            actual = F.skeletonize(torch.tensor(mask,device=device))
            np.testing.assert_array_equal(actual.cpu(), reference(mask,method='lee'))
            assert actual.device.type == device.type
            assert actual.dtype == torch.bool
            assert not np.any(actual.cpu().numpy() & ~mask)


def test_cldice_known_line_gap(device):
    target = torch.zeros(1,1,7,9,dtype=torch.bool,device=device)
    target[0,0,3,1:8] = True
    pred = target.clone(); pred[0,0,3,4] = False
    expected = 2*(6/7)/(1+6/7)
    actual = F.cldice(pred,target,input_format='one-hot')
    assert actual.item() == pytest.approx(expected)
    sp = F.skeletonize(pred[0,0])[None,None]
    st = F.skeletonize(target[0,0])[None,None]
    torch.testing.assert_close(actual,F.cldice(pred,target,input_format='one-hot',pred_skeleton=sp,target_skeleton=st))
    metric = metrics.ClDice(input_format='one-hot')
    metric.update(pred,target,target_skeleton=st)
    assert metric.compute().mean.item() == pytest.approx(expected)
    together = F.evaluate(pred,target,input_format='one-hot',metrics=['dice','cldice'])
    torch.testing.assert_close(together['cldice'],actual)


def test_cldice_empty_skeleton_policy(device):
    empty = torch.zeros(1,1,4,4,4,dtype=torch.bool,device=device)
    full = ~empty
    assert F.cldice(empty,empty,input_format='one-hot').isnan().all()
    assert F.cldice(empty,empty,input_format='one-hot',both_empty=1).item() == 1
    assert F.cldice(full,empty,input_format='one-hot').item() == 0
    assert F.cldice(empty,full,input_format='one-hot').item() == 0
    # Known Lee limitation: an even-width solid cube can have an empty skeleton.
    assert F.cldice(full,full,input_format='one-hot').item() == 0


def test_bad_skeleton():
    with pytest.raises(ValueError):
        F.skeletonize(np.full((4,5),.5))
    with pytest.raises(ValueError):
        F.cldice(np.zeros((1,4,5),int),np.zeros((1,4,5),int),num_classes=2,
                 pred_skeleton=np.ones((1,2,4,5),bool))


def test_lee_frozen_reference(device):
    from pathlib import Path
    with np.load(Path(__file__).parent/'fixtures'/'skeleton_lee.npz') as data:
        for key in data.files:
            if key.startswith('mask_'):
                actual=F.skeletonize(torch.tensor(data[key],device=device))
                np.testing.assert_array_equal(actual.cpu(),data[key.replace('mask_','skeleton_')])
