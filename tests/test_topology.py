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
            actual = F.skeletonize(torch.tensor(mask,device=device),method='lee')
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
                actual=F.skeletonize(torch.tensor(data[key],device=device),method='lee')
                np.testing.assert_array_equal(actual.cpu(),data[key.replace('mask_','skeleton_')])


@pytest.mark.parametrize('shape', [(1,1),(1,9),(11,13),(32,29)])
def test_zhang_matches_scikit_image(device, shape):
    reference = pytest.importorskip('skimage.morphology').skeletonize
    ndimage = pytest.importorskip('scipy.ndimage')
    rng = np.random.default_rng(1984)
    for density in [0.,.2,.5,.9,1.]:
        for closing in (0, 1, 2):
            mask = rng.random(shape) < density
            if closing:
                mask = ndimage.binary_closing(mask, iterations=closing)
            actual = F.skeletonize(torch.tensor(mask,device=device), method='zhang')
            np.testing.assert_array_equal(actual.cpu(), reference(mask, method='zhang'))
            assert actual.device.type == device.type and actual.dtype == torch.bool
            # None follows scikit-image's default: Zhang in 2-D.
            torch.testing.assert_close(F.skeletonize(torch.tensor(mask,device=device)), actual)


def test_skeleton_method_contract():
    with pytest.raises(ValueError, match='2-D'):
        F.skeletonize(np.zeros((3,3,3),bool), method='zhang')
    with pytest.raises(ValueError, match='method'):
        F.skeletonize(np.zeros((3,3),bool), method='medial')
    with pytest.raises(ValueError, match='method'):
        metrics.ClDice(method='medial')
    mask = np.zeros((5,7,9),bool); mask[1:4,2:5,1:8] = True
    np.testing.assert_array_equal(F.skeletonize(mask), F.skeletonize(mask, method='lee'))


def _authors_cldice(p, t, skeletonize):
    """clDice as the authors' cldice_metric/cldice.py computes it, with scikit-image skeletons."""
    tprec = np.sum(p * skeletonize(t)) / np.sum(skeletonize(t))
    tsens = np.sum(t * skeletonize(p)) / np.sum(skeletonize(p))
    return 2 * tprec * tsens / (tprec + tsens)


@pytest.mark.parametrize('shape', [(48,52),(12,14,16)])
def test_cldice_default_reproduces_authors_metric(device, shape):
    skeletonize = pytest.importorskip('skimage.morphology').skeletonize
    ndimage = pytest.importorskip('scipy.ndimage')
    rng = np.random.default_rng(2021)
    target = ndimage.binary_dilation(rng.random(shape) < .03, iterations=2)
    pred = target.copy()
    pred[rng.random(shape) < .08] ^= True
    pred = ndimage.binary_opening(pred)
    expected = _authors_cldice(pred, target, skeletonize)
    actual = F.cldice(torch.tensor(pred,device=device)[None,None], torch.tensor(target,device=device)[None,None],
                      input_format='one-hot')
    assert actual.item() == pytest.approx(expected, rel=1e-12)
    if len(shape) == 2:
        lee = F.cldice(pred[None,None], target[None,None], input_format='one-hot', method='lee').item()
        assert lee == pytest.approx(_authors_cldice(pred, target, lambda m: skeletonize(m, method='lee')), rel=1e-12)
        metric = metrics.ClDice(input_format='one-hot', method='lee')
        metric.update(pred[None,None], target[None,None])
        assert metric.compute().mean.item() == pytest.approx(lee)
        together = F.evaluate(pred[None,None], target[None,None], input_format='one-hot',
                              metrics={'cldice': {'method': 'lee'}})
        assert together['cldice'].item() == pytest.approx(lee)


def _shapes():
    grid = np.indices((20, 20, 20)) - 9.5
    r = np.sqrt(grid[1]**2 + grid[2]**2)
    radius = np.sqrt((grid**2).sum(0))
    plane = np.indices((20, 20)) - 9.5
    return {
        'solid torus': ((r - 5.5)**2 + grid[0]**2 < 2.5**2, [1, 1, 0]),
        'hollow sphere': ((radius < 8) & (radius > 5), [1, 0, 1]),
        'two balls': (((grid[0] - 4)**2 + (grid[1:]**2).sum(0) < 9) | ((grid[0] + 4)**2 + (grid[1:]**2).sum(0) < 9), [2, 0, 0]),
        'annulus': ((np.hypot(*plane) < 8) & (np.hypot(*plane) > 4), [1, 1]),
    }


@pytest.mark.parametrize('name', ['solid torus', 'hollow sphere', 'two balls', 'annulus'])
def test_betti_numbers_of_known_shapes(device, name):
    mask, expected = _shapes()[name]
    for connectivity in (1, mask.ndim):
        got = F.betti_numbers(torch.tensor(mask, device=device)[None, None], connectivity=connectivity, input_format='one-hot')
        assert got.device.type == device.type and got.dtype == torch.int64
        assert got[0, 0].tolist() == expected


def test_betti_numbers_match_scikit_image(device):
    measure = pytest.importorskip('skimage.measure')
    rng = np.random.default_rng(1871)
    for trial in range(120):
        ndim = 2 + trial % 2
        mask = rng.random(tuple(rng.integers(1, 12 if ndim == 3 else 20, ndim))) < rng.uniform(.1, .9)
        for connectivity in (1, ndim):
            b = F.betti_numbers(torch.tensor(mask, device=device)[None, None], connectivity=connectivity,
                                input_format='one-hot')[0, 0].cpu().numpy()
            assert b[0] == measure.label(mask, connectivity=connectivity).max()
            # Holes (2-D) and cavities (3-D): background components of the padded image, dual neighbourhood.
            dual = ndim if connectivity == 1 else 1
            assert b[-1] == measure.label(~np.pad(mask, 1), connectivity=dual).max() - 1
            assert b[0] - b[1] + (b[2] if ndim == 3 else 0) == measure.euler_number(mask, connectivity=connectivity)


def test_betti_error_and_collections(device):
    target = torch.zeros((1, 16, 16), dtype=torch.long, device=device)
    target[0, 2:6, 2:14] = 1
    target[0, 9:13, 2:14] = 2
    pred = target.clone()
    pred[0, 2:6, 8] = 0  # vessel 1 broken into two components
    pred[0, 10, 4:6] = 0  # a hole in structure 2
    got = F.betti_error(pred, target, connectivity=2, num_classes=3, include_background=False)
    torch.testing.assert_close(got, torch.tensor([[1., 0.]], dtype=torch.float64, device=device))
    holes = F.betti_error(pred, target, connectivity=2, dimension=1, num_classes=3, include_background=False)
    torch.testing.assert_close(holes, torch.tensor([[0., 1.]], dtype=torch.float64, device=device))
    metric = metrics.BettiError(connectivity=2, num_classes=3, include_background=False)
    metric.update(pred, target)
    torch.testing.assert_close(metric.compute().per_class, torch.tensor([1., 0.], dtype=torch.float64, device=device))
    together = F.evaluate(pred, target, num_classes=3, include_background=False,
                          metrics={'dice': {}, 'betti_error': {'connectivity': 2}})
    torch.testing.assert_close(together['betti_error'], got)
    with pytest.raises(ValueError, match='connectivity'):
        F.evaluate(pred, target, num_classes=3, metrics=['betti_error'])
    with pytest.raises(ValueError, match='complete segmentations'):
        F.evaluate(pred, target, num_classes=3, metrics={'betti_error': {'connectivity': 2}}, ignore_index=0)
    with pytest.raises(ValueError, match='connectivity'):
        F.betti_numbers(np.zeros((1, 1, 4, 4, 4), bool), connectivity=2, input_format='one-hot')
    with pytest.raises(ValueError, match='dimension'):
        F.betti_error(pred, target, connectivity=2, num_classes=3, dimension=2)
