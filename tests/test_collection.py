"""Shared evaluation must preserve individual metric semantics and avoid repeated volume work."""

from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import exacta.functional as EF
from exacta.functional import overlap, surface
from exacta.metrics import MetricCollection


def _inputs(device, shape=(2, 9, 11, 7)):
    generator = torch.Generator().manual_seed(57)
    return [torch.randint(3, shape, generator=generator).to(device) for _ in range(2)]


@pytest.mark.parametrize('representation', ['voxels', 'surfels', 'mesh'])
@pytest.mark.parametrize('shape', [(2, 9, 11), (2, 7, 9, 11)])
@pytest.mark.parametrize('one_hot', [False, True])
def test_collection_matches_individual_functions(device, representation, shape, one_hot):
    pred, target = _inputs(device, shape)
    pred[0] = 0  # one-empty cases alongside ordinary comparisons
    common = dict(include_background=False, num_classes=4)
    if one_hot:
        pred, target = [torch.nn.functional.one_hot(x, 4).movedim(-1, 1).bool() for x in (pred, target)]
        common['input_format'] = 'one-hot'
    # Strided arrays, per-case anisotropic spacing, an absent class and nondefault policies.
    pred, target = pred.transpose(-1, -2), target.transpose(-1, -2)
    spacing = [[1.3, 0.7, 2.1][:len(shape)-1], [0.9, 2.2, 1.7][:len(shape)-1]]
    metrics = {
        'dice': {'both_empty': 1}, 'iou': {}, 'sensitivity': {'zero_division': 0},
        'precision': {}, 'specificity': {}, 'volume_similarity': {'convention': 'itk'},
        'relative_volume_difference': {'absolute': True},
        'hausdorff_distance': {'percentile': 80, 'combine': 'mean', 'surface': representation},
        'hd95': {'combine': 'pooled', 'surface': representation, 'one_empty': 99},
        'assd': {'combine': 'mean', 'surface': representation},
        'nsd': {'tolerance': [0.7, 1.0, 2.0], 'surface': representation, 'both_empty': 1},
    }
    actual = EF.evaluate(pred, target, metrics=metrics, spacing=spacing, **common)
    assert list(actual) == list(metrics)
    for name, options in metrics.items():
        extra = dict(spacing=spacing) if 'surface' in options else {}
        expected = getattr(EF, name)(pred, target, **common, **extra, **options)
        torch.testing.assert_close(actual[name], expected, rtol=0, atol=0, equal_nan=True)


def test_groups_reuse_work_and_do_not_cache_mutated_arrays(monkeypatch):
    pred, target = _inputs('cpu', (1, 9, 9))
    calls = {'counts': 0, 'surfaces': 0}
    count, distances = overlap._confusion_counts_and_seg, surface._edt_stack
    def counted(*args, **kwargs):
        calls['counts'] += 1
        return count(*args, **kwargs)
    def measured(*args, **kwargs):
        calls['surfaces'] += 1
        return distances(*args, **kwargs)
    monkeypatch.setattr(overlap, '_confusion_counts_and_seg', counted)
    monkeypatch.setattr(surface, '_edt_stack', measured)
    metrics = {'dice': {}, 'iou': {}, 'precision': {}, 'hd95': {}, 'assd': {}, 'nsd': {'tolerance': 1}}
    EF.evaluate(pred, target, metrics=metrics, spacing=1, num_classes=3)
    assert calls == {'counts': 1, 'surfaces': 3}  # one transform pair per class, not per metric
    pred.copy_(target)
    changed = EF.evaluate(pred, target, metrics=metrics, spacing=1, num_classes=3)
    assert changed['dice'].eq(1).all() and changed['hd95'].eq(0).all()


def test_mixed_surfaces_numpy_and_state():
    pred, target = [x.numpy()[:, ::-1] for x in _inputs('cpu')]
    metrics = {'dice': {}, 'hd95': {'surface': 'surfels'}, 'assd': {'surface': 'voxels'}}
    values = EF.evaluate(pred, target, metrics=metrics, spacing=1, num_classes=3)
    assert all(isinstance(x, np.ndarray) for x in values.values())
    collection = MetricCollection(metrics=metrics, num_classes=3, spacing=2)
    with pytest.raises(RuntimeError, match='before update'):
        collection.compute()
    for b in range(2):
        collection.update(pred[b:b+1], target[b:b+1], spacing=1)
    for name, result in collection.compute().items():
        np.testing.assert_allclose(result.per_case.numpy(), values[name], equal_nan=True)
        torch.testing.assert_close(result.mean, result.per_case.nanmean(0).nanmean())
    collection.reset()
    with pytest.raises(RuntimeError, match='before update'):
        collection.compute()


@pytest.mark.parametrize('metrics', [[], ['dice', 'dice'], ['unknown'], {'nsd': {}},
                                   {'dice': {'tolerance': 1}}, 'dice', {'dice': None},
                                   {'hd95': {'combine': 'bad'}}, {'hd95': {'surface': 'bad'}},
                                   {'hausdorff_distance': {'percentile': 101}}])
def test_invalid_selections(metrics):
    pred, target = _inputs('cpu')
    with pytest.raises((ValueError, TypeError)):
        EF.evaluate(pred, target, metrics=metrics, num_classes=3, spacing=1)


def _distributed_collection(rank, init_file):
    dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        collection = MetricCollection(metrics=['dice', 'iou'], num_classes=3)
        pred, target = _inputs('cpu')
        if rank == 1:
            collection.update(pred, target)
        actual = collection.compute()
        for name, result in actual.items():
            torch.testing.assert_close(result.per_case, getattr(EF, name)(pred, target, num_classes=3))
        collection.reset()
        with pytest.raises(RuntimeError, match='before update'):
            collection.compute()
    finally:
        dist.destroy_process_group()


def test_distributed_collection_with_empty_rank(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip('Gloo unavailable')
    mp.spawn(_distributed_collection, args=(tmp_path / 'init',), nprocs=2)


def test_empty_batch(device):
    empty = torch.empty(0, 3, 4, dtype=torch.long, device=device)
    collection = MetricCollection(metrics=['dice', 'hd95'], num_classes=3, spacing=1)
    values = collection.update(empty, empty)
    assert all(value.shape == (0, 3) for value in values.values())
    for result in collection.compute().values():
        assert result.per_case.shape == (0, 3)
        assert result.per_class.isnan().all() and result.mean.isnan()


@pytest.mark.parametrize('dtype', [torch.float32, torch.uint8, torch.int64])
@pytest.mark.parametrize('representation', ['voxels', 'surfels', 'mesh'])
def test_numeric_binary_targets_preserve_surface_semantics(device, dtype, representation):
    pred, target = _inputs(device, (1, 5, 7, 9))
    pred, target = [torch.nn.functional.one_hot(x, 3).movedim(-1, 1).to(dtype) for x in (pred, target)]
    selection = {'dice': {}, 'hd95': {'surface': representation}}
    actual = EF.evaluate(pred, target, metrics=selection, input_format='one-hot', spacing=1)
    expected = EF.hd95(pred, target, input_format='one-hot', spacing=1, surface=representation)
    torch.testing.assert_close(actual['hd95'], expected, rtol=0, atol=0, equal_nan=True)
    pred = pred.float()
    pred[0, 0, 0, 0, 0] = 0.5
    with pytest.raises(ValueError):
        EF.evaluate(pred, target, metrics=selection, input_format='one-hot', spacing=1)
