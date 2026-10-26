"""Physical units, independent voxel counts, and shared evaluation."""
import numpy as np
import pytest
import torch

from exacta import functional as f, metrics


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("one_hot", [False, True])
@pytest.mark.parametrize("unit,divisor", [("mm3", 1), ("ml", 1000)])
def test_physical_volume_counts(device, one_hot, unit, divisor):
    rng = np.random.default_rng(414)
    p, t = rng.integers(0, 3, (2, 3, 4, 5)), rng.integers(0, 3, (2, 3, 4, 5))
    mask = rng.random(t.shape) > .2
    spacing = np.array([[.5, .7, 2], [1, 2, 3]])
    expected_p = np.array([[np.count_nonzero((p[b] == c) & mask[b]) for c in (1, 2)] for b in (0, 1)])
    expected_t = np.array([[np.count_nonzero((t[b] == c) & mask[b]) for c in (1, 2)] for b in (0, 1)])
    scale = spacing.prod(1)[:, None] / divisor
    if one_hot:
        p, t = (np.moveaxis(np.eye(3, dtype=bool)[x], -1, 1) for x in (p, t))
    p, t, mask = (torch.as_tensor(x, device=device) for x in (p, t, mask))
    kwargs = dict(spacing=spacing, unit=unit, include_background=False, num_classes=3,
                  input_format="one-hot" if one_hot else "index", mask=mask)
    for fn, expected in [(f.predicted_volume, expected_p), (f.target_volume, expected_t),
                         (f.volume_error, expected_p-expected_t)]:
        np.testing.assert_allclose(fn(p, t, **kwargs).cpu(), expected * scale)
    np.testing.assert_allclose(f.volume_error(p, t, absolute=True, **kwargs).cpu(),
                               np.abs(expected_p-expected_t) * scale)
    collection = f.evaluate(p, t, metrics={"dice": {}, "predicted_volume": {"unit": unit},
                            "target_volume": {"unit": unit}, "volume_error": {"unit": unit, "absolute": True}},
                            **{k: v for k, v in kwargs.items() if k != "unit"})
    torch.testing.assert_close(collection["predicted_volume"], f.predicted_volume(p, t, **kwargs))
    torch.testing.assert_close(collection["volume_error"], f.volume_error(p, t, absolute=True, **kwargs))


def test_ignored_and_unobserved_volume():
    p = np.zeros((2, 2, 3, 4), dtype=np.int64)
    t = p.copy()
    t[1] = -1
    p[1] = 999
    kwargs = dict(num_classes=2, spacing=2, ignore_index=-1)
    for fn in (f.predicted_volume, f.target_volume, f.volume_error):
        value = fn(p, t, **kwargs)
        assert value[0, 1] == 0
        assert np.isnan(value[1]).all()
    assert f.predicted_volume(p, t, **kwargs)[0, 0] == 24 * 8
    for cls in (metrics.PredictedVolume, metrics.TargetVolume, metrics.VolumeError):
        metric = cls(**kwargs)
        metric.update(p, t)
        result = metric.compute()
        assert result.per_case.shape == (2, 2)
        torch.testing.assert_close(result.per_class, result.per_case[0])
        metric.reset()
        metric.update(p[:1], t[:1], spacing=(1, 1, 1))
        assert metric.compute().per_class[0] in (0, 24)


@pytest.mark.parametrize("spacing,unit", [(None, "mm3"), (0, "mm3"), (-1, "mm3"),
    (float("inf"), "mm3"), ([1, 2], "mm3"), (1, "litres"), (1e150, "mm3")])
def test_volume_invalid_units_spacing(spacing, unit):
    p = torch.zeros((1, 2, 3, 4), dtype=torch.int64)
    with pytest.raises(ValueError):
        f.volume_error(p, p, spacing=spacing, unit=unit, num_classes=2)


def test_volume_2d_and_empty_batch():
    p = torch.zeros((1, 3, 4), dtype=torch.int64)
    with pytest.raises(ValueError, match="3D"):
        f.predicted_volume(p, p, spacing=1, num_classes=2)
    p = torch.empty((0, 2, 3, 4), dtype=torch.int64)
    assert f.target_volume(p, p, spacing=1, num_classes=2).shape == (0, 2)
