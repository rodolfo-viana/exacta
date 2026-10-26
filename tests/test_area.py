"""Physical pixel areas: independent counts, units, missing observations and aggregation."""
from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist

from _process import destroy_process_group, spawn
from exacta import functional as f, metrics
from exacta.functional import overlap

NAMES = ("predicted_area", "target_area", "area_error")


@pytest.mark.parametrize("layout", ["index", "one-hot", "multilabel"])
@pytest.mark.parametrize("unit,divisor", [("mm2", 1), ("cm2", 100)])
def test_area_matches_pixel_counts(device, layout, unit, divisor):
    rng = np.random.default_rng(607)
    shape = (2, 3, 7, 9) if layout == "multilabel" else (2, 7, 9)
    p, t = (rng.integers(0, 2 if layout == "multilabel" else 3, shape) for _ in range(2))
    mask = rng.random((2, 7, 9)) > .2
    spacing = np.array([[.3, .7], [1.5, 2.0]])
    expected = []
    for x in (p, t):
        expected.append(np.array([[np.count_nonzero((x[b, c] if layout == "multilabel" else x[b] == c)
                                                   & mask[b]) for c in (1, 2)] for b in range(2)])
                        * spacing.prod(1)[:, None] / divisor)
    if layout == "one-hot":
        p, t = (np.moveaxis(np.eye(3, dtype=bool)[x], -1, 1) for x in (p, t))
    p, t, mask = (torch.as_tensor(x, device=device).transpose(-1, -2) for x in (p, t, mask))
    kwargs = dict(spacing=spacing[:, ::-1].copy(), unit=unit, include_background=False,
                  input_format="index" if layout == "index" else "one-hot", num_classes=3, mask=mask)
    for name, reference in zip(NAMES, [*expected, expected[0] - expected[1]]):
        result = getattr(f, name)(p, t, **kwargs)
        assert result.device == p.device and result.dtype == torch.float64
        np.testing.assert_allclose(result.cpu(), reference, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(f.area_error(p, t, absolute=True, **kwargs).cpu(),
                               np.abs(expected[0] - expected[1]), rtol=1e-12, atol=1e-12)


def test_area_ignored_pixels_spacing_override_and_state(device):
    p = torch.tensor([[[0, 1, 1], [2, 0, 1]], [[99, 99, 99], [99, 99, 99]]], device=device)
    t = torch.tensor([[[0, 1, -1], [2, 1, 1]], [[-1, -1, -1], [-1, -1, -1]]], device=device)
    options = dict(num_classes=4, include_background=False, ignore_index=-1)
    expected = torch.tensor([[2, 1, 0], [3, 1, 0], [-1, 0, 0]], device=device, dtype=torch.float64)
    classes = (metrics.PredictedArea, metrics.TargetArea, metrics.AreaError)
    for name, cls, reference in zip(NAMES, classes, expected):
        metric = cls(spacing=2, **options)
        result = metric.update(p, t)
        torch.testing.assert_close(result[0], reference * 4)
        assert result[1].isnan().all()
        metric.update(p[:1], t[:1], spacing=(.5, 1))
        torch.testing.assert_close(metric.compute().per_class, reference * 2.25)
        metric.reset()
        with pytest.raises(RuntimeError, match="before update"):
            metric.compute()
        assert getattr(f, name)(p[:0], t[:0], spacing=1, **options).shape == (0, 3)
    metric = metrics.AreaError(spacing=1, absolute=True, **options)
    torch.testing.assert_close(metric.update(p[:1], t[:1])[0], expected[2].abs())


def test_area_collection_shares_counts_and_preserves_dimensions(monkeypatch):
    p = np.array([[[0, 1, 1], [0, 1, 0]]])[:, :, ::-1]
    t = np.array([[[0, 1, 0], [0, 1, 0]]])[:, :, ::-1]
    count = overlap._confusion_counts_and_seg
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return count(*args, **kwargs)
    monkeypatch.setattr(overlap, "_confusion_counts_and_seg", counted)
    selection = {"dice": {}, "predicted_area": {}, "target_area": {"unit": "cm2"},
                 "area_error": {"absolute": True}}
    collection = metrics.MetricCollection(metrics=selection, num_classes=2, spacing=(.5, 2))
    collection.update(p, t)
    assert len(calls) == 1
    for name, result in collection.compute().items():
        options = dict(spacing=(.5, 2)) if name != "dice" else {}
        np.testing.assert_allclose(result.per_case, getattr(f, name)(p, t, num_classes=2,
                                                                    **options, **selection[name]))
    # A unit shared with a prior request must not bypass dimensional validation.
    with pytest.raises(ValueError, match="3D"):
        f.evaluate(p, t, metrics={"predicted_area": {}, "predicted_volume": {"unit": "mm2"}},
                   num_classes=2, spacing=1)


@pytest.mark.parametrize("spacing,unit", [(None, "mm2"), (0, "mm2"), (-1, "mm2"),
    (float("inf"), "mm2"), ([1, 2, 3], "mm2"), ([[1, 2], [1, 2]], "mm2"), (1, "ml"), (1e150, "mm2")])
def test_area_rejects_invalid_units_and_spacing(spacing, unit):
    p = torch.zeros((1, 3, 4), dtype=torch.int64)
    with pytest.raises(ValueError):
        f.area_error(p, p, spacing=spacing, unit=unit, num_classes=2)


def test_area_rejects_3d_and_handles_unobserved_pixels(device):
    p = torch.zeros((1, 2, 3, 4), dtype=torch.int64, device=device)
    with pytest.raises(ValueError, match="2D"):
        f.predicted_area(p, p, spacing=1, num_classes=2)
    for shape in [(2, 0, 3), (2, 3, 4)]:
        p = torch.zeros(shape, dtype=torch.int64, device=device)
        for name in NAMES:
            result = getattr(f, name)(p, p, spacing=1, num_classes=2, mask=torch.zeros_like(p, dtype=torch.bool))
            assert result.isnan().all()
    for cls in (metrics.PredictedArea, metrics.TargetArea, metrics.AreaError):
        with pytest.raises(ValueError, match="unit"):
            cls(unit="mm3")


def _distributed_area(rank, init_file):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        metric = metrics.PredictedArea(spacing=(.5, 2), num_classes=2)
        collection = metrics.MetricCollection(metrics=NAMES, spacing=(.5, 2), num_classes=2)
        if rank == 1:
            p = torch.tensor([[[0, 1], [1, 1]]])
            metric.update(p, p)
            collection.update(p, p)
        expected = torch.tensor([[1., 3.]], dtype=torch.float64)
        torch.testing.assert_close(metric.compute().per_case, expected)
        for name, result in collection.compute().items():
            torch.testing.assert_close(result.per_case, expected if name != "area_error" else expected * 0)
    finally:
        destroy_process_group()


def test_distributed_area_with_empty_rank(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    spawn(_distributed_area, args=(tmp_path / "init",), directory=tmp_path / "workers")
