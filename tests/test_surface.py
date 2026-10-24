import math
import warnings

import numpy as np
import pytest
import torch

import exacta.functional as EF
import reference
from _data import blobs, pair
from exacta.functional.surface import _boundary, _sqrt


@pytest.mark.parametrize("shape", [(30, 40), (16, 20, 24)])
def test_boundary_matches_scipy(shape):
    labels = blobs(shape, 2, seed=0)
    mask = torch.from_numpy(labels == 1)
    got = _boundary(mask[None, None], len(shape))[0, 0].numpy()
    np.testing.assert_array_equal(got, reference.boundary(labels == 1))


def _backends(device):
    return ["torch", "triton"] if device.type == "cuda" else ["torch", "native"]


@pytest.mark.parametrize(
    ("shape", "spacing"), [((40, 36), (0.5, 1.25)), ((20, 24, 28), (3.0, 0.8, 0.8))]
)
@pytest.mark.parametrize("percentile", [95.0, 100.0, 50.0])
@pytest.mark.parametrize("combine", ["max", "mean", "pooled"])
def test_matches_reference(device, shape, spacing, percentile, combine):
    pred, target = pair(shape, 4, seed=5)
    spacings = [spacing] * pred.shape[0]
    expected = reference.per_case(
        reference.hausdorff, pred, target, 4, spacings=spacings, percentile=percentile, combine=combine
    )
    for backend in _backends(device):
        got = EF.hausdorff_distance(
            torch.from_numpy(pred).to(device),
            torch.from_numpy(target).to(device),
            spacing=spacing,
            percentile=percentile,
            combine=combine,
            num_classes=4,
            backend=backend,
        )
        assert got.device.type == device.type
        np.testing.assert_allclose(got.cpu().numpy(), expected, rtol=1e-10, err_msg=backend)


def test_per_case_spacing(device):
    pred, target = pair((16, 18, 20), 3, seed=6, batch=3)
    spacings = [(1.0, 1.0, 1.0), (2.5, 0.7, 0.7), (0.5, 3.0, 1.0)]
    expected = reference.per_case(reference.hausdorff, pred, target, 3, spacings=spacings, percentile=95)
    for backend in _backends(device):
        got = EF.hd95(
            torch.from_numpy(pred).to(device),
            torch.from_numpy(target).to(device),
            spacing=torch.tensor(spacings, dtype=torch.float64),
            num_classes=3,
            backend=backend,
        )
        np.testing.assert_allclose(got.cpu().numpy(), expected, rtol=1e-10, err_msg=backend)


def test_empty_handling(device):
    empty = torch.zeros(1, 8, 8, 8, dtype=torch.long, device=device)
    filled = empty.clone()
    filled[0, 2:5, 2:5, 2:5] = 1
    assert EF.hd95(empty, empty, spacing=1, num_classes=2)[0, 1].isnan()
    assert EF.hd95(empty, empty, spacing=1, num_classes=2, both_empty=0.0)[0, 1] == 0
    assert EF.hd95(empty, filled, spacing=1, num_classes=2)[0, 1] == math.inf
    assert EF.hd95(filled, empty, spacing=1, num_classes=2, one_empty=99.0)[0, 1] == 99


@pytest.mark.parametrize(
    ("shape", "spacing"), [((40, 36), (0.5, 1.25)), ((20, 24, 28), (3.0, 0.8, 0.8))]
)
@pytest.mark.parametrize("combine", ["max", "mean", "pooled"])
def test_assd_matches_reference(device, shape, spacing, combine):
    pred, target = pair(shape, 4, seed=8)
    spacings = [spacing] * pred.shape[0]
    expected = reference.per_case(reference.assd, pred, target, 4, spacings=spacings, combine=combine)
    for backend in _backends(device):
        got = EF.assd(
            torch.from_numpy(pred).to(device),
            torch.from_numpy(target).to(device),
            spacing=spacing,
            combine=combine,
            num_classes=4,
            backend=backend,
        )
        assert got.device.type == device.type
        np.testing.assert_allclose(got.cpu().numpy(), expected, rtol=1e-10, err_msg=backend)


# Tolerances include exact multiples of the spacing, where boundary voxels tie with the tolerance.
@pytest.mark.parametrize(
    ("shape", "spacing", "tolerance"),
    [((40, 36), (0.5, 1.25), 0.5), ((40, 36), (0.5, 1.25), 1.75), ((20, 24, 28), (3.0, 0.8, 0.8), 1.6),
     ((20, 24, 28), (3.0, 0.8, 0.8), 3.3)],
)
def test_nsd_matches_reference(device, shape, spacing, tolerance):
    pred, target = pair(shape, 4, seed=9)
    spacings = [spacing] * pred.shape[0]
    expected = reference.per_case(reference.nsd, pred, target, 4, spacings=spacings, tolerance=tolerance)
    for backend in _backends(device):
        got = EF.nsd(
            torch.from_numpy(pred).to(device),
            torch.from_numpy(target).to(device),
            spacing=spacing,
            tolerance=tolerance,
            num_classes=4,
            backend=backend,
        )
        assert got.device.type == device.type
        np.testing.assert_array_equal(got.cpu().numpy(), expected, err_msg=backend)


def test_nsd_per_class_tolerance(device):
    pred, target = pair((20, 24, 28), 4, seed=10)
    spacing, tolerances = (2.0, 0.9, 0.9), [1.0, 1.8, 2.5]  # classes 1, 2, 3
    got = EF.nsd(
        torch.from_numpy(pred).to(device),
        torch.from_numpy(target).to(device),
        spacing=spacing,
        tolerance=tolerances,
        num_classes=4,
        include_background=False,
    ).cpu().numpy()
    for k, tol in enumerate(tolerances):
        expected = reference.per_case(reference.nsd, pred, target, 4, spacings=[spacing] * 2, tolerance=tol)
        np.testing.assert_array_equal(got[:, k], expected[:, k + 1])


@pytest.mark.parametrize(
    "tolerance",
    [2 * 0.7, math.sqrt(0.7 * 0.7 + 0.9 * 0.9)],  # two voxels along axis 0; one voxel diagonal
    ids=["axis", "diagonal"],
)
def test_nsd_counts_distances_equal_to_tolerance(device, tolerance):
    pred, target = pair((20, 24, 28), 3, seed=11)
    spacing = (0.7, 0.9, 1.3)
    expected = reference.per_case(reference.nsd, pred, target, 3, spacings=[spacing] * 2, tolerance=tolerance)
    for backend in _backends(device):
        nsd = lambda tol: EF.nsd(  # noqa: E731
            torch.from_numpy(pred).to(device),
            torch.from_numpy(target).to(device),
            spacing=spacing,
            tolerance=tol,
            num_classes=3,
            backend=backend,
        ).cpu().numpy()
        at = nsd(tolerance)
        np.testing.assert_array_equal(at, expected, err_msg=backend)
        assert (at > nsd(math.nextafter(tolerance, 0))).any(), "no boundary voxel tied with the tolerance"


def test_sqrt_is_correctly_rounded(device):
    x = torch.arange(200_001, dtype=torch.float64)
    expected = torch.tensor([math.sqrt(v) for v in x.tolist()], dtype=torch.float64)
    assert torch.equal(_sqrt(x.to(device)).cpu(), expected)


def test_nsd_empty_handling(device):
    empty = torch.zeros(1, 8, 8, 8, dtype=torch.long, device=device)
    filled = empty.clone()
    filled[0, 2:5, 2:5, 2:5] = 1
    assert EF.nsd(empty, empty, spacing=1, tolerance=1, num_classes=2)[0, 1].isnan()
    assert EF.nsd(empty, filled, spacing=1, tolerance=1, num_classes=2)[0, 1] == 0
    assert EF.nsd(filled, empty, spacing=1, tolerance=1, num_classes=2, one_empty=math.nan)[0, 1].isnan()
    assert EF.nsd(filled, filled, spacing=1, tolerance=0, num_classes=2)[0, 1] == 1
    assert EF.assd(empty, filled, spacing=1, num_classes=2)[0, 1] == math.inf


def test_nsd_rejects_bad_tolerance():
    x = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    with pytest.raises(ValueError, match=r"one value per evaluated class \(2 values, for classes 1..2\)"):
        EF.nsd(x, x, spacing=1, tolerance=[1.0, 1.0, 1.0], num_classes=3, include_background=False)
    with pytest.raises(ValueError, match="non-negative"):
        EF.nsd(x, x, spacing=1, tolerance=-1.0, num_classes=2)


@pytest.mark.parametrize("fn", [EF.hausdorff_distance, EF.assd])
def test_rejects_unknown_combine(fn):
    x = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="combine must be"):
        fn(x, x, spacing=1, num_classes=2, combine="symmetric")


def test_spacing_is_required():
    x = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="spacing is required"):
        EF.hd95(x, x, spacing=None, num_classes=2)


def test_native_backend_rejects_cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    x = torch.zeros(1, 4, 4, 4, dtype=torch.long, device="cuda")
    x[0, 1:3, 1:3, 1:3] = 1
    with pytest.raises(NotImplementedError, match="runs on CPU"):
        EF.hd95(x, x, spacing=1, num_classes=2, backend="native")


def test_triton_backend_rejects_cpu():
    x = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    x[0, 1:3, 1:3, 1:3] = 1
    with pytest.raises(NotImplementedError, match="runs on GPU"):
        EF.hd95(x, x, spacing=1, num_classes=2, backend="triton")


def test_rejects_unknown_backend():
    x = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="backend must be"):
        EF.hd95(x, x, spacing=1, num_classes=2, backend="cuda")


@pytest.mark.parametrize("percentile", [95.0, 100.0])
def test_matches_monai(percentile):
    monai = pytest.importorskip("monai.metrics")
    pred, target = pair((20, 24, 28), 4, seed=7)
    spacing = (2.0, 0.9, 0.9)
    one_hot = lambda x: torch.nn.functional.one_hot(torch.from_numpy(x), 4).movedim(-1, 1)  # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = monai.compute_hausdorff_distance(
            one_hot(pred),
            one_hot(target),
            include_background=True,
            percentile=percentile,
            spacing=spacing,
        )
    got = EF.hausdorff_distance(pred, target, spacing=spacing, percentile=percentile, num_classes=4)
    np.testing.assert_allclose(got, expected.double().numpy(), rtol=1e-5)


def _monai_one_hot(x):
    return torch.nn.functional.one_hot(torch.from_numpy(x), 4).movedim(-1, 1)


def test_assd_matches_monai():
    monai = pytest.importorskip("monai.metrics")
    pred, target = pair((20, 24, 28), 4, seed=12)
    spacing = (2.0, 0.9, 0.9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = monai.compute_average_surface_distance(
            _monai_one_hot(pred),
            _monai_one_hot(target),
            include_background=True,
            symmetric=True,
            spacing=spacing,
        )
    got = EF.assd(pred, target, spacing=spacing, num_classes=4)
    np.testing.assert_allclose(got, expected.double().numpy(), rtol=1e-5)


def test_nsd_matches_monai():
    monai = pytest.importorskip("monai.metrics")
    pred, target = pair((20, 24, 28), 4, seed=13)
    spacing, tolerances = (2.0, 0.9, 0.9), [1.0, 1.5, 2.5, 3.5]  # none is a distance on this grid
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = monai.compute_surface_dice(
            _monai_one_hot(pred),
            _monai_one_hot(target),
            class_thresholds=tolerances,
            include_background=True,
            spacing=spacing,
        )
    got = EF.nsd(pred, target, spacing=spacing, tolerance=tolerances, num_classes=4)
    np.testing.assert_allclose(got, expected.double().numpy(), rtol=1e-6)


def _medpy_per_case(fn, pred, target, num_classes, spacing):
    """MedPy raises on empty masks, so cases where a class is absent stay NaN."""
    out = np.full((pred.shape[0], num_classes), np.nan)
    for b, c in np.ndindex(out.shape):
        p, t = pred[b] == c, target[b] == c
        if p.any() and t.any():
            out[b, c] = fn(p, t, voxelspacing=spacing)
    return out


def test_pooled_hd95_matches_medpy():
    medpy = pytest.importorskip("medpy.metric.binary")
    pred, target = pair((20, 24, 28), 4, seed=14)
    spacing = (2.0, 0.9, 0.9)
    expected = _medpy_per_case(medpy.hd95, pred, target, 4, spacing)
    got = EF.hd95(pred, target, spacing=spacing, num_classes=4, combine="pooled", one_empty=math.nan)
    np.testing.assert_allclose(got, expected, rtol=1e-10)
    default = EF.hd95(pred, target, spacing=spacing, num_classes=4, one_empty=math.nan)
    assert not np.allclose(default, expected, equal_nan=True), "combine had no effect"


def test_pooled_assd_matches_medpy():
    medpy = pytest.importorskip("medpy.metric.binary")
    from importlib.metadata import version

    if tuple(int(v) for v in version("medpy").split(".")[:3]) < (0, 5, 2):
        pytest.skip("MedPy < 0.5.2 defines assd differently (0.4 averages the directed means)")
    pred, target = pair((20, 24, 28), 4, seed=15)
    spacing = (2.0, 0.9, 0.9)
    expected = _medpy_per_case(medpy.assd, pred, target, 4, spacing)
    got = EF.assd(pred, target, spacing=spacing, num_classes=4, one_empty=math.nan)
    np.testing.assert_allclose(got, expected, rtol=1e-10)


def _many_structures(shape, classes, seed):
    """Small ellipsoids, one per class, and a prediction shifted by a voxel or two."""
    rng = np.random.default_rng(seed)
    grid = np.indices(shape, dtype=np.float64)
    target = np.zeros(shape, np.int64)
    for c in range(1, classes):
        center = rng.uniform(4, np.array(shape) - 4)
        radii = rng.uniform(1.5, 5, len(shape))
        column = (-1,) + (1,) * len(shape)
        inside = (((grid - center.reshape(column)) / radii.reshape(column)) ** 2).sum(0) <= 1
        target[inside] = c
    pred = np.zeros_like(target)
    pred[1:, :-1] = target[:-1, 1:]  # shift without wrapping around
    return pred[None], target[None]


@pytest.mark.parametrize("surface", ["voxels", "surfels"])
def test_many_classes_match_per_class_computation(device, surface):
    # Many small structures go through batched transforms on GPU; the values must not change.
    pred, target = _many_structures((30, 34, 38), 25, seed=30)
    spacing = (1.2, 0.8, 0.9)
    if surface == "voxels":
        expected = reference.per_case(reference.hausdorff, pred, target, 25, spacings=[spacing], percentile=95)
    else:
        expected = EF.hd95(pred, target, spacing=spacing, num_classes=25, surface=surface, backend="native")
    got = EF.hd95(
        torch.from_numpy(pred).to(device), torch.from_numpy(target).to(device), spacing=spacing,
        num_classes=25, surface=surface,
    )  # fmt: skip
    np.testing.assert_allclose(got.cpu().numpy(), expected, rtol=1e-12)
