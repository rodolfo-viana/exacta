"""The ``surface="surfels"`` representation, checked against DeepMind's surface-distance and MONAI."""

import itertools
import math
import warnings

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy import ndimage

import exacta.functional as EF
from _data import pair
from exacta.functional._surfel_tables import surfel_measures
from exacta.functional.surface import _neighbour_code
from exacta.metrics import NSD

SPACINGS_3D = [(1.0, 1.0, 1.0), (1.5, 0.8, 0.8), (0.7, 0.9, 1.3), (3.0, 0.5, 2.0)]


def _bit(*voxel):
    """Bit of the block voxel (a, b[, c]) in a neighbour code."""
    n = len(voxel)
    return 1 << ((1 << n) - 1 - int("".join(map(str, voxel)), 2))


def _transform(code, ndim, f):
    out = 0
    for voxel in itertools.product((0, 1), repeat=ndim):
        if code & _bit(*voxel):
            out |= _bit(*f(*voxel))
    return out


def _mirror(axis):
    return lambda *voxel: tuple(1 - x if i == axis else x for i, x in enumerate(voxel))


# --- Tables -------------------------------------------------------------------------------------


@pytest.mark.parametrize("spacing", SPACINGS_3D + [(1.0, 1.0), (1.5, 0.8), (0.7, 1.3)])
def test_tables_match_deepmind_bit_for_bit(spacing):
    lookup = pytest.importorskip("surface_distance.lookup_tables")
    if len(spacing) == 3:
        expected = lookup.create_table_neighbour_code_to_surface_area(spacing)
    else:
        expected = lookup.create_table_neighbour_code_to_contour_length(spacing)
    np.testing.assert_array_equal(surfel_measures(spacing), expected)


@pytest.mark.parametrize("spacing", SPACINGS_3D)
def test_area_table_has_the_symmetries_of_the_cube(spacing):
    areas = surfel_measures(spacing)
    for code in range(256):
        # a block and its complement hold the same surface
        assert math.isclose(areas[code], areas[255 - code], rel_tol=1e-13)
        # mirroring the block along an axis does not change the area, whatever the spacing
        for axis in range(3):
            mirrored = _transform(code, 3, _mirror(axis))
            assert math.isclose(areas[code], areas[mirrored], rel_tol=1e-13)
    # permuting the axes permutes the spacing
    for perm in itertools.permutations(range(3)):
        permuted = surfel_measures(tuple(spacing[perm[i]] for i in range(3)))
        for code in range(256):
            moved = _transform(code, 3, lambda *v: tuple(v[perm[i]] for i in range(3)))
            assert math.isclose(permuted[moved], areas[code], rel_tol=1e-13)


@pytest.mark.parametrize("spacing", SPACINGS_3D)
def test_area_table_known_values(spacing):
    h0, h1, h2 = spacing
    areas = surfel_measures(spacing)
    # one corner inside: the triangle through the midpoints of its three edges
    assert math.isclose(areas[_bit(1, 1, 1)], math.hypot(h1 * h2, h0 * h2, h0 * h1) / 8, rel_tol=1e-14)
    # half of the block inside: a flat face across the block
    half = sum(_bit(1, b, c) for b in (0, 1) for c in (0, 1))
    assert math.isclose(areas[half], h1 * h2, rel_tol=1e-14)
    assert areas[0] == areas[255] == 0


def test_contour_lengths_known_values():
    h0, h1 = 1.5, 0.8
    lengths = surfel_measures((h0, h1))
    diag = math.hypot(h0, h1) / 2
    assert math.isclose(lengths[_bit(0, 0)], diag)
    assert lengths[_bit(0, 0) | _bit(0, 1)] == h1  # one row inside: contour along axis 1
    assert lengths[_bit(0, 0) | _bit(1, 0)] == h0  # one column inside: contour along axis 0
    assert math.isclose(lengths[_bit(0, 0) | _bit(1, 1)], 2 * diag)
    assert lengths[0] == lengths[15] == 0


@pytest.mark.parametrize("shape", [(9, 11), (7, 8, 9)])
def test_neighbour_code_matches_deepmind_construction(device, shape):
    """DeepMind pads the mask with one zero layer on the high side and correlates it with the
    kernel; scipy's correlate treats the low side as zero too, so every corner is covered."""
    mask = np.random.default_rng(0).random(shape) < 0.5
    kernel = np.array([[8, 4], [2, 1]] if len(shape) == 2 else [[[128, 64], [32, 16]], [[8, 4], [2, 1]]])
    padded = np.zeros(tuple(n + 1 for n in shape), np.uint8)
    padded[tuple(slice(0, n) for n in shape)] = mask
    expected = ndimage.correlate(padded, kernel, mode="constant", cval=0)
    got = _neighbour_code(torch.from_numpy(mask).to(device)).cpu().numpy()
    np.testing.assert_array_equal(got, expected)


# --- Metrics against DeepMind -------------------------------------------------------------------


def _deepmind(p, t, spacing):
    sd = pytest.importorskip("surface_distance")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # scipy.ndimage.morphology namespace
        return sd, sd.compute_surface_distances(t, p, spacing)  # DeepMind's order is (gt, pred)


def _weighted_percentile(d, a, q):
    """DeepMind's compute_robust_hausdorff rule, for directions it does not combine itself."""
    order = np.lexsort((a, d))
    d, a = d[order], a[order]
    cum = np.cumsum(a) / np.sum(a)
    return d[min(np.searchsorted(cum, q / 100), len(d) - 1)]


def _deepmind_hd(p, t, spacing, percentile, combine):
    sd, s = _deepmind(p, t, spacing)
    d_tp, a_t = s["distances_gt_to_pred"], s["surfel_areas_gt"]
    d_pt, a_p = s["distances_pred_to_gt"], s["surfel_areas_pred"]
    if combine == "max":
        return sd.compute_robust_hausdorff(s, percentile)
    if combine == "mean":
        return (_weighted_percentile(d_pt, a_p, percentile) + _weighted_percentile(d_tp, a_t, percentile)) / 2
    return _weighted_percentile(np.concatenate([d_pt, d_tp]), np.concatenate([a_p, a_t]), percentile)


def _deepmind_assd(p, t, spacing, combine):
    sd, s = _deepmind(p, t, spacing)
    directed = sd.compute_average_surface_distance(s)
    if combine == "max":
        return max(directed)
    if combine == "mean":
        return np.mean(directed)
    d = np.concatenate([s["distances_gt_to_pred"], s["distances_pred_to_gt"]])
    a = np.concatenate([s["surfel_areas_gt"], s["surfel_areas_pred"]])
    return np.sum(d * a) / np.sum(a)


def _deepmind_nsd(p, t, spacing, tolerance):
    sd, s = _deepmind(p, t, spacing)
    return sd.compute_surface_dice_at_tolerance(s, tolerance)


def _per_case(fn, pred, target, num_classes, **kw):
    out = np.full((pred.shape[0], num_classes), np.nan)
    for b, c in np.ndindex(out.shape):
        p, t = pred[b] == c, target[b] == c
        if p.any() and t.any():
            out[b, c] = fn(p, t, **kw)
    return out


def _backends(device):
    return ["torch", "triton"] if device.type == "cuda" else ["torch", "native"]


CASES = [((40, 36), (0.5, 1.25)), ((20, 24, 28), (1.5, 0.8, 0.8)), ((18, 22, 26), (3.0, 0.7, 0.9))]


@pytest.mark.parametrize(("shape", "spacing"), CASES)
@pytest.mark.parametrize("percentile", [100.0, 95.0, 50.0])
@pytest.mark.parametrize("combine", ["max", "mean", "pooled"])
def test_hd_matches_deepmind(device, shape, spacing, percentile, combine):
    pred, target = pair(shape, 4, seed=20)
    expected = _per_case(_deepmind_hd, pred, target, 4, spacing=spacing, percentile=percentile, combine=combine)
    for backend in _backends(device):
        got = EF.hausdorff_distance(
            torch.from_numpy(pred).to(device),
            torch.from_numpy(target).to(device),
            spacing=spacing,
            percentile=percentile,
            combine=combine,
            surface="surfels",
            num_classes=4,
            one_empty=math.nan,
            backend=backend,
        )
        np.testing.assert_allclose(got.cpu().numpy(), expected, rtol=1e-12, err_msg=backend)


@pytest.mark.parametrize(("shape", "spacing"), CASES)
@pytest.mark.parametrize("combine", ["max", "mean", "pooled"])
def test_assd_matches_deepmind(device, shape, spacing, combine):
    pred, target = pair(shape, 4, seed=21)
    expected = _per_case(_deepmind_assd, pred, target, 4, spacing=spacing, combine=combine)
    for backend in _backends(device):
        got = EF.assd(
            torch.from_numpy(pred).to(device),
            torch.from_numpy(target).to(device),
            spacing=spacing,
            combine=combine,
            surface="surfels",
            num_classes=4,
            one_empty=math.nan,
            backend=backend,
        )
        np.testing.assert_allclose(got.cpu().numpy(), expected, rtol=1e-12, err_msg=backend)


@pytest.mark.parametrize(("shape", "spacing"), CASES)
@pytest.mark.parametrize("tolerance", [0.5, 1.6, 2.5])  # 1.6 = 2 x 0.8, a tie on the second grid
def test_nsd_matches_deepmind(device, shape, spacing, tolerance):
    pred, target = pair(shape, 4, seed=22)
    expected = _per_case(_deepmind_nsd, pred, target, 4, spacing=spacing, tolerance=tolerance)
    for backend in _backends(device):
        got = EF.nsd(
            torch.from_numpy(pred).to(device),
            torch.from_numpy(target).to(device),
            spacing=spacing,
            tolerance=tolerance,
            surface="surfels",
            num_classes=4,
            one_empty=math.nan,
            backend=backend,
        )
        np.testing.assert_allclose(got.cpu().numpy(), expected, rtol=1e-12, err_msg=backend)


@pytest.mark.parametrize(("shape", "spacing"), [((40, 36), (0.5, 1.25)), ((20, 24, 28), (2.0, 0.9, 0.9))])
def test_nsd_matches_monai_subvoxels(shape, spacing):
    monai = pytest.importorskip("monai.metrics")
    pred, target = pair(shape, 4, seed=23)
    tolerances = [1.0, 1.5, 2.5, 3.5]  # none is a distance on these grids
    one_hot = lambda x: torch.nn.functional.one_hot(torch.from_numpy(x), 4).movedim(-1, 1)  # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = monai.compute_surface_dice(
            one_hot(pred),
            one_hot(target),
            class_thresholds=tolerances,
            include_background=True,
            spacing=spacing,
            use_subvoxels=True,
        )
    got = EF.nsd(pred, target, spacing=spacing, tolerance=tolerances, surface="surfels", num_classes=4)
    np.testing.assert_allclose(got, expected.double().numpy(), rtol=1e-5)  # MONAI works in float32


def test_surfels_differ_from_voxels():
    pred, target = pair((20, 24, 28), 3, seed=24)
    kw = dict(spacing=(1.5, 0.8, 0.8), num_classes=3, tolerance=1.0)
    assert not np.allclose(EF.nsd(pred, target, **kw), EF.nsd(pred, target, surface="surfels", **kw))


def test_empty_handling():
    empty = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    filled = empty.clone()
    filled[0, 2:5, 2:5, 2:5] = 1
    kw = dict(spacing=1, num_classes=2, surface="surfels")
    assert EF.hd95(empty, empty, **kw)[0, 1].isnan()
    assert EF.hd95(empty, filled, **kw)[0, 1] == math.inf
    assert EF.assd(filled, empty, **kw)[0, 1] == math.inf
    assert EF.nsd(empty, filled, tolerance=1, **kw)[0, 1] == 0
    assert EF.nsd(filled, filled, tolerance=0, **kw)[0, 1] == 1


def test_rejects_unknown_surface():
    x = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    x[0, 1:3, 1:3, 1:3] = 1
    with pytest.raises(ValueError, match="surface must be"):
        EF.hd95(x, x, spacing=1, num_classes=2, surface="points")


def test_metric_class_accepts_surface(device):
    pred, target = (torch.from_numpy(a).to(device) for a in pair((16, 18, 20), 3, seed=25))
    metric = NSD(tolerance=1.0, surface="surfels", num_classes=3, include_background=False)
    metric.update(pred, target, spacing=(1.5, 1.0, 1.0))
    expected = EF.nsd(
        pred,
        target,
        spacing=(1.5, 1.0, 1.0),
        tolerance=1.0,
        surface="surfels",
        num_classes=3,
        include_background=False,
    )
    torch.testing.assert_close(metric.compute().per_case, expected)


# --- Properties ---------------------------------------------------------------------------------


@st.composite
def mask_pairs(draw):
    shape = draw(st.tuples(*[st.integers(1, 7)] * 3))
    return draw(arrays(np.bool_, shape)), draw(arrays(np.bool_, shape))


@settings(max_examples=150, deadline=None)
@given(mask_pairs(), st.tuples(*[st.floats(0.25, 4.0)] * 3), st.floats(0.0, 5.0))
def test_random_masks_match_deepmind(masks, spacing, tolerance):
    pytest.importorskip("surface_distance")
    p, t = masks
    if not (p.any() and t.any()):
        return
    expected = {
        "hd95": _deepmind_hd(p, t, spacing, 95, "max"),
        "assd": _deepmind_assd(p, t, spacing, "mean"),
        "nsd": _deepmind_nsd(p, t, spacing, tolerance),
    }
    P, T = p[None, None], t[None, None]
    for backend in ("native", "torch"):
        kw = dict(spacing=spacing, surface="surfels", input_format="one-hot", backend=backend)
        got = {
            "hd95": EF.hd95(P, T, **kw)[0, 0],
            "assd": EF.assd(P, T, combine="mean", **kw)[0, 0],
            "nsd": EF.nsd(P, T, tolerance=tolerance, **kw)[0, 0],
        }
        for name in expected:
            assert math.isclose(got[name], expected[name], rel_tol=1e-12), (name, backend)


@settings(max_examples=150, deadline=None)
@given(mask_pairs(), st.tuples(*[st.floats(0.25, 4.0)] * 3))
def test_symmetry_and_bounds(masks, spacing):
    p, t = masks
    if not (p.any() and t.any()):
        return
    P, T = p[None, None], t[None, None]
    kw = dict(spacing=spacing, surface="surfels", input_format="one-hot")
    assert EF.hausdorff_distance(P, T, **kw)[0, 0] == EF.hausdorff_distance(T, P, **kw)[0, 0]
    assert math.isclose(EF.assd(P, T, **kw)[0, 0], EF.assd(T, P, **kw)[0, 0], rel_tol=1e-12)
    n = EF.nsd(P, T, tolerance=1.0, **kw)[0, 0]
    assert 0 <= n <= 1
    assert EF.nsd(P, P, tolerance=0.0, **kw)[0, 0] == 1
    assert EF.assd(P, P, **kw)[0, 0] == 0
