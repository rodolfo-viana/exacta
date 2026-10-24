"""Property-based tests on small random binary volumes."""

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import exacta.functional as EF
import reference

shapes = st.tuples(*[st.integers(1, 7)] * 3)
spacings = st.tuples(*[st.floats(0.25, 4.0)] * 3)


@st.composite
def mask_pairs(draw):
    shape = draw(shapes)
    a = draw(arrays(np.bool_, shape))
    b = draw(arrays(np.bool_, shape))
    return a[None, None], b[None, None]


def _hd(a, b, spacing, percentile=100.0, backend="auto"):
    return EF.hausdorff_distance(
        a, b, spacing=spacing, percentile=percentile, input_format="one-hot", backend=backend
    )[0, 0]


@settings(max_examples=200, deadline=None)
@given(mask_pairs(), spacings)
def test_hd_matches_reference_and_backends_agree(masks, spacing):
    a, b = masks
    expected = reference.hausdorff(a[0, 0], b[0, 0], spacing, 95)
    for backend in ("native", "torch"):
        got = _hd(a, b, spacing, 95, backend)
        assert (math.isnan(got) and math.isnan(expected)) or math.isclose(got, expected, rel_tol=1e-10)


@settings(max_examples=200, deadline=None)
@given(mask_pairs(), spacings)
def test_hd_is_symmetric_and_ordered(masks, spacing):
    a, b = masks
    if not (a.any() and b.any()):
        return
    hd, hd_ba = _hd(a, b, spacing), _hd(b, a, spacing)
    assert hd == hd_ba
    assert _hd(a, b, spacing, 95) <= hd
    assert _hd(a, a, spacing) == 0
    # at 100%, pooling both directions gives the largest distance in either: the classic HD
    pooled = EF.hausdorff_distance(a, b, spacing=spacing, combine="pooled", input_format="one-hot")
    assert pooled[0, 0] == hd


@settings(max_examples=100, deadline=None)
@given(mask_pairs(), spacings, st.floats(0.5, 3.0))
def test_hd_scales_with_spacing(masks, spacing, k):
    a, b = masks
    if not (a.any() and b.any()):
        return
    scaled = tuple(k * s for s in spacing)
    assert math.isclose(_hd(a, b, scaled), k * _hd(a, b, spacing), rel_tol=1e-10)


@settings(max_examples=200, deadline=None)
@given(mask_pairs())
def test_dice_bounds_and_symmetry(masks):
    a, b = masks
    d = EF.dice(a, b, input_format="one-hot")[0, 0]
    if not (a.any() or b.any()):
        assert math.isnan(d)
        return
    assert 0 <= d <= 1
    assert d == EF.dice(b, a, input_format="one-hot")[0, 0]
    assert math.isclose(d, reference.dice(a, b), rel_tol=1e-12)


def _assd(a, b, spacing, combine="pooled", backend="auto"):
    return EF.assd(a, b, spacing=spacing, combine=combine, input_format="one-hot", backend=backend)[0, 0]


def _nsd(a, b, spacing, tolerance, backend="auto"):
    return EF.nsd(a, b, spacing=spacing, tolerance=tolerance, input_format="one-hot", backend=backend)[0, 0]


def _same(got, expected, rel_tol=1e-10):
    return (math.isnan(got) and math.isnan(expected)) or math.isclose(got, expected, rel_tol=rel_tol)


@settings(max_examples=200, deadline=None)
@given(mask_pairs(), spacings, st.sampled_from(["max", "mean", "pooled"]))
def test_assd_matches_reference_and_backends_agree(masks, spacing, combine):
    a, b = masks
    expected = reference.assd(a[0, 0], b[0, 0], spacing, combine)
    for backend in ("native", "torch"):
        assert _same(_assd(a, b, spacing, combine, backend), expected)


@settings(max_examples=200, deadline=None)
@given(mask_pairs(), spacings, st.floats(0.0, 10.0))
def test_nsd_matches_reference_and_backends_agree(masks, spacing, tolerance):
    a, b = masks
    expected = reference.nsd(a[0, 0], b[0, 0], spacing, tolerance)
    for backend in ("native", "torch"):
        assert _same(_nsd(a, b, spacing, tolerance, backend), expected, rel_tol=0)


@settings(max_examples=200, deadline=None)
@given(mask_pairs(), spacings)
def test_assd_is_symmetric_and_ordered(masks, spacing):
    a, b = masks
    if not (a.any() and b.any()):
        return
    values = {c: _assd(a, b, spacing, c) for c in ("max", "mean", "pooled")}
    for combine, value in values.items():
        assert math.isclose(_assd(b, a, spacing, combine), value, rel_tol=1e-12)
    hi = values["max"]
    lo = 2 * values["mean"] - hi  # the smaller of the two directed means
    assert lo - 1e-12 * hi <= values["pooled"] <= hi * (1 + 1e-12)  # a weighted average of them
    assert _assd(a, a, spacing) == 0


@settings(max_examples=200, deadline=None)
@given(mask_pairs(), spacings, st.floats(0.0, 5.0), st.floats(0.0, 5.0))
def test_nsd_bounds_symmetry_and_monotonicity(masks, spacing, t1, t2):
    a, b = masks
    if not (a.any() and b.any()):
        return
    lo, hi = sorted((t1, t2))
    n_lo, n_hi = _nsd(a, b, spacing, lo), _nsd(a, b, spacing, hi)
    assert 0 <= n_lo <= n_hi <= 1
    assert n_lo == _nsd(b, a, spacing, lo)
    assert _nsd(a, a, spacing, 0.0) == 1
    assert _nsd(a, b, spacing, _hd(a, b, spacing)) == 1  # every boundary voxel is within HD
