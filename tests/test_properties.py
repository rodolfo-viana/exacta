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
