import math

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy import ndimage

from exacta import _C, _native
from exacta.functional.surface import _edt_sq_torch

CASES = [
    ((50,), (0.7,)),
    ((31, 47), (1.0, 2.5)),
    ((17, 23, 29), (3.0, 0.8, 0.8)),
    ((5, 6, 7, 8), (1.0, 2.0, 0.5, 1.5)),
]


def _features(shape, density):
    features = np.random.default_rng(0).random(shape) < density
    features.flat[0] = True
    return features


@pytest.mark.parametrize(("shape", "spacing"), CASES)
@pytest.mark.parametrize("density", [0.001, 0.05, 0.5])
def test_native_matches_scipy(shape, spacing, density):
    features = _features(shape, density)
    expected = ndimage.distance_transform_edt(~features, sampling=spacing) ** 2
    got = _native.edt_sq(torch.from_numpy(features), spacing).numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize(("shape", "spacing"), CASES)
@pytest.mark.parametrize("density", [0.001, 0.05, 0.5])
def test_torch_matches_scipy(device, shape, spacing, density):
    features = _features(shape, density)
    expected = ndimage.distance_transform_edt(~features, sampling=spacing) ** 2
    got = _edt_sq_torch(torch.from_numpy(features).to(device), spacing).cpu().numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-9)


def _brute_force(features: np.ndarray, spacing) -> np.ndarray:
    """min over feature voxels p of sum_i ((p_i - q_i) * h_i)^2, rounded and summed in axis
    order: the expression SciPy's distance_transform_edt evaluates for its chosen feature."""
    pts = np.argwhere(features)
    grid = np.indices(features.shape).reshape(features.ndim, -1).T
    total = 0.0
    for i, h in enumerate(spacing):
        total = total + ((pts[None, :, i] - grid[:, None, i]).astype(np.float64) * h) ** 2
    return total.min(1).reshape(features.shape)


@settings(max_examples=300, deadline=None)
@given(
    arrays(np.bool_, st.tuples(st.integers(1, 6), st.integers(1, 7), st.integers(1, 8))),
    st.tuples(*[st.floats(0.1, 5.0)] * 3),
)
def test_is_the_exact_floating_point_minimum(features, spacing):
    if not features.any():
        return
    expected = _brute_force(features, spacing)
    for edt_sq in (_native.edt_sq, _edt_sq_torch):
        np.testing.assert_array_equal(edt_sq(torch.from_numpy(features), spacing).numpy(), expected)


# Non-dyadic spacings, where mathematically tied features round to different values. The
# isotropic 1.1 grid has many points where three parabolas meet, which the plain lower-envelope
# algorithm resolves to a value an ulp too high.
@pytest.mark.parametrize("spacing", [(1.1, 1.1, 1.1), (0.7, 0.9, 1.3), (2.0, 0.6, 0.6), (3.0, 0.8, 0.8)])
@pytest.mark.parametrize("density", [0.002, 0.02, 0.2])
def test_backends_agree_bit_for_bit(device, spacing, density):
    features = torch.from_numpy(_features((30, 36, 42), density))
    native = _native.edt_sq(features, spacing)
    assert torch.equal(_edt_sq_torch(features.to(device), spacing).cpu(), native)


@pytest.mark.parametrize("edt_sq", [_native.edt_sq, _edt_sq_torch])
def test_no_features_is_inf(edt_sq):
    got = edt_sq(torch.zeros(4, 5, 6, dtype=torch.bool), (1.0, 1.0, 1.0))
    assert torch.isinf(got).all()


@pytest.mark.parametrize("edt_sq", [_native.edt_sq, _edt_sq_torch])
def test_all_features_is_zero(edt_sq):
    got = edt_sq(torch.ones(4, 5, dtype=torch.bool), (1.0, 2.0))
    assert (got == 0).all()


def test_single_thread_matches_default():
    features = torch.rand(20, 30, 40) < 0.01
    single = torch.empty(features.shape, dtype=torch.float64)
    _C.edt_sq(features.to(torch.uint8), single, [1.0, 1.0, 1.0], num_threads=1)
    torch.testing.assert_close(single, _native.edt_sq(features, (1.0, 1.0, 1.0)), rtol=0, atol=0)


@pytest.mark.parametrize("spacing", [[1.0, 0.0], [1.0, -2.0], [1.0, math.inf], [1.0]])
def test_rejects_bad_spacing(spacing):
    features = torch.zeros(3, 4, dtype=torch.uint8)
    with pytest.raises(ValueError):
        _C.edt_sq(features, torch.empty(3, 4, dtype=torch.float64), spacing)


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        _C.edt_sq(torch.zeros(3, 4, dtype=torch.uint8), torch.empty(4, 3, dtype=torch.float64), [1.0, 1.0])
