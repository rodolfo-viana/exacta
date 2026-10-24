import numpy as np
import pytest
import torch

import exacta.functional as EF
from exacta._inputs import as_spacing


def test_index_rejects_float():
    x = torch.zeros(1, 4, 4, 4)
    with pytest.raises(TypeError, match="argmax"):
        EF.dice(x, x, num_classes=2)


def test_index_requires_num_classes():
    x = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="num_classes"):
        EF.dice(x, x)


def test_index_out_of_range():
    x = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    y = x.clone()
    y[0, 0, 0, 0] = 3
    with pytest.raises(ValueError, match=r"\[0, 3\]"):
        EF.dice(x, y, num_classes=3)


def test_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        a, b = torch.zeros(1, 4, 4, 4, dtype=torch.long), torch.zeros(1, 4, 4, 5, dtype=torch.long)
        EF.dice(a, b, num_classes=2)


def test_one_hot_rejects_probabilities():
    x = torch.full((1, 2, 4, 4, 4), 0.5)
    with pytest.raises(ValueError, match="binary"):
        EF.dice(x, x, input_format="one-hot")


def test_one_hot_accepts_float_zero_one():
    x = torch.zeros(1, 2, 4, 4, 4)
    x[:, 1, :2] = 1
    assert EF.dice(x, x, input_format="one-hot")[0, 1] == 1


def test_include_background_needs_two_classes():
    x = torch.zeros(1, 1, 4, 4, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="include_background"):
        EF.dice(x, x, input_format="one-hot", include_background=False)


def test_numpy_in_numpy_out():
    x = np.zeros((1, 4, 4, 4), dtype=np.int64)
    out = EF.dice(x, x, num_classes=2, both_empty=1.0)
    assert isinstance(out, np.ndarray) and out.dtype == np.float64
    np.testing.assert_array_equal(out, [[1.0, 1.0]])


def test_spacing_forms():
    assert as_spacing(2.0, batch=2, ndim=3) == [(2.0, 2.0, 2.0)] * 2
    assert as_spacing((1, 2, 3), batch=2, ndim=3) == [(1.0, 2.0, 3.0)] * 2
    per_case = [[1, 1, 1], [2, 2, 2]]
    assert as_spacing(np.array(per_case), batch=2, ndim=3) == [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]


@pytest.mark.parametrize("spacing", [None, (1, 1), [[1, 1, 1]], (1, 0, 1), (1, float("nan"), 1)])
def test_spacing_rejects(spacing):
    with pytest.raises(ValueError):
        as_spacing(spacing, batch=2, ndim=3)


EXTENT_SHAPES = [(50,), (31, 47), (17, 23, 29), (1, 9, 1), (0, 4)]


@pytest.mark.parametrize("shape", EXTENT_SHAPES)
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.int32, torch.int64])
def test_class_extents_native_matches_torch(shape, dtype):
    from exacta import _native
    from exacta._inputs import _extent

    rng = np.random.default_rng(0)
    labels = torch.from_numpy(rng.integers(0, 6, shape)).to(dtype)
    labels[labels == 5] = 0  # class 5 absent
    present, lo, hi = _native.class_extents(labels, 7)  # class 6 out of the data's range
    expected_present, expected_lo, expected_hi = _extent(labels, 7, one_hot=False, native=False)
    assert torch.equal(present, expected_present)
    assert not present[5] and not present[6]
    assert torch.equal(lo[present], expected_lo[present])
    assert torch.equal(hi[present], expected_hi[present])
    for c in torch.nonzero(present).flatten().tolist():  # against the definition
        idx = torch.nonzero(labels == c)
        assert torch.equal(lo[c], idx.amin(0)) and torch.equal(hi[c], idx.amax(0))


def test_metrics_on_zero_size_volumes(device):
    empty = torch.zeros(2, 0, 5, 6, dtype=torch.long, device=device)
    assert EF.hd95(empty, empty, spacing=1, num_classes=3).isnan().all()
    assert EF.nsd(empty, empty, spacing=1, tolerance=1, num_classes=3, surface="surfels").isnan().all()


@pytest.mark.parametrize("spacing", [1e-160, 1e160, (1.0, 1.0, 1e101)])
def test_spacing_outside_the_exact_range_is_rejected(spacing):
    x = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="spacing values must lie in"):
        EF.hd95(x, x, spacing=spacing, num_classes=2)


def test_negative_stride_arrays_are_accepted():
    from scipy import ndimage

    from exacta import components

    rng = np.random.default_rng(1)
    mask = rng.random((12, 13, 14)) < 0.4
    labels, count = components.label(mask[::-1], connectivity=1)
    expected, n = ndimage.label(mask[::-1])
    assert count == n
    np.testing.assert_array_equal(labels, expected)
    pred, target = rng.integers(0, 3, (2, 9, 10, 11)), rng.integers(0, 3, (2, 9, 10, 11))
    flipped, copied = (pred[:, ::-1], target[:, ::-1]), (pred[:, ::-1].copy(), target[:, ::-1].copy())
    np.testing.assert_array_equal(EF.dice(*flipped, num_classes=3), EF.dice(*copied, num_classes=3))
