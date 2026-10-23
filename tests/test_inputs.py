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
        EF.dice(torch.zeros(1, 4, 4, 4, dtype=torch.long), torch.zeros(1, 4, 4, 5, dtype=torch.long), num_classes=2)


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
