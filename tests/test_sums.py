import math

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import exacta.functional as EF
from exacta import _native, _triton_edt

GPU = torch.cuda.is_available() and _triton_edt.available()

finite = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e300, max_value=1e300)


@settings(max_examples=300, deadline=None)
@given(st.lists(finite, max_size=60), st.data())
def test_exact_sum_is_math_fsum_in_any_order(values, data):
    x = np.array(values, dtype=np.float64)
    w = np.array(data.draw(st.lists(st.floats(0, 4), min_size=x.size, max_size=x.size)), dtype=np.float64)
    select = np.array(data.draw(st.lists(st.booleans(), min_size=x.size, max_size=x.size)), dtype=bool)
    terms = (x * w)[select]
    try:
        expected = math.fsum(terms.tolist())
    except OverflowError:
        return
    got = _native.exact_sum(torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(select))
    assert got == expected
    order = np.random.default_rng(0).permutation(x.size)
    args = (torch.from_numpy(x[order]), torch.from_numpy(w[order]), torch.from_numpy(select[order]))
    assert _native.exact_sum(*args) == expected


def test_exact_sum_cancellation_and_special_values():
    assert _native.exact_sum(torch.tensor([1e16, 1.0, -1e16], dtype=torch.float64)) == 1.0
    assert _native.exact_sum(torch.tensor([0.1] * 10, dtype=torch.float64)) == 1.0
    assert _native.exact_sum(torch.zeros(0, dtype=torch.float64)) == 0.0
    assert _native.exact_sum(torch.tensor([1.0, math.inf], dtype=torch.float64)) == math.inf
    assert math.isnan(_native.exact_sum(torch.tensor([math.inf, -math.inf], dtype=torch.float64)))


def _spheres(h=0.25):
    axes = [(np.arange(int(26 / h)) + 0.5) * h - 13 for _ in range(3)]
    z, y, x = np.meshgrid(*axes, indexing="ij")
    a = z**2 + y**2 + x**2 <= 100.0
    b = (z - 1.3) ** 2 + y**2 + x**2 <= 100.0
    return torch.from_numpy(a)[None, None], torch.from_numpy(b)[None, None], (h, h, h)


@pytest.mark.parametrize("surface", ["voxels", "surfels", "mesh"])
def test_surface_means_do_not_depend_on_the_thread_count(surface):
    # Tens of thousands of elements: PyTorch's parallel sum would split them differently.
    pred, target, spacing = _spheres()
    kw = dict(spacing=spacing, surface=surface, input_format="one-hot")
    threads = torch.get_num_threads()
    try:
        results = set()
        for n in (1, 2, 3):
            torch.set_num_threads(n)
            results.add((EF.assd(pred, target, **kw).item(), EF.nsd(pred, target, tolerance=0.5, **kw).item()))
    finally:
        torch.set_num_threads(threads)
    assert len(results) == 1


@pytest.mark.skipif(not GPU, reason="needs CUDA and Triton")
@pytest.mark.parametrize("surface", ["voxels", "surfels"])
def test_surface_means_are_the_same_bits_on_gpu(surface):
    pred, target, spacing = _spheres()
    kw = dict(spacing=spacing, surface=surface, input_format="one-hot")
    for metric in (EF.assd, lambda p, t, **k: EF.nsd(p, t, tolerance=0.5, **k)):
        cpu = metric(pred, target, **kw).item()
        gpu = metric(pred.cuda(), target.cuda(), **kw).item()
        assert cpu == gpu
