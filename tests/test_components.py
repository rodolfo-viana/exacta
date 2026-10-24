import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays
from scipy import ndimage

from exacta import _triton_label, components

GPU = torch.cuda.is_available() and _triton_label.available()
needs_gpu = pytest.mark.skipif(not GPU, reason="needs CUDA and Triton")


def _scipy_label(mask, connectivity):
    return ndimage.label(mask, ndimage.generate_binary_structure(mask.ndim, connectivity))


@pytest.mark.parametrize("shape", [(200,), (37, 41), (17, 23, 29), (1, 9, 1)])
@pytest.mark.parametrize("density", [0.05, 0.3, 0.6])
def test_label_matches_scipy(shape, density):
    mask = np.random.default_rng(0).random(shape) < density
    for connectivity in range(1, len(shape) + 1):
        expected, n = _scipy_label(mask, connectivity)
        got, count = components.label(mask, connectivity=connectivity)
        assert count == n
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, expected)


@settings(max_examples=300, deadline=None)
@given(arrays(np.bool_, array_shapes(min_dims=1, max_dims=3, min_side=0, max_side=9)), st.data())
def test_label_matches_scipy_on_random_masks(mask, data):
    connectivity = data.draw(st.integers(1, mask.ndim))
    expected, n = _scipy_label(mask, connectivity)
    got, count = components.label(mask, connectivity=connectivity)
    assert count == n
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("threads", [1, 3, 32])
@pytest.mark.parametrize("density", [0.25, 0.5, 0.75])
def test_label_matches_scipy_on_large_volumes_with_any_thread_count(threads, density):
    # The rows are labelled in blocks, one per thread, then joined; near the percolation density
    # most components cross several blocks.
    rng = np.random.default_rng(10)
    masks = [rng.random((48, 40, 36)) < density, rng.random((300, 170)) < density]
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)  # the kernels share PyTorch's OpenMP runtime
    try:
        for mask in masks:
            for connectivity in range(1, mask.ndim + 1):
                expected, n = _scipy_label(mask, connectivity)
                got, count = components.label(mask, connectivity=connectivity)
                assert count == n
                np.testing.assert_array_equal(got, expected)
    finally:
        torch.set_num_threads(previous)


def test_numbering_follows_first_element_in_c_order():
    mask = np.array(
        [
            [0, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    labels, count = components.label(mask, connectivity=1)
    # the right-hand component starts at (0, 3), before the left one at (1, 0)
    np.testing.assert_array_equal(labels, [[0, 0, 0, 1], [2, 0, 0, 1], [2, 0, 3, 0]])
    assert count == 3
    labels, count = components.label(mask, connectivity=2)  # (2, 2) now touches (1, 3)
    np.testing.assert_array_equal(labels, [[0, 0, 0, 1], [2, 0, 0, 1], [2, 0, 1, 0]])
    assert count == 2


def test_u_shape_merges_into_the_first_label():
    # Two arms that start as separate provisional labels and meet at the bottom.
    mask = np.zeros((5, 5), bool)
    mask[:, 0] = mask[:, 4] = mask[4] = True
    labels, count = components.label(mask, connectivity=1)
    assert count == 1
    assert set(np.unique(labels)) == {0, 1}


def test_empty_full_and_zero_size():
    labels, count = components.label(np.zeros((4, 5, 6), bool), connectivity=1)
    assert count == 0 and not labels.any()
    labels, count = components.label(np.ones((4, 5, 6), bool), connectivity=1)
    assert count == 1 and (labels == 1).all()
    labels, count = components.label(np.zeros((0, 5), bool), connectivity=2)
    assert count == 0 and labels.shape == (0, 5)


def test_nonzero_is_foreground():
    mask = np.array([0.0, 0.5, -2.0, 0.0, 3.0])
    labels, count = components.label(mask, connectivity=1)
    np.testing.assert_array_equal(labels, [0, 1, 1, 0, 2])
    assert count == 2


def test_keeps_array_library_and_device(device):
    mask = torch.from_numpy(np.random.default_rng(1).random((12, 13, 14)) < 0.4).to(device)
    labels, count = components.label(mask, connectivity=3)
    assert isinstance(labels, torch.Tensor) and labels.device.type == device.type
    expected, n = _scipy_label(mask.cpu().numpy(), 3)
    assert count == n
    np.testing.assert_array_equal(labels.cpu().numpy(), expected)
    assert components.largest(mask, connectivity=1).device.type == device.type
    assert components.remove_small(mask, 3, connectivity=1).device.type == device.type


@pytest.mark.parametrize("connectivity", [1, 2, 3])
def test_largest_matches_reference(connectivity):
    mask = np.random.default_rng(2).random((20, 22, 24)) < 0.3
    labels, n = _scipy_label(mask, connectivity)
    sizes = np.bincount(labels.ravel())
    expected = labels == 1 + np.argmax(sizes[1:])
    got = components.largest(mask, connectivity=connectivity)
    assert got.dtype == np.bool_
    np.testing.assert_array_equal(got, expected)


def test_largest_ties_go_to_the_first_component():
    mask = np.array([1, 1, 0, 1, 1, 0, 1])
    np.testing.assert_array_equal(components.largest(mask, connectivity=1), [1, 1, 0, 0, 0, 0, 0])
    assert not components.largest(np.zeros((3, 3)), connectivity=2).any()


@pytest.mark.parametrize("min_size", [0, 1, 2, 5, 50, 10**6])
def test_remove_small_matches_reference(min_size):
    mask = np.random.default_rng(3).random((20, 22, 24)) < 0.25
    labels, n = _scipy_label(mask, 1)
    sizes = np.bincount(labels.ravel())
    expected = (sizes >= min_size)[labels] & mask
    np.testing.assert_array_equal(components.remove_small(mask, min_size, connectivity=1), expected)


@pytest.mark.parametrize("threads", [1, 5, 32])
def test_native_label_map_kernels_match_numpy(threads):
    from exacta import _native

    rng = np.random.default_rng(11)
    # Runs of equal labels, as labelling gives, and isolated values.
    a = np.repeat(rng.integers(0, 40, 3000), rng.integers(1, 9, 3000)).astype(np.int32)
    b = np.repeat(rng.integers(0, 25, 4000), rng.integers(1, 7, 4000)).astype(np.int32)[: a.size]
    a = a[: b.size]
    a[::7] = rng.integers(0, 40, a[::7].size)
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        ta, tb = torch.from_numpy(a), torch.from_numpy(b)
        np.testing.assert_array_equal(_native.histogram(ta, 40).numpy(), np.bincount(a, minlength=40))
        select = rng.random(40) < 0.5
        got = _native.select_labels(ta, torch.from_numpy(select))
        np.testing.assert_array_equal(got.numpy(), select[a])
        size_a, size_b, pairs = _native.label_overlap(ta, tb, 39, 24)
        np.testing.assert_array_equal(size_a.numpy(), np.bincount(a, minlength=40))
        np.testing.assert_array_equal(size_b.numpy(), np.bincount(b, minlength=25))
        both = (a > 0) & (b > 0)
        keys, counts = np.unique(a[both].astype(np.int64) * 25 + b[both], return_counts=True)
        expected = np.stack([keys // 25, keys % 25, counts], axis=1)
        np.testing.assert_array_equal(pairs.numpy(), expected)
    finally:
        torch.set_num_threads(previous)


def test_native_label_map_kernels_report_labels_out_of_range():
    from exacta import _C, _native

    a = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    assert _native.histogram(a, 3) is None
    assert _native.histogram(-a, 4) is None
    assert _native.select_labels(a, torch.ones(3, dtype=torch.bool)) is None
    assert _native.label_overlap(a, a, 2, 3) is None
    assert _native.label_overlap(a, a, 3, 2) is None
    assert _native.label_overlap(a, a, 3, 3) is not None
    # With less room than pairs, the count comes back and the rows are left alone.
    sizes = [torch.empty(4, dtype=torch.int64) for _ in range(2)]
    pairs = torch.full((2, 3), -7, dtype=torch.int64)
    assert _C.label_overlap(a, a, *sizes, pairs) == 3
    assert (pairs == -7).all()


def test_rejects_bad_arguments():
    with pytest.raises(ValueError, match="connectivity must be"):
        components.label(np.ones((3, 3)), connectivity=3)
    with pytest.raises(ValueError, match="connectivity must be"):
        components.label(np.ones((3, 3)), connectivity=0)
    with pytest.raises(ValueError, match="1 to 3 dimensions"):
        components.label(np.ones((2, 2, 2, 2)), connectivity=1)
    with pytest.raises(ValueError, match="min_size"):
        components.remove_small(np.ones(3), -1, connectivity=1)
    with pytest.raises(TypeError):
        components.label(np.ones(3))  # connectivity is required


# --- GPU (Triton) -------------------------------------------------------------------------------


@needs_gpu
@pytest.mark.parametrize("shape", [(200,), (37, 41), (17, 23, 29), (1, 9, 1), (40, 300, 7)])
@pytest.mark.parametrize("density", [0.0, 0.05, 0.3, 0.6, 1.0])
def test_gpu_label_matches_scipy(shape, density):
    mask = np.random.default_rng(4).random(shape) < density
    for connectivity in range(1, len(shape) + 1):
        expected, n = _scipy_label(mask, connectivity)
        got, count = components.label(torch.from_numpy(mask).cuda(), connectivity=connectivity)
        assert got.device.type == "cuda" and got.dtype == torch.int32
        assert count == n
        np.testing.assert_array_equal(got.cpu().numpy(), expected)


@needs_gpu
@settings(max_examples=200, deadline=None)
@given(arrays(np.bool_, array_shapes(min_dims=1, max_dims=3, min_side=0, max_side=9)), st.data())
def test_gpu_label_matches_scipy_on_random_masks(mask, data):
    connectivity = data.draw(st.integers(1, mask.ndim))
    expected, n = _scipy_label(mask, connectivity)
    got, count = components.label(torch.from_numpy(mask).cuda(), connectivity=connectivity, backend="triton")
    assert count == n
    np.testing.assert_array_equal(got.cpu().numpy(), expected)


@needs_gpu
def test_gpu_matches_native_on_a_large_noisy_volume():
    mask = torch.from_numpy(np.random.default_rng(5).random((96, 128, 160)) < 0.3)
    for connectivity in (1, 3):
        native, n = components.label(mask, connectivity=connectivity, backend="native")
        gpu, m = components.label(mask.cuda(), connectivity=connectivity, backend="triton")
        assert n == m
        assert torch.equal(gpu.cpu(), native)


@needs_gpu
def test_gpu_largest_and_remove_small_match_cpu():
    mask = torch.from_numpy(np.random.default_rng(6).random((20, 22, 24)) < 0.3)
    gpu = mask.cuda()
    assert torch.equal(components.largest(gpu, connectivity=2).cpu(), components.largest(mask, connectivity=2))
    small_gpu = components.remove_small(gpu, 4, connectivity=1).cpu()
    assert torch.equal(small_gpu, components.remove_small(mask, 4, connectivity=1))


def test_backend_validation():
    with pytest.raises(ValueError, match="backend must be"):
        components.label(np.ones(3), connectivity=1, backend="cuda")
    with pytest.raises(NotImplementedError, match="runs on GPU"):
        components.label(np.ones(3), connectivity=1, backend="triton")
