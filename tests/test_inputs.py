import numpy as np
import pytest
import torch

import exacta.functional as EF
from exacta._inputs import as_spacing


@pytest.mark.parametrize("library", ["numpy", "cupy", "jax"])
def test_array_library_round_trip(library):
    from exacta import components

    if library == "numpy":
        xp = np
    elif library == "cupy":
        if not torch.cuda.is_available():
            pytest.skip("CuPy round trip needs CUDA")
        xp = pytest.importorskip("cupy")
    else:
        xp = pytest.importorskip("jax.numpy")
    mask = xp.asarray([[[1, 0], [0, 0]]], dtype=xp.int32)
    score = EF.dice(mask, mask, num_classes=2)
    assert isinstance(score, type(mask))
    np.testing.assert_array_equal(np.asarray(score.get() if library == "cupy" else score), [[1, 1]])
    labels, count = components.label(mask[0], connectivity=1)
    kept = components.largest(mask[0], connectivity=1)
    assert isinstance(labels, type(mask)) and isinstance(kept, type(mask))
    assert count == 1
    if library == "cupy":
        assert score.device.id == mask.device.id == labels.device.id == kept.device.id


@pytest.mark.parametrize("surface", ["surfels", "mesh"])
@pytest.mark.parametrize("spacing", [1e-100, 1e100])
@pytest.mark.parametrize("empty", [False, True])
def test_area_surfaces_reject_unsupported_spacing(device, surface, spacing, empty):
    from exacta import lesions

    pred = torch.zeros((1, 1, 3, 3, 3), dtype=torch.bool, device=device)
    target = pred.clone()
    if not empty:
        pred[0, 0, 1, 1, 1] = True
        target[0, 0, 2, 1, 1] = True
    for fn, extra in [(EF.hd95, {}), (EF.assd, {}), (EF.nsd, {"tolerance": 1.0})]:
        with pytest.raises(ValueError, match="spacing values must lie in"):
            fn(pred, target, spacing=spacing, surface=surface, input_format="one-hot", **extra)
    with pytest.raises(ValueError, match="spacing values must lie in"):
        lesions.evaluate(pred, target, matching="overlap", connectivity=1, spacing=spacing,
                         surface=surface, input_format="one-hot")


@pytest.mark.parametrize("surface", ["voxels", "surfels", "mesh"])
@pytest.mark.parametrize("scale", [1e-50, 1e50])
def test_surface_metrics_scale_at_supported_limits(device, surface, scale):
    pred = torch.zeros((1, 1, 3, 3, 3), dtype=torch.bool, device=device)
    target = pred.clone()
    pred[0, 0, 1, 1, 1] = True
    target[0, 0, 2, 1, 1] = True
    kw = dict(input_format="one-hot", surface=surface)
    for fn in (EF.hd95, EF.assd):
        expected = fn(pred, target, spacing=1.0, **kw)
        actual = fn(pred, target, spacing=scale, **kw) / scale
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        EF.nsd(pred, target, spacing=scale, tolerance=1.1 * scale, **kw),
        EF.nsd(pred, target, spacing=1.0, tolerance=1.1, **kw),
    )


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


@pytest.mark.parametrize("library", ["numpy", "cupy"])
def test_negative_strides_on_contiguous_singleton_axes(library):
    from exacta._inputs import to_torch

    xp = np if library == "numpy" else pytest.importorskip("cupy")
    if library == "cupy" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    x = xp.arange(12).reshape(1, 3, 4)[::-1]
    assert x.flags.c_contiguous and x.strides[0] < 0
    got = to_torch(x)
    torch.testing.assert_close(got.cpu(), torch.arange(12).reshape(1, 3, 4))


def test_negative_strides_without_dlpack():
    from exacta._inputs import to_torch

    class LegacyArray(np.ndarray):
        @property
        def __dlpack__(self):
            raise AttributeError("NumPy 1.22 does not expose DLPack")

    x = np.arange(12).reshape(1, 3, 4).view(LegacyArray)[::-1]
    assert not hasattr(x, "__dlpack__") and x.flags.c_contiguous
    torch.testing.assert_close(to_torch(x), torch.arange(12).reshape(1, 3, 4))


@pytest.mark.parametrize("dtype,value", [(torch.float64, 1 + 2**-40), (torch.int64, 2**40 + 1)])
@pytest.mark.parametrize("x64", [False, True])
def test_jax_round_trip_preserves_precision_and_configuration(dtype, value, x64):
    jax = pytest.importorskip("jax")
    from exacta._inputs import like_input, to_torch

    enable_x64 = getattr(jax, "enable_x64", None)
    if enable_x64 is None:
        from jax.experimental import enable_x64
    before = jax.config.x64_enabled
    with enable_x64(x64):
        original = jax.numpy.zeros((1,), dtype=jax.numpy.int32)
        tensor = torch.tensor([value], dtype=dtype)
        result = like_input(tensor, original)
        torch.testing.assert_close(to_torch(result), tensor, rtol=0, atol=0)
        assert jax.config.x64_enabled == x64
    assert jax.config.x64_enabled == before


@pytest.mark.parametrize("spacing", [0, -1, float("nan"), float("inf"), [1, 0]])
def test_empty_batch_still_validates_shared_spacing(spacing):
    with pytest.raises(ValueError, match="spacing"):
        as_spacing(spacing, batch=0, ndim=2)


@pytest.mark.parametrize("library", ["numpy", "cupy", "jax"])
def test_physical_area_array_libraries_preserve_float64(library):
    from exacta._inputs import to_torch

    if library == "cupy" and not torch.cuda.is_available():
        pytest.skip("CuPy round trip needs CUDA")
    xp = np if library == "numpy" else pytest.importorskip("jax.numpy" if library == "jax" else library)
    p = xp.asarray([[[0, 1], [1, 1]]], dtype=xp.int32)
    scale = 1 + 2**-40
    result = EF.evaluate(p, p, num_classes=2, spacing=(scale, 1),
                         metrics=["dice", "predicted_area", "target_area", "area_error"])
    expected = torch.tensor([[scale, 3*scale]], dtype=torch.float64)
    for name in ("predicted_area", "target_area"):
        assert isinstance(result[name], type(p))
        torch.testing.assert_close(to_torch(result[name]).cpu(), expected, rtol=0, atol=0)
    assert to_torch(result["area_error"]).eq(0).all()
