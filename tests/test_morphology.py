"""Native binary morphology: dilation and mask bounding boxes."""

import numpy as np
import pytest
import torch
from scipy import ndimage

from exacta import _native
from exacta._inputs import _extent
from exacta.lesions import _dilate

SHAPES = [(40,), (31, 47), (17, 23, 29), (1, 9, 1), (12, 1, 14)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("steps", [0, 1, 3])
def test_dilation_matches_scipy(device, shape, steps):
    mask = np.random.default_rng(0).random(shape) < 0.03
    for connectivity in range(1, len(shape) + 1):
        structure = ndimage.generate_binary_structure(len(shape), connectivity)
        expected = ndimage.binary_dilation(mask, structure, iterations=steps) if steps else mask
        got = _dilate(torch.from_numpy(mask).to(device), steps, connectivity)
        assert got.dtype == torch.bool and got.device.type == device.type
        np.testing.assert_array_equal(got.cpu().numpy(), expected)


def test_dilation_reaches_the_image_border_without_wrapping():
    mask = np.zeros((5, 6), bool)
    mask[0, 0] = True
    got = _native.dilate(torch.from_numpy(mask), 2, 2).numpy()
    np.testing.assert_array_equal(got, ndimage.binary_dilation(mask, np.ones((3, 3), bool), iterations=2))
    assert not got[-1].any() and not got[:, -1].any()


@pytest.mark.parametrize("shape", [(5, 31, 47), (3, 17, 23, 29), (2, 0, 4), (4, 9), (1, 1, 1, 1)])
def test_mask_extents_match_definition(shape):
    masks = torch.from_numpy(np.random.default_rng(1).random(shape) < 0.02)
    if masks.numel():
        masks[0] = False  # one empty mask
    present, lo, hi = _native.mask_extents(masks)
    expected_present, expected_lo, expected_hi = _extent(masks, shape[0], True, native=False)
    assert torch.equal(present, expected_present)
    for c in torch.nonzero(present).flatten().tolist():
        idx = torch.nonzero(masks[c])
        assert torch.equal(lo[c], idx.amin(0)) and torch.equal(hi[c], idx.amax(0))


@pytest.mark.parametrize(
    "shape", [(1,), (5,), (7, 1), (1, 7), (1, 1, 1), (9, 10, 11), (0, 3, 3), (3, 1, 4), (40, 33)]
)
@pytest.mark.parametrize("density", [0.0, 0.5, 0.9, 1.0])
def test_native_boundary_matches_torch(shape, density):
    from exacta.functional.surface import _boundary_torch

    big = torch.from_numpy(np.random.default_rng(5).random(tuple(s + 4 for s in shape)) < density)
    crop = big[tuple(slice(1, 1 + s) for s in shape)]  # a strided view, read in place
    expected = _boundary_torch(crop, crop.ndim)
    for mask in (crop, crop.contiguous(), crop.to(torch.uint8), crop.to(torch.int64) * 3):
        got = _native.boundary(mask)
        assert got.dtype == torch.bool and got.is_contiguous()
        assert torch.equal(got, expected)


def test_native_masked_sqrt_is_correctly_rounded():
    rng = np.random.default_rng(6)
    values = torch.from_numpy(rng.random((30, 40)) * 50)
    mask = torch.from_numpy(rng.random((30, 40)) < 0.3)
    assert torch.equal(_native.masked_sqrt(values, mask), torch.from_numpy(np.sqrt(values[mask].numpy())))
    assert _native.masked_sqrt(values, torch.zeros_like(mask)).numel() == 0
