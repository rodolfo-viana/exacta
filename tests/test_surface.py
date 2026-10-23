import math
import warnings

import numpy as np
import pytest
import torch

import exacta.functional as EF
import reference
from _data import blobs, pair
from exacta.functional.surface import _boundary, _sqrt


@pytest.mark.parametrize("shape", [(30, 40), (16, 20, 24)])
def test_boundary_matches_scipy(shape):
    labels = blobs(shape, 2, seed=0)
    mask = torch.from_numpy(labels == 1)
    got = _boundary(mask[None, None], len(shape))[0, 0].numpy()
    np.testing.assert_array_equal(got, reference.boundary(labels == 1))


def _backends(device):
    return ["torch"] if device.type == "cuda" else ["torch", "native"]


@pytest.mark.parametrize(
    ("shape", "spacing"), [((40, 36), (0.5, 1.25)), ((20, 24, 28), (3.0, 0.8, 0.8))]
)
@pytest.mark.parametrize("percentile", [95.0, 100.0, 50.0])
def test_matches_reference(device, shape, spacing, percentile):
    pred, target = pair(shape, 4, seed=5)
    spacings = [spacing] * pred.shape[0]
    expected = reference.per_case(
        reference.hausdorff, pred, target, 4, spacings=spacings, percentile=percentile
    )
    for backend in _backends(device):
        got = EF.hausdorff_distance(
            torch.from_numpy(pred).to(device),
            torch.from_numpy(target).to(device),
            spacing=spacing,
            percentile=percentile,
            num_classes=4,
            backend=backend,
        )
        assert got.device.type == device.type
        np.testing.assert_allclose(got.cpu().numpy(), expected, rtol=1e-10, err_msg=backend)


def test_per_case_spacing(device):
    pred, target = pair((16, 18, 20), 3, seed=6, batch=3)
    spacings = [(1.0, 1.0, 1.0), (2.5, 0.7, 0.7), (0.5, 3.0, 1.0)]
    expected = reference.per_case(reference.hausdorff, pred, target, 3, spacings=spacings, percentile=95)
    for backend in _backends(device):
        got = EF.hd95(
            torch.from_numpy(pred).to(device),
            torch.from_numpy(target).to(device),
            spacing=torch.tensor(spacings, dtype=torch.float64),
            num_classes=3,
            backend=backend,
        )
        np.testing.assert_allclose(got.cpu().numpy(), expected, rtol=1e-10, err_msg=backend)


def test_empty_handling(device):
    empty = torch.zeros(1, 8, 8, 8, dtype=torch.long, device=device)
    filled = empty.clone()
    filled[0, 2:5, 2:5, 2:5] = 1
    assert EF.hd95(empty, empty, spacing=1, num_classes=2)[0, 1].isnan()
    assert EF.hd95(empty, empty, spacing=1, num_classes=2, both_empty=0.0)[0, 1] == 0
    assert EF.hd95(empty, filled, spacing=1, num_classes=2)[0, 1] == math.inf
    assert EF.hd95(filled, empty, spacing=1, num_classes=2, one_empty=99.0)[0, 1] == 99


def test_sqrt_is_correctly_rounded(device):
    x = torch.arange(200_001, dtype=torch.float64)
    expected = torch.tensor([math.sqrt(v) for v in x.tolist()], dtype=torch.float64)
    assert torch.equal(_sqrt(x.to(device)).cpu(), expected)


def test_spacing_is_required():
    x = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="spacing is required"):
        EF.hd95(x, x, spacing=None, num_classes=2)


def test_native_backend_rejects_cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    x = torch.zeros(1, 4, 4, 4, dtype=torch.long, device="cuda")
    with pytest.raises(NotImplementedError):
        EF.hd95(x, x, spacing=1, num_classes=2, backend="native")


@pytest.mark.parametrize("percentile", [95.0, 100.0])
def test_matches_monai(percentile):
    monai = pytest.importorskip("monai.metrics")
    pred, target = pair((20, 24, 28), 4, seed=7)
    spacing = (2.0, 0.9, 0.9)
    one_hot = lambda x: torch.nn.functional.one_hot(torch.from_numpy(x), 4).movedim(-1, 1)  # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = monai.compute_hausdorff_distance(
            one_hot(pred),
            one_hot(target),
            include_background=True,
            percentile=percentile,
            spacing=spacing,
        )
    got = EF.hausdorff_distance(pred, target, spacing=spacing, percentile=percentile, num_classes=4)
    np.testing.assert_allclose(got, expected.double().numpy(), rtol=1e-5)
