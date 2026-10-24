import numpy as np
import pytest
import torch

import exacta.functional as EF
import reference
from _data import pair
from exacta import _triton_counts


@pytest.mark.parametrize("shape", [(32, 40), (24, 28, 20)])
@pytest.mark.parametrize(("fn", "ref"), [(EF.dice, reference.dice), (EF.iou, reference.iou)])
def test_matches_reference(device, shape, fn, ref):
    pred, target = pair(shape, 4, seed=1)
    expected = reference.per_case(ref, pred, target, 4)
    got = fn(torch.from_numpy(pred).to(device), torch.from_numpy(target).to(device), num_classes=4)
    assert got.device.type == device.type and got.dtype == torch.float64
    np.testing.assert_allclose(got.cpu().numpy(), expected, rtol=1e-12)


def test_one_hot_equals_index(device):
    pred, target = (torch.from_numpy(a).to(device) for a in pair((20, 20, 20), 3, seed=2))
    one_hot = lambda x: torch.nn.functional.one_hot(x, 3).movedim(-1, 1)  # noqa: E731
    torch.testing.assert_close(
        EF.dice(one_hot(pred), one_hot(target), input_format="one-hot"),
        EF.dice(pred, target, num_classes=3),
    )


def test_include_background_drops_channel_zero():
    pred, target = (torch.from_numpy(a) for a in pair((20, 20, 20), 3, seed=3))
    full = EF.dice(pred, target, num_classes=3)
    torch.testing.assert_close(EF.dice(pred, target, num_classes=3, include_background=False), full[:, 1:])


def test_empty_handling():
    empty = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    filled = empty.clone()
    filled[0, 2:5, 2:5, 2:5] = 1
    assert EF.dice(empty, empty, num_classes=2)[0, 1].isnan()
    assert EF.dice(empty, empty, num_classes=2, both_empty=1.0)[0, 1] == 1
    assert EF.dice(empty, filled, num_classes=2)[0, 1] == 0
    assert EF.iou(filled, empty, num_classes=2)[0, 1] == 0


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.int32, torch.int64])
def test_native_counts_match_torch_for_label_dtypes(dtype):
    # CPU counts come from native kernels; the GPU and fallback path from PyTorch.
    from exacta.functional.overlap import _counts, _native_counts
    from exacta._inputs import validate

    pred, target = (torch.from_numpy(a).to(dtype) for a in pair((20, 22, 24), 5, seed=7))
    seg = validate(pred, target, input_format="index", num_classes=5, include_background=False)
    for ours, theirs in zip(_native_counts(seg), _counts(seg)):
        assert torch.equal(ours, theirs)


@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.uint8, torch.int32, torch.int64, torch.float32, torch.float64]
)
def test_native_counts_match_torch_for_one_hot_dtypes(dtype):
    from exacta.functional.overlap import _counts, _native_counts
    from exacta._inputs import validate

    rng = np.random.default_rng(8)
    pred, target = (torch.from_numpy(rng.random((2, 3, 9, 10, 11)) < 0.3).to(dtype) for _ in range(2))
    kw = dict(input_format="one-hot", num_classes=None, include_background=True)
    raw, checked = validate(pred, target, check_values=False, **kw), validate(pred, target, **kw)
    for ours, theirs in zip(_native_counts(raw), _counts(checked)):
        assert torch.equal(ours, theirs)


def test_counting_kernels_keep_the_validation_errors(device):
    zeros = torch.zeros(1, 4, 4, dtype=torch.long, device=device)
    with pytest.raises(ValueError, match=r"found values in \[0, 7\]"):
        EF.dice(torch.full((1, 4, 4), 7, device=device), zeros, num_classes=4)
    with pytest.raises(ValueError, match=r"found values in \[-1, 0\]"):
        EF.dice(torch.full((1, 4, 4), -1, device=device), zeros, num_classes=4)
    with pytest.raises(ValueError, match="must be binary"):
        one_hot = torch.full((1, 2, 4, 4), 0.5, device=device)
        EF.dice(one_hot, torch.zeros(1, 2, 4, 4, device=device), input_format="one-hot")


GPU_COUNTS = pytest.mark.skipif(
    not (torch.cuda.is_available() and _triton_counts.available()), reason="needs CUDA and Triton"
)


@GPU_COUNTS
@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.int16, torch.int32, torch.int64])
@pytest.mark.parametrize("classes", [2, 5, 31])
def test_triton_counts_match_torch_for_label_dtypes(dtype, classes):
    from exacta.functional.overlap import _counts, _triton_counts_of
    from exacta._inputs import validate

    if dtype == torch.bool and classes > 2:
        pytest.skip("a boolean label map has two classes")
    pred, target = (torch.from_numpy(a).to(dtype).cuda() for a in pair((20, 22, 24), classes, seed=7))
    seg = validate(pred, target, input_format="index", num_classes=classes, include_background=False)
    for ours, theirs in zip(_triton_counts_of(seg), _counts(seg)):
        assert torch.equal(ours, theirs)


@GPU_COUNTS
@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.uint8, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64]
)
def test_triton_counts_match_torch_for_one_hot_dtypes(dtype):
    from exacta.functional.overlap import _counts, _triton_counts_of
    from exacta._inputs import validate

    rng = np.random.default_rng(8)
    pred, target = (torch.from_numpy(rng.random((2, 3, 9, 10, 11)) < 0.3).to(dtype).cuda() for _ in range(2))
    kw = dict(input_format="one-hot", num_classes=None, include_background=True)
    raw, checked = validate(pred, target, check_values=False, **kw), validate(pred, target, **kw)
    for ours, theirs in zip(_triton_counts_of(raw), _counts(checked)):
        assert torch.equal(ours, theirs)


@GPU_COUNTS
def test_gpu_dice_with_many_classes_matches_cpu():
    assert not _triton_counts.fits(40)  # counted with bincount
    pred, target = (torch.from_numpy(a) for a in pair((20, 22, 24), 40, seed=9))
    on_gpu = EF.dice(pred.cuda(), target.cuda(), num_classes=40).cpu()
    assert torch.equal(on_gpu, EF.dice(pred, target, num_classes=40))
