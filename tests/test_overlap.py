import numpy as np
import pytest
import torch

import exacta.functional as EF
import reference
from _data import pair


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
