import math

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import exacta.functional as EF
from _data import pair
from exacta.metrics import HD95, Dice, IoU


def _batches(device, n=3):
    return [
        tuple(torch.from_numpy(a).to(device) for a in pair((16, 18, 20), 3, seed=10 * i, batch=2))
        for i in range(n)
    ]


@pytest.mark.parametrize(("metric_cls", "fn"), [(Dice, EF.dice), (IoU, EF.iou)])
def test_accumulates_like_functional(device, metric_cls, fn):
    batches = _batches(device)
    metric = metric_cls(num_classes=3, include_background=False)
    for pred, target in batches:
        metric.update(pred, target)
    result = metric.compute()

    expected = torch.cat([fn(p, t, num_classes=3, include_background=False) for p, t in batches])
    torch.testing.assert_close(result.per_case, expected)
    torch.testing.assert_close(result.per_class, expected.nanmean(0))
    torch.testing.assert_close(result.mean, expected.nanmean(0).nanmean())


def test_nan_cases_are_skipped():
    metric = Dice(num_classes=2)
    empty = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    filled = empty.clone()
    filled[0, :2] = 1
    metric.update(empty, empty)  # class 1 absent from both: NaN
    metric.update(filled, filled)
    assert metric.compute().per_class[1] == 1


def test_hd95_spacing_per_update(device):
    batches = _batches(device, n=2)
    spacings = [(1.0, 1.0, 1.0), (2.0, 0.5, 0.5)]
    metric = HD95(num_classes=3)
    for (pred, target), spacing in zip(batches, spacings):
        metric.update(pred, target, spacing=spacing)
    expected = torch.cat(
        [EF.hd95(p, t, spacing=s, num_classes=3) for (p, t), s in zip(batches, spacings)]
    )
    torch.testing.assert_close(metric.compute().per_case, expected)


def test_hd95_spacing_from_constructor():
    pred, target = _batches("cpu", n=1)[0]
    metric = HD95(spacing=(1.5, 1.0, 1.0), num_classes=3)
    metric.update(pred, target)
    torch.testing.assert_close(
        metric.compute().per_case, EF.hd95(pred, target, spacing=(1.5, 1.0, 1.0), num_classes=3)
    )


def test_compute_before_update_and_reset():
    metric = Dice(num_classes=2)
    with pytest.raises(RuntimeError, match="before update"):
        metric.compute()
    x = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    metric.update(x, x)
    metric.reset()
    with pytest.raises(RuntimeError):
        metric.compute()


def _ddp_worker(rank, world, init_file, batches, out):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    metric = Dice(num_classes=3)
    for pred, target in batches[rank::world]:  # uneven split: rank 0 gets 2 batches, rank 1 gets 1
        metric.update(pred, target)
    result = metric.compute()
    if rank == 0:
        out.put(result.per_case.numpy())
    dist.destroy_process_group()


def test_ddp_gathers_all_ranks(tmp_path):
    if not dist.is_available():
        pytest.skip("torch.distributed unavailable")
    batches = _batches("cpu", n=3)
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(
        _ddp_worker, args=(2, tmp_path / "init", batches, out), nprocs=2, start_method="spawn"
    )
    gathered = out.get(timeout=60)
    order = [0, 2, 1]  # rank 0 holds batches 0 and 2, rank 1 holds batch 1
    expected = torch.cat([EF.dice(*batches[i], num_classes=3) for i in order]).numpy()
    np.testing.assert_allclose(gathered, expected)
    assert not math.isnan(gathered[0, 1])
