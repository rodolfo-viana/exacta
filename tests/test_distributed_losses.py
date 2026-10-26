"""DDP parameter gradients must match one concatenated global batch."""

from copy import deepcopy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from _process import destroy_process_group
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from exacta.losses import (
    DiceLoss, TverskyLoss, FocalTverskyLoss, GeneralizedDiceLoss, FocalLoss, DiceCELoss, GeneralizedDiceCELoss, DiceFocalLoss,
)

CLASSES = [DiceLoss, TverskyLoss, FocalTverskyLoss, GeneralizedDiceLoss, FocalLoss, DiceCELoss, GeneralizedDiceCELoss, DiceFocalLoss]


class Affine(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([0.7, 1.1, 1.5], dtype=torch.float64, device=device))
        self.bias = nn.Parameter(torch.tensor([-0.1, 0.3, 0.2], dtype=torch.float64, device=device))

    def forward(self, x):
        return x * self.scale[None, :, None, None, None] + self.bias[None, :, None, None, None]


def _worker(rank, rendezvous, device, transport):
    if transport == "nccl":
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    dist.init_process_group(transport, init_method=f"file://{rendezvous}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=180))
    try:
        torch.set_num_threads(1)
        model = DDP(Affine(device))
        # The CUDA/Gloo test intentionally uses one GPU for both workers. It tests
        # distributed fused math without claiming multi-GPU/NCCL throughput.
        configurations = [(cls, False) for cls in CLASSES] + [(DiceLoss, True), (DiceCELoss, True)]
        for cls, squared in configurations:
            for activation in ("softmax", "sigmoid"):
                for soft in (False, True):
                    for scenario in ("uneven", "empty_rank", "ignored_rank", "all_ignored", "all_empty", "unmasked"):
                        torch.manual_seed(352)
                        n = 0 if scenario == "all_empty" else 3
                        x = torch.randn(n, 3, 2, 3, 4, dtype=torch.float64, device=device)
                        labels = torch.randint(3, (n, 2, 3, 4), device=device)
                        y = torch.rand_like(x).softmax(1) if soft else labels
                        mask = torch.rand(labels.shape, device=device) > 0.2
                        cut = 0 if scenario in ("empty_rank", "all_empty") else 1
                        if scenario == "ignored_rank":
                            mask[:cut] = False
                        if scenario == "all_ignored":
                            mask[:] = False
                        if not soft:
                            y = torch.where(mask, y, -100)
                        if scenario == "unmasked":
                            mask = None
                            y = labels if not soft else y
                        selected = slice(0, cut) if rank == 0 else slice(cut, n)
                        options = dict(activation=activation, target_format="one-hot" if soft else "index")
                        if squared:
                            options["squared_pred"] = True
                        if not soft and scenario != "unmasked":
                            options["ignore_index"] = -100
                        if cls is not FocalLoss:
                            options.update(batch_dice=True, include_background=False, smooth=0.01)
                        if cls in (DiceCELoss, GeneralizedDiceCELoss):
                            options["class_weight"] = [0.01, 0.0, 0.02]
                        if cls in (FocalLoss, DiceFocalLoss):
                            options.update(gamma=0.5, alpha=0.3)
                        reference = deepcopy(model.module)
                        expected = cls(backend="torch", **options)(reference(x), y, mask=mask)
                        expected.backward()
                        model.zero_grad(set_to_none=True)
                        actual = cls(distributed=True, **options)(
                            model(x[selected]), y[selected], mask=None if mask is None else mask[selected],
                        )
                        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
                        actual.backward()
                        for p, gold in zip(model.module.parameters(), reference.parameters()):
                            torch.testing.assert_close(p.grad, gold.grad, rtol=1e-9, atol=1e-12)
        # Raw logits and differentiable soft targets have the documented world-size
        # factor before DDP averages parameter gradients. Auto must select Torch here.
        torch.manual_seed(353)
        x = torch.randn(3, 3, 2, 3, dtype=torch.float64, device=device, requires_grad=True)
        y = torch.rand_like(x, requires_grad=True)
        options = dict(batch_dice=True, target_format="one-hot", class_weight=[0.3, 0.7, 1.1])
        gold = GeneralizedDiceCELoss(backend="torch", **options)(x, y)
        gx, gy = torch.autograd.grad(gold, (x, y))
        selected = slice(0, 1) if rank == 0 else slice(1, 3)
        local_x = x[selected].detach().requires_grad_()
        local_y = y[selected].detach().requires_grad_()
        value = GeneralizedDiceCELoss(distributed=True, **options)(local_x, local_y)
        ax, ay = torch.autograd.grad(value, (local_x, local_y))
        torch.testing.assert_close(ax / 2, gx[selected], rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(ay / 2, gy[selected], rtol=1e-10, atol=1e-12)
    finally:
        destroy_process_group()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_two_rank_ddp_matches_global_batch(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    mp.spawn(_worker, args=(str(tmp_path / "gloo"), device, "gloo"), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two GPUs required for NCCL")
def test_nccl_ddp_matches_global_batch(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "nccl"), "cuda", "nccl"), nprocs=2, join=True)


def test_distributed_requires_group_and_batch_dice():
    with pytest.raises(ValueError, match="batch_dice"):
        DiceLoss(distributed=True)
    with pytest.raises(RuntimeError, match="initialized"):
        DiceLoss(distributed=True, batch_dice=True)(torch.randn(1, 2, 4), torch.zeros(1, 4, dtype=torch.long))
