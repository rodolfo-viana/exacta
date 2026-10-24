"""Differentiable sums of small loss statistics, with DDP gradient semantics."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.nn.functional import all_reduce


def sum_ranks(value: Tensor) -> Tensor:
    """Sum statistics and their adjoints over the default process group.

    A replicated global objective contributes once on every rank. Summing its
    adjoints here compensates for DDP's subsequent parameter-gradient average.
    Gloo uses CPU buffers, including when CUDA computation is tested on one GPU.
    NCCL keeps statistics on the rank's CUDA device. Transfers remain in the graph.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("distributed=True requires an initialized torch.distributed process group")
    if dist.get_world_size() == 1:
        return value
    device = value.device if dist.get_backend() == "nccl" else torch.device("cpu")
    return all_reduce(value.to(device), op=dist.ReduceOp.SUM).to(value.device)
