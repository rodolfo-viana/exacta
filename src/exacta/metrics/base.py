"""Base class for metrics that collect one value per case and class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from exacta._inputs import to_torch


@dataclass(frozen=True)
class MetricResult:
    """Accumulated metric values.

    Attributes:
        per_case: (N, C) values for every case and class, gathered from all processes.
        per_class: (C,) mean over cases, ignoring NaN (cases where ``both_empty`` was NaN).
        mean: Mean of ``per_class``, ignoring NaN.
    """

    per_case: Tensor
    per_class: Tensor
    mean: Tensor


class PerCaseMetric:
    """Collects (B, C) per-case values in ``update`` and aggregates them in ``compute``.

    Under ``torch.distributed``, ``compute`` gathers values from every process, so every rank
    must call it, including ranks without updates. With NCCL, set each rank's current CUDA
    device before computing. An error is raised if no rank has any cases.
    ``DistributedSampler`` pads the dataset by repeating cases, which then count
    twice; use a sampler without padding for evaluation.
    """

    def __init__(self) -> None:
        self._values: list[Tensor] = []

    def update(self, pred: Any, target: Any) -> Tensor:
        """Add a batch and return its (B, C) per-case values."""
        return self._record(self._batch(to_torch(pred), to_torch(target)))

    def compute(self) -> MetricResult:
        per_case = _all_gather_cat(torch.cat(self._values) if self._values else None)
        if per_case is None:
            raise RuntimeError(f"{type(self).__name__}.compute() called before update()")
        per_class = per_case.nanmean(0)
        return MetricResult(per_case, per_class, per_class.nanmean())

    def reset(self) -> None:
        self._values.clear()

    def _batch(self, pred: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError

    def _record(self, values: Tensor) -> Tensor:
        self._values.append(values.detach())
        return values


def _all_gather_cat(x: Tensor | None) -> Tensor | None:
    """Gather float64 case scores, including ranks with no updates; None if all ranks are empty.

    Exchange row and channel counts before allocating the payload, so an empty rank need not
    know the channel count. Gloo uses CPU collectives; NCCL uses the rank's CUDA device.
    """
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return x
    world = dist.get_world_size()
    if dist.get_backend() == "nccl":
        device = x.device if x is not None and x.is_cuda else torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    output_device = x.device if x is not None else device
    size = torch.tensor([x.shape[0], x.shape[1]] if x is not None else [0, -1], device=device)
    sizes = [torch.zeros_like(size) for _ in range(world)]
    dist.all_gather(sizes, size)
    shapes = [s.tolist() for s in sizes]
    counts = [rows for rows, _ in shapes]
    if not any(counts):
        return None
    channels = {cols for _, cols in shapes if cols >= 0}
    if len(channels) != 1:
        raise ValueError("metric channel counts must agree across distributed ranks")
    padded = torch.full((max(counts), channels.pop()), float("nan"), dtype=torch.float64, device=device)
    if x is not None:
        padded[: x.shape[0]] = x.to(device)
    gathered = [torch.empty_like(padded) for _ in range(world)]
    dist.all_gather(gathered, padded)
    return torch.cat([g[:n] for g, n in zip(gathered, counts)]).to(output_device)
