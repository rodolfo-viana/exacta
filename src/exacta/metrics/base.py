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
    must call it. ``DistributedSampler`` pads the dataset by repeating cases, which then count
    twice; use a sampler without padding for evaluation.
    """

    def __init__(self) -> None:
        self._values: list[Tensor] = []

    def update(self, pred: Any, target: Any, **kwargs: Any) -> Tensor:
        """Add a batch and return its (B, C) per-case values."""
        values = self._batch(to_torch(pred), to_torch(target), **kwargs)
        self._values.append(values.detach())
        return values

    def compute(self) -> MetricResult:
        if not self._values:
            raise RuntimeError(f"{type(self).__name__}.compute() called before update()")
        per_case = _all_gather_cat(torch.cat(self._values))
        per_class = per_case.nanmean(0)
        return MetricResult(per_case, per_class, per_class.nanmean())

    def reset(self) -> None:
        self._values.clear()

    def _batch(self, pred: Tensor, target: Tensor, **kwargs: Any) -> Tensor:
        raise NotImplementedError


def _all_gather_cat(x: Tensor) -> Tensor:
    """Concatenate ``x`` along dim 0 across processes (rows may differ per rank)."""
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return x
    world = dist.get_world_size()
    size = torch.tensor([x.shape[0]], device=x.device)
    sizes = [torch.zeros_like(size) for _ in range(world)]
    dist.all_gather(sizes, size)
    counts = [int(s.item()) for s in sizes]
    padded = x.new_full((max(counts), *x.shape[1:]), float("nan"))
    padded[: x.shape[0]] = x
    gathered = [torch.empty_like(padded) for _ in range(world)]
    dist.all_gather(gathered, padded)
    return torch.cat([g[:n] for g, n in zip(gathered, counts)])
