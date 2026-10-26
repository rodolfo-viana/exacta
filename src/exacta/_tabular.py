"""Shared validation and distributed gathering for dataset-level metrics."""
from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from exacta._inputs import to_torch


def real_tensor(value: Any, name: str, device: torch.device | None = None) -> Tensor:
    x = to_torch(value).detach()
    if x.is_complex() or x.layout != torch.strided:
        raise ValueError(f"{name} must be a dense real array")
    if device is not None and x.device != device:
        raise ValueError(f"{name} and pred must be on the same device")
    return x


def observations(target: Tensor, mask: Any, sample_weight: Any) -> tuple[Tensor, Tensor | None]:
    """Boolean observations, and optional nonnegative weights along the sample axis."""
    valid = torch.ones_like(target, dtype=torch.bool)
    if mask is not None:
        m = real_tensor(mask, "mask", target.device)
        if m.dtype != torch.bool:
            raise ValueError("mask must be boolean")
        if target.ndim == 2 and m.shape == target.shape[:1]:
            m = m[:, None]
        if m.shape != target.shape and m.shape != (target.shape[0], 1):
            raise ValueError("mask must have target shape or shape (N,)")
        # Do not accept (N,1) for a one-dimensional target: broadcasting would make (N,N).
        if m.ndim != target.ndim:
            raise ValueError("mask must have target shape or shape (N,)")
        valid &= m
    weight = None
    if sample_weight is not None:
        weight = real_tensor(sample_weight, "sample_weight", target.device).double()
        if weight.shape != target.shape[:1]:
            raise ValueError("sample_weight must have shape (N,)")
        if not bool((torch.isfinite(weight) & (weight >= 0)).all()):
            raise ValueError("sample_weight must be finite and nonnegative")
        if target.ndim == 2:
            weight = weight[:, None]
        valid &= weight > 0
    return valid, weight


def check_config(config: tuple[Any, ...]) -> None:
    """Reject mismatched distributed options before entering payload collectives."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        configs: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(configs, config)
        if any(c != config for c in configs):
            raise ValueError("metric configuration must agree across distributed ranks")


def gathered(state: Tensor | None, config: tuple[Any, ...]) -> list[Tensor]:
    """Gather compact states, checking configuration even on ranks without updates."""
    # Lazy import avoids a functional -> metrics -> functional initialization cycle.
    from exacta.metrics.base import _all_gather_cat

    check_config(config)
    packed = _all_gather_cat(None if state is None else state.reshape(1, -1))
    if packed is None:
        raise RuntimeError("compute() called before update() on every rank")
    return list(packed.unbind(0))


def divide(numerator: Tensor, denominator: Tensor, zero: float = float("nan")) -> Tensor:
    return torch.where(denominator != 0, numerator / denominator, zero)
