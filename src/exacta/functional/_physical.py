"""Physical measures from shared counts; no additional image traversal."""
from __future__ import annotations

import math
from typing import Any, Literal

import torch
from torch import Tensor

from exacta._inputs import Segmentations, as_spacing


def _physical_scale(seg: Segmentations, spacing: Any, unit: str, ndim: Literal[2, 3]) -> Tensor:
    kind, units, divisor = ("area", ("mm2", "cm2"), 100) if ndim == 2 else ("volume", ("mm3", "ml"), 1000)
    if seg.ndim != ndim:
        raise ValueError(f"physical {kind} requires {ndim}D inputs")
    if unit not in units:
        raise ValueError(f"unit must be {units[0]!r} or {units[1]!r}; spacing is in millimetres")
    if spacing is None:
        raise ValueError(f"spacing in millimetres is required for physical {kind}")
    values = [math.prod(s) / (divisor if unit == units[1] else 1)
              for s in as_spacing(spacing, batch=seg.batch, ndim=ndim)]
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError(f"spacing product must be finite and positive in the requested {kind} unit")
    return torch.tensor(values, device=seg.pred.device, dtype=torch.float64).reshape(-1, 1)


def _score_size(name: str, counts: tuple[Tensor, Tensor, Tensor, Tensor],
                scale: Tensor, absolute: bool = False) -> Tensor:
    tp, fp, fn, tn = counts
    value = tp + fp if name.startswith("predicted_") else tp + fn if name.startswith("target_") else fp - fn
    if absolute:
        value = value.abs()
    return torch.where(tp + fp + fn + tn > 0, value.double() * scale, torch.nan)
