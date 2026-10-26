"""Validated sufficient statistics for independent Bernoulli predictions.

Scores store (count, sum) per case/label; reliability stores count, confidence sum and
positive-target sum per bin. Backends never choose the public averaging convention.
"""
from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor

from exacta._inputs import to_torch
from exacta import _native, _triton_probability

Average = Literal["none", "macro", "micro"]
Statistic = Literal["brier", "nll", "bins"]


def check_options(multilabel: bool, average: Average, ignore_index: int | None) -> None:
    if not isinstance(multilabel, bool):
        raise TypeError("multilabel must be a bool")
    if average not in ("none", "macro", "micro"):
        raise ValueError("average must be 'none', 'macro' or 'micro'")
    if multilabel and ignore_index is not None:
        raise ValueError("multilabel targets use boolean masks instead of ignore_index")
    if not multilabel and average != "none":
        raise ValueError("average is supported only with multilabel=True")


def inputs(pred: Any, target: Any, mask: Any) -> tuple[Tensor, Tensor, Tensor | None]:
    """Validate layout/dtypes without scanning values, preserving target precision."""
    p, t = to_torch(pred).detach(), to_torch(target).detach()
    if p.ndim not in (4, 5) or p.shape[1] < 1:
        raise ValueError("pred must have shape (B,C,H,W) or (B,C,D,H,W), with C >= 1")
    if p.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("pred must contain real floating-point logits or probabilities")
    if t.shape != p.shape:
        raise ValueError("multilabel target must have the same (B,C,*spatial) shape as pred")
    if t.device != p.device:
        raise ValueError("pred and target must be on the same device")
    if t.is_complex():
        raise TypeError("multilabel target must contain real binary values")
    m = None if mask is None else to_torch(mask)
    if m is not None:
        if m.dtype != torch.bool:
            raise TypeError("mask must be a boolean array")
        if m.device != p.device:
            raise ValueError("mask and inputs must be on the same device")
        if m.shape == (p.shape[0], *p.shape[2:]):
            m = m[:, None]
        if m.shape not in ((p.shape[0], 1, *p.shape[2:]), p.shape):
            raise ValueError("mask must have shape (B,*spatial), (B,1,*spatial) or (B,C,*spatial)")
    return p, t, m


def statistics(pred: Any, target: Any, mask: Any, from_logits: bool,
               kind: Statistic, bins: int = 0) -> Tensor:
    p, t, m = inputs(pred, target, mask)
    if p.device.type == "cpu" and _native.available():
        result = _native.probability_stats(p, t, m, from_logits, ("brier", "nll", "bins").index(kind), bins)
        if result is not None:
            return result
    if p.device.type == "cuda" and _triton_probability.available() and (kind != "bins" or bins <= _triton_probability._MAX_BINS):
        result = _triton_probability.statistics(p, t, m, from_logits, ("brier", "nll", "bins").index(kind), bins)
        if result is not None:
            return result
    return torch_statistics(p, t, m, from_logits, kind, bins)


def torch_statistics(p: Tensor, t: Tensor, mask: Tensor | None, from_logits: bool,
                     kind: Statistic, bins: int) -> Tensor:
    """Native PyTorch baseline, with validation before any lossy target conversion."""
    valid = torch.ones_like(t, dtype=torch.bool) if mask is None else mask.expand_as(t)
    bad_target = (((t != 0) & (t != 1)) & valid).any()
    bad_finite = ((~torch.isfinite(p)) & valid).any()
    bad_range = (((p < 0) | (p > 1)) & valid).any() if not from_logits else p.new_zeros((), dtype=torch.bool)
    bad = torch.stack((bad_target, bad_finite, bad_range)).tolist()
    if bad[0]:
        raise ValueError("included multilabel targets must be binary (0/1)")
    if bad[1]:
        raise ValueError("included predictions must be finite")
    if bad[2]:
        raise ValueError("probabilities must lie in [0, 1]")
    # Float64 arithmetic defines the metric independently of the input's storage precision.
    # Fused backends avoid these full-volume work buffers.
    x = torch.where(valid, p, 0).double()
    positive = (t == 1) & valid
    if kind == "nll":
        if from_logits:
            signed = torch.where(positive, -x, x)
            terms = torch.logaddexp(torch.zeros_like(x), signed)
        else:
            terms = torch.where(positive, -x.log(), -torch.log1p(-x))
    else:
        probability = x.sigmoid() if from_logits else x
        if kind == "brier":
            terms = (probability - positive.double()).square()
        else:
            batch, classes = p.shape[:2]
            rows = batch * classes
            indices = (probability.flatten(2) * bins).long().clamp_max(bins - 1).flatten(0, 1)
            keep = valid.flatten(2).flatten(0, 1)
            count = torch.zeros((rows, bins), dtype=torch.float64, device=p.device)
            confidence = torch.zeros_like(count)
            target_sum = torch.zeros_like(count)
            count.scatter_add_(1, indices, keep.double())
            confidence.scatter_add_(1, indices, torch.where(keep, probability.flatten(2).flatten(0, 1), 0))
            target_sum.scatter_add_(1, indices, positive.flatten(2).flatten(0, 1).double())
            return torch.stack((count, confidence, target_sum), 1).reshape(batch, classes, 3, bins)
    count = valid.flatten(2).sum(-1, dtype=torch.float64)
    total = torch.where(valid, terms, 0).flatten(2).sum(-1)
    return torch.stack((count, total), -1)


def score(sums: Tensor, average: Average) -> Tensor:
    count, total = sums.unbind(-1)
    if average == "micro":
        count, total = count.sum(1, keepdim=True), total.sum(1, keepdim=True)
    values = torch.where(count > 0, total / count.clamp_min(1), torch.nan)
    return values.nanmean(1, keepdim=True) if average == "macro" else values


def ece(sums: Tensor, average: Average) -> Tensor:
    """Micro pools label bins before absolute errors; macro averages label ECE values."""
    if average == "micro":
        sums = sums.sum(1, keepdim=True)
    count, confidence, positives = sums.unbind(-2)
    total = count.sum(-1)
    values = (positives - confidence).abs().sum(-1) / total.clamp_min(1)
    values = torch.where(total > 0, values, torch.nan)
    return values.nanmean(1, keepdim=True) if average == "macro" else values
