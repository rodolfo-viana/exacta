"""Regression objectives, using native PyTorch autograd and reductions.

Standard objectives are the original torch.nn classes, with their exact PyTorch
contracts and performance. Custom objectives require identical floating input and
target shapes; they do not silently broadcast targets.
"""
from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import GaussianNLLLoss, HuberLoss, L1Loss, MSELoss, PoissonNLLLoss, SmoothL1Loss

__all__ = ["L1Loss", "MSELoss", "HuberLoss", "SmoothL1Loss", "GaussianNLLLoss", "PoissonNLLLoss",
           "QuantileLoss", "LogCoshLoss"]
Reduction = Literal["none", "mean", "sum"]


class _RegressionLoss(nn.Module):
    def __init__(self, reduction: Reduction) -> None:
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be none, mean or sum")
        self.reduction = reduction

    def _error(self, input: Tensor, target: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor | None]:
        if input.shape != target.shape or input.device != target.device:
            raise ValueError("input and target must have identical shapes and devices")
        if not input.is_floating_point() or not target.is_floating_point():
            raise ValueError("input and target must be floating tensors")
        if mask is not None:
            if mask.shape != input.shape or mask.device != input.device or mask.dtype != torch.bool:
                raise ValueError("mask must be boolean with identical shape and device")
            input, target = torch.where(mask, input, 0), torch.where(mask, target, 0)
        dtype = torch.promote_types(input.dtype, target.dtype)
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        return input.to(dtype) - target.to(dtype), mask

    def _reduce(self, terms: Tensor, mask: Tensor | None) -> Tensor:
        if mask is not None:
            terms = torch.where(mask, terms, 0)
        if self.reduction == "none":
            return terms
        total = terms.sum()
        if self.reduction == "sum":
            return total
        count = terms.numel() if mask is None else mask.sum().clamp_min(1)
        return total / max(count, 1) if isinstance(count, int) else total / count


class QuantileLoss(_RegressionLoss):
    """Pinball loss for a single conditional quantile, 0 < quantile < 1.

    For residual r=target-input the term is max(q*r, (q-1)*r). The selected
    subgradient at zero is the midpoint of the one-sided slopes. Mean divides
    by observed elements; fully masked/empty inputs give differentiable zero.
    half/bfloat16 accumulate in float32; float64 and higher derivatives are kept.
    Inputs are expected finite on observed elements, as with torch.nn losses.
    """

    def __init__(self, quantile: float = .5, *, reduction: Reduction = "mean") -> None:
        super().__init__(reduction)
        if not math.isfinite(quantile) or not 0 < quantile < 1:
            raise ValueError("quantile must be finite and strictly between 0 and 1")
        self.quantile = quantile

    def forward(self, input: Tensor, target: Tensor, *, mask: Tensor | None = None) -> Tensor:
        error, mask = self._error(input, target, mask)
        terms = torch.maximum((1 - self.quantile) * error, -self.quantile * error)
        return self._reduce(terms, mask)


class LogCoshLoss(_RegressionLoss):
    """Stable log(cosh(input-target)); quadratic near zero, linear for large errors.

    Near zero, log1p(2*sinh(error/2)²) avoids cancellation and preserves the
    second derivative. The large-error expression avoids overflow, including in
    higher derivatives. Mask/dtype/reduction semantics match QuantileLoss.
    """

    def __init__(self, *, reduction: Reduction = "mean") -> None:
        super().__init__(reduction)

    def forward(self, input: Tensor, target: Tensor, *, mask: Tensor | None = None) -> Tensor:
        error, mask = self._error(input, target, mask)
        small = torch.log1p(2 * (error.clamp(-1, 1) / 2).sinh().square())
        magnitude = error.abs()
        large = magnitude + torch.nn.functional.softplus(-2 * magnitude) - math.log(2)
        terms = torch.where(magnitude <= 1, small, large)
        return self._reduce(terms, mask)
