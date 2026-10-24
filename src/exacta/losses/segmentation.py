"""Segmentation losses.

Inputs are raw network outputs of shape (B, C, *spatial). Targets are class indices of shape
(B, *spatial) with ``target_format="index"`` or one-hot masks of shape (B, C, *spatial) with
``target_format="one-hot"``. Half-precision inputs are computed in float32.

``backend="auto"`` runs fused Triton kernels on GPU (one pass forward, one backward, nothing of
the input's size kept for the backward pass but the input itself) and plain PyTorch elsewhere.
Both give the same loss and gradients up to floating-point rounding.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from exacta import _triton_losses

Activation = Literal["softmax", "sigmoid", "none"]
TargetFormat = Literal["index", "one-hot"]
Backend = Literal["auto", "triton", "torch"]


class DiceLoss(nn.Module):
    """Soft Dice loss ``1 - (2 Σ p·g + smooth) / (Σ p + Σ g + smooth)``.

    Sums run over spatial dimensions (and the batch if ``batch_dice``); the result is averaged
    over the remaining case and class entries. Equals MONAI's ``DiceLoss`` with
    ``smooth_nr = smooth_dr = smooth``.

    Args:
        activation: Applied to the inputs: ``"softmax"`` over channels, ``"sigmoid"`` per
            channel, or ``"none"`` if the inputs are already probabilities.
        target_format: ``"index"`` or ``"one-hot"``.
        include_background: If False, ignore channel 0.
        batch_dice: Sum over the whole batch before taking the ratio (nnU-Net's batch Dice).
        smooth: Added to numerator and denominator; makes empty classes score 1.
        backend: ``"triton"`` (fused GPU kernels), ``"torch"``, or ``"auto"`` for Triton on GPU
            when available and PyTorch otherwise.
    """

    def __init__(
        self,
        *,
        activation: Activation = "softmax",
        target_format: TargetFormat = "index",
        include_background: bool = True,
        batch_dice: bool = False,
        smooth: float = 1e-5,
        backend: Backend = "auto",
    ) -> None:
        super().__init__()
        if activation not in ("softmax", "sigmoid", "none"):
            raise ValueError(f"activation must be 'softmax', 'sigmoid' or 'none', got {activation!r}")
        _check_backend(backend)
        self.activation = activation
        self.target_format = target_format
        self.include_background = include_background
        self.batch_dice = batch_dice
        self.smooth = smooth
        self.backend = backend

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if _use_triton(self.backend, input):
            return self._fused(input, target, lambda_dice=1.0, lambda_ce=0.0, cross_entropy=False)
        input = _upcast(input)
        if self.activation == "softmax":
            probs = input.softmax(1)
        elif self.activation == "sigmoid":
            probs = input.sigmoid()
        else:
            probs = input
        onehot = _one_hot(target, input, self.target_format).to(probs.dtype)
        if not self.include_background:
            if probs.shape[1] < 2:
                raise ValueError("include_background=False needs at least 2 channels")
            probs, onehot = probs[:, 1:], onehot[:, 1:]
        dims = tuple(range(2, probs.ndim))
        if self.batch_dice:
            dims = (0, *dims)
        intersection = (probs * onehot).sum(dims)
        denominator = probs.sum(dims) + onehot.sum(dims)
        return 1 - ((2 * intersection + self.smooth) / (denominator + self.smooth)).mean()

    def _fused(self, input: Tensor, target: Tensor, **weights: float | bool) -> Tensor:
        _check_input(input)
        if self.target_format == "index":
            _check_index_target(target, input)
            _check_index_values(target, input.shape[1])
            if target.dtype.is_floating_point:
                target = target.long()
        else:
            _one_hot(target, input, self.target_format)  # shape and format checks
        if not self.include_background and input.shape[1] < 2:
            raise ValueError("include_background=False needs at least 2 channels")
        return _triton_losses.seg_loss(
            input,
            target,
            activation=self.activation,
            one_hot=self.target_format == "one-hot",
            include_background=self.include_background,
            batch_dice=self.batch_dice,
            smooth=self.smooth,
            **weights,
        )


class DiceCELoss(nn.Module):
    """``lambda_dice * DiceLoss + lambda_ce * CE`` on raw logits.

    CE is cross-entropy with ``activation="softmax"`` and binary cross-entropy with
    ``"sigmoid"``. It always covers every class, whatever ``include_background`` says (as in
    MONAI's ``DiceCELoss``). Other arguments as in :class:`DiceLoss`.
    """

    def __init__(
        self,
        *,
        activation: Literal["softmax", "sigmoid"] = "softmax",
        target_format: TargetFormat = "index",
        include_background: bool = True,
        batch_dice: bool = False,
        smooth: float = 1e-5,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        backend: Backend = "auto",
    ) -> None:
        super().__init__()
        if activation not in ("softmax", "sigmoid"):
            raise ValueError(f"activation must be 'softmax' or 'sigmoid', got {activation!r}")
        self.dice = DiceLoss(
            activation=activation,
            target_format=target_format,
            include_background=include_background,
            batch_dice=batch_dice,
            smooth=smooth,
            backend=backend,
        )
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if _use_triton(self.dice.backend, input):
            return self.dice._fused(
                input, target, lambda_dice=self.lambda_dice, lambda_ce=self.lambda_ce, cross_entropy=True
            )
        dice = self.dice(input, target)
        input = _upcast(input)
        if self.dice.activation == "sigmoid":
            onehot = _one_hot(target, input, self.dice.target_format).to(input.dtype)
            ce = F.binary_cross_entropy_with_logits(input, onehot)
        elif self.dice.target_format == "index":
            _check_index_target(target, input)
            ce = F.cross_entropy(input, target.long())
        else:
            ce = F.cross_entropy(input, _one_hot(target, input, "one-hot").to(input.dtype))
        return self.lambda_dice * dice + self.lambda_ce * ce


def _check_backend(backend: str) -> None:
    if backend not in ("auto", "triton", "torch"):
        raise ValueError(f"backend must be 'auto', 'triton' or 'torch', got {backend!r}")


def _use_triton(backend: Backend, input: Tensor) -> bool:
    if backend == "torch":
        return False
    on_gpu = input.device.type == "cuda" and _triton_losses.available()
    if backend == "triton":
        if not on_gpu:
            raise NotImplementedError("backend='triton' needs a CUDA input and Triton")
        if input.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise TypeError(f"backend='triton' needs floating-point logits, got {input.dtype}")
        return True
    return on_gpu and input.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _check_index_values(target: Tensor, num_classes: int) -> None:
    """The fused kernels cannot raise, so check what F.one_hot would."""
    if target.numel():
        lo, hi = torch.aminmax(target)
        lo, hi = int(lo), int(hi)
        if lo < 0 or hi >= num_classes:
            raise ValueError(f"class indices must lie in [0, {num_classes}), found values in [{lo}, {hi}]")


def _check_input(input: Tensor) -> None:
    if input.ndim < 3:
        raise ValueError(f"input must have shape (B, C, *spatial), got {tuple(input.shape)}")


def _upcast(input: Tensor) -> Tensor:
    _check_input(input)
    return input.float() if input.dtype in (torch.float16, torch.bfloat16) else input


def _check_index_target(target: Tensor, input: Tensor) -> None:
    expected = (input.shape[0], *input.shape[2:])
    if target.shape != expected:
        raise ValueError(f"index target must have shape {expected}, got {tuple(target.shape)}")


def _one_hot(target: Tensor, input: Tensor, target_format: TargetFormat) -> Tensor:
    if target_format == "index":
        _check_index_target(target, input)
        return F.one_hot(target.long(), input.shape[1]).movedim(-1, 1)
    if target_format == "one-hot":
        if target.shape != input.shape:
            raise ValueError(
                f"one-hot target must have the input's shape {tuple(input.shape)}, got {tuple(target.shape)}"
            )
        return target
    raise ValueError(f"target_format must be 'index' or 'one-hot', got {target_format!r}")
