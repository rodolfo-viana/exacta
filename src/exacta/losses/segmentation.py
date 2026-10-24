"""Segmentation losses.

Inputs are raw network outputs of shape (B, C, *spatial). Targets are class indices of shape
(B, *spatial) with ``target_format="index"`` or one-hot masks of shape (B, C, *spatial) with
``target_format="one-hot"``. Half-precision inputs are computed in float32.

``backend="auto"`` runs fused Triton kernels on GPU (one pass forward, one backward, saving
inputs and small statistics instead of volume-sized activations) and plain PyTorch elsewhere.
Both give the same loss and gradients up to floating-point rounding. The fused kernels do not
differentiate their own backward pass (double backward raises; use ``backend="torch"`` for
gradient penalties) and do not propagate gradients into the target (``"auto"`` then uses
PyTorch).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from exacta import _triton_losses

Activation = Literal["softmax", "sigmoid", "none"]
TargetFormat = Literal["index", "one-hot"]
Backend = Literal["auto", "triton", "torch"]


class _OverlapLoss(nn.Module):
    """Shared activation, reduction and dispatch for overlap losses."""

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
        _check_target_format(target_format)
        _nonnegative_finite("smooth", smooth)
        self.activation = activation
        self.target_format = target_format
        self.include_background = include_background
        self.batch_dice = batch_dice
        self.smooth = smooth
        self.backend = backend

    def _tversky_weights(self) -> tuple[float, float] | None:
        return None

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if _use_triton(self.backend, input, target):
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
        weights = self._tversky_weights()
        if weights is not None:
            alpha, beta = weights
            fp = (probs * (1 - onehot)).sum(dims)
            fn = ((1 - probs) * onehot).sum(dims)
            return 1 - ((intersection + self.smooth) / (
                intersection + alpha * fp + beta * fn + self.smooth
            )).mean()
        denominator = probs.sum(dims) + onehot.sum(dims)
        return 1 - ((2 * intersection + self.smooth) / (denominator + self.smooth)).mean()

    def _fused(
        self, input: Tensor, target: Tensor, *, lambda_dice: float, lambda_ce: float, cross_entropy: bool
    ) -> Tensor:
        _check_input(input)
        _check_fused_target(target, input, self.target_format)
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
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
            cross_entropy=cross_entropy,
            tversky=self._tversky_weights(),
        )


class DiceLoss(_OverlapLoss):
    """Soft Dice loss ``1 - (2 Σ p·g + smooth) / (Σ p + Σ g + smooth)``.

    Sums cover spatial dimensions and optionally the batch (``batch_dice=True``); the result
    is averaged over remaining cases and classes. ``include_background=False`` excludes
    channel 0 after activation. ``activation`` is ``"softmax"`` (default), ``"sigmoid"`` or
    ``"none"`` for probabilities. Targets use ``target_format="index"`` (default) or
    ``"one-hot"`` (also accepts soft labels). ``smooth=1e-5`` adds the same nonnegative value
    to numerator and denominator, matching MONAI's equal ``smooth_nr`` and ``smooth_dr``.
    ``backend`` selects ``"auto"``, ``"torch"`` or ``"triton"``.
    """


class TverskyLoss(_OverlapLoss):
    """``1 - (TP + smooth) / (TP + alpha·FP + beta·FN + smooth)``.

    ``TP=Σ p·g``, ``FP=Σ p·(1-g)``, ``FN=Σ (1-p)·g``. Increasing ``beta`` penalizes missed
    foreground; increasing ``alpha`` penalizes false positives. Both weights must be finite
    and nonnegative; they need not sum to one. The default ``alpha=beta=0.5`` equals
    :class:`DiceLoss` with **twice** this loss's smoothing constant. Other arguments and
    reductions follow :class:`DiceLoss`.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        beta: float = 0.5,
        activation: Activation = "softmax",
        target_format: TargetFormat = "index",
        include_background: bool = True,
        batch_dice: bool = False,
        smooth: float = 1e-5,
        backend: Backend = "auto",
    ) -> None:
        super().__init__(
            activation=activation, target_format=target_format, include_background=include_background,
            batch_dice=batch_dice, smooth=smooth, backend=backend,
        )
        _nonnegative_finite("alpha", alpha)
        _nonnegative_finite("beta", beta)
        self.alpha, self.beta = alpha, beta

    def _tversky_weights(self) -> tuple[float, float]:
        return self.alpha, self.beta


class FocalLoss(nn.Module):
    """Focal cross-entropy on finite logits, with a mean reduction.

    For ``activation="softmax"`` (default), sum ``-w·g·(1-p)**gamma·log(p)`` over classes,
    then average over cases and voxels. For ``"sigmoid"``, average
    ``-w_pos·g·(1-p)**gamma·log(p) - w_neg·(1-g)·p**gamma·log(1-p)`` over all entries.
    These formulas also define the extension to soft targets: a weighted sum of hard-label
    losses. Half/bfloat16 inputs compute in float32, and float64 precision is preserved.

    ``gamma`` must be finite and nonnegative (default 2). ``alpha=None`` disables weighting;
    a scalar in [0, 1] weights positive sigmoid labels by alpha and negatives by 1-alpha.
    For softmax it weights background (channel 0) by 1-alpha and each foreground class by
    alpha. Every channel participates. With ``gamma=0, alpha=None`` this is ordinary CE/BCE.
    Target formats and backends follow :class:`DiceLoss`.
    """

    def __init__(
        self,
        *,
        activation: Literal["softmax", "sigmoid"] = "softmax",
        target_format: TargetFormat = "index",
        gamma: float = 2.0,
        alpha: float | None = None,
        backend: Backend = "auto",
    ) -> None:
        super().__init__()
        if activation not in ("softmax", "sigmoid"):
            raise ValueError(f"activation must be 'softmax' or 'sigmoid', got {activation!r}")
        _check_target_format(target_format)
        _check_backend(backend)
        _nonnegative_finite("gamma", gamma)
        if alpha is not None and (not math.isfinite(alpha) or not 0 <= alpha <= 1):
            raise ValueError("alpha must be None or a finite number in [0, 1]")
        self.activation, self.target_format = activation, target_format
        self.gamma, self.alpha, self.backend = gamma, alpha, backend

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if _use_triton(self.backend, input, target):
            _check_input(input)
            _check_fused_target(target, input, self.target_format)
            return _triton_losses.focal_loss(
                input, target, activation=self.activation, one_hot=self.target_format == "one-hot",
                gamma=self.gamma, alpha=self.alpha,
            )
        input = _upcast(input)
        truth = _one_hot(target, input, self.target_format).to(input.dtype)
        if self.activation == "softmax":
            logp = input.log_softmax(1)
            if self.gamma == 0:
                modulation = torch.ones_like(logp)
            else:
                complement = -logp.expm1()
                # At rounded p=1, the positive-gamma loss and its gradient are both zero.
                # Avoid the undefined derivative of 0**gamma for fractional gamma < 1.
                safe = torch.where(complement > 0, complement, 1)
                modulation = torch.where(complement > 0, safe.pow(self.gamma), 0)
            terms = -truth * modulation * logp
            if self.alpha is not None:
                weights = torch.where(
                    torch.arange(input.shape[1], device=input.device) == 0,
                    input.new_tensor(1 - self.alpha), input.new_tensor(self.alpha),
                )
                terms = terms * weights.reshape(1, -1, *([1] * (input.ndim - 2)))
            return terms.sum(1).mean()
        pos = F.logsigmoid(input)
        neg = F.logsigmoid(-input)
        positive = -truth * (self.gamma * neg).exp() * pos
        negative = -(1 - truth) * (self.gamma * pos).exp() * neg
        if self.alpha is not None:
            positive, negative = self.alpha * positive, (1 - self.alpha) * negative
        return (positive + negative).mean()


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
        if _use_triton(self.dice.backend, input, target):
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


def _check_target_format(target_format: str) -> None:
    if target_format not in ("index", "one-hot"):
        raise ValueError(f"target_format must be 'index' or 'one-hot', got {target_format!r}")


def _nonnegative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")


def _check_fused_target(target: Tensor, input: Tensor, target_format: TargetFormat) -> None:
    if target_format == "index":
        _check_index_target(target, input)
        if not torch.compiler.is_compiling():
            # Eager execution raises. Compiled kernels avoid a host sync and return NaN instead.
            _check_index_values(target, input.shape[1])
    else:
        _one_hot(target, input, target_format)


def _use_triton(backend: Backend, input: Tensor, target: Tensor) -> bool:
    if backend == "torch":
        return False
    on_gpu = input.device.type == "cuda" and _triton_losses.available()
    floating = input.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    if backend == "triton":
        if not on_gpu:
            raise NotImplementedError("backend='triton' needs a CUDA input and Triton")
        if not floating:
            raise TypeError(f"backend='triton' needs floating-point logits, got {input.dtype}")
        if target.requires_grad:
            raise NotImplementedError(
                "backend='triton' does not propagate gradients into the target; use backend='torch'"
            )
    elif not (on_gpu and floating) or target.requires_grad:
        return False
    # Empty inputs have nothing to fuse; the PyTorch path returns NaN for them.
    return input.numel() > 0


def _check_index_values(target: Tensor, num_classes: int) -> None:
    """The fused kernels cannot raise, so check what F.one_hot would."""
    if target.numel():
        bounds = torch.aminmax(target)
        lo, hi = int(bounds.min), int(bounds.max)
        if lo < 0 or hi >= num_classes:
            raise ValueError(f"class indices must lie in [0, {num_classes}), found values in [{lo}, {hi}]")


def _check_input(input: Tensor) -> None:
    if input.ndim < 3:
        raise ValueError(f"input must have shape (B, C, *spatial), got {tuple(input.shape)}")
    if input.shape[1] < 1:
        raise ValueError("input must have at least one channel")
    if not input.is_floating_point():
        raise TypeError("input must contain floating-point logits or probabilities")


def _upcast(input: Tensor) -> Tensor:
    _check_input(input)
    return input.float() if input.dtype in (torch.float16, torch.bfloat16) else input


def _check_index_target(target: Tensor, input: Tensor) -> None:
    if target.device != input.device:
        raise ValueError("input and target must be on the same device")
    if target.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("index target must have an integer dtype")
    expected = (input.shape[0], *input.shape[2:])
    if target.shape != expected:
        raise ValueError(f"index target must have shape {expected}, got {tuple(target.shape)}")


def _one_hot(target: Tensor, input: Tensor, target_format: TargetFormat) -> Tensor:
    if target.device != input.device:
        raise ValueError("input and target must be on the same device")
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
