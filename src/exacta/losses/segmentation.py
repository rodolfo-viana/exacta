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
from collections.abc import Sequence
from numbers import Integral
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from exacta import _triton_losses, _loss_reductions
from exacta._distributed import sum_ranks
from exacta._loss_reductions import EmptyTarget, WeightType, generalized_dice_terms, masked_overlap_terms

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
        distributed: bool = False,
        smooth: float = 1e-5,
        ignore_index: int | None = None,
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
        if distributed and not batch_dice:
            raise ValueError("distributed=True requires batch_dice=True for overlap losses")
        self.distributed = distributed
        self.smooth = smooth
        self.ignore_index = _check_ignore_index(ignore_index, target_format)
        self.backend = backend

    def _tversky_weights(self) -> tuple[float, float] | None:
        return None

    def _generalized_config(self) -> tuple[WeightType, EmptyTarget] | None:
        return None

    def _overlap_power(self) -> float:
        return 1.0

    def _squared_denominator(self) -> bool:
        return False

    def forward(self, input: Tensor, target: Tensor, *, mask: Tensor | None = None) -> Tensor:
        mask = _check_mask(mask, input)
        if _use_triton(self.backend, input, target, allow_empty=self.distributed):
            return self._fused(input, target, mask=mask, lambda_dice=1.0, lambda_ce=0.0, cross_entropy=False)
        input, target, valid = _prepare_masked_input(
            input, target, self.target_format, self.ignore_index, mask,
        )
        if self.activation == "softmax":
            probs = input.softmax(1)
        elif self.activation == "sigmoid":
            probs = input.sigmoid()
        else:
            probs = input
        onehot = _one_hot(target, input, self.target_format).to(probs.dtype)
        if valid is not None:
            probs = torch.where(valid[:, None], probs, 0)
            onehot = torch.where(valid[:, None], onehot, 0)
        if not self.include_background:
            if probs.shape[1] < 2:
                raise ValueError("include_background=False needs at least 2 channels")
            probs, onehot = probs[:, 1:], onehot[:, 1:]
        dims = tuple(range(2, probs.ndim))
        if self.batch_dice:
            dims = (0, *dims)
        intersection = (probs * onehot).sum(dims)
        weights = self._tversky_weights()
        valid_counts = None if valid is None else valid.flatten(1).sum(1)
        if valid_counts is not None and self.batch_dice:
            valid_counts = valid_counts.sum()
        if weights is not None:
            alpha, beta = weights
            fp = (probs * (1 - onehot)).sum(dims)
            fn = ((1 - probs) * onehot).sum(dims)
            if self.distributed:
                intersection, fp, fn, valid_counts = _global_overlap(intersection, fp, fn, valid_counts, input)
            return _overlap_mean(intersection + self.smooth,
                                 intersection + alpha * fp + beta * fn + self.smooth, valid_counts,
                                 power=self._overlap_power()).to(input.dtype)
        if self._squared_denominator():
            prediction, truth = probs.square().sum(dims), onehot.square().sum(dims)
        else:
            prediction, truth = probs.sum(dims), onehot.sum(dims)
        if self.distributed:
            intersection, prediction, truth, valid_counts = _global_overlap(
                intersection, prediction, truth, valid_counts, input,
            )
        generalized = self._generalized_config()
        if generalized is not None:
            numerator, denominator, _ = generalized_dice_terms(
                intersection, prediction, truth, weight_type=generalized[0],
                empty_target=generalized[1], smooth=self.smooth,
            )
            return _overlap_mean(numerator, denominator, valid_counts).to(input.dtype)
        denominator = prediction + truth
        return _overlap_mean(2 * intersection + self.smooth, denominator + self.smooth, valid_counts).to(input.dtype)

    def _fused(
        self, input: Tensor, target: Tensor, *, lambda_dice: float, lambda_ce: float, cross_entropy: bool,
        mask: Tensor | None = None,
        class_weight: Tensor | None = None, focal: tuple[float, float | None] | None = None,
    ) -> Tensor:
        _check_input(input)
        _check_fused_target(target, input, self.target_format,
                            masked=mask is not None or self.ignore_index is not None or self.distributed)
        if lambda_dice != 0 and not self.include_background and input.shape[1] < 2:
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
            generalized=self._generalized_config(),
            mask=mask,
            ignore_index=self.ignore_index,
            class_weight=class_weight, focal=focal,
            distributed=self.distributed,
            overlap_power=self._overlap_power(),
            squared_pred=self._squared_denominator(),
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
    ``squared_pred=True`` changes the denominator to ``Σ p² + Σ g² + smooth``;
    the product numerator is unchanged, including for soft targets. This option
    is supported by the fused forward/backward and distributed reductions.

    ``ignore_index`` excludes that integer label. All losses also accept a keyword-only
    boolean ``mask`` of shape ``(B, *spatial)`` or ``(B, 1, *spatial)`` in forward; True
    includes a voxel. Both exclusions combine. Excluded values have zero gradient, including
    nonfinite logits or soft targets there. Entirely ignored cases are omitted from overlap
    averages; an entirely ignored batch returns differentiable zero, also with smooth=0.
    An annotated but empty foreground remains governed by the ordinary loss definition.

    ``distributed=True`` requires ``batch_dice=True`` and an initialized default
    process group. Reduce statistics across ranks before taking ratios. Every rank
    must use identical settings and participate in forward and backward, including
    empty or fully ignored ranks. This gives the concatenated-batch objective and
    parameter gradients with ordinary DDP gradient averaging. Logit gradients before
    DDP averaging are multiplied by world size. Local behavior remains the default.
    """

    def __init__(self, *, squared_pred: bool = False, activation: Activation = "softmax",
                 target_format: TargetFormat = "index", include_background: bool = True,
                 batch_dice: bool = False, distributed: bool = False, smooth: float = 1e-5,
                 ignore_index: int | None = None, backend: Backend = "auto") -> None:
        super().__init__(activation=activation, target_format=target_format,
                         include_background=include_background, batch_dice=batch_dice,
                         distributed=distributed, smooth=smooth, ignore_index=ignore_index, backend=backend)
        if not isinstance(squared_pred, bool):
            raise TypeError("squared_pred must be a bool")
        self.squared_pred = squared_pred

    def _squared_denominator(self) -> bool:
        return self.squared_pred


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
        distributed: bool = False,
        smooth: float = 1e-5,
        ignore_index: int | None = None,
        backend: Backend = "auto",
    ) -> None:
        super().__init__(
            activation=activation, target_format=target_format, include_background=include_background,
            batch_dice=batch_dice, distributed=distributed, smooth=smooth, ignore_index=ignore_index, backend=backend,
        )
        _nonnegative_finite("alpha", alpha)
        _nonnegative_finite("beta", beta)
        self.alpha, self.beta = alpha, beta

    def _tversky_weights(self) -> tuple[float, float]:
        return self.alpha, self.beta


class FocalTverskyLoss(TverskyLoss):
    """Mean of ``(1 - TverskyIndex)**gamma`` over selected cases/classes.

    ``gamma`` is the actual exponent (default 0.75, as in the authors' code),
    not its reciprocal. ``gamma=1`` equals TverskyLoss, including its gradient.
    For gamma != 1, an exactly perfect prediction has zero gradient, including
    the otherwise singular endpoint when gamma < 1.
    All other options, masking and distributed reduction follow TverskyLoss.
    """

    def __init__(self, *, gamma: float = 0.75, alpha: float = 0.3, beta: float = 0.7,
                 activation: Activation = "softmax", target_format: TargetFormat = "index",
                 include_background: bool = True, batch_dice: bool = False,
                 distributed: bool = False, smooth: float = 1e-5,
                 ignore_index: int | None = None, backend: Backend = "auto") -> None:
        super().__init__(alpha=alpha, beta=beta, activation=activation, target_format=target_format,
                         include_background=include_background, batch_dice=batch_dice,
                         distributed=distributed, smooth=smooth, ignore_index=ignore_index, backend=backend)
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be finite and positive")
        self.gamma = gamma

    def _overlap_power(self) -> float:
        return self.gamma


class GeneralizedDiceLoss(_OverlapLoss):
    """Class-volume-weighted Dice following Sudre et al. (2017).

    ``1 - (2 Σ_c w_c Σ p_c*g_c + smooth) / (Σ_c w_c Σ (p_c+g_c) + smooth)``,
    averaged over cases. ``weight_type="square"`` (default) uses inverse squared target
    volume; ``"simple"`` uses inverse volume and ``"uniform"`` uses unit weights.
    This is a ratio of class-weighted sums, unlike DiceLoss's mean of classwise ratios.
    With ``batch_dice=True``, accumulate volumes over the batch before computing weights
    and one ratio. Smoothing is added after class weighting.

    Absent classes receive the largest weight of the present selected classes, as in MONAI.
    With inverse-volume weighting, ``empty_target="zero"`` (default) gives zero loss and
    gradient if every selected class is absent, even with false-positive predictions.
    Use ``empty_target="uniform"`` to assign unit weights to those cases and penalize
    false positives. Zero-weight cases return zero also with ``smooth=0``. Other zero
    denominators remain NaN. Empty-target cases still count in the batch mean.

    Soft targets use the same product-based formula, including their volume-dependent
    weights; no implicit background channel is added for single-channel sigmoid inputs.
    Activation, target format, background selection and backends follow :class:`DiceLoss`.
    """

    def __init__(
        self,
        *,
        weight_type: WeightType = "square",
        empty_target: EmptyTarget = "zero",
        activation: Activation = "softmax",
        target_format: TargetFormat = "index",
        include_background: bool = True,
        batch_dice: bool = False,
        distributed: bool = False,
        smooth: float = 1e-5,
        ignore_index: int | None = None,
        backend: Backend = "auto",
    ) -> None:
        super().__init__(
            activation=activation, target_format=target_format, include_background=include_background,
            batch_dice=batch_dice, distributed=distributed, smooth=smooth, ignore_index=ignore_index, backend=backend,
        )
        if weight_type not in ("square", "simple", "uniform"):
            raise ValueError("weight_type must be 'square', 'simple' or 'uniform'")
        if empty_target not in ("zero", "uniform"):
            raise ValueError("empty_target must be 'zero' or 'uniform'")
        self.weight_type, self.empty_target = weight_type, empty_target

    def _generalized_config(self) -> tuple[WeightType, EmptyTarget]:
        return self.weight_type, self.empty_target


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
    Target formats, masking and backends follow :class:`DiceLoss`. With exclusions, average
    over included voxels (and channels for sigmoid); fully ignored batches return zero.
    ``distributed=True`` sums the numerator and denominator over the default process
    group, with the same DDP gradient and participation requirements as :class:`DiceLoss`.
    """

    def __init__(
        self,
        *,
        activation: Literal["softmax", "sigmoid"] = "softmax",
        target_format: TargetFormat = "index",
        gamma: float = 2.0,
        alpha: float | None = None,
        ignore_index: int | None = None,
        distributed: bool = False,
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
        self.distributed = distributed
        self.ignore_index = _check_ignore_index(ignore_index, target_format)

    def forward(self, input: Tensor, target: Tensor, *, mask: Tensor | None = None) -> Tensor:
        mask = _check_mask(mask, input)
        if _use_triton(self.backend, input, target, allow_empty=self.distributed):
            _check_input(input)
            _check_fused_target(target, input, self.target_format,
                                masked=mask is not None or self.ignore_index is not None or self.distributed)
            return _triton_losses.focal_loss(
                input, target, activation=self.activation, one_hot=self.target_format == "one-hot",
                gamma=self.gamma, alpha=self.alpha, mask=mask, ignore_index=self.ignore_index, distributed=self.distributed,
            )
        input, target, valid = _prepare_masked_input(
            input, target, self.target_format, self.ignore_index, mask,
        )
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
            return _voxel_mean(terms.sum(1), valid, channels=1, distributed=self.distributed)
        pos = F.logsigmoid(input)
        neg = F.logsigmoid(-input)
        positive = -truth * (self.gamma * neg).exp() * pos
        negative = -(1 - truth) * (self.gamma * pos).exp() * neg
        if self.alpha is not None:
            positive, negative = self.alpha * positive, (1 - self.alpha) * negative
        return _voxel_mean(positive + negative, valid, channels=input.shape[1], distributed=self.distributed)


class DiceCELoss(nn.Module):
    """``lambda_dice * DiceLoss + lambda_ce * CE`` on raw logits.

    CE is cross-entropy with ``activation="softmax"`` and binary cross-entropy with
    ``"sigmoid"``. It always covers every class, whatever ``include_background`` says (as in
    MONAI's ``DiceCELoss``). With exclusions, CE averages included voxels (and channels for
    sigmoid). ``class_weight`` optionally weights each CE class; it does not weight
    Dice. Softmax index targets normalize by the sum of included target weights,
    matching PyTorch CE. Soft targets normalize by voxel count; sigmoid weights
    both positive and negative terms and normalizes by voxel count times channels.
    Zero total target weight returns zero CE. Other arguments as in :class:`DiceLoss`.
    """

    class_weight: Tensor | None
    dice: _OverlapLoss

    def __init__(
        self,
        *,
        activation: Literal["softmax", "sigmoid"] = "softmax",
        target_format: TargetFormat = "index",
        include_background: bool = True,
        batch_dice: bool = False,
        distributed: bool = False,
        smooth: float = 1e-5,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        class_weight: Tensor | Sequence[float] | None = None,
        squared_pred: bool = False,
        ignore_index: int | None = None,
        backend: Backend = "auto",
    ) -> None:
        super().__init__()
        if activation not in ("softmax", "sigmoid"):
            raise ValueError(f"activation must be 'softmax' or 'sigmoid', got {activation!r}")
        self.dice = DiceLoss(
            squared_pred=squared_pred,
            activation=activation,
            target_format=target_format,
            include_background=include_background,
            batch_dice=batch_dice,
            distributed=distributed,
            smooth=smooth,
            ignore_index=ignore_index,
            backend=backend,
        )
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        _check_lambdas(lambda_dice, lambda_ce)
        self.register_buffer("class_weight", _class_weight(class_weight))

    def _focal_config(self) -> tuple[float, float | None] | None:
        return None

    def forward(self, input: Tensor, target: Tensor, *, mask: Tensor | None = None) -> Tensor:
        mask = _check_mask(mask, input)
        weight = self.class_weight
        if weight is not None:
            _check_input(input)
            if weight.numel() != input.shape[1]:
                raise ValueError("class_weight must have one entry per input channel")
            weight = weight.to(device=input.device, dtype=torch.float64 if input.dtype == torch.float64 else torch.float32)
        if _use_triton(self.dice.backend, input, target, allow_empty=self.dice.distributed):
            return self.dice._fused(
                input, target, mask=mask, lambda_dice=self.lambda_dice, lambda_ce=self.lambda_ce,
                cross_entropy=self.lambda_ce != 0, class_weight=weight, focal=self._focal_config(),
            )
        dice = self.dice(input, target, mask=mask) if self.lambda_dice else None
        if not self.lambda_ce:
            assert dice is not None
            return self.lambda_dice * dice
        ce = self._pointwise(input, target, mask, weight)
        return self.lambda_ce * ce if dice is None else self.lambda_dice * dice + self.lambda_ce * ce

    def _pointwise(self, input: Tensor, target: Tensor, mask: Tensor | None, weight: Tensor | None) -> Tensor:
        input, target, valid = _prepare_masked_input(
            input, target, self.dice.target_format, self.dice.ignore_index, mask,
        )
        # Weighted index CE has a target-weight denominator; use the unreduced terms
        # to define zero-weight batches and exclusions explicitly.
        reduction = "mean" if valid is None and weight is None and not self.dice.distributed else "none"
        if self.dice.activation == "sigmoid":
            onehot = _one_hot(target, input, self.dice.target_format).to(input.dtype)
            shaped_weight = None if weight is None else weight.reshape(1, -1, *([1] * (input.ndim - 2)))
            ce = F.binary_cross_entropy_with_logits(input, onehot, weight=shaped_weight, reduction=reduction)
        elif self.dice.target_format == "index":
            _check_index_target(target, input)
            ce = F.cross_entropy(input, target.long(), weight=weight, reduction=reduction)
            if weight is not None:
                target_weights = weight[target.long()]
                if valid is not None:
                    target_weights = torch.where(valid, target_weights, 0)
                    ce = torch.where(valid, ce, 0)
                denominator = target_weights.sum()
                return _sum_mean(ce.sum(), denominator, self.dice.distributed)
        else:
            ce = F.cross_entropy(input, _one_hot(target, input, "one-hot").to(input.dtype),
                                 weight=weight, reduction=reduction)
        if reduction == "none":
            ce = _voxel_mean(ce, valid, channels=input.shape[1] if self.dice.activation == "sigmoid" else 1,
                             distributed=self.dice.distributed)
        return ce


class GeneralizedDiceCELoss(DiceCELoss):
    """Generalized Dice plus CE/BCE, sharing the fused GPU passes.

    Generalized-Dice options follow :class:`GeneralizedDiceLoss`; CE weights,
    coefficients, masking and normalization follow :class:`DiceCELoss`.
    """

    def __init__(
        self, *, weight_type: WeightType = "square", empty_target: EmptyTarget = "zero",
        activation: Literal["softmax", "sigmoid"] = "softmax", target_format: TargetFormat = "index",
        include_background: bool = True, batch_dice: bool = False, distributed: bool = False, smooth: float = 1e-5,
        lambda_dice: float = 1.0, lambda_ce: float = 1.0,
        class_weight: Tensor | Sequence[float] | None = None,
        ignore_index: int | None = None, backend: Backend = "auto",
    ) -> None:
        super().__init__(activation=activation, target_format=target_format, include_background=include_background,
                         batch_dice=batch_dice, distributed=distributed, smooth=smooth, lambda_dice=lambda_dice, lambda_ce=lambda_ce,
                         class_weight=class_weight, ignore_index=ignore_index, backend=backend)
        self.dice = GeneralizedDiceLoss(
            weight_type=weight_type, empty_target=empty_target, activation=activation, target_format=target_format,
            include_background=include_background, batch_dice=batch_dice, distributed=distributed, smooth=smooth,
            ignore_index=ignore_index, backend=backend,
        )


class DiceFocalLoss(DiceCELoss):
    """``lambda_dice * DiceLoss + lambda_focal * FocalLoss`` with shared GPU passes.

    Overlap options follow :class:`DiceLoss`; ``gamma`` and ``alpha`` and pointwise
    normalization follow :class:`FocalLoss`. A zero coefficient skips that term.
    """

    def __init__(
        self, *, activation: Literal["softmax", "sigmoid"] = "softmax", target_format: TargetFormat = "index",
        include_background: bool = True, batch_dice: bool = False, distributed: bool = False, smooth: float = 1e-5,
        lambda_dice: float = 1.0, lambda_focal: float = 1.0, gamma: float = 2.0, alpha: float | None = None,
        ignore_index: int | None = None, backend: Backend = "auto", squared_pred: bool = False,
    ) -> None:
        super().__init__(activation=activation, target_format=target_format, include_background=include_background,
                         batch_dice=batch_dice, distributed=distributed, smooth=smooth, lambda_dice=lambda_dice, lambda_ce=lambda_focal,
                         ignore_index=ignore_index, backend=backend, squared_pred=squared_pred)
        self.focal = FocalLoss(activation=activation, target_format=target_format, gamma=gamma, alpha=alpha,
                               ignore_index=ignore_index, distributed=distributed, backend="torch")
        self.lambda_focal = lambda_focal

    @property
    def lambda_focal(self) -> float:
        return self.lambda_ce

    @lambda_focal.setter
    def lambda_focal(self, value: float) -> None:
        self.lambda_ce = value

    def _focal_config(self) -> tuple[float, float | None]:
        return self.focal.gamma, self.focal.alpha

    def _pointwise(self, input: Tensor, target: Tensor, mask: Tensor | None, weight: Tensor | None) -> Tensor:
        return self.focal(input, target, mask=mask)


def _check_lambdas(overlap: float, pointwise: float) -> None:
    _nonnegative_finite("lambda_dice", overlap)
    _nonnegative_finite("pointwise coefficient", pointwise)
    if overlap == 0 and pointwise == 0:
        raise ValueError("at least one loss coefficient must be positive")


def _class_weight(value: Tensor | Sequence[float] | None) -> Tensor | None:
    if value is None:
        return None
    weight = torch.as_tensor(value, dtype=torch.float64).detach().clone()
    if weight.ndim != 1 or not weight.numel() or not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
        raise ValueError("class_weight must be a nonempty vector of finite nonnegative weights")
    return weight


def _overlap_mean(numerator: Tensor, denominator: Tensor, valid_counts: Tensor | None,
                  *, power: float = 1.0) -> Tensor:
    if valid_counts is None:
        if power == 1:
            return 1 - (numerator / denominator).mean()
        return _loss_reductions.overlap_power(1 - numerator / denominator, power).mean()
    numerator, denominator, scale = masked_overlap_terms(numerator, denominator, valid_counts)
    return (_loss_reductions.overlap_power(1 - numerator / denominator, power) * scale).sum()


def _global_overlap(
    intersection: Tensor, prediction: Tensor, truth: Tensor, count: Tensor | None, input: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    count = (torch.tensor(input.numel() // input.shape[1], device=input.device, dtype=torch.float64)
             if count is None else count.double())
    packed = torch.cat((intersection.double(), prediction.double(), truth.double(), count.reshape(1)))
    packed = sum_ranks(packed)
    classes = intersection.numel()
    return packed[:classes], packed[classes:2*classes], packed[2*classes:3*classes], packed[-1]


def _sum_mean(numerator: Tensor, denominator: Tensor, distributed: bool) -> Tensor:
    dtype = numerator.dtype
    if distributed:
        numerator, denominator = sum_ranks(torch.stack((numerator.double(), denominator.double()))).unbind()
    return (numerator / torch.where(denominator > 0, denominator, 1)).to(dtype)


def _voxel_mean(terms: Tensor, valid: Tensor | None, *, channels: int, distributed: bool = False) -> Tensor:
    if valid is None:
        if not distributed:
            return terms.mean()
        return _sum_mean(terms.sum(), terms.new_tensor(terms.numel(), dtype=torch.float64), True)
    included = valid[:, None] if terms.ndim == valid.ndim + 1 else valid
    return _sum_mean(torch.where(included, terms, 0).sum(), valid.sum() * channels, distributed)


def _check_ignore_index(ignore_index: int | None, target_format: TargetFormat) -> int | None:
    if ignore_index is None:
        return None
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, Integral):
        raise TypeError("ignore_index must be an integer or None")
    if not -(2**63) <= ignore_index < 2**63:
        raise ValueError("ignore_index must fit in a signed 64-bit integer")
    if target_format != "index":
        raise ValueError("ignore_index requires target_format='index'; use mask for one-hot targets")
    return int(ignore_index)


def _check_mask(mask: Tensor | None, input: Tensor) -> Tensor | None:
    if mask is None:
        return None
    _check_input(input)
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
        raise TypeError("mask must be a boolean tensor")
    if mask.device != input.device:
        raise ValueError("mask and input must be on the same device")
    shape = (input.shape[0], *input.shape[2:])
    if mask.shape == (input.shape[0], 1, *input.shape[2:]):
        mask = mask[:, 0]
    if mask.shape != shape:
        raise ValueError(f"mask must have shape {shape} or {(input.shape[0], 1, *input.shape[2:])}")
    return mask


def _prepare_masked_input(
    input: Tensor, target: Tensor, target_format: TargetFormat, ignore_index: int | None,
    mask: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    input = _upcast(input)
    valid = mask
    if ignore_index is not None:
        _check_index_target(target, input)
        # Widen before scalar comparison: e.g. uint8 would wrap ignore_index=256 to label 0.
        keep = target.long() != ignore_index
        valid = keep if valid is None else valid & keep
    if valid is None:
        return input, target, None
    if target_format == "index":
        _check_index_target(target, input)
        target = torch.where(valid, target, 0)
    else:
        target = _one_hot(target, input, target_format)  # Validate shape/device before broadcasting.
        target = torch.where(valid[:, None], target, 0)
    # Sanitize before activation: masking NaN afterward would still poison its backward.
    return torch.where(valid[:, None], input, 0), target, valid


def _check_backend(backend: str) -> None:
    if backend not in ("auto", "triton", "torch"):
        raise ValueError(f"backend must be 'auto', 'triton' or 'torch', got {backend!r}")


def _check_target_format(target_format: str) -> None:
    if target_format not in ("index", "one-hot"):
        raise ValueError(f"target_format must be 'index' or 'one-hot', got {target_format!r}")


def _nonnegative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")


def _check_fused_target(target: Tensor, input: Tensor, target_format: TargetFormat, *, masked: bool = False) -> None:
    if target_format == "index":
        _check_index_target(target, input)
        if not masked and not torch.compiler.is_compiling():
            # Eager execution raises. Compiled kernels avoid a host sync and return NaN instead.
            _check_index_values(target, input.shape[1])
    else:
        _one_hot(target, input, target_format)


def _use_triton(backend: Backend, input: Tensor, target: Tensor, *, allow_empty: bool = False) -> bool:
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
    return allow_empty or input.numel() > 0


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
