"""Sorting-based losses using PyTorch's native CPU/CUDA selection kernels."""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from exacta.losses.segmentation import (
    Backend, DiceLoss, _check_ignore_index, _check_index_target, _check_index_values,
    _check_lambdas, _check_mask, _prepare_masked_input,
    _upcast,
)


def _indexed(input: Tensor, target: Tensor, ignore_index: int | None,
             mask: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
    mask = _check_mask(mask, input)
    input, target, valid = _prepare_masked_input(input, target, "index", ignore_index, mask)
    _check_index_target(target, input)
    if not torch.compiler.is_compiling():
        _check_index_values(target, input.shape[1])
    if valid is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    return input, target.long(), valid


class TopKCELoss(nn.Module):
    """Mean softmax cross entropy of the hardest ``ceil(fraction * N)`` valid voxels.

    N counts valid voxels across the local batch. Targets are integer class indices;
    ignored/masked voxels never enter selection. An entirely ignored batch returns
    differentiable zero. ``fraction=1`` is ordinary unweighted cross entropy.
    Native ``topk`` runs on the input device, without transferring the selected count
    to the CPU. Tied losses may receive different valid subgradients on different devices.
    No global distributed top-k is implied by DDP.
    """

    def __init__(self, *, fraction: float = 0.1, ignore_index: int | None = None) -> None:
        super().__init__()
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("fraction must lie in (0, 1]")
        self.fraction = fraction
        self.ignore_index = _check_ignore_index(ignore_index, "index")

    def forward(self, input: Tensor, target: Tensor, *, mask: Tensor | None = None) -> Tensor:
        input, target, valid = _indexed(input, target, self.ignore_index, mask)
        if not target.numel():
            return input.sum() * 0
        terms = F.cross_entropy(input, target, reduction="none")
        if self.fraction == 1:
            return torch.where(valid, terms, 0).sum() / valid.sum().clamp_min(1)
        upper = math.ceil(self.fraction * target.numel())
        ranked = torch.topk(torch.where(valid, terms, -torch.inf).flatten(), upper, sorted=True).values
        count = (valid.sum().double() * self.fraction).ceil().long()
        selected = torch.arange(upper, device=input.device) < count
        return torch.where(selected, ranked, 0).sum() / count.clamp_min(1)


class DiceTopKCELoss(nn.Module):
    """Weighted sum of soft Dice and TopKCELoss, with integer targets and softmax logits.

    Dice uses Exacta's fused GPU backend where available; top-k uses native PyTorch
    CPU/CUDA selection. Coefficients are not normalized. Zero-weight terms are skipped.
    """

    def __init__(self, *, fraction: float = 0.1, lambda_dice: float = 1.0,
                 lambda_ce: float = 1.0, include_background: bool = True,
                 batch_dice: bool = False, smooth: float = 1e-5,
                 ignore_index: int | None = None, backend: Backend = "auto", squared_pred: bool = False) -> None:
        super().__init__()
        _check_lambdas(lambda_dice, lambda_ce)
        self.lambda_dice, self.lambda_ce = lambda_dice, lambda_ce
        self.dice = DiceLoss(include_background=include_background, batch_dice=batch_dice,
                             smooth=smooth, ignore_index=ignore_index, backend=backend, squared_pred=squared_pred)
        self.ce = TopKCELoss(fraction=fraction, ignore_index=ignore_index)

    def forward(self, input: Tensor, target: Tensor, *, mask: Tensor | None = None) -> Tensor:
        if self.lambda_dice == 0:
            return self.lambda_ce * self.ce(input, target, mask=mask)
        value = self.lambda_dice * self.dice(input, target, mask=mask)
        if self.lambda_ce:
            value = value + self.lambda_ce * self.ce(input, target, mask=mask)
        return value


class LovaszSoftmaxLoss(nn.Module):
    """Lovasz extension of multiclass Jaccard error (Berman et al., 2018).

    Integer targets, softmax logits by default (``from_logits=False`` accepts
    probabilities). ``classes='present'`` averages only classes present in the valid
    target; ``'all'`` includes absent classes. ``per_image=False`` sorts over the local
    batch; True computes one loss per case and omits entirely ignored cases.
    Channel 0 can be excluded. Sorting is stable, so ties use flattened voxel order.
    Masking does not require a device-to-host transfer or a full one-hot target.
    """

    def __init__(self, *, from_logits: bool = True, classes: Literal["present", "all"] = "present",
                 per_image: bool = False, include_background: bool = True,
                 ignore_index: int | None = None) -> None:
        super().__init__()
        if classes not in ("present", "all"):
            raise ValueError("classes must be 'present' or 'all'")
        self.from_logits, self.classes, self.per_image = from_logits, classes, per_image
        self.include_background = include_background
        self.ignore_index = _check_ignore_index(ignore_index, "index")

    def forward(self, input: Tensor, target: Tensor, *, mask: Tensor | None = None) -> Tensor:
        input, target, valid = _indexed(input, target, self.ignore_index, mask)
        first = 0 if self.include_background else 1
        if input.shape[1] <= first:
            raise ValueError("include_background=False needs at least 2 channels")
        probs = input.softmax(1) if self.from_logits else input
        if not target.numel():
            return probs.sum() * 0
        if self.per_image:
            values = torch.stack([self._flat(p, t, v, first) for p, t, v in zip(probs, target, valid)])
            active = valid.flatten(1).any(1)
            return torch.where(active, values, 0).sum() / active.sum().clamp_min(1)
        return self._flat(probs.movedim(1, 0), target, valid, first)

    def _flat(self, probs: Tensor, target: Tensor, valid: Tensor, first: int) -> Tensor:
        target, valid = target.flatten(), valid.flatten()
        losses, active = [], []
        for c in range(first, probs.shape[0]):
            foreground = ((target == c) & valid).to(probs.dtype)
            errors = (foreground - probs[c].flatten()).abs()
            ranked, order = torch.sort(torch.where(valid, errors, -1), descending=True, stable=True)
            fg = foreground[order]
            keep = valid[order]
            total = foreground.sum()
            intersection = total - fg.cumsum(0)
            union = total + (keep.to(probs.dtype) - fg).cumsum(0)
            jaccard = 1 - intersection / union.clamp_min(1)
            gradient = torch.cat((jaccard[:1], jaccard[1:] - jaccard[:-1]))
            losses.append((torch.where(keep, ranked, 0) * gradient).sum())
            active.append((total > 0) if self.classes == "present" else valid.any())
        selected = torch.stack(active)
        return torch.where(selected, torch.stack(losses), 0).sum() / selected.sum().clamp_min(1)


class LovaszHingeLoss(nn.Module):
    """Binary Lovasz-hinge on real-valued logits (Berman et al., 2018).

    Input is (B,1,*spatial) with binary target (B,*spatial). ``multilabel=True``
    accepts independent channels and a binary target matching the input shape.
    No sigmoid is applied. ``per_image=True`` averages observed case/label losses;
    False pools the local batch per label before averaging observed labels.
    Absent positive classes still penalize false positives. An unobserved label
    is omitted; an entirely excluded input returns differentiable zero.

    Masks may be (B,*spatial), (B,1,*spatial), or, for multilabel, (B,C,*spatial).
    ``ignore_index`` is for integer binary targets only. Sorting is stable: ties
    use flattened voxel order, defining a reproducible valid subgradient. At a
    zero hinge margin, ReLU gives zero derivative. Native PyTorch sort runs on
    CPU/CUDA; no global distributed sorting is implied by DDP.
    """

    def __init__(self, *, multilabel: bool = False, per_image: bool = True,
                 ignore_index: int | None = None) -> None:
        super().__init__()
        if not isinstance(multilabel, bool) or not isinstance(per_image, bool):
            raise TypeError("multilabel and per_image must be bools")
        if multilabel and ignore_index is not None:
            raise ValueError("multilabel targets use mask instead of ignore_index")
        self.multilabel, self.per_image = multilabel, per_image
        self.ignore_index = _check_ignore_index(ignore_index, "index")

    def forward(self, input: Tensor, target: Tensor, *, mask: Tensor | None = None) -> Tensor:
        input = _upcast(input)
        if target.device != input.device:
            raise ValueError("input and target must be on the same device")
        expected = input.shape if self.multilabel else (input.shape[0], *input.shape[2:])
        if not self.multilabel and input.shape[1] != 1:
            raise ValueError("binary Lovasz-hinge requires one channel; use multilabel=True for independent labels")
        if target.shape != expected:
            raise ValueError(f"target must have shape {tuple(expected)}")
        if target.is_complex():
            raise TypeError("target must contain real binary values")
        if not self.multilabel:
            if self.ignore_index is not None:
                _check_index_target(target, input)
            target = target[:, None]
        if mask is None:
            valid = torch.ones_like(target, dtype=torch.bool)
        else:
            if mask.dtype != torch.bool:
                raise TypeError("mask must be a boolean tensor")
            if mask.device != input.device:
                raise ValueError("mask and input must be on the same device")
            if mask.shape == (input.shape[0], *input.shape[2:]):
                mask = mask[:, None]
            if mask.shape not in (input.shape, (input.shape[0], 1, *input.shape[2:])):
                raise ValueError("mask must have shape (B,*spatial), (B,1,*spatial) or (B,C,*spatial)")
            valid = mask.expand_as(target)
        if self.ignore_index is not None:
            valid = valid & (target.long() != self.ignore_index)
        if not torch.compiler.is_compiling():
            bad = torch.stack(((((target != 0) & (target != 1)) & valid).any(),
                               ((~torch.isfinite(input)) & valid).any())).tolist()
            if bad[0]:
                raise ValueError("included targets must be binary (0/1)")
            if bad[1]:
                raise ValueError("included logits must be finite")
        clean = torch.where(valid, input, 0)
        if not input.numel():
            return clean.sum() * 0
        positive = (target == 1) & valid
        if self.per_image:
            clean, positive, valid = (x.flatten(2).flatten(0, 1) for x in (clean, positive, valid))
        else:
            clean, positive, valid = (x.movedim(1, 0).flatten(1) for x in (clean, positive, valid))
        errors = 1 - clean * (2 * positive.to(clean.dtype) - 1)
        ranked, order = torch.sort(torch.where(valid, errors, -torch.inf), descending=True, stable=True)
        fg, keep = positive.gather(1, order), valid.gather(1, order)
        total = positive.sum(1, keepdim=True)
        intersection = total - fg.cumsum(1)
        union = total + (keep.long() - fg.long()).cumsum(1)
        jaccard = 1 - intersection.double() / union.clamp_min(1)
        gradient = torch.cat((jaccard[:, :1], jaccard[:, 1:] - jaccard[:, :-1]), 1).to(clean.dtype)
        values = (torch.where(keep, ranked, 0).relu() * gradient).sum(1)
        active = valid.any(1)
        return torch.where(active, values, 0).sum() / active.sum().clamp_min(1)
