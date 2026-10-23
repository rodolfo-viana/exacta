"""Overlap metrics computed from per-case, per-class confusion counts."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from exacta._inputs import InputFormat, Segmentations, like_input, to_torch, validate

# Voxels per bincount call, bounding the int64 index buffer to 128 MiB.
_CHUNK = 1 << 24


def dice(
    pred: Any,
    target: Any,
    *,
    input_format: InputFormat = "index",
    num_classes: int | None = None,
    include_background: bool = True,
    both_empty: float = math.nan,
) -> Any:
    """Dice coefficient ``2|P ∩ T| / (|P| + |T|)`` for every case and class.

    Args:
        pred, target: Hard segmentations. With ``input_format="index"``, class indices of shape
            (B, H, W) or (B, D, H, W); with ``"one-hot"``, binary masks of shape (B, C, *spatial).
            Any array library that supports DLPack works (PyTorch, NumPy, CuPy, JAX).
        num_classes: Number of classes; required with ``input_format="index"``.
        include_background: If False, drop class/channel 0.
        both_empty: Value when the class is absent from both prediction and target.

    Returns:
        A float64 array of shape (B, C), in the array library of ``pred``.
    """
    seg = validate(
        to_torch(pred),
        to_torch(target),
        input_format=input_format,
        num_classes=num_classes,
        include_background=include_background,
    )
    tp, fp, fn = _counts(seg)
    return like_input(_ratio(2 * tp, 2 * tp + fp + fn, both_empty), pred)


def iou(
    pred: Any,
    target: Any,
    *,
    input_format: InputFormat = "index",
    num_classes: int | None = None,
    include_background: bool = True,
    both_empty: float = math.nan,
) -> Any:
    """Intersection over union (Jaccard index) ``|P ∩ T| / |P ∪ T|`` for every case and class.

    Arguments and return value as in :func:`dice`.
    """
    seg = validate(
        to_torch(pred),
        to_torch(target),
        input_format=input_format,
        num_classes=num_classes,
        include_background=include_background,
    )
    tp, fp, fn = _counts(seg)
    return like_input(_ratio(tp, tp + fp + fn, both_empty), pred)


def _counts(seg: Segmentations) -> tuple[Tensor, Tensor, Tensor]:
    """True positives, false positives and false negatives, each (B, C)."""
    if seg.one_hot:
        p, t = seg.pred[:, seg.first_class :], seg.target[:, seg.first_class :]
        tp = _count(p & t)
        return tp, _count(p) - tp, _count(t) - tp
    cm = _confusion(seg.pred, seg.target, seg.num_classes)
    tp = cm.diagonal(dim1=1, dim2=2)
    keep = slice(seg.first_class, None)  # drop excluded classes only after summing full rows/columns
    return tp[:, keep], (cm.sum(2) - tp)[:, keep], (cm.sum(1) - tp)[:, keep]


def _confusion(pred: Tensor, target: Tensor, num_classes: int) -> Tensor:
    """(B, C, C) confusion matrices, ``[b, i, j]`` = voxels predicted ``i`` with target ``j``.

    One bincount pass per case: O(voxels) memory however many classes there are.
    """
    batch = pred.shape[0]
    p, t = pred.reshape(batch, -1), target.reshape(batch, -1)
    cm = torch.zeros(batch, num_classes * num_classes, dtype=torch.int64, device=pred.device)
    for b in range(batch):
        for start in range(0, p.shape[1], _CHUNK):
            index = p[b, start : start + _CHUNK].long() * num_classes + t[b, start : start + _CHUNK]
            cm[b] += torch.bincount(index, minlength=num_classes * num_classes)
    return cm.view(batch, num_classes, num_classes)


def _count(mask: Tensor) -> Tensor:
    """Voxels per (case, class). Summing uint8 into int32 is ~2x faster on CPU than ``bool.sum``."""
    flat = mask.flatten(2).view(torch.uint8)
    dtype = torch.int32 if flat.shape[-1] < 2**31 else torch.int64
    return flat.sum(-1, dtype=dtype).long()


def _ratio(num: Tensor, den: Tensor, both_empty: float) -> Tensor:
    num, den = num.double(), den.double()
    return torch.where(den > 0, num / den.clamp_min(1), torch.full_like(den, both_empty))
