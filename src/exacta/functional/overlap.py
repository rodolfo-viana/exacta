"""Overlap and volume metrics, computed from per-case, per-class confusion counts.

TP, FP, FN and TN count voxels: predicted and in the target, predicted only, in the target
only, and in neither. |P| = TP + FP and |T| = TP + FN are the predicted and target volumes, in
voxels; the metrics here are ratios, so voxel spacing does not change them.

Where a ratio's denominator is zero, tools disagree. MONAI returns NaN. MedPy 0.5.2 means to
return 0 (and raises for the relative volume difference), and does with NumPy 1.x; with NumPy 2,
``np.count_nonzero`` returns a NumPy integer, dividing it by zero no longer raises, and MedPy
returns NaN. ``zero_division`` (``both_empty`` for Dice and IoU, whose denominator is zero exactly
when both masks are empty) sets the value; NaN by default, which the stateful metrics skip when
averaging.
"""

from __future__ import annotations

import math
from typing import Any, Literal

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


def sensitivity(
    pred: Any,
    target: Any,
    *,
    input_format: InputFormat = "index",
    num_classes: int | None = None,
    include_background: bool = True,
    zero_division: float = math.nan,
) -> Any:
    """Sensitivity (recall, true positive rate) ``TP / (TP + FN) = |P ∩ T| / |T|``.

    Args:
        zero_division: Value when the class is absent from the target.
        Others: As in :func:`dice`.
    """
    tp, fp, fn, tn = _confusion_counts(pred, target, input_format, num_classes, include_background)
    return like_input(_ratio(tp, tp + fn, zero_division), pred)


def precision(
    pred: Any,
    target: Any,
    *,
    input_format: InputFormat = "index",
    num_classes: int | None = None,
    include_background: bool = True,
    zero_division: float = math.nan,
) -> Any:
    """Precision (positive predictive value) ``TP / (TP + FP) = |P ∩ T| / |P|``.

    Args:
        zero_division: Value when the class is absent from the prediction.
        Others: As in :func:`dice`.
    """
    tp, fp, fn, tn = _confusion_counts(pred, target, input_format, num_classes, include_background)
    return like_input(_ratio(tp, tp + fp, zero_division), pred)


def specificity(
    pred: Any,
    target: Any,
    *,
    input_format: InputFormat = "index",
    num_classes: int | None = None,
    include_background: bool = True,
    zero_division: float = math.nan,
) -> Any:
    """Specificity (true negative rate) ``TN / (TN + FP)``.

    Args:
        zero_division: Value when the target covers the whole image (no negatives).
        Others: As in :func:`dice`.
    """
    tp, fp, fn, tn = _confusion_counts(pred, target, input_format, num_classes, include_background)
    return like_input(_ratio(tn, tn + fp, zero_division), pred)


def volume_similarity(
    pred: Any,
    target: Any,
    *,
    convention: Literal["taha", "itk"] = "taha",
    input_format: InputFormat = "index",
    num_classes: int | None = None,
    include_background: bool = True,
    zero_division: float = math.nan,
) -> Any:
    """Volume similarity: how close the predicted and target volumes are, whatever the overlap.

    Two different quantities go by this name:

    * ``convention="taha"``: ``1 - ||P| - |T|| / (|P| + |T|)``, in [0, 1] with 1 for equal
      volumes (Taha & Hanbury, BMC Medical Imaging 2015).
    * ``convention="itk"``: ``2 (|P| - |T|) / (|P| + |T|)``, signed, in [-2, 2] with 0 for equal
      volumes (SimpleITK's ``LabelOverlapMeasuresImageFilter.GetVolumeSimilarity``).

    Args:
        zero_division: Value when the class is absent from both prediction and target.
        Others: As in :func:`dice`.
    """
    if convention not in ("taha", "itk"):
        raise ValueError(f"convention must be 'taha' or 'itk', got {convention!r}")
    tp, fp, fn, tn = _confusion_counts(pred, target, input_format, num_classes, include_background)
    size_p, size_t = tp + fp, tp + fn
    if convention == "taha":
        similarity = 1 - (size_p - size_t).abs().double() / (size_p + size_t).clamp_min(1)
    else:
        similarity = 2 * (size_p - size_t).double() / (size_p + size_t).clamp_min(1)
    both_empty = size_p + size_t == 0
    return like_input(torch.where(both_empty, torch.full_like(similarity, zero_division), similarity), pred)


def relative_volume_difference(
    pred: Any,
    target: Any,
    *,
    absolute: bool = False,
    input_format: InputFormat = "index",
    num_classes: int | None = None,
    include_background: bool = True,
    zero_division: float = math.nan,
) -> Any:
    """Relative volume difference ``(|P| - |T|) / |T|``, as MedPy's ``ravd``.

    Positive when the prediction is too large. ``absolute=True`` gives ``||P| - |T|| / |T|``,
    the unsigned form some challenges report (often as a percentage).

    Args:
        zero_division: Value when the class is absent from the target (MedPy raises).
        Others: As in :func:`dice`.
    """
    tp, fp, fn, tn = _confusion_counts(pred, target, input_format, num_classes, include_background)
    difference = fp - fn  # = |P| - |T|
    return like_input(_ratio(difference.abs() if absolute else difference, tp + fn, zero_division), pred)


def _confusion_counts(
    pred: Any, target: Any, input_format: InputFormat, num_classes: int | None, include_background: bool
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """TP, FP, FN and TN, each (B, C)."""
    seg = validate(
        to_torch(pred),
        to_torch(target),
        input_format=input_format,
        num_classes=num_classes,
        include_background=include_background,
    )
    tp, fp, fn = _counts(seg)
    voxels = math.prod(seg.pred.shape[seg.pred.ndim - seg.ndim :])
    return tp, fp, fn, voxels - tp - fp - fn


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
