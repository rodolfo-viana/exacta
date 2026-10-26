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
from numbers import Integral
from typing import Any, Literal

import torch
from torch import Tensor

from exacta import _native, _triton_counts
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
    ignore_index: int | None = None,
    mask: Any = None,
    both_empty: float = math.nan,
) -> Any:
    """Dice coefficient ``2|P ∩ T| / (|P| + |T|)`` for every case and class.

    Args:
        pred, target: Hard segmentations. With ``input_format="index"``, class indices of shape
            (B, H, W) or (B, D, H, W); with ``"one-hot"``, binary masks of shape (B, C, *spatial).
            Any array library that supports DLPack works (PyTorch, NumPy, CuPy, JAX).
        num_classes: Number of classes; required with ``input_format="index"``.
        include_background: If False, drop class/channel 0.
        ignore_index: Target label to exclude before validating or counting either input;
            index inputs only. This does not remove the corresponding class from the output.
        mask: Boolean array on the input device, shape (B, *spatial) or (B, 1, *spatial).
            True includes a voxel in every channel; False excludes it from TP, FP, FN and TN.
            Combined with ``ignore_index`` by intersection. Excluded values are not validated.
        both_empty: Value when the class is absent from both prediction and target.

    Returns:
        A float64 array of shape (B, C), in the array library of ``pred``. Fully excluded
        cases follow ``both_empty`` (NaN by default); other overlap metrics use their
        corresponding zero-denominator policy.
    """
    counts = _confusion_counts(
        pred, target, input_format, num_classes, include_background, ignore_index=ignore_index, mask=mask,
    )
    return like_input(_score_counts("dice", counts, both_empty=both_empty), pred)


def iou(
    pred: Any,
    target: Any,
    *,
    input_format: InputFormat = "index",
    num_classes: int | None = None,
    include_background: bool = True,
    ignore_index: int | None = None,
    mask: Any = None,
    both_empty: float = math.nan,
) -> Any:
    """Intersection over union (Jaccard index) ``|P ∩ T| / |P ∪ T|`` for every case and class.

    Arguments and return value as in :func:`dice`.
    """
    counts = _confusion_counts(
        pred, target, input_format, num_classes, include_background, ignore_index=ignore_index, mask=mask,
    )
    return like_input(_score_counts("iou", counts, both_empty=both_empty), pred)


def sensitivity(
    pred: Any,
    target: Any,
    *,
    input_format: InputFormat = "index",
    num_classes: int | None = None,
    include_background: bool = True,
    ignore_index: int | None = None,
    mask: Any = None,
    zero_division: float = math.nan,
) -> Any:
    """Sensitivity (recall, true positive rate) ``TP / (TP + FN) = |P ∩ T| / |T|``.

    Args:
        zero_division: Value when the class is absent from the target.
        Others: As in :func:`dice`.
    """
    counts = _confusion_counts(
        pred, target, input_format, num_classes, include_background, ignore_index=ignore_index, mask=mask,
    )
    return like_input(_score_counts("sensitivity", counts, zero_division=zero_division), pred)


def precision(
    pred: Any,
    target: Any,
    *,
    input_format: InputFormat = "index",
    num_classes: int | None = None,
    include_background: bool = True,
    ignore_index: int | None = None,
    mask: Any = None,
    zero_division: float = math.nan,
) -> Any:
    """Precision (positive predictive value) ``TP / (TP + FP) = |P ∩ T| / |P|``.

    Args:
        zero_division: Value when the class is absent from the prediction.
        Others: As in :func:`dice`.
    """
    counts = _confusion_counts(
        pred, target, input_format, num_classes, include_background, ignore_index=ignore_index, mask=mask,
    )
    return like_input(_score_counts("precision", counts, zero_division=zero_division), pred)


def specificity(
    pred: Any,
    target: Any,
    *,
    input_format: InputFormat = "index",
    num_classes: int | None = None,
    include_background: bool = True,
    ignore_index: int | None = None,
    mask: Any = None,
    zero_division: float = math.nan,
) -> Any:
    """Specificity (true negative rate) ``TN / (TN + FP)``.

    Args:
        zero_division: Value when the target covers the whole image (no negatives).
        Others: As in :func:`dice`.
    """
    counts = _confusion_counts(
        pred, target, input_format, num_classes, include_background, ignore_index=ignore_index, mask=mask,
    )
    return like_input(_score_counts("specificity", counts, zero_division=zero_division), pred)


def volume_similarity(
    pred: Any,
    target: Any,
    *,
    convention: Literal["taha", "itk"] = "taha",
    input_format: InputFormat = "index",
    num_classes: int | None = None,
    include_background: bool = True,
    ignore_index: int | None = None,
    mask: Any = None,
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
    counts = _confusion_counts(
        pred, target, input_format, num_classes, include_background, ignore_index=ignore_index, mask=mask,
    )
    return like_input(_score_counts("volume_similarity", counts, zero_division=zero_division, convention=convention), pred)


def relative_volume_difference(
    pred: Any,
    target: Any,
    *,
    absolute: bool = False,
    input_format: InputFormat = "index",
    num_classes: int | None = None,
    include_background: bool = True,
    ignore_index: int | None = None,
    mask: Any = None,
    zero_division: float = math.nan,
) -> Any:
    """Relative volume difference ``(|P| - |T|) / |T|``, as MedPy's ``ravd``.

    Positive when the prediction is too large. ``absolute=True`` gives ``||P| - |T|| / |T|``,
    the unsigned form some challenges report (often as a percentage).

    Args:
        zero_division: Value when the class is absent from the target (MedPy raises).
        Others: As in :func:`dice`.
    """
    counts = _confusion_counts(
        pred, target, input_format, num_classes, include_background, ignore_index=ignore_index, mask=mask,
    )
    return like_input(_score_counts("relative_volume_difference", counts, zero_division=zero_division, absolute=absolute), pred)


def _score_counts(
    name: str, counts: tuple[Tensor, Tensor, Tensor, Tensor], *,
    both_empty: float = math.nan, zero_division: float = math.nan,
    convention: str = "taha", absolute: bool = False,
) -> Tensor:
    """Reduce shared integer statistics without rescanning a segmentation."""
    tp, fp, fn, tn = counts
    if name == "dice":
        return _ratio(2 * tp, 2 * tp + fp + fn, both_empty)
    if name == "iou":
        return _ratio(tp, tp + fp + fn, both_empty)
    if name == "sensitivity":
        return _ratio(tp, tp + fn, zero_division)
    if name == "precision":
        return _ratio(tp, tp + fp, zero_division)
    if name == "specificity":
        return _ratio(tn, tn + fp, zero_division)
    if name == "relative_volume_difference":
        difference = fp - fn
        return _ratio(difference.abs() if absolute else difference, tp + fn, zero_division)
    if name == "volume_similarity":
        if convention not in ("taha", "itk"):
            raise ValueError(f"convention must be 'taha' or 'itk', got {convention!r}")
        size_p, size_t = tp + fp, tp + fn
        if convention == "taha":
            value = 1 - (size_p - size_t).abs().double() / (size_p + size_t).clamp_min(1)
        else:
            value = 2 * (size_p - size_t).double() / (size_p + size_t).clamp_min(1)
        return torch.where(size_p + size_t == 0, torch.full_like(value, zero_division), value)
    raise ValueError(f"unknown overlap metric {name!r}")


def _confusion_counts(
    pred: Any, target: Any, input_format: InputFormat, num_classes: int | None, include_background: bool,
    *, ignore_index: int | None = None, mask: Any = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _confusion_counts_and_seg(
        pred, target, input_format, num_classes, include_background, ignore_index=ignore_index, mask=mask,
    )[0]


def _confusion_counts_and_seg(
    pred: Any, target: Any, input_format: InputFormat, num_classes: int | None, include_background: bool,
    *, ignore_index: int | None = None, mask: Any = None,
) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor], Segmentations]:
    """TP, FP, FN and TN, each (B, C).

    Native kernels (CPU) and Triton kernels (GPU) count in one pass and check values as they go
    (labels in range, one-hot masks binary), so validation skips its own passes over the values.
    """
    p, t = to_torch(pred), to_torch(target)
    kw = dict(input_format=input_format, num_classes=num_classes, include_background=include_background)
    native = p.device.type == "cpu" and _native.available()
    gpu = (
        p.device.type == "cuda"
        and _triton_counts.available()
        and (input_format == "one-hot" or (num_classes is not None and _triton_counts.fits(num_classes)))
    )
    if mask is not None or ignore_index is not None:
        seg = validate(p, t, check_values=False, **kw)  # type: ignore[arg-type]
        ignore_index = _check_ignore_index(ignore_index, input_format)
        m = _voxel_mask(mask, seg)
        return _masked_counts(seg, m, ignore_index, native=native, gpu=gpu), seg
    seg = validate(p, t, check_values=not (native or gpu), **kw)  # type: ignore[arg-type]
    if native:
        counts = _native_counts(seg)
    elif gpu:
        counts = _triton_counts_of(seg)
    else:
        counts = _counts(seg)
    if counts is None:  # values the kernels reject: the full validation raises the right error
        validate(p, t, **kw)  # type: ignore[arg-type]
        raise AssertionError("validation accepted values that the counting kernels rejected")
    tp, fp, fn = counts
    voxels = math.prod(seg.pred.shape[seg.pred.ndim - seg.ndim :])
    return (tp, fp, fn, voxels - tp - fp - fn), seg


def _native_counts(seg: Segmentations) -> tuple[Tensor, Tensor, Tensor] | None:
    """TP, FP, FN (each (B, C)) from the native kernels, or None if they reject a value."""
    if seg.one_hot:
        return _from_channels(seg, _native.channel_counts(seg.pred.flatten(0, 1), seg.target.flatten(0, 1)))
    if seg.batch == 0:
        empty = torch.empty((0, len(seg.classes)), dtype=torch.int64, device=seg.pred.device)
        return empty, empty.clone(), empty.clone()
    matrices = []
    for b in range(seg.batch):
        cm = _native.confusion(seg.pred[b], seg.target[b], seg.num_classes)
        if cm is None:
            return None
        matrices.append(cm)
    return _from_confusion(seg, torch.stack(matrices))


def _triton_counts_of(seg: Segmentations) -> tuple[Tensor, Tensor, Tensor] | None:
    """TP, FP, FN (each (B, C)) from the Triton kernels, or None if they reject a value."""
    if seg.one_hot:
        found = _triton_counts.channel_counts(seg.pred.flatten(0, 1), seg.target.flatten(0, 1))
        return _from_channels(seg, found)
    cm = _triton_counts.confusion(seg.pred, seg.target, seg.num_classes)
    return None if cm is None else _from_confusion(seg, cm)


def _counts(seg: Segmentations) -> tuple[Tensor, Tensor, Tensor]:
    """True positives, false positives and false negatives, each (B, C)."""
    if seg.one_hot:
        p, t = seg.pred[:, seg.first_class :], seg.target[:, seg.first_class :]
        tp = _count(p & t)
        return tp, _count(p) - tp, _count(t) - tp
    return _from_confusion(seg, _confusion(seg.pred, seg.target, seg.num_classes))


def _from_channels(
    seg: Segmentations, found: tuple[Tensor, Tensor, Tensor] | None
) -> tuple[Tensor, Tensor, Tensor] | None:
    """TP, FP, FN from per-channel counts (in both, in pred, in target; each (B * C,))."""
    if found is None:
        return None
    batch, classes = seg.pred.shape[:2]
    both, in_pred, in_target = (x.view(batch, classes)[:, seg.first_class :] for x in found)
    return both, in_pred - both, in_target - both


def _from_confusion(seg: Segmentations, cm: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """TP, FP, FN from (B, C, C) confusion matrices."""
    tp = cm.diagonal(dim1=1, dim2=2)
    keep = slice(seg.first_class, None)  # drop excluded classes only after summing full rows/columns
    return tp[:, keep], (cm.sum(2) - tp)[:, keep], (cm.sum(1) - tp)[:, keep]


def _confusion(pred: Tensor, target: Tensor, num_classes: int) -> Tensor:
    """(B, C, C) confusion matrices, ``[b, i, j]`` = voxels predicted ``i`` with target ``j``.

    One bincount pass per case: O(voxels) memory however many classes there are.
    """
    batch = pred.shape[0]
    p, t = pred.flatten(1), target.flatten(1)
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


def _check_ignore_index(ignore_index: int | None, input_format: InputFormat) -> int | None:
    if ignore_index is None:
        return None
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, Integral):
        raise TypeError("ignore_index must be an integer or None")
    if not -(2**63) <= ignore_index < 2**63:
        raise ValueError("ignore_index must fit in a signed 64-bit integer")
    if input_format != "index":
        raise ValueError("ignore_index requires input_format='index'; use mask for one-hot inputs")
    return int(ignore_index)


def _voxel_mask(mask: Any, seg: Segmentations) -> Tensor | None:
    if mask is None:
        return None
    m = to_torch(mask)
    if m.dtype != torch.bool:
        raise TypeError("mask must be a boolean array")
    if m.device != seg.pred.device:
        raise ValueError("mask and inputs must be on the same device")
    shape = (seg.batch, *seg.pred.shape[-seg.ndim:])
    if m.shape == (seg.batch, 1, *shape[1:]):
        m = m[:, 0]
    if m.shape != shape:
        raise ValueError(f"mask must have shape {shape} or {(seg.batch, 1, *shape[1:])}")
    return m


def _masked_counts(
    seg: Segmentations, mask: Tensor | None, ignore_index: int | None, *, native: bool, gpu: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Count valid voxels inside the native scan; allocate a combined mask only on fallback."""
    if seg.batch == 0 or seg.num_classes == 0:
        empty = torch.empty((seg.batch, len(seg.classes)), dtype=torch.int64, device=seg.pred.device)
        return empty, empty.clone(), empty.clone(), empty.clone()
    # uint64 values above INT64_MAX become negative when widened by a counting backend.
    # They must be rejected, not mistaken for a negative ignored target label.
    kernel_ignore = None if seg.target.dtype == torch.uint64 and ignore_index is not None and ignore_index < 0 else ignore_index
    engine = _native if native else _triton_counts
    # The kernels support real arrays; complex one-hot values are checked by the fallback.
    if (native or gpu) and not (seg.pred.is_complex() or seg.target.is_complex()):
        if seg.one_hot:
            assert mask is not None
            found = engine.masked_channel_counts(
                seg.pred.flatten(0, 1), seg.target.flatten(0, 1), mask, seg.num_classes,
            )
            if found is not None:
                both, p, t, valid = (v.view(seg.batch, seg.num_classes)[:, seg.first_class:] for v in found)
                return both, p - both, t - both, valid - p - t + both
        else:
            if native:
                matrices = [_native.confusion(seg.pred[b], seg.target[b], seg.num_classes,
                            mask=None if mask is None else mask[b], ignore_index=kernel_ignore)
                            for b in range(seg.batch)]
                cm = None if any(x is None for x in matrices) else torch.stack(matrices)  # type: ignore[arg-type]
            else:
                cm = _triton_counts.confusion(seg.pred, seg.target, seg.num_classes,
                                              mask=mask, ignore_index=kernel_ignore)
            if cm is not None:
                tp, fp, fn = _from_confusion(seg, cm)
                return tp, fp, fn, cm.sum((1, 2))[:, None] - tp - fp - fn
    # Also provides the precise validation error when a kernel rejects an included value.
    return _masked_torch_counts(seg, mask, ignore_index)


def _masked_torch_counts(seg: Segmentations, mask: Tensor | None,
                         ignore_index: int | None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    keep = mask
    if ignore_index is not None:
        # Widen before comparison so e.g. ignore_index=256 cannot wrap to uint8 label 0.
        labels = seg.target.long()
        observed = labels != ignore_index
        if seg.target.dtype == torch.uint64 and ignore_index < 0:
            observed |= labels < 0
        keep = observed if keep is None else keep & observed
    assert keep is not None
    valid = keep.flatten(1).sum(1, dtype=torch.int64)[:, None]
    if seg.one_hot:
        for name, x in (("pred", seg.pred), ("target", seg.target)):
            if bool((((x != 0) & (x != 1)) & keep[:, None]).any()):
                raise ValueError(f"{name} must be binary (0/1) per channel with input_format='one-hot'; "
                                 "threshold probabilities first")
        p = seg.pred[:, seg.first_class:].bool() & keep[:, None]
        t = seg.target[:, seg.first_class:].bool() & keep[:, None]
        tp = _count(p & t)
        fp, fn = _count(p) - tp, _count(t) - tp
    else:
        # Validation here costs additional passes only without a supported counting kernel.
        p = torch.where(keep, seg.pred.long(), 0)
        t = torch.where(keep, seg.target.long(), 0)
        validate(p, t, input_format="index", num_classes=seg.num_classes,
                 include_background=seg.first_class == 0)
        # Reserve a discarded bin for excluded voxels. This avoids boolean indexing's dynamic
        # allocation and GPU synchronization, and keeps the working buffer bounded by _CHUNK.
        cm = torch.zeros((seg.batch, seg.num_classes**2 + 1), dtype=torch.int64, device=p.device)
        p, t, keep_flat = p.flatten(1), t.flatten(1), keep.flatten(1)
        for b in range(seg.batch):
            for start in range(0, p.shape[1], _CHUNK):
                sl = slice(start, start + _CHUNK)
                indices = torch.where(keep_flat[b, sl], p[b, sl].long() * seg.num_classes + t[b, sl],
                                      seg.num_classes**2)
                cm[b] += torch.bincount(indices, minlength=seg.num_classes**2 + 1)
        tp, fp, fn = _from_confusion(seg, cm[:, :-1].reshape(seg.batch, seg.num_classes, seg.num_classes))
    return tp, fp, fn, valid - tp - fp - fn
