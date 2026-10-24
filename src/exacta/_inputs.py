"""Input contract shared by all metrics: array conversion, segmentation layout and voxel spacing."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from exacta import _native

InputFormat = Literal["index", "one-hot"]

# Voxel sizes beyond these overflow (or make subnormal) the squared distances the exact distance
# transforms compute; any physical unit is far inside them.
MIN_SPACING, MAX_SPACING = 1e-100, 1e100
# Areas and point-to-triangle predicates involve fourth powers of physical coordinates.
# Leave headroom for image extents and sums, rather than applying the EDT's wider range.
MIN_SURFACE_SPACING, MAX_SURFACE_SPACING = 1e-50, 1e50


def to_torch(x: Any) -> Tensor:
    """Return ``x`` as a tensor, sharing memory through DLPack when the source allows it."""
    if isinstance(x, Tensor):
        return x
    if hasattr(x, "__dlpack__"):
        try:
            return torch.from_dlpack(x)
        except (BufferError, RuntimeError, TypeError):
            pass  # e.g. read-only NumPy arrays
    try:
        return torch.as_tensor(x)
    except ValueError:  # NumPy arrays with negative strides (e.g. a[::-1]) need a copy
        return torch.as_tensor(np.ascontiguousarray(x))


def like_input(result: Tensor, original: Any) -> Any:
    """Return ``result`` in the array library ``original`` came from (NumPy, CuPy, JAX, ...)."""
    if isinstance(original, Tensor):
        return result
    if isinstance(original, np.ndarray):
        # NumPy < 2 has no __array_namespace__; NumPy < 1.23 also lacks from_dlpack.
        return result.detach().cpu().numpy()
    # An existing CuPy array implies CuPy is loaded. Do not import an optional dependency on
    # every call (or classify unrelated CUDA-array-interface providers as CuPy).
    cupy = sys.modules.get("cupy")
    if cupy is not None and isinstance(original, cupy.ndarray):
        return cupy.from_dlpack(result)
    namespace = getattr(original, "__array_namespace__", None)
    if namespace is None:
        return result
    return namespace().from_dlpack(result)


@dataclass(frozen=True)
class Segmentations:
    """A validated prediction/target pair.

    Index inputs stay as label maps: expanding them to one-hot would cost B x C x voxels of
    memory, which is prohibitive for 3D volumes with many classes.
    """

    pred: Tensor  # index: (B, *spatial) integers; one-hot: (B, C, *spatial) bool
    target: Tensor
    one_hot: bool
    num_classes: int
    first_class: int  # 1 when the background class is excluded
    ndim: int  # spatial dimensions

    @property
    def batch(self) -> int:
        return self.pred.shape[0]

    @property
    def classes(self) -> range:
        return range(self.first_class, self.num_classes)

    def masks(self, case: int, cls: int, box: tuple[slice, ...] = ()) -> tuple[Tensor, Tensor]:
        """Boolean (*spatial) masks of class ``cls`` in one case, restricted to ``box``."""
        if self.one_hot:
            return self.pred[case, cls][box], self.target[case, cls][box]
        return self.pred[case][box] == cls, self.target[case][box] == cls

    def extents(self, case: int) -> tuple[Tensor, Tensor, Tensor]:
        """Where each class lies in one case: ``present`` (2, C) for prediction and target, and
        the first and last index of each along every axis, ``lo`` and ``hi`` (2, C, ndim).

        One pass over the volume whatever the number of classes, instead of one per class.
        """
        found = [_extent(x, self.num_classes, self.one_hot) for x in (self.pred[case], self.target[case])]
        return tuple(torch.stack(parts) for parts in zip(*found))  # type: ignore[return-value]


def _extent(
    volume: Tensor, num_classes: int, one_hot: bool, native: bool = True
) -> tuple[Tensor, Tensor, Tensor]:
    """Presence (C,), first and last index (C, ndim) of every class in one (C, *spatial) one-hot
    volume or one (*spatial) label map; indices are meaningless where a class is absent."""
    if native and volume.device.type == "cpu" and _native.available():  # one multithreaded pass
        return _native.mask_extents(volume) if one_hot else _native.class_extents(volume, num_classes)
    spatial = volume.shape[1:] if one_hot else volume.shape
    if volume.numel() == 0:  # no class anywhere
        corners = torch.zeros((num_classes, len(spatial)), dtype=torch.int64, device=volume.device)
        return torch.zeros(num_classes, dtype=torch.bool, device=volume.device), corners, corners
    if one_hot:
        dims = range(1, volume.ndim)
        planes = [
            volume.any(dim=tuple(d for d in dims if d != axis)) if volume.ndim > 2 else volume for axis in dims
        ]
    else:
        # lines[..., c]: whether class c occurs on each line along the last axis; one scatter over
        # the volume gives every class's projection on the leading axes.
        labels = volume.long()
        lead = labels.ndim - 1
        lines = torch.zeros((*labels.shape[:-1], num_classes), dtype=torch.bool, device=labels.device)
        lines.scatter_(-1, labels, True)
        planes = [
            (lines.any(dim=tuple(d for d in range(lead) if d != axis)) if lead > 1 else lines).T
            for axis in range(lead)
        ]
        last = torch.zeros((num_classes, labels.shape[-1]), dtype=torch.bool, device=labels.device)
        planes.append(last.scatter_(0, labels.reshape(-1, labels.shape[-1]), True))
    first = torch.stack([plane.to(torch.uint8).argmax(1) for plane in planes], 1)
    final = torch.stack([plane.shape[1] - 1 - plane.flip(1).to(torch.uint8).argmax(1) for plane in planes], 1)
    return planes[0].any(1), first, final


def validate(
    pred: Tensor,
    target: Tensor,
    *,
    input_format: InputFormat,
    num_classes: int | None,
    include_background: bool,
    check_values: bool = True,
) -> Segmentations:
    """Check shapes, formats and (unless ``check_values`` is False) values.

    ``check_values=False`` leaves the label range unchecked and one-hot inputs as they are, for
    callers whose kernels check values as they read them; such callers must validate again with
    the default when a kernel rejects the input, for the error message.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {tuple(pred.shape)} and {tuple(target.shape)}"
        )
    if pred.device != target.device:
        raise ValueError(f"pred and target are on different devices ({pred.device} and {target.device})")

    if input_format == "index":
        num_classes = _check_index(pred, target, num_classes, check_values)
        one_hot, spatial_ndim = False, pred.ndim - 1
    elif input_format == "one-hot":
        if pred.ndim not in (4, 5):
            raise ValueError(
                "one-hot inputs must have shape (B, C, H, W) or (B, C, D, H, W), "
                f"got {tuple(pred.shape)}"
            )
        if num_classes is not None and num_classes != pred.shape[1]:
            raise ValueError(f"num_classes={num_classes} but inputs have {pred.shape[1]} channels")
        if check_values:
            pred, target = _binary(pred, "pred"), _binary(target, "target")
        num_classes = pred.shape[1]
        one_hot, spatial_ndim = True, pred.ndim - 2
    else:
        raise ValueError(f"input_format must be 'index' or 'one-hot', got {input_format!r}")

    if not include_background and num_classes < 2:
        raise ValueError("include_background=False needs at least 2 classes (class 0 is background)")
    return Segmentations(
        pred, target, one_hot, num_classes, first_class=0 if include_background else 1, ndim=spatial_ndim
    )


def _check_index(pred: Tensor, target: Tensor, num_classes: int | None, check_values: bool = True) -> int:
    """Validate index label maps and return ``num_classes``, which they require."""
    if pred.ndim not in (3, 4):
        raise ValueError(
            f"index inputs must have shape (B, H, W) or (B, D, H, W), got {tuple(pred.shape)}"
        )
    if num_classes is None or num_classes < 1:
        raise ValueError("num_classes (>= 1) is required with input_format='index'")
    for name, x in (("pred", pred), ("target", target)):
        if x.dtype.is_floating_point or x.dtype.is_complex:
            raise TypeError(
                f"{name} must hold integer class indices, got {x.dtype}; "
                "apply argmax (multi-class) or a threshold (binary) first"
            )
    if check_values and pred.numel():
        lo_p, lo_t, hi_p, hi_t = torch.stack(
            [pred.min().long(), target.min().long(), pred.max().long(), target.max().long()]
        ).tolist()
        lo, hi = min(lo_p, lo_t), max(hi_p, hi_t)
        if lo < 0 or hi >= num_classes:
            raise ValueError(
                f"class indices must lie in [0, {num_classes}), found values in [{lo}, {hi}]"
            )
    return num_classes


def _binary(x: Tensor, name: str) -> Tensor:
    if x.dtype == torch.bool:
        return x
    if x.device.type == "cpu" and _native.available():  # one multithreaded pass
        mask = _native.binary_mask(x)
    else:
        mask = None if x.numel() and bool(((x != 0) & (x != 1)).any()) else x != 0
    if mask is None:
        raise ValueError(
            f"{name} must be binary (0/1) per channel with input_format='one-hot'; "
            "threshold probabilities first"
        )
    return mask


def as_spacing(
    spacing: Any, *, batch: int, ndim: int, area_weighted: bool = False
) -> list[tuple[float, ...]]:
    """Normalise voxel spacing to one tuple of ``ndim`` values per case.

    Accepts a scalar (isotropic), ``ndim`` values shared by the batch, or a (batch, ndim) array.
    Area-weighted surfel/mesh metrics use a narrower range than squared voxel distances.
    """
    if spacing is None:
        raise ValueError(
            "spacing is required for distance metrics: pass the voxel size in physical units "
            "(e.g. mm), in the same axis order as the spatial dimensions of the tensor"
        )
    s = torch.as_tensor(spacing, dtype=torch.float64).cpu()
    if s.ndim == 0:
        s = s.expand(ndim)
    if s.ndim == 1:
        if s.numel() != ndim:
            raise ValueError(f"spacing has {s.numel()} values but inputs have {ndim} spatial dimensions")
        s = s.expand(batch, ndim)
    elif s.shape != (batch, ndim):
        raise ValueError(
            f"spacing must be a scalar, shape ({ndim},) or ({batch}, {ndim}); got {tuple(s.shape)}"
        )
    low, high = (MIN_SURFACE_SPACING, MAX_SURFACE_SPACING) if area_weighted else (MIN_SPACING, MAX_SPACING)
    if not bool(torch.isfinite(s).all() and (s >= low).all() and (s <= high).all()):
        quantities = "surface areas and mesh geometry" if area_weighted else "squared distances"
        raise ValueError(
            f"spacing values must lie in [{low:g}, {high:g}]; outside, {quantities} "
            "overflow or lose precision and exact distances cannot be guaranteed"
        )
    return [tuple(row) for row in s.tolist()]
