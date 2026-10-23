"""Input contract shared by all metrics: array conversion, segmentation layout and voxel spacing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

InputFormat = Literal["index", "one-hot"]


def to_torch(x: Any) -> Tensor:
    """Return ``x`` as a tensor, sharing memory through DLPack when the source allows it."""
    if isinstance(x, Tensor):
        return x
    if hasattr(x, "__dlpack__"):
        try:
            return torch.from_dlpack(x)
        except (BufferError, RuntimeError, TypeError):
            pass  # e.g. read-only NumPy arrays
    return torch.as_tensor(x)


def like_input(result: Tensor, original: Any) -> Any:
    """Return ``result`` in the array library ``original`` came from (NumPy, CuPy, JAX, ...)."""
    if isinstance(original, Tensor):
        return result
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

    def masks(self, case: int, cls: int) -> tuple[Tensor, Tensor]:
        """Boolean (*spatial) masks of class ``cls`` in one case."""
        if self.one_hot:
            return self.pred[case, cls], self.target[case, cls]
        return self.pred[case] == cls, self.target[case] == cls


def validate(
    pred: Tensor,
    target: Tensor,
    *,
    input_format: InputFormat,
    num_classes: int | None,
    include_background: bool,
) -> Segmentations:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {tuple(pred.shape)} and {tuple(target.shape)}"
        )
    if pred.device != target.device:
        raise ValueError(f"pred and target are on different devices ({pred.device} and {target.device})")

    if input_format == "index":
        _check_index(pred, target, num_classes)
        one_hot, spatial_ndim = False, pred.ndim - 1
    elif input_format == "one-hot":
        if pred.ndim not in (4, 5):
            raise ValueError(
                "one-hot inputs must have shape (B, C, H, W) or (B, C, D, H, W), "
                f"got {tuple(pred.shape)}"
            )
        if num_classes is not None and num_classes != pred.shape[1]:
            raise ValueError(f"num_classes={num_classes} but inputs have {pred.shape[1]} channels")
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


def _check_index(pred: Tensor, target: Tensor, num_classes: int | None) -> None:
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
    if pred.numel():
        lo_p, lo_t, hi_p, hi_t = torch.stack(
            [pred.min().long(), target.min().long(), pred.max().long(), target.max().long()]
        ).tolist()
        lo, hi = min(lo_p, lo_t), max(hi_p, hi_t)
        if lo < 0 or hi >= num_classes:
            raise ValueError(
                f"class indices must lie in [0, {num_classes}), found values in [{lo}, {hi}]"
            )


def _binary(x: Tensor, name: str) -> Tensor:
    if x.dtype == torch.bool:
        return x
    if x.numel() and bool(((x != 0) & (x != 1)).any()):
        raise ValueError(
            f"{name} must be binary (0/1) per channel with input_format='one-hot'; "
            "threshold probabilities first"
        )
    return x != 0


def as_spacing(spacing: Any, *, batch: int, ndim: int) -> list[tuple[float, ...]]:
    """Normalise voxel spacing to one tuple of ``ndim`` values per case.

    Accepts a scalar (isotropic), ``ndim`` values shared by the batch, or a (batch, ndim) array.
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
    if not bool(torch.isfinite(s).all() and (s > 0).all()):
        raise ValueError("spacing values must be positive and finite")
    return [tuple(row) for row in s.tolist()]
