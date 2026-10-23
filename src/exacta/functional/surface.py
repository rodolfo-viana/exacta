"""Surface-distance metrics: the Hausdorff distance and its percentile variants (HD95).

Definition:
    * The boundary of a mask is the set of mask voxels with at least one face-adjacent
      neighbour outside the mask. Voxels outside the image count as background.
    * Distances are Euclidean, between voxel centres, in the physical units of ``spacing``.
    * ``HD_q = max(P_q(d(P -> T)), P_q(d(T -> P)))``, where ``d(A -> B)`` holds, for every
      boundary voxel of A, the distance to the nearest boundary voxel of B, and ``P_q`` is the
      q-th percentile with linear interpolation (NumPy's default).

This is the definition used by MONAI's ``compute_hausdorff_distance``. MedPy's ``hd95`` instead
takes one percentile over both directions pooled, which gives different values.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
from torch import Tensor

from exacta import _native
from exacta._inputs import InputFormat, as_spacing, like_input, to_torch, validate

Backend = Literal["auto", "native", "torch"]

# Elements of the (lines, n, n) block evaluated at once by the PyTorch distance transform (128 MiB).
_CHUNK = 1 << 24


def hausdorff_distance(
    pred: Any,
    target: Any,
    *,
    spacing: Any,
    percentile: float = 100.0,
    input_format: InputFormat = "index",
    num_classes: int | None = None,
    include_background: bool = True,
    both_empty: float = math.nan,
    one_empty: float = math.inf,
    backend: Backend = "auto",
) -> Any:
    """Percentile Hausdorff distance for every case and class (see the module docstring).

    Args:
        pred, target: Hard segmentations, as in :func:`exacta.functional.dice`.
        spacing: Voxel size in physical units, in the axis order of the tensor's spatial
            dimensions (a SimpleITK array is (z, y, x) while ``GetSpacing()`` is (x, y, z)).
            A scalar, one value per spatial axis, or a (B, ndim) array for per-case spacing.
        percentile: 100 gives the Hausdorff distance, 95 gives HD95.
        num_classes, include_background: As in :func:`exacta.functional.dice`.
        both_empty: Value when the class is absent from both prediction and target.
        one_empty: Value when the class is absent from exactly one of them.
        backend: Exact distance transform to use. ``"native"`` is the compiled kernel (CPU),
            ``"torch"`` runs on any device, ``"auto"`` picks native on CPU and torch elsewhere.

    Returns:
        A float64 array of shape (B, C), in the array library of ``pred``.
    """
    if not 0 <= percentile <= 100:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    seg = validate(
        to_torch(pred),
        to_torch(target),
        input_format=input_format,
        num_classes=num_classes,
        include_background=include_background,
    )
    spacings = as_spacing(spacing, batch=seg.batch, ndim=seg.ndim)
    edt_sq = _native.edt_sq if _use_native(backend, seg.pred.device) else _edt_sq_torch

    out = torch.full((seg.batch, len(seg.classes)), both_empty, dtype=torch.float64)
    for b in range(seg.batch):
        for k, cls in enumerate(seg.classes):
            p, t = seg.masks(b, cls)
            has_p, has_t = torch.stack([p.any(), t.any()]).tolist()
            if not (has_p or has_t):
                continue
            if not (has_p and has_t):
                out[b, k] = one_empty
                continue
            d_pt, d_tp = _surface_distances(_boundary(p, seg.ndim), _boundary(t, seg.ndim), spacings[b], edt_sq)
            out[b, k] = max(_percentile(d_pt, percentile), _percentile(d_tp, percentile))
    return like_input(out.to(seg.pred.device), pred)


def hd95(pred: Any, target: Any, *, spacing: Any, **kwargs: Any) -> Any:
    """95th-percentile Hausdorff distance; see :func:`hausdorff_distance`."""
    return hausdorff_distance(pred, target, spacing=spacing, percentile=95.0, **kwargs)


def _use_native(backend: Backend, device: torch.device) -> bool:
    if backend == "torch":
        return False
    if backend == "native":
        if device.type != "cpu":
            raise NotImplementedError(
                "native kernels run on CPU only for now; use backend='torch' or 'auto' on GPU"
            )
        if not _native.available():
            raise RuntimeError("exacta was installed without its compiled extension")
        return True
    if backend == "auto":
        return device.type == "cpu" and _native.available()
    raise ValueError(f"backend must be 'auto', 'native' or 'torch', got {backend!r}")


def _boundary(mask: Tensor, ndim: int) -> Tensor:
    """Mask voxels with a face-adjacent neighbour outside the mask or outside the image."""
    interior = mask.clone()
    for axis in range(mask.ndim - ndim, mask.ndim):
        n = mask.shape[axis]
        if n == 0:
            continue
        interior.narrow(axis, 0, n - 1).logical_and_(mask.narrow(axis, 1, n - 1))
        interior.narrow(axis, 1, n - 1).logical_and_(mask.narrow(axis, 0, n - 1))
        interior.narrow(axis, 0, 1).fill_(False)
        interior.narrow(axis, n - 1, 1).fill_(False)
    return mask & ~interior


def _surface_distances(
    a: Tensor, b: Tensor, spacing: Sequence[float], edt_sq: Callable[[Tensor, Sequence[float]], Tensor]
) -> tuple[Tensor, Tensor]:
    """Distances from each boundary voxel of ``a`` to boundary ``b``, and from ``b`` to ``a``."""
    # Cropping to the bounding box of both boundaries is exact: every query voxel and every
    # feature voxel lies inside the box.
    idx = torch.nonzero(a | b)
    box = tuple(slice(lo, hi + 1) for lo, hi in zip(idx.amin(0).tolist(), idx.amax(0).tolist()))
    a, b = a[box], b[box]
    to_a, to_b = edt_sq(a, spacing), edt_sq(b, spacing)
    return to_b[a].sqrt(), to_a[b].sqrt()


def _edt_sq_torch(features: Tensor, spacing: Sequence[float]) -> Tensor:
    """Exact squared Euclidean distance transform in PyTorch, for any device.

    Squared distance is separable, so one 1-D min-plus transform per axis is exact:
    ``out[q] = min_p (h*(q - p))^2 + in[p]``. Each pass is brute force, O(n) per voxel along
    the axis, but fully vectorised.
    """
    out = torch.zeros(features.shape, dtype=torch.float64, device=features.device)
    out.masked_fill_(~features, math.inf)
    for axis, h in enumerate(spacing):
        n = out.shape[axis]
        lines = out.movedim(axis, -1)
        shape = lines.shape
        lines = lines.reshape(-1, n)
        x = torch.arange(n, dtype=torch.float64, device=out.device) * h
        cost = (x[:, None] - x[None, :]).square()  # cost[q, p]
        result = torch.empty_like(lines)
        step = max(1, _CHUNK // (n * n))
        for start in range(0, lines.shape[0], step):
            result[start : start + step] = (lines[start : start + step, None, :] + cost).amin(-1)
        out = result.view(shape).movedim(-1, axis)
    return out


def _percentile(x: Tensor, q: float) -> float:
    if q == 100:
        return x.max().item()
    x = x.sort().values
    pos = q / 100 * (x.numel() - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, x.numel() - 1)
    return (x[lo] + (x[hi] - x[lo]) * (pos - lo)).item()
