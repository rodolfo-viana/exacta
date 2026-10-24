"""Surface-distance metrics: Hausdorff distance (HD, HD95), average symmetric surface distance
(ASSD) and normalized surface Dice (NSD).

Shared definitions:
    * The boundary of a mask is the set of mask voxels with at least one face-adjacent
      neighbour outside the mask. Voxels outside the image count as background. Every boundary
      voxel has equal weight: the area-weighted surface elements of DeepMind's surface-distance
      (and MONAI's ``use_subvoxels=True``) are not implemented yet.
    * Distances are Euclidean, between voxel centres, in the physical units of ``spacing``.
    * ``d(A -> B)`` holds, for every boundary voxel of A, the distance to the nearest boundary
      voxel of B. P is the prediction and T the target.

HD and ASSD reduce each direction with a statistic ``s`` (for HD the q-th percentile with linear
interpolation, NumPy's default; for ASSD the mean) and join the two directions with ``combine``:

    * ``"max"``: ``max(s(d(P -> T)), s(d(T -> P)))``
    * ``"mean"``: ``(s(d(P -> T)) + s(d(T -> P))) / 2``
    * ``"pooled"``: ``s`` over both sets of distances together

Tools disagree on this choice, so it is an argument rather than a fixed definition:

    ===========  =======  ==================================
    Tool         HD/HD95  ASSD
    ===========  =======  ==================================
    MONAI        max      pooled (``symmetric=True``)
    MedPy 0.5.2  pooled   pooled
    MedPy 0.4    pooled   mean
    ===========  =======  ==================================

NSD at tolerance ``tau`` (Nikolov et al., JMIR 2021) is the fraction of boundary voxels, of both
masks together, that lie within ``tau`` of the other boundary:
``(#{d(P -> T) <= tau} + #{d(T -> P) <= tau}) / (#d(P -> T) + #d(T -> P))``. This is MONAI's
``compute_surface_dice`` with its default ``use_subvoxels=False``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Literal, NamedTuple

import numpy as np
import torch
from torch import Tensor

from exacta import _native
from exacta._inputs import InputFormat, Segmentations, as_spacing, like_input, to_torch, validate

Backend = Literal["auto", "native", "torch"]
Combine = Literal["max", "mean", "pooled"]

EdtSq = Callable[[Tensor, Sequence[float]], Tensor]

# Elements of the (lines, n, n) block evaluated at once by the PyTorch distance transform (128 MiB).
_CHUNK = 1 << 24


class _Distances(NamedTuple):
    """Distances from the boundary voxels of one mask to the other mask's boundary, and the
    voxels' weights (``None`` when every voxel counts once)."""

    d: Tensor
    w: Tensor | None


def hausdorff_distance(
    pred: Any,
    target: Any,
    *,
    spacing: Any,
    percentile: float = 100.0,
    combine: Combine = "max",
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
        combine: How the two directions are joined: ``"max"`` (MONAI), ``"pooled"`` (MedPy)
            or ``"mean"``.
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
    _check_combine(combine)
    seg = validate(
        to_torch(pred),
        to_torch(target),
        input_format=input_format,
        num_classes=num_classes,
        include_background=include_background,
    )

    def score(pt: _Distances, tp: _Distances, k: int) -> float:
        return _combine(pt, tp, combine, lambda x: _percentile(x, percentile))

    out = _per_case(
        seg, spacing, score, both_empty=both_empty, one_empty=one_empty, backend=backend
    )
    return like_input(out, pred)


def hd95(pred: Any, target: Any, *, spacing: Any, **kwargs: Any) -> Any:
    """95th-percentile Hausdorff distance; see :func:`hausdorff_distance`."""
    return hausdorff_distance(pred, target, spacing=spacing, percentile=95.0, **kwargs)


def assd(
    pred: Any,
    target: Any,
    *,
    spacing: Any,
    combine: Combine = "pooled",
    input_format: InputFormat = "index",
    num_classes: int | None = None,
    include_background: bool = True,
    both_empty: float = math.nan,
    one_empty: float = math.inf,
    backend: Backend = "auto",
) -> Any:
    """Average symmetric surface distance for every case and class (see the module docstring).

    Args:
        combine: How the two directions are joined: ``"pooled"`` (MONAI, MedPy 0.5.2), ``"mean"``
            of the two directed means (MedPy 0.4) or ``"max"`` of them.
        Others: As in :func:`hausdorff_distance`.

    Returns:
        A float64 array of shape (B, C), in the array library of ``pred``.
    """
    _check_combine(combine)
    seg = validate(
        to_torch(pred),
        to_torch(target),
        input_format=input_format,
        num_classes=num_classes,
        include_background=include_background,
    )

    def score(pt: _Distances, tp: _Distances, k: int) -> float:
        return _combine(pt, tp, combine, _mean)

    out = _per_case(
        seg, spacing, score, both_empty=both_empty, one_empty=one_empty, backend=backend
    )
    return like_input(out, pred)


def nsd(
    pred: Any,
    target: Any,
    *,
    spacing: Any,
    tolerance: Any,
    input_format: InputFormat = "index",
    num_classes: int | None = None,
    include_background: bool = True,
    both_empty: float = math.nan,
    one_empty: float = 0.0,
    backend: Backend = "auto",
) -> Any:
    """Normalized surface Dice for every case and class (see the module docstring).

    Distances equal to the tolerance count as within it. Each distance is the exact
    floating-point minimum of the expression SciPy's ``distance_transform_edt`` evaluates, so a
    distance that ties with the tolerance lands on the same side as in MedPy, which uses SciPy.
    (SciPy itself returns a value one ulp above that minimum at a few voxels per
    million.) MONAI casts distances to float32 before comparing them with the tolerance, which
    agrees on exact ties but also counts distances within float32 resolution of the tolerance.

    Args:
        tolerance: Accepted boundary deviation, in the units of ``spacing``. A scalar, or one
            value per evaluated class (``num_classes - 1`` values when the background is excluded).
        one_empty: Value when the class is absent from exactly one of prediction and target.
            The default 0 is the limit of the definition: no boundary voxel has a partner.
        Others: As in :func:`hausdorff_distance`.

    Returns:
        A float64 array of shape (B, C) with values in [0, 1], in the array library of ``pred``.
    """
    seg = validate(
        to_torch(pred),
        to_torch(target),
        input_format=input_format,
        num_classes=num_classes,
        include_background=include_background,
    )
    tau = _tolerances(tolerance, seg.classes)

    def score(pt: _Distances, tp: _Distances, k: int) -> float:
        within_p, total_p = _within(pt, tau[k])
        within_t, total_t = _within(tp, tau[k])
        return (within_p + within_t) / (total_p + total_t)

    out = _per_case(
        seg, spacing, score, both_empty=both_empty, one_empty=one_empty, backend=backend
    )
    return like_input(out, pred)


def _per_case(
    seg: Segmentations,
    spacing: Any,
    score: Callable[[_Distances, _Distances, int], float],
    *,
    both_empty: float,
    one_empty: float,
    backend: Backend,
) -> Tensor:
    """Apply ``score(d(P -> T), d(T -> P), class position)`` to every case and class."""
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
            out[b, k] = score(*_voxel_distances(p, t, spacings[b], edt_sq), k)
    return out.to(seg.pred.device)


def _check_combine(combine: str) -> None:
    if combine not in ("max", "mean", "pooled"):
        raise ValueError(f"combine must be 'max', 'mean' or 'pooled', got {combine!r}")


def _combine(pt: _Distances, tp: _Distances, combine: Combine, stat: Callable[[_Distances], float]) -> float:
    if combine == "pooled":
        w = None if pt.w is None or tp.w is None else torch.cat([pt.w, tp.w])
        return stat(_Distances(torch.cat([pt.d, tp.d]), w))
    a, b = stat(pt), stat(tp)
    return max(a, b) if combine == "max" else (a + b) / 2


def _percentile(x: _Distances, q: float) -> float:
    if q == 100:
        return x.d.max().item()
    d = x.d.sort().values
    pos = q / 100 * (d.numel() - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, d.numel() - 1)
    return (d[lo] + (d[hi] - d[lo]) * (pos - lo)).item()


def _mean(x: _Distances) -> float:
    if x.w is None:
        return x.d.mean().item()
    return ((x.d * x.w).sum() / x.w.sum()).item()


def _within(x: _Distances, tau: float) -> tuple[float, float]:
    """Amount of boundary (voxel count, or total weight) within ``tau``, and the total amount."""
    inside = x.d <= tau
    if x.w is None:
        return inside.sum().item(), x.d.numel()
    return x.w[inside].sum().item(), x.w.sum().item()


def _tolerances(tolerance: Any, classes: range) -> list[float]:
    """One NSD tolerance per evaluated class."""
    t = torch.as_tensor(tolerance, dtype=torch.float64).cpu()
    if t.ndim == 0:
        t = t.expand(len(classes))
    elif t.shape != (len(classes),):
        raise ValueError(
            f"tolerance must be a scalar or one value per evaluated class ({len(classes)} values, "
            f"for classes {classes.start}..{classes.stop - 1}); got shape {tuple(t.shape)}"
        )
    if not bool(torch.isfinite(t).all() and (t >= 0).all()):
        raise ValueError("tolerance values must be non-negative and finite")
    return t.tolist()


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


def _voxel_distances(
    p: Tensor, t: Tensor, spacing: Sequence[float], edt_sq: EdtSq
) -> tuple[_Distances, _Distances]:
    d_pt, d_tp = _surface_distances(_boundary(p, p.ndim), _boundary(t, t.ndim), spacing, edt_sq)
    return _Distances(d_pt, None), _Distances(d_tp, None)


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


def _surface_distances(a: Tensor, b: Tensor, spacing: Sequence[float], edt_sq: EdtSq) -> tuple[Tensor, Tensor]:
    """Distances from each boundary voxel of ``a`` to boundary ``b``, and from ``b`` to ``a``."""
    # Cropping to the bounding box of both boundaries is exact: every query voxel and every
    # feature voxel lies inside the box.
    idx = torch.nonzero(a | b)
    box = tuple(slice(lo, hi + 1) for lo, hi in zip(idx.amin(0).tolist(), idx.amax(0).tolist()))
    a, b = a[box], b[box]
    to_a, to_b = edt_sq(a, spacing), edt_sq(b, spacing)
    return _sqrt(to_b[a]), _sqrt(to_a[b])


def _sqrt(x: Tensor) -> Tensor:
    """Correctly rounded square root of a float64 tensor.

    PyTorch's CPU float64 sqrt is one ulp off for about 1% of inputs (seen in 2.14), which can
    decide NSD's ``d <= tolerance`` when the tolerance is itself a distance between voxel centres
    (e.g. a voxel diagonal). NumPy's sqrt is correctly rounded, as is CUDA's.
    """
    if x.device.type == "cpu":
        return torch.from_numpy(np.sqrt(x.numpy()))
    return x.sqrt()


def _edt_sq_torch(features: Tensor, spacing: Sequence[float]) -> Tensor:
    """Exact squared Euclidean distance transform in PyTorch, for any device.

    Squared distance is separable, so one 1-D min-plus transform per axis is exact:
    ``out[q] = min_p (h*(q - p))^2 + in[p]``. Rounding is monotonic, so the per-pass minima give
    the exact floating-point minimum of the whole expression, as :func:`exacta._native.edt_sq`
    does. Each pass is brute force, O(n) per voxel along the axis, but fully vectorised.
    """
    out = torch.zeros(features.shape, dtype=torch.float64, device=features.device)
    out.masked_fill_(~features, math.inf)
    for axis, h in enumerate(spacing):
        n = out.shape[axis]
        lines = out.movedim(axis, -1)
        shape = lines.shape
        lines = lines.reshape(-1, n)
        i = torch.arange(n, dtype=torch.float64, device=out.device)
        cost = ((i[:, None] - i[None, :]) * h).square()  # cost[q, p]; offsets rounded once, as SciPy
        result = torch.empty_like(lines)
        step = max(1, _CHUNK // (n * n))
        for start in range(0, lines.shape[0], step):
            result[start : start + step] = (lines[start : start + step, None, :] + cost).amin(-1)
        out = result.view(shape).movedim(-1, axis)
    return out
