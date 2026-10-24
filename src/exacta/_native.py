"""Thin Python layer over the compiled kernels in ``exacta._C``."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

try:
    from exacta import _C  # type: ignore[attr-defined]  # compiled, no stub
except ImportError:  # running from a source tree that was never built
    _C = None


def available() -> bool:
    return _C is not None


def edt_sq(features: Tensor, spacing: Sequence[float], batch_ndim: int = 0) -> Tensor:
    """Squared Euclidean distance from every voxel to the nearest nonzero voxel of ``features``.

    Each value is the exact floating-point minimum, over nonzero voxels p, of
    ``sum_i ((p_i - q_i) * h_i)^2`` rounded and summed in axis order, which is the expression
    SciPy's ``distance_transform_edt`` evaluates. The first ``batch_ndim`` axes index separate
    images, and ``spacing`` covers the others. CPU only. Returns float64; ``inf`` throughout an
    image without nonzero voxels.
    """
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    features = features.to(torch.uint8).contiguous()
    out = torch.empty(features.shape, dtype=torch.float64)
    _C.edt_sq(features, out, [float(s) for s in spacing], batch_ndim)
    return out


def label(mask: Tensor, connectivity: int) -> tuple[Tensor, int]:
    """Connected-component labels (int32) of a CPU mask and the number of components.

    Components are numbered 1, 2, ... in C order of their first voxel, as scipy.ndimage.label
    numbers them; see ``exacta_label_u8`` in ``kernels.h``.
    """
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    # A bool tensor already holds one byte of 0 or 1 per element; viewing it avoids a copy.
    mask = mask.contiguous().view(torch.uint8) if mask.dtype == torch.bool else (mask != 0).to(torch.uint8)
    out = torch.empty(mask.shape, dtype=torch.int32)
    count = _C.label(mask, out, int(connectivity))
    return out, count


def class_extents(labels: Tensor, num_classes: int) -> tuple[Tensor, Tensor, Tensor]:
    """Presence (C,), first and last index (C, ndim) of every class of a CPU label map.

    One pass over the volume; see ``exacta_class_extents_*`` in ``kernels.h``.
    """
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    if labels.dtype not in (torch.uint8, torch.int32, torch.int64):
        labels = labels.long()
    labels = labels.contiguous()
    lo = torch.empty((num_classes, labels.ndim), dtype=torch.int64)
    hi = torch.empty((num_classes, labels.ndim), dtype=torch.int64)
    _C.class_extents(labels, lo, hi)
    return hi[:, 0] >= 0, lo, hi


def point_mesh_distances(points: Tensor, elements: Tensor) -> Tensor:
    """Distance from each point (N, d) to the nearest element (M, d, d) of a CPU mesh: triangles
    in 3-D, segments in 2-D; float64, inf for an empty mesh."""
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    points = points.to(torch.float64).contiguous()
    elements = elements.to(torch.float64).contiguous()
    out = torch.empty(points.shape[0], dtype=torch.float64)
    _C.point_mesh_distances(points, elements, out)
    return out
