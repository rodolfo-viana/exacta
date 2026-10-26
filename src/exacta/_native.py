"""Thin Python layer over the compiled kernels in ``exacta._C``."""

from __future__ import annotations

import math

from collections.abc import Sequence

import numpy as np
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
    features = _as_bytes(features)
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
    mask = _as_bytes(mask)
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


def _as_bytes(mask: Tensor) -> Tensor:
    """0/1 (or nonzero) mask as contiguous uint8; a bool tensor is viewed without a copy."""
    if mask.dtype == torch.bool:
        return mask.contiguous().view(torch.uint8)
    return (mask != 0).to(torch.uint8).contiguous()


def neighbour_codes(mask: Tensor, counts: Tensor | None = None) -> tuple[Tensor, int]:
    """DeepMind's neighbour code (uint8, shape + 1 along each axis) at every voxel corner of a CPU
    2-D or 3-D mask, and, given ``counts`` (int64, elements per code), the number of mesh
    elements; see ``exacta_neighbour_codes_u8`` in ``kernels.h``."""
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    mask = _as_bytes(mask)
    code = torch.empty(tuple(n + 1 for n in mask.shape), dtype=torch.uint8)
    counts = counts.contiguous() if counts is not None else torch.empty(0, dtype=torch.int64)
    total = _C.neighbour_codes(mask, code, counts)
    return code, total


def mesh_elements(
    code: Tensor, table: Tensor, counts: Tensor, spacing: Sequence[float], num_elements: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Elements (M, d, d), centres (M, d), blocks, codes and slots (M,) of the mesh that the
    neighbour codes select from ``table``; see ``exacta_mesh_elements_f64`` in ``kernels.h``."""
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    ndim = code.ndim
    elements = torch.empty((num_elements, ndim, ndim), dtype=torch.float64)
    centres = torch.empty((num_elements, ndim), dtype=torch.float64)
    block, element_code, slot = (torch.empty(num_elements, dtype=torch.int64) for _ in range(3))
    _C.mesh_elements(
        code.contiguous(), table.contiguous(), counts.contiguous(), [float(s) for s in spacing],
        elements, centres, block, element_code, slot,
    )
    return elements, centres, block, element_code, slot


def mask_extents(masks: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Presence (C,), first and last index (C, ndim) of each of the CPU masks (C, *spatial)."""
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    masks = _as_bytes(masks)
    lo = torch.empty((masks.shape[0], masks.ndim - 1), dtype=torch.int64)
    hi = torch.empty_like(lo)
    _C.mask_extents(masks, lo, hi)
    return hi[:, 0] >= 0, lo, hi


def dilate(mask: Tensor, connectivity: int, steps: int) -> Tensor:
    """Binary dilation of a CPU mask (bool), as scipy.ndimage.binary_dilation with ``iterations``."""
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    out = torch.empty(mask.shape, dtype=torch.uint8)
    _C.dilate(_as_bytes(mask), out, int(connectivity), int(steps))
    return out.view(torch.bool)


def confusion(pred: Tensor, target: Tensor, classes: int, *, mask: Tensor | None = None,
              ignore_index: int | None = None) -> Tensor | None:
    """(classes, classes) int64 confusion matrix of two CPU label maps (any shape), counted in one
    pass; None if a label lies outside [0, classes)."""
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    if pred.dtype == target.dtype == torch.bool:
        pred, target = pred.view(torch.uint8), target.view(torch.uint8)
    elif pred.dtype != target.dtype or pred.dtype not in (torch.uint8, torch.int32, torch.int64):
        # Both operands must select the same binding overload, without narrowing either label.
        pred, target = pred.long(), target.long()
    out = torch.empty((classes, classes), dtype=torch.int64)
    p, t = pred.contiguous().reshape(-1), target.contiguous().reshape(-1)
    if mask is None and ignore_index is None:
        ok = _C.confusion(p, t, out)
    else:
        m = torch.empty(0, dtype=torch.uint8) if mask is None else _as_bytes(mask).reshape(-1)
        ok = _C.masked_confusion(p, t, m, ignore_index is not None, ignore_index or 0, out)
    return out if ok else None


_BINARY_DTYPES = (torch.uint8, torch.int32, torch.int64, torch.float32, torch.float64)


def binary_mask(values: Tensor) -> Tensor | None:
    """``values != 0`` (bool) of a CPU tensor, in one parallel pass that also checks the values are
    0 or 1; None if one is not. Dtypes without a kernel are converted first."""
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    if values.dtype not in _BINARY_DTYPES:
        values = values.to(torch.float64 if values.dtype.is_floating_point else torch.int64)
    out = torch.empty(values.shape, dtype=torch.bool)
    ok = _C.binary_mask(values.contiguous().reshape(-1), out.view(torch.uint8).reshape(-1))
    return out if ok else None


def boundary(mask: Tensor) -> Tensor:
    """Elements of a CPU 1-D to 3-D mask (any strides) with a face neighbour outside the mask or
    outside the image, as a contiguous bool tensor."""
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    if mask.dtype == torch.bool:
        mask = mask.view(torch.uint8)  # keeps the strides of a view
    elif mask.dtype != torch.uint8:
        mask = (mask != 0).to(torch.uint8)
    out = torch.empty(mask.shape, dtype=torch.bool)
    _C.boundary(mask, out.view(torch.uint8))
    return out


def masked_sqrt(values: Tensor, mask: Tensor) -> Tensor:
    """``sqrt(values[mask])`` of a CPU float64 tensor, correctly rounded, in one parallel pass."""
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    flat_mask = _as_bytes(mask).reshape(-1)
    out = torch.empty(int(torch.count_nonzero(flat_mask)), dtype=torch.float64)
    _C.masked_sqrt(values.to(torch.float64).contiguous().reshape(-1), flat_mask, out)
    return out


def sort_pairs(keys: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Copies of two float64 arrays with their (key, value) pairs sorted by key, then by value, as
    ``np.lexsort((values, keys))`` orders them; parallel. None if one is NaN."""
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    keys = np.array(keys, dtype=np.float64, order="C")
    values = np.array(values, dtype=np.float64, order="C")
    return (keys, values) if _C.sort_pairs(keys, values) else None


def histogram(values: Tensor, bins: int) -> Tensor | None:
    """Number of elements (int64, (bins,)) of each value of a CPU int32 label map, in one parallel
    pass; None if a value lies outside [0, bins)."""
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    out = torch.empty(bins, dtype=torch.int64)
    ok = _C.histogram(values.to(torch.int32).contiguous().reshape(-1), out)
    return out if ok else None


def select_labels(labels: Tensor, select: Tensor) -> Tensor | None:
    """Boolean mask of the elements of a CPU int32 label map whose label is selected,
    ``select[labels]``; None if a label lies outside [0, len(select))."""
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    out = torch.empty(labels.shape, dtype=torch.bool)
    flat = labels.to(torch.int32).contiguous().reshape(-1)
    table = select.to(torch.uint8).contiguous()
    ok = _C.select_labels(flat, table, out.view(torch.uint8).reshape(-1))
    return out if ok else None


def label_overlap(a: Tensor, b: Tensor, na: int, nb: int) -> tuple[Tensor, Tensor, Tensor] | None:
    """Overlap of two CPU int32 label maps of the same shape, labels in [0, na] and [0, nb], in one
    pass: the size of every label (int64, (na + 1,) and (nb + 1,)) and the (i, j, count) rows
    (int64, (M, 3)), in increasing order of (i, j), of the pairs of nonzero labels that meet.
    None if a label is out of range."""
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    flat = [x.to(torch.int32).contiguous().reshape(-1) for x in (a, b)]
    size_a, size_b = torch.empty(na + 1, dtype=torch.int64), torch.empty(nb + 1, dtype=torch.int64)
    # Each pair needs a shared element; pages of the buffer that stay unused are never touched.
    capacity = min(na * nb, flat[0].numel(), 1 << 20)
    pairs = torch.empty((capacity, 3), dtype=torch.int64)
    count = _C.label_overlap(*flat, size_a, size_b, pairs)
    if count > capacity:  # rare: more pairs than the first guess
        pairs = torch.empty((count, 3), dtype=torch.int64)
        count = _C.label_overlap(*flat, size_a, size_b, pairs)
    return (size_a, size_b, pairs[:count]) if count >= 0 else None


def channel_counts(pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor] | None:
    """Per mask of two CPU stacks of binary masks (C, ...): elements in both, in pred and in
    target (int64, (C,)), counted in one pass; None if a value is not 0 or 1."""
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    count = pred.shape[0]
    flat = _channel_inputs(pred, target)
    both, in_pred, in_target = (torch.empty(count, dtype=torch.int64) for _ in range(3))
    ok = _C.channel_counts(*flat, both, in_pred, in_target)
    return (both, in_pred, in_target) if ok else None


def _channel_inputs(pred: Tensor, target: Tensor) -> list[Tensor]:
    """Select a common kernel dtype without converting invalid values to binary ones."""
    if pred.dtype == target.dtype == torch.bool:
        pred, target = pred.view(torch.uint8), target.view(torch.uint8)
    elif pred.dtype != target.dtype or pred.dtype not in (
        torch.uint8, torch.int64, torch.float32, torch.float64
    ):
        # Preserve every floating-point value that could round to 0 or 1 before validation.
        # Integer-to-float conversion can round large integers, but never turns them into 0/1.
        if torch.float64 in (pred.dtype, target.dtype):
            dtype = torch.float64
        elif pred.dtype.is_floating_point or target.dtype.is_floating_point:
            dtype = torch.float32
        else:
            dtype = torch.int64
        pred, target = pred.to(dtype), target.to(dtype)
    count = pred.shape[0]
    flat = [x.contiguous().reshape(count, math.prod(x.shape[1:])) for x in (pred, target)]
    return flat


def masked_channel_counts(pred: Tensor, target: Tensor, mask: Tensor,
                          channels: int) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
    """Binary counts and included elements, sharing each case's mask across its channels."""
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    count = pred.shape[0]
    flat = _channel_inputs(pred, target)
    m = _as_bytes(mask).reshape(mask.shape[0], math.prod(mask.shape[1:]))
    out = tuple(torch.empty(count, dtype=torch.int64) for _ in range(4))
    ok = _C.masked_channel_counts(*flat, m, channels, *out)
    return out if ok else None  # type: ignore[return-value]
