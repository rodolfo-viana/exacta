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


def regression_stats(pred: Tensor, target: Tensor, mask: Tensor | None, weight: Tensor | None,
                     *, moments: bool) -> Tensor:
    """CPU (N,D) errors/moments with fused numerical validation; shapes checked by caller."""
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    dtype = torch.float32 if pred.dtype == target.dtype == torch.float32 else torch.float64
    p, t = (x.to(dtype).contiguous() for x in (pred, target))
    m = torch.empty((0, 0), dtype=torch.uint8) if mask is None else _as_bytes(mask)
    w = torch.empty(0, dtype=torch.float64) if weight is None else weight.double().contiguous()
    out = torch.empty((11, p.shape[1]), dtype=torch.float64)
    if not _C.regression_stats(p, t, m, w, moments, out, torch.get_num_threads()):
        raise ValueError("weights must be finite/nonnegative and observed predictions/targets finite")
    return out


# Largest value of M / h that a pass of the exact distance transform may reach, where h is the
# spacing of the pass and M bounds the squared distances along it. Breakpoints are about
# M / (2 h) and tie tolerances 128 u M / h, so this keeps every intermediate quantity far below
# the binary64 maximum (about 1.8e308), as the exactness argument assumes.
EDT_MAX_SCALE = 1e300

# Longest line the exactness proof covers; far beyond any array that fits in memory.
EDT_MAX_LINE = 2**40

# Spacings for which no quantity of a pass underflows or overflows on its own.
EDT_SPACING_RANGE = (1e-100, 1e100)


def check_edt_extent(shape: Sequence[int], spacing: Sequence[float]) -> None:
    """Raise ``ValueError`` if a grid could overflow the exact squared distance transform.

    ``shape`` and ``spacing`` describe the transformed axes, in order. The pass along an axis of
    spacing ``h`` and ``n`` samples has squared distances up to ``M``, the squared extents of the
    axes before it plus ``(h * n)^2``. Spacings in [1e-100, 1e100] keep ``M`` finite, but an
    extremely anisotropic grid with long lines can still make ``M / h`` overflow, and then the
    result would no longer be the exact minimum. Ordinary images are many orders of magnitude
    inside the limit. The compiled kernel (``exacta_edt_sq_f64``) enforces the same limits, and
    lines of more than ``EDT_MAX_LINE`` samples, so that no caller of the C interface can leave
    the range of the exactness proof.
    """
    valid = all(math.isfinite(float(h)) and float(h) > 0 for h in spacing)
    if len(spacing) != len(shape) or not valid:
        return  # invalid spacings are reported by the kernels' own validation
    bound = 0.0
    for n, h in zip(shape, spacing):
        h = float(h)
        if n > EDT_MAX_LINE:
            raise ValueError(
                f"a line of {n} samples is too long for the exact distance transform, "
                f"which is proved for at most {EDT_MAX_LINE}"
            )
        xmax = h * float(n)
        scale = (bound + xmax * xmax) / h
        if not scale <= EDT_MAX_SCALE:
            raise ValueError(
                f"spacing {tuple(float(s) for s in spacing)} on a grid of shape {tuple(shape)} "
                f"is too anisotropic for the exact distance transform: a pass would reach "
                f"squared distance / spacing = {scale:.3g}, above {EDT_MAX_SCALE:g}"
            )
        extent = h * float(n - 1)
        bound += extent * extent


def edt_sq(features: Tensor, spacing: Sequence[float], batch_ndim: int = 0) -> Tensor:
    """Squared Euclidean distance from every voxel to the nearest nonzero voxel of ``features``.

    Each value is the exact floating-point minimum, over nonzero voxels p, of
    ``sum_i ((p_i - q_i) * h_i)^2`` rounded and summed in axis order, which is the expression
    SciPy's ``distance_transform_edt`` evaluates. The first ``batch_ndim`` axes index separate
    images, and ``spacing`` covers the others. CPU only. Returns float64; ``inf`` throughout an
    image without nonzero voxels. Raises ``ValueError`` for grids so anisotropic that the
    transform could overflow (see :func:`check_edt_extent`); the compiled kernel applies the
    same check, so calling ``exacta._C.edt_sq`` directly raises the same error.
    """
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    check_edt_extent(features.shape[batch_ndim:], spacing)
    features = _as_bytes(features)
    out = torch.empty(features.shape, dtype=torch.float64)
    _C.edt_sq(features, out, [float(s) for s in spacing], batch_ndim)
    return out


def check_edt_lines(f: Tensor, h: float, bound: float) -> None:
    """Raise ``ValueError`` unless ``f`` (``(lines, n)``), ``h`` and ``bound`` meet the assumptions
    of the exactness proof for one pass of :func:`edt_sq`.

    Every entry of ``f`` must be ``inf`` (a sample without a parabola) or lie in ``[0, bound]``, so
    NaN, negative and ``-inf`` values are rejected; ``h`` must lie in ``[1e-100, 1e100]`` and
    ``bound`` be finite and nonnegative; and the pass must satisfy the range limit of
    :func:`check_edt_extent`, ``(bound + (h n)^2) / h <= EDT_MAX_SCALE``, with ``n`` at most
    ``EDT_MAX_LINE``. The compiled kernel ``exacta_edt_lines_f64`` checks the same conditions.
    """
    h, bound = float(h), float(bound)
    if not EDT_SPACING_RANGE[0] <= h <= EDT_SPACING_RANGE[1] or not 0.0 <= bound < math.inf:
        raise ValueError("edt_lines: h must lie in [1e-100, 1e100] and bound be finite and >= 0")
    n = f.shape[-1] if f.ndim else 0
    if n > EDT_MAX_LINE:
        raise ValueError(f"edt_lines: {n} samples per line exceed the {EDT_MAX_LINE} of the proof")
    if not (bound + (h * float(n)) ** 2) / h <= EDT_MAX_SCALE:
        raise ValueError(
            f"edt_lines: a pass with spacing {h:g}, bound {bound:g} and {n} samples is too "
            f"anisotropic for the exact distance transform: (bound + (h n)^2) / h exceeds "
            f"{EDT_MAX_SCALE:g}"
        )
    if f.numel() and not bool(((f == math.inf) | ((f >= 0) & (f <= bound))).all()):
        raise ValueError("edt_lines: every value of f must be inf or lie in [0, bound]")


def edt_lines(f: Tensor, h: float, bound: float) -> Tensor:
    """One pass of :func:`edt_sq` over each row of ``f`` (float64, ``(lines, n)``), for testing.

    ``out[q] = min_p fl(fl(fl(h * (q - p))^2) + f[p])`` over ``p`` with finite ``f[p]``; finite
    values must lie in ``[0, bound]``, and ``inf`` marks samples without a parabola. Raises
    ``ValueError`` if the input leaves the assumptions of the exactness proof (see
    :func:`check_edt_lines`): a value that is NaN, negative or above ``bound``, or a pass whose
    ``(bound + (h n)^2) / h`` exceeds ``EDT_MAX_SCALE``.
    """
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    f = f.to(torch.float64).contiguous()
    check_edt_lines(f, h, bound)
    out = torch.empty_like(f)
    _C.edt_lines(f, out, float(h), float(bound))
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


def exact_sum(x: Tensor, w: Tensor | None = None, select: Tensor | None = None) -> float:
    """Correctly rounded sum of ``x * w`` (each product rounded once) over ``select``, as
    :func:`math.fsum` would return it: the same bits for any order of the terms, number of
    threads or device. Tensors are copied to the CPU as float64.

    The sum is computed serially on the host, so a GPU input pays for the copy and nothing else
    can make the result depend on the device. If a partial sum overflows, the result is ``inf``
    (or ``-inf``) even when the exact sum is finite, whereas :func:`math.fsum` raises
    ``OverflowError``; the order of the terms matters only in that case.
    """
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    flat = x.detach().to("cpu", torch.float64).contiguous().reshape(-1)
    weights = None if w is None else w.detach().to("cpu", torch.float64).contiguous().reshape(-1)
    chosen = None if select is None else _as_bytes(select.detach().cpu()).reshape(-1)
    return float(_C.exact_sum(flat, weights, chosen))


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


def probability_stats(pred: Tensor, target: Tensor, mask: Tensor | None, from_logits: bool,
                      kind: int, bins: int) -> Tensor | None:
    """Fused CPU validation and float64 Bernoulli sufficient statistics."""
    if _C is None:
        raise RuntimeError("exacta was installed without its compiled extension")
    batch, channels = pred.shape[:2]
    rows, n = batch * channels, math.prod(pred.shape[2:])
    if pred.dtype not in (torch.float32, torch.float64):
        pred = pred.float()
    if target.dtype == torch.bool:
        target = target.view(torch.uint8)
    elif target.dtype not in (torch.uint8, torch.int64, torch.float32, torch.float64):
        target = target.float() if target.is_floating_point() else target.long()
    p, t = (x.contiguous().reshape(rows, n) for x in (pred, target))
    m = (torch.empty((0, 0), dtype=torch.uint8) if mask is None
         else _as_bytes(mask).reshape(batch * mask.shape[1], n))
    out = torch.empty((rows, 3 * bins if kind == 2 else 2), dtype=torch.float64)
    if not _C.probability_stats(p, t, m, channels, from_logits, kind, bins, out):
        return None
    return out.reshape(batch, channels, 3, bins) if kind == 2 else out.reshape(batch, channels, 2)
