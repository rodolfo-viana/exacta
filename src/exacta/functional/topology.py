"""Hard centreline Dice on thinned skeletons, independent of the soft training surrogate."""
from __future__ import annotations

import math
from typing import Any, Literal

import torch
from torch import Tensor

from exacta import _native
from exacta._inputs import InputFormat, like_input, to_torch, validate


SkeletonMethod = Literal["zhang", "lee"]


@torch.no_grad()
def skeletonize(mask: Any, *, method: SkeletonMethod | None = None) -> Any:
    """Thin one binary 2-D/3-D mask to a one-voxel-wide skeleton, returning a boolean array.

    ``method`` follows scikit-image's ``skeletonize``, whose output the compiled C++ kernels
    reproduce exactly: ``"zhang"`` (Zhang & Suen 1984, with scikit-image's lookup table; 2-D
    only) or ``"lee"`` (Lee, Kashyap & Chu 1994; 2-D or 3-D). ``None`` picks scikit-image's
    default: Zhang in 2-D, Lee in 3-D. The two give different 2-D skeletons.

    CUDA inputs are copied to CPU and back: both algorithms are sequential across
    subiterations. For repeated GPU metric evaluation, precompute skeletons explicitly.
    SoftClDiceLoss stays on the GPU throughout. The image exterior is background. No spatial
    resampling is performed.
    """
    x = to_torch(mask)
    if x.ndim not in (2, 3):
        raise ValueError("skeletonize expects one 2-D or 3-D mask")
    method = _skeleton_method(method, x.ndim)
    if not bool(((x == 0) | (x == 1)).all()):
        raise ValueError("mask must contain only 0 and 1")
    if not _native.available():
        raise RuntimeError("skeletonize requires Exacta's compiled extension")
    source = x.to(device="cpu", dtype=torch.uint8).contiguous()
    result = torch.empty_like(source)
    (_native._C.skeleton_zhang if method == "zhang" else _native._C.skeleton)(source, result)
    return like_input(result.bool().to(x.device), mask)


def _skeleton_method(method: str | None, ndim: int) -> SkeletonMethod:
    if method is None:
        return "zhang" if ndim == 2 else "lee"
    if method == "zhang":
        if ndim != 2:
            raise ValueError("Zhang thinning is defined for 2-D masks only; use method='lee' in 3-D")
        return "zhang"
    if method == "lee":
        return "lee"
    raise ValueError(f"method must be 'zhang', 'lee' or None, got {method!r}")


@torch.no_grad()
def cldice(pred: Any, target: Any, *, input_format: InputFormat = "index",
           num_classes: int | None = None, include_background: bool = True,
           both_empty: float = math.nan, pred_skeleton: Any = None,
           target_skeleton: Any = None, method: SkeletonMethod | None = None) -> Any:
    """Hard clDice: harmonic mean of skeleton precision and skeleton sensitivity.

    Returns (B, C) float64 in the prediction's array library. One-empty masks give 0;
    both-empty use ``both_empty``. A nonempty mask with an empty skeleton gets
    score 0 (thinning can erase even-width objects). Precomputed skeletons must
    be binary subsets of their masks, with shape (B, selected_C, *spatial).
    Skeleton voxels are counted uniformly; anisotropic voxels are not resampled.

    ``method`` selects the thinning (see :func:`skeletonize`). The default, Zhang in 2-D and
    Lee in 3-D, is scikit-image's and therefore the clDice authors' reference metric
    (Shit et al., CVPR 2021); ``method="lee"`` gives the 2-D values of Exacta < 0.0.1a6.
    """
    seg = validate(to_torch(pred), to_torch(target), input_format=input_format,
                   num_classes=num_classes, include_background=include_background)
    method = _skeleton_method(method, seg.ndim)
    shape = (seg.batch, len(seg.classes), *seg.pred.shape[-seg.ndim:])
    cached: list[Tensor | None] = []
    for value in (pred_skeleton, target_skeleton):
        if value is None:
            cached.append(None)
            continue
        x = to_torch(value)
        if x.shape != shape or x.device != seg.pred.device:
            raise ValueError(f"skeleton must have shape {shape} and the segmentation's device")
        if not bool(((x == 0) | (x == 1)).all()):
            raise ValueError("skeleton must be binary")
        cached.append(x.bool())
    out = torch.empty(shape[:2], dtype=torch.float64, device=seg.pred.device)
    for b in range(seg.batch):
        for col, cls in enumerate(seg.classes):
            p, t = seg.masks(b, cls)
            sp = skeletonize(p, method=method) if cached[0] is None else cached[0][b, col]
            st = skeletonize(t, method=method) if cached[1] is None else cached[1][b, col]
            if bool((sp & ~p).any() | (st & ~t).any()):
                raise ValueError("skeleton must be a subset of its mask")
            np, nt = sp.sum().double(), st.sum().double()
            a, b_hits = (sp & t).sum().double(), (st & p).sum().double()
            denominator = a * nt + b_hits * np
            value = 2 * a * b_hits / denominator.clamp_min(1)
            out[b, col] = torch.where(~p.any() & ~t.any(), both_empty, value)
    return like_input(out, pred)


@torch.no_grad()
def betti_numbers(seg: Any, *, connectivity: int, input_format: InputFormat = "index",
                  num_classes: int | None = None, include_background: bool = True) -> Any:
    """Betti numbers of every case and class: (B, C, 2) ``[β0, β1]`` in 2-D (components,
    holes) and (B, C, 3) ``[β0, β1, β2]`` in 3-D (components, tunnels, cavities), int64.

    ``connectivity`` is the foreground neighbourhood, as in :func:`exacta.components.label`:
    1 (4 / 6 neighbours) or ``ndim`` (8 / 26). The background uses the complementary
    neighbourhood (8 / 26 and 4 / 6 respectively), the pairing under which digital topology
    matches the continuous one; 18-connectivity has no such partner here and is rejected. The
    image exterior is background, so a hole or cavity must be enclosed inside the image.

    β0 counts foreground components; the last Betti number (holes in 2-D, cavities in 3-D)
    counts background components that do not reach the exterior; in 3-D, β1 follows from the
    Euler characteristic ``χ = β0 - β1 + β2``. χ is counted on the cell complex matching the
    connectivity: closed voxels for the full neighbourhood (a vertex, edge or face belongs to
    the set when any voxel around it does) and open ones for faces only (when every voxel does),
    as scikit-image's ``euler_number`` counts it. ``seg`` follows the layout of
    :func:`exacta.functional.dice` (index or binary one-hot, 2-D or 3-D).
    """
    s = validate(to_torch(seg), to_torch(seg), input_format=input_format, num_classes=num_classes,
                 include_background=include_background)
    full = _check_topology_connectivity(connectivity, s.ndim)
    out = torch.zeros((s.batch, len(s.classes), s.ndim), dtype=torch.int64)
    for b in range(s.batch):
        for col, cls in enumerate(s.classes):
            mask = s.masks(b, cls)[0]
            out[b, col] = torch.tensor(_betti(mask, connectivity, full))
    return like_input(out.to(s.pred.device), seg)


@torch.no_grad()
def betti_error(pred: Any, target: Any, *, connectivity: int, dimension: int = 0,
                input_format: InputFormat = "index", num_classes: int | None = None,
                include_background: bool = True) -> Any:
    """``|β_k(P) - β_k(T)|`` for ``k = dimension`` (0: components, 1: holes or tunnels, 2:
    cavities in 3-D), per case and class, as float64 (B, C). The TopCoW challenges report the
    Betti-0 error. Betti numbers ignore where structures are: two errors can cancel (a missed
    component and a spurious one). See :func:`betti_numbers` for the connectivity convention.
    """
    p, t = to_torch(pred), to_torch(target)
    validate(p, t, input_format=input_format, num_classes=num_classes, include_background=include_background)
    ndim = p.ndim - (1 if input_format == "index" else 2)
    if isinstance(dimension, bool) or not isinstance(dimension, int) or not 0 <= dimension < ndim:
        raise ValueError(f"dimension must be an integer in [0, {ndim - 1}], got {dimension!r}")
    bp, bt = (to_torch(betti_numbers(x, connectivity=connectivity, input_format=input_format,
                                     num_classes=num_classes, include_background=include_background))
              for x in (p, t))
    return like_input((bp[..., dimension] - bt[..., dimension]).abs().double(), pred)


def _check_topology_connectivity(connectivity: int, ndim: int) -> bool:
    if isinstance(connectivity, bool) or connectivity not in (1, ndim):
        raise ValueError(f"connectivity must be 1 or {ndim} for {ndim}-D Betti numbers, got {connectivity!r}")
    return connectivity == ndim


def _betti(mask: Tensor, connectivity: int, full: bool) -> list[int]:
    from exacta.components import _label

    ndim = mask.ndim
    padded = torch.nn.functional.pad(mask.to(torch.uint8), (1, 1) * ndim).bool()
    b0 = _label(padded, connectivity, "auto")[1] if bool(padded.any()) else 0
    # Background components of the padded image, with the complementary neighbourhood; the one
    # that holds the padding is the exterior.
    last = _label(~padded, 1 if full else ndim, "auto")[1] - 1
    if ndim == 2:
        return [b0, last]
    return [b0, b0 + last - _euler(padded, full), last]


def _euler(x: Tensor, full: bool) -> int:
    """Euler characteristic of a zero-padded 2-D/3-D mask: cells of every dimension, each present
    when any (closed voxels, full connectivity) or every (open voxels) voxel around it is set."""
    combine = torch.logical_or if full else torch.logical_and
    ndim = x.ndim
    total = 0
    # A cell of dimension ndim - len(axes) is shared by the 2**len(axes) voxels obtained by
    # stepping along ``axes``.
    for size in range(ndim + 1):
        for axes in _subsets(ndim, size):
            cells = x
            for axis in axes:
                cells = combine(cells.narrow(axis, 0, cells.shape[axis] - 1), cells.narrow(axis, 1, cells.shape[axis] - 1))
            total += (-1) ** size * int(cells.sum())
    # total = Σ (-1)^codimension · cells: F - E + V in 2-D, C - F + E - V in 3-D. Closed voxels
    # form a compact complex, whose χ is the alternating sum by dimension: total in 2-D, -total
    # in 3-D. Open voxels form an open set O (the interior of the closed union), whose cell count
    # is the compactly supported χ_c(O) = (-1)^n χ(O): χ = total in both dimensions.
    return -total if full and ndim == 3 else total


def _subsets(n: int, size: int) -> list[tuple[int, ...]]:
    from itertools import combinations

    return list(combinations(range(n), size))
