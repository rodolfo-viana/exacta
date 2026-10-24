"""Surface-distance metrics: Hausdorff distance (HD, HD95), average symmetric surface distance
(ASSD) and normalized surface Dice (NSD).

Every metric measures, for each element of one mask's surface, the Euclidean distance to the
nearest element of the other mask's surface, in the physical units of ``spacing``. ``d(A -> B)``
denotes these distances from the surface of A to that of B; P is the prediction and T the target.
Voxels outside the image count as background.

``surface`` chooses what a surface element is:

    * ``"voxels"``: the boundary voxels, i.e. mask voxels with at least one face-adjacent
      neighbour outside the mask, located at voxel centres, each with equal weight. This is
      MONAI's default and MedPy's definition.
    * ``"surfels"``: DeepMind's surface elements (surface-distance; MONAI's
      ``use_subvoxels=True``). Each 2x2(x2) block of voxels that straddles the surface gives one
      element, located at the voxel corner the block shares and weighted by the area (3-D) or
      length (2-D) of the marching-cubes surface inside the block. Weights replace counts in every
      statistic below, and percentiles become area-weighted: the smallest distance at which the
      cumulative area reaches q% (no interpolation), as in DeepMind's ``compute_robust_hausdorff``.
    * ``"mesh"``: MeshMetrics' definition (Podobnik & Vrtovec). Each mask is meshed with VTK's
      discrete marching cubes (contoured with discrete flying edges in 2-D), and every triangle
      (segment) is an element located at its centre and weighted by its area (length). Distances
      run from those centres to the nearest point of the other mesh, not to its elements, and are
      computed exactly in float64 on the CPU. Percentiles are area-weighted as for surfels.
      Elements that both meshes share (wherever the masks agree locally) are at distance exactly 0;
      MeshMetrics reports ~1e-16 for them (a centre involves thirds, which binary cannot represent),
      which matters for an NSD tolerance of exactly 0. Other ties between a distance and the
      tolerance remain subject to floating-point rounding, as in any implementation.

HD and ASSD reduce each direction with a statistic ``s`` (for HD the q-th percentile, with linear
interpolation for voxels as in NumPy's default; for ASSD the mean) and join the two directions
with ``combine``:

    * ``"max"``: ``max(s(d(P -> T)), s(d(T -> P)))``
    * ``"mean"``: ``(s(d(P -> T)) + s(d(T -> P))) / 2``
    * ``"pooled"``: ``s`` over both sets of distances together

Tools disagree on these choices, so they are arguments rather than fixed definitions:

    ===========  ========  =======  ==================================
    Tool         surface   HD/HD95  ASSD
    ===========  ========  =======  ==================================
    MONAI        voxels    max      pooled (``symmetric=True``)
    MedPy 0.5.2  voxels    pooled   pooled
    MedPy 0.4    voxels    pooled   mean
    DeepMind     surfels   max      both directed means are returned
    MeshMetrics  mesh      max      pooled (``assd``), mean (``masd``)
    ===========  ========  =======  ==================================

MeshMetrics measures 2-D distances to an extruded contour that VTK stores in float32, so its 2-D
values differ from exacta's by float32 rounding (about 1e-7 relative); 3-D values agree to 1e-9.

NSD at tolerance ``tau`` (Nikolov et al., JMIR 2021) is the fraction of the surface, of both
masks together, that lies within ``tau`` of the other surface:
``(#{d(P -> T) <= tau} + #{d(T -> P) <= tau}) / (#d(P -> T) + #d(T -> P))``, with ``#`` counting
voxels or summing surfel areas. This is MONAI's ``compute_surface_dice`` (``use_subvoxels`` picks
the representation) and, for surfels, DeepMind's ``compute_surface_dice_at_tolerance``.
"""

from __future__ import annotations

import functools
import itertools
import math
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Literal, NamedTuple

import numpy as np
import torch
from torch import Tensor

from exacta import _native, _triton_edt
from exacta._inputs import InputFormat, Segmentations, as_spacing, like_input, to_torch, validate
from exacta.functional._mesh_tables import SEGMENTS_2D, TRIANGLES_3D
from exacta.functional._surfel_tables import surfel_measures

Backend = Literal["auto", "native", "triton", "torch"]
Combine = Literal["max", "mean", "pooled"]
Surface = Literal["voxels", "surfels", "mesh"]

EdtSq = Callable[..., Tensor]  # (features, spacing, batch_ndim=0) -> squared distances

# Elements of the (lines, n, n) block evaluated at once by the PyTorch distance transform (128 MiB).
_CHUNK = 1 << 24
# Memory for the distance maps of the classes transformed together (one batched transform). Only
# on GPU: there it turns many small launches into a few full ones, while on CPU each class's
# transform already uses every core and padding classes to a common shape only adds work.
_BATCH_BYTES = 512 << 20

Box = tuple[slice, ...]


class _Distances(NamedTuple):
    """Distances from the surface elements of one mask to the other mask's surface, and the
    elements' areas or lengths (``None`` when every element counts once)."""

    d: Tensor
    w: Tensor | None


def hausdorff_distance(
    pred: Any,
    target: Any,
    *,
    spacing: Any,
    percentile: float = 100.0,
    combine: Combine = "max",
    surface: Surface = "voxels",
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
            Values must lie in [1e-100, 1e100] for voxels and [1e-50, 1e50] for surfels/mesh.
        percentile: 100 gives the Hausdorff distance, 95 gives HD95.
        combine: How the two directions are joined: ``"max"`` (MONAI, DeepMind), ``"pooled"``
            (MedPy) or ``"mean"``.
        surface: ``"voxels"`` (MONAI, MedPy), area-weighted ``"surfels"`` (DeepMind) or
            ``"mesh"`` (MeshMetrics).
        num_classes, include_background: As in :func:`exacta.functional.dice`.
        both_empty: Value when the class is absent from both prediction and target.
        one_empty: Value when the class is absent from exactly one of them.
        backend: Exact distance transform to use: ``"native"`` (compiled kernel, CPU),
            ``"triton"`` (GPU kernel) or ``"torch"`` (plain PyTorch, any device, much slower).
            ``"auto"`` picks native on CPU and Triton on GPU. All give the same values, bit for bit.

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
        seg, spacing, score, surface=surface, both_empty=both_empty, one_empty=one_empty, backend=backend
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
    surface: Surface = "voxels",
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
            of the two directed means (MedPy 0.4; the mean of the pair DeepMind returns) or
            ``"max"`` of them.
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
        seg, spacing, score, surface=surface, both_empty=both_empty, one_empty=one_empty, backend=backend
    )
    return like_input(out, pred)


def nsd(
    pred: Any,
    target: Any,
    *,
    spacing: Any,
    tolerance: Any,
    surface: Surface = "voxels",
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
    distance that ties with the tolerance lands on the same side as in MedPy and DeepMind, which
    use SciPy. (SciPy itself returns a value one ulp above that minimum at a few voxels per
    million.) MONAI casts distances to float32 before comparing them with the tolerance, which
    agrees on exact ties but also counts distances within float32 resolution of the tolerance.

    Args:
        tolerance: Accepted boundary deviation, in the units of ``spacing``. A scalar, or one
            value per evaluated class (``num_classes - 1`` values when the background is excluded).
        one_empty: Value when the class is absent from exactly one of prediction and target.
            The default 0 is the limit of the definition: no surface element has a partner.
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
        seg, spacing, score, surface=surface, both_empty=both_empty, one_empty=one_empty, backend=backend
    )
    return like_input(out, pred)


def _per_case(
    seg: Segmentations,
    spacing: Any,
    score: Callable[[_Distances, _Distances, int], float],
    *,
    surface: Surface,
    both_empty: float,
    one_empty: float,
    backend: Backend,
) -> Tensor:
    """Apply ``score(d(P -> T), d(T -> P), class position)`` to every case and class."""
    return _per_case_many(seg, spacing, [(score, both_empty, one_empty)], surface=surface, backend=backend)[..., 0]


def _per_case_many(
    seg: Segmentations,
    spacing: Any,
    scores: Sequence[tuple[Callable[[_Distances, _Distances, int], float], float, float]],
    *,
    surface: Surface,
    backend: Backend,
) -> Tensor:
    """Reduce several metrics immediately, retaining distance buffers only for one chunk."""
    if surface == "voxels":
        elements = _voxel_distances
    elif surface == "surfels":
        elements = _surfel_distances
    elif surface == "mesh":
        elements = _mesh_distances
    else:
        raise ValueError(f"surface must be 'voxels', 'surfels' or 'mesh', got {surface!r}")
    spacings = as_spacing(spacing, batch=seg.batch, ndim=seg.ndim, area_weighted=surface != "voxels")
    edt_sq = _edt_function(backend, seg.pred.device)

    out = torch.tensor([s[1] for s in scores], dtype=torch.float64).expand(seg.batch, len(seg.classes), -1).clone()
    one_empty = torch.tensor([s[2] for s in scores], dtype=torch.float64)
    for b in range(seg.batch):
        # Bounding boxes of every class at once, read back in one sync. Working inside the box of
        # both masks is exact: outside it both are background, as outside the image.
        present, lo, hi = (x.tolist() for x in seg.extents(b))
        jobs: list[tuple[int, int, Box]] = []
        for k, cls in enumerate(seg.classes):
            has_p, has_t = present[0][cls], present[1][cls]
            if not (has_p or has_t):
                continue
            if not (has_p and has_t):
                out[b, k] = one_empty
                continue
            box = tuple(
                slice(min(lo_p, lo_t), max(hi_p, hi_t) + 1)
                for lo_p, lo_t, hi_p, hi_t in zip(lo[0][cls], lo[1][cls], hi[0][cls], hi[1][cls])
            )
            jobs.append((k, cls, box))
        # Classes share batched distance transforms, so many small structures cost a few kernel
        # launches rather than several each.
        for chunk in _chunks(jobs, _BATCH_BYTES if seg.pred.device.type == "cuda" else 0):
            masks = [seg.masks(b, cls, box) for _, cls, box in chunk]
            for (k, _, _), distances in zip(chunk, elements(masks, spacings[b], edt_sq)):
                for j, (score, _, _) in enumerate(scores):
                    out[b, k, j] = score(*distances, k)
    return out.to(seg.pred.device)


def _chunks(jobs: list[tuple[int, int, Box]], budget: int) -> Iterator[list[tuple[int, int, Box]]]:
    """Group classes whose crops, padded to a common shape, fit ``budget`` bytes."""
    jobs = sorted(jobs, key=lambda job: math.prod(s.stop - s.start for s in job[2]))  # similar sizes
    chunk: list[tuple[int, int, Box]] = []
    shape: list[int] = []
    for job in jobs:
        size = [s.stop - s.start + 1 for s in job[2]]  # + 1: surfels live on the corner grid
        padded = [max(a, b) for a, b in zip(shape, size)] if chunk else size
        # two masks per class; a bool input and two float64 maps per element
        if chunk and 2 * (len(chunk) + 1) * math.prod(padded) * 17 > budget:
            yield chunk
            chunk, padded = [], size
        chunk.append(job)
        shape = padded
    if chunk:
        yield chunk


def _edt_stack(features: list[Tensor], spacing: Sequence[float], edt_sq: EdtSq) -> list[Tensor]:
    """Squared distance maps of several images, in one batched transform on GPU.

    Each image is padded with background to a common shape. Padding adds no features, so the
    distances inside each image are unchanged. On CPU, where batching does not pay (see
    ``_BATCH_BYTES``), each image is transformed on its own, without the copy.
    """
    if features[0].device.type != "cuda":
        return [edt_sq(f, spacing) for f in features]
    shape = [max(f.shape[axis] for f in features) for axis in range(features[0].ndim)]
    stack = features[0].new_zeros((len(features), *shape))
    for i, f in enumerate(features):
        stack[i][_region(f)] = f
    maps = edt_sq(stack, spacing, 1)
    return [maps[i][_region(f)] for i, f in enumerate(features)]


def _region(x: Tensor) -> Box:
    """Where ``x`` sits in an array padded at the end of every axis."""
    return tuple(slice(0, n) for n in x.shape)


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
    if x.w is not None:
        return _weighted_percentile(x.d.cpu().numpy(), x.w.cpu().numpy(), q)
    if q == 100:
        return x.d.max().item()
    # NumPy's CPU sort is vectorised and many times faster than PyTorch's; same order, NaN last.
    d = torch.from_numpy(np.sort(x.d.numpy())) if x.d.device.type == "cpu" else x.d.sort().values
    pos = q / 100 * (d.numel() - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, d.numel() - 1)
    return (d[lo] + (d[hi] - d[lo]) * (pos - lo)).item()


def _weighted_percentile(d: np.ndarray, w: np.ndarray, q: float) -> float:
    """Smallest distance at which the cumulative weight reaches q%, computed as DeepMind does.

    NumPy, not PyTorch, so the cumulative sum and the total round as DeepMind's do: with equal
    areas the cumulative fraction can hit q% exactly, and a one-ulp difference moves the result.
    """
    # DeepMind sorts (distance, area) pairs; ties have equal values, so any such sort will do.
    pairs = _native.sort_pairs(d, w) if _native.available() else None
    if pairs is None:  # NaN, or no compiled extension
        order = np.lexsort((w, d))
        pairs = d[order], w[order]
    d, w = pairs
    cum = np.cumsum(w) / np.sum(w)
    return float(d[min(int(np.searchsorted(cum, q / 100.0)), d.size - 1)])


def _mean(x: _Distances) -> float:
    if x.w is None:
        return x.d.mean().item()
    return ((x.d * x.w).sum() / x.w.sum()).item()


def _within(x: _Distances, tau: float) -> tuple[float, float]:
    """Amount of surface (voxel count, or surfel area) within ``tau``, and the total amount."""
    inside = x.d <= tau
    if x.w is None:
        return torch.count_nonzero(inside).item(), x.d.numel()
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


def _edt_function(backend: Backend, device: torch.device) -> EdtSq:
    if backend == "auto":
        if device.type == "cpu":
            backend = "native" if _native.available() else "torch"
        else:
            backend = "triton" if device.type == "cuda" and _triton_edt.available() else "torch"
    if backend == "torch":
        return _edt_sq_torch
    if backend == "native":
        if device.type != "cpu":
            raise NotImplementedError("the native kernel runs on CPU; use backend='triton' or 'auto' on GPU")
        if not _native.available():
            raise RuntimeError("exacta was installed without its compiled extension")
        return _native.edt_sq
    if backend == "triton":
        if device.type != "cuda":
            raise NotImplementedError("the Triton kernel runs on GPU; use backend='native' or 'auto' on CPU")
        if not _triton_edt.available():
            raise RuntimeError("backend='triton' needs Triton, which this PyTorch build does not include")
        return _triton_edt.edt_sq
    raise ValueError(f"backend must be 'auto', 'native', 'triton' or 'torch', got {backend!r}")


def _voxel_distances(
    masks: list[tuple[Tensor, Tensor]], spacing: Sequence[float], edt_sq: EdtSq
) -> list[tuple[_Distances, _Distances]]:
    """For each (prediction, target) pair, cropped to the bounding box of both masks (which is
    also that of their boundaries): distances between the boundary voxels."""
    boundaries = [(_boundary(p, p.ndim), _boundary(t, t.ndim)) for p, t in masks]
    maps = _edt_stack([b for pair in boundaries for b in pair], spacing, edt_sq)
    out = []
    for i, (on_p, on_t) in enumerate(boundaries):
        to_p, to_t = maps[2 * i], maps[2 * i + 1]
        out.append((_Distances(_sqrt_at(to_t, on_p), None), _Distances(_sqrt_at(to_p, on_t), None)))
    return out


def _surfel_distances(
    masks: list[tuple[Tensor, Tensor]], spacing: Sequence[float], edt_sq: EdtSq
) -> list[tuple[_Distances, _Distances]]:
    """For each (prediction, target) pair, cropped to the bounding box of both masks: distances
    between surface elements, and their areas. The crop is exact here too, as in DeepMind's
    implementation: every block that straddles a surface has its corner on the box or inside it."""
    codes = [(_neighbour_code(p), _neighbour_code(t)) for p, t in masks]
    full = (1 << (1 << masks[0][0].ndim)) - 1  # every voxel of the block inside: no surface
    on = [((cp != 0) & (cp != full), (ct != 0) & (ct != full)) for cp, ct in codes]
    maps = _edt_stack([x for pair in on for x in pair], spacing, edt_sq)
    measure = torch.from_numpy(surfel_measures(tuple(spacing))).to(masks[0][0].device)
    out = []
    for i, ((code_p, code_t), (on_p, on_t)) in enumerate(zip(codes, on)):
        to_p, to_t = maps[2 * i], maps[2 * i + 1]
        out.append(
            (
                _Distances(_sqrt_at(to_t, on_p), measure[code_p[on_p].long()]),
                _Distances(_sqrt_at(to_p, on_t), measure[code_t[on_t].long()]),
            )
        )
    return out


def _mesh_distances(
    masks: list[tuple[Tensor, Tensor]], spacing: Sequence[float], edt_sq: EdtSq
) -> list[tuple[_Distances, _Distances]]:
    """For each (prediction, target) pair: distances from the centre of every triangle (3-D) or
    segment (2-D) of one mesh to the other mesh, and the elements' areas or lengths. MeshMetrics'
    definition, computed on the CPU."""
    out = []
    for p, t in masks:
        mesh_p, mesh_t = _mesh(p, spacing), _mesh(t, spacing)
        d_pt = _native.point_mesh_distances(mesh_p.centres, mesh_t.elements)
        d_tp = _native.point_mesh_distances(mesh_t.centres, mesh_p.elements)
        # An element both meshes share (wherever the masks agree locally) holds its own centre, so
        # its distance is 0. Floating point cannot show it (a centre involves thirds, which binary
        # cannot represent, so it lands ~1e-16 off the plane), but the block codes can.
        d_pt[_shared(mesh_p, mesh_t)] = 0.0
        d_tp[_shared(mesh_t, mesh_p)] = 0.0
        out.append((_Distances(d_pt, mesh_p.sizes), _Distances(d_tp, mesh_t.sizes)))
    return out


def _shared(a: _Mesh, b: _Mesh) -> Tensor:
    """Which elements of mesh ``a`` mesh ``b`` also has, for masks of the same shape.

    No element lies on the boundary of its block (checked in tests/test_mesh.py), so two meshes
    can only share an element within the same block, where it depends on the two codes alone.
    """
    other = b.code.reshape(-1)[a.block].long()  # b's code at the blocks of a's elements
    return _shared_table(a.code.ndim)[a.element_code, a.slot, other]


@functools.lru_cache(maxsize=None)
def _shared_table(ndim: int) -> Tensor:
    """``[a, k, b]``: whether element k of neighbour code a is also an element of code b."""
    cells = TRIANGLES_3D if ndim == 3 else SEGMENTS_2D
    # an element as its set of vertices, so that vertex order does not matter
    vertex_sets = [[frozenset(e[i * ndim : (i + 1) * ndim] for i in range(ndim)) for e in c] for c in cells]
    by_code = [set(sets) for sets in vertex_sets]
    table = torch.zeros((len(cells), max(map(len, cells)), len(cells)), dtype=torch.bool)
    for a, sets in enumerate(vertex_sets):
        for k, element in enumerate(sets):
            table[a, k] = torch.tensor([element in other for other in by_code])
    return table


class _Mesh(NamedTuple):
    """A mask's mesh: elements (M, d, d), centres (M, d) and sizes (M,) in physical units (float64,
    CPU); the neighbour codes of the corner grid, and for each element its block (flat index into
    that grid), the block's code and its position among that code's elements."""

    elements: Tensor
    centres: Tensor
    sizes: Tensor
    code: Tensor
    block: Tensor
    element_code: Tensor
    slot: Tensor


def _mesh(mask: Tensor, spacing: Sequence[float]) -> _Mesh:
    """MeshMetrics' mesh of a mask: its triangles (3-D) or segments (2-D), see :class:`_Mesh`.

    VTK's discrete marching cubes (flying edges in 2-D), tabulated per 2x2(x2) block of voxels:
    the block whose neighbour code sits at corner index ``c`` starts at voxel ``c - 1``.
    """
    if not _native.available():
        raise RuntimeError("surface='mesh' needs exacta's compiled extension")
    table, counts = _mesh_table(mask.ndim)
    code, total = _native.neighbour_codes(mask.cpu(), counts)
    # Vertices (h / 2) * spacing for half-voxel coordinates h, and VTK's parametric centre of a
    # triangle (the midpoint of a segment), evaluated as VTK does.
    elements, centres, block, element_code, slot = _native.mesh_elements(
        code, table, counts, spacing, total
    )
    if mask.ndim == 3:
        edges = elements[:, 1:] - elements[:, :1]
        sizes = 0.5 * torch.linalg.cross(edges[:, 0], edges[:, 1]).norm(dim=1)
    else:
        sizes = (elements[:, 1] - elements[:, 0]).norm(dim=1)
    return _Mesh(elements, centres, sizes, code, block, element_code, slot)


@functools.lru_cache(maxsize=None)
def _mesh_table(ndim: int) -> tuple[Tensor, Tensor]:
    """Elements of each neighbour code (codes, most per code, d, d) in half-voxels, and their counts."""
    cells = TRIANGLES_3D if ndim == 3 else SEGMENTS_2D
    table = torch.zeros((len(cells), max(map(len, cells)), ndim, ndim), dtype=torch.int64)
    for code, elements in enumerate(cells):
        for k, element in enumerate(elements):
            table[code, k] = torch.tensor([int(x) for x in element]).view(ndim, ndim)
    return table, torch.tensor([len(c) for c in cells])


def _neighbour_code(mask: Tensor) -> Tensor:
    """DeepMind's neighbour code at every voxel corner, including those on the image border.

    The code of a corner has one bit per voxel of the 2x2(x2) block around it, set when the
    voxel is in the mask: in 3-D the block's voxel (a, b, c) carries bit 7 - (4a + 2b + c),
    so (0, 0, 0) is 128 and (1, 1, 1) is 1. Voxels outside the image count as outside.
    """
    if mask.device.type == "cpu" and _native.available():
        return _native.neighbour_codes(mask)[0]
    return _neighbour_code_torch(mask)


def _neighbour_code_torch(mask: Tensor) -> Tensor:
    padded = torch.nn.functional.pad(mask.to(torch.uint8), (1, 1) * mask.ndim)
    code = torch.zeros(tuple(n + 1 for n in mask.shape), dtype=torch.uint8, device=mask.device)
    bits = 1 << mask.ndim
    for i, offset in enumerate(itertools.product((0, 1), repeat=mask.ndim)):
        block = padded[tuple(slice(o, o + n + 1) for o, n in zip(offset, mask.shape))]
        code += block << (bits - 1 - i)
    return code


def _boundary(mask: Tensor, ndim: int) -> Tensor:
    """Mask voxels with a face-adjacent neighbour outside the mask or outside the image."""
    if mask.ndim == ndim and 1 <= ndim <= 3 and mask.device.type == "cpu" and _native.available():
        return _native.boundary(mask)  # one pass, reading crops in place
    return _boundary_torch(mask, ndim)


def _boundary_torch(mask: Tensor, ndim: int) -> Tensor:
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


def _sqrt_at(x: Tensor, mask: Tensor) -> Tensor:
    """``_sqrt(x[mask])``, natively on CPU: one pass instead of a gather and a square root."""
    if x.device.type == "cpu" and _native.available():
        return _native.masked_sqrt(x, mask)
    return _sqrt(x[mask])


def _sqrt(x: Tensor) -> Tensor:
    """Correctly rounded square root of a float64 tensor.

    PyTorch's CPU float64 sqrt is one ulp off for about 1% of inputs (seen in 2.14), which can
    decide NSD's ``d <= tolerance`` when the tolerance is itself a distance between voxel centres
    (e.g. a voxel diagonal). NumPy's sqrt is correctly rounded, as is CUDA's.
    """
    if x.device.type == "cpu":
        return torch.from_numpy(np.sqrt(x.numpy()))
    return x.sqrt()


def _edt_sq_torch(features: Tensor, spacing: Sequence[float], batch_ndim: int = 0) -> Tensor:
    """Exact squared Euclidean distance transform in PyTorch, for any device.

    Squared distance is separable, so one 1-D min-plus transform per axis is exact:
    ``out[q] = min_p (h*(q - p))^2 + in[p]``. Rounding is monotonic, so the per-pass minima give
    the exact floating-point minimum of the whole expression, as :func:`exacta._native.edt_sq`
    does. Each pass is brute force, O(n) per voxel along the axis, but fully vectorised.
    """
    out = torch.zeros(features.shape, dtype=torch.float64, device=features.device)
    out.masked_fill_(~features, math.inf)
    if out.numel() == 0:
        return out
    for axis, h in enumerate(spacing, start=batch_ndim):
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
