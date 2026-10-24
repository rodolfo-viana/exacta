"""Connected-component labelling on GPU, in Triton.

Lock-free union-find over voxel indices. Every foreground voxel starts as its own parent; a merge
kernel unites each voxel with its earlier neighbours by linking the larger root to the smaller
with an atomic minimum, retrying when another lane changed that root first. Links only point to
smaller indices, so each component's root is its smallest linear index, i.e. its first voxel in
C order. A compression kernel points every voxel at its root, and a prefix sum over roots then
numbers components 1, 2, ... in order of first appearance: the numbering of scipy.ndimage.label
and of the CPU kernel, independent of how threads are scheduled.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only PyTorch builds ship without Triton
    triton = None

_BLOCK = 32  # lanes per program, one warp: block-wide loop conditions are then a warp vote
# One thread per lane. With more threads than lanes, Triton gives a lane to several threads and
# lets only one issue its atomic minimum; the others can retry a union forever.
_WARPS = _BLOCK // 32
_FLAT_BLOCK = 1024  # the elementwise kernels, which have neither loops nor atomics


def available() -> bool:
    return triton is not None


def label(mask: Tensor, connectivity: int) -> tuple[Tensor, int]:
    """int32 labels and the number of components of a 1-D to 3-D CUDA mask.

    Same contract and same labels as :func:`exacta._native.label`.
    """
    if triton is None:
        raise RuntimeError("Triton is not installed")
    shape = (1,) * (3 - mask.ndim) + tuple(mask.shape)  # work in 3-D, as the CPU kernel does
    total = mask.numel()
    if total >= 2**31:
        raise ValueError("labelling supports fewer than 2^31 elements")
    if total == 0:
        return torch.zeros(mask.shape, dtype=torch.int32, device=mask.device), 0
    values = mask.reshape(-1).contiguous()
    if values.dtype == torch.bool:
        values = values.view(torch.uint8)

    # Earlier neighbours in raster order, at most `connectivity` nonzero offsets.
    offsets = [
        (a, b, c)
        for a in (-1, 0, 1)
        for b in (-1, 0, 1)
        for c in (-1, 0, 1)
        if 0 < (a != 0) + (b != 0) + (c != 0) <= connectivity and (a, b, c) < (0, 0, 0)
    ]
    table = torch.tensor(offsets, dtype=torch.int32, device=mask.device).reshape(-1)

    parent = torch.empty(total, dtype=torch.int32, device=mask.device)
    roots = torch.empty(total, dtype=torch.int32, device=mask.device)
    labels = torch.empty(total, dtype=torch.int32, device=mask.device)
    grid, flat = (triton.cdiv(total, _BLOCK),), (triton.cdiv(total, _FLAT_BLOCK),)
    with torch.cuda.device(mask.device):  # Triton launches on the current device
        _init[flat](values, parent, total, BLOCK=_FLAT_BLOCK)
        _merge[grid](parent, table, len(offsets), *shape, total, BLOCK=_BLOCK, num_warps=_WARPS)
        _compress[grid](parent, roots, total, BLOCK=_BLOCK, num_warps=_WARPS)
        number = torch.cumsum(roots, 0, dtype=torch.int32)  # components in order of first voxel
        _number[flat](parent, number, labels, total, BLOCK=_FLAT_BLOCK)
    return labels.reshape(mask.shape), int(number[-1])


if triton is not None:

    @triton.jit
    def _find(parent, x, active):
        """Root of x for active lanes (x itself elsewhere)."""
        up = tl.load(parent + x, mask=active, other=0)
        climbing = active & (up != x)
        while tl.max(climbing.to(tl.int32), axis=0) > 0:
            x = tl.where(climbing, up, x)
            up = tl.load(parent + x, mask=climbing, other=0)
            climbing = climbing & (up != x)
        return x

    @triton.jit(do_not_specialize=["count", "n0", "n1", "n2", "total"])
    def _merge(parent, table, count, n0, n1, n2, total, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        inside = i < total
        fg = inside & (tl.load(parent + i, mask=inside, other=-1) >= 0)
        x0 = i // (n1 * n2)
        x1 = (i // n2) % n1
        x2 = i % n2
        for k in range(0, count):
            y0 = x0 + tl.load(table + 3 * k)
            y1 = x1 + tl.load(table + 3 * k + 1)
            y2 = x2 + tl.load(table + 3 * k + 2)
            j = (y0 * n1 + y1) * n2 + y2
            in_bounds = (y0 >= 0) & (y1 >= 0) & (y1 < n1) & (y2 >= 0) & (y2 < n2)
            pair = fg & in_bounds
            pair = pair & (tl.load(parent + j, mask=pair, other=-1) >= 0)
            # Union of i and j: link the larger root to the smaller; if that root gained a
            # parent meanwhile, the atomic minimum still recorded the smaller of the two, and
            # the merge continues from the parent it had.
            a = i
            b = tl.where(pair, j, i)
            active = pair
            while tl.max(active.to(tl.int32), axis=0) > 0:
                a = _find(parent, a, active)
                b = _find(parent, b, active)
                active = active & (a != b)
                hi = tl.maximum(a, b)
                lo = tl.minimum(a, b)
                old = tl.atomic_min(parent + hi, lo, mask=active)
                active = active & (old != hi)
                a = tl.where(active, old, a)
                b = tl.where(active, lo, b)

    @triton.jit(do_not_specialize=["total"])
    def _init(mask, parent, total, BLOCK: tl.constexpr):
        # Every foreground voxel (nonzero) its own parent, background -1.
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        fg = tl.load(mask + i, mask=i < total, other=0) != 0
        tl.store(parent + i, tl.where(fg, i, -1), mask=i < total)

    @triton.jit(do_not_specialize=["total"])
    def _compress(parent, roots, total, BLOCK: tl.constexpr):
        # Points every voxel at its root, and flags the roots (1) for the prefix sum.
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        fg = (i < total) & (tl.load(parent + i, mask=i < total, other=-1) >= 0)
        root = _find(parent, i, fg)
        tl.store(parent + i, root, mask=fg)
        tl.store(roots + i, (fg & (root == i)).to(tl.int32), mask=i < total)

    @triton.jit(do_not_specialize=["total"])
    def _number(parent, number, labels, total, BLOCK: tl.constexpr):
        # A voxel's label is its root's position among the roots: the prefix sum at the root.
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        root = tl.load(parent + i, mask=i < total, other=-1)
        label = tl.load(number + root, mask=root >= 0, other=0)
        tl.store(labels + i, label, mask=i < total)
