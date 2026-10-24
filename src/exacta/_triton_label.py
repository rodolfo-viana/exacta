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

_BLOCK = 256


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
    fg = (mask != 0).reshape(-1).contiguous()
    if total == 0:
        return torch.zeros(mask.shape, dtype=torch.int32, device=mask.device), 0

    # Earlier neighbours in raster order, at most `connectivity` nonzero offsets.
    offsets = [
        (a, b, c)
        for a in (-1, 0, 1)
        for b in (-1, 0, 1)
        for c in (-1, 0, 1)
        if 0 < (a != 0) + (b != 0) + (c != 0) <= connectivity and (a, b, c) < (0, 0, 0)
    ]
    table = torch.tensor(offsets, dtype=torch.int32, device=mask.device).reshape(-1)

    index = torch.arange(total, dtype=torch.int32, device=mask.device)
    parent = torch.where(fg, index, -1)
    grid = (triton.cdiv(total, _BLOCK),)
    with torch.cuda.device(mask.device):  # Triton launches on the current device
        _merge[grid](parent, table, len(offsets), shape[0], shape[1], shape[2], total, BLOCK=_BLOCK)
        _compress[grid](parent, total, BLOCK=_BLOCK)

    roots = fg & (parent == index)
    number = torch.cumsum(roots, 0, dtype=torch.int32)  # components in order of first voxel
    labels = torch.where(fg, number[parent.clamp_min(0).long()], 0)
    return labels.reshape(mask.shape), int(roots.sum())


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
    def _compress(parent, total, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        fg = (i < total) & (tl.load(parent + i, mask=i < total, other=-1) >= 0)
        root = _find(parent, i, fg)
        tl.store(parent + i, root, mask=fg)
