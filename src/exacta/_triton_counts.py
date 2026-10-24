"""Voxel counts behind the overlap metrics on GPU, in Triton.

The CUDA counterpart of the native counting kernels (``csrc/cpu/counts.cpp``): one pass over the
inputs counts and checks the values as it goes (labels in range, one-hot masks binary), so
validation skips its own passes. Each block counts its voxels in a histogram of its own and adds
it to the totals once, where PyTorch's bincount adds every voxel atomically to the few bins of a
confusion matrix, one at a time.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only PyTorch builds ship without Triton
    triton = None

_BLOCK = 4096
_MAX_BINS = 1024  # a block's histogram lives in registers; more classes use PyTorch's bincount


def available() -> bool:
    return triton is not None


def fits(classes: int) -> bool:
    """Whether :func:`confusion` handles ``classes`` classes."""
    return classes * classes + 1 <= _MAX_BINS


def confusion(pred: Tensor, target: Tensor, classes: int) -> Tensor | None:
    """(B, classes, classes) int64 confusion matrices of two CUDA label maps (B, ...),
    ``[b, i, j]`` = voxels predicted i with target j; None if a label lies outside [0, classes)."""
    batch = pred.shape[0]
    p, t = (_flat(x, batch) for x in (pred, target))
    n = p.shape[1]
    bins = triton.next_power_of_2(classes * classes)
    out = torch.zeros((batch, bins), dtype=torch.int64, device=pred.device)
    bad = torch.zeros(1, dtype=torch.int32, device=pred.device)
    if n and batch:
        with torch.cuda.device(pred.device):
            _confusion[(triton.cdiv(n, _BLOCK), batch)](p, t, out, bad, n, classes, BINS=bins, BLOCK=_BLOCK)
    if bad.item():
        return None
    return out[:, : classes * classes].view(batch, classes, classes)


def channel_counts(pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor] | None:
    """Per mask of two CUDA stacks of binary masks (C, ...): elements in both, in pred and in
    target (int64, (C,)); None if a value is not 0 or 1."""
    count = pred.shape[0]
    p, t = (_flat(x, count) for x in (pred, target))
    n = p.shape[1]
    out = torch.zeros((count, 3), dtype=torch.int64, device=pred.device)
    bad = torch.zeros(1, dtype=torch.int32, device=pred.device)
    if n and count:
        with torch.cuda.device(pred.device):
            _channel_counts[(triton.cdiv(n, _BLOCK), count)](p, t, out, bad, n, BLOCK=_BLOCK)
    if bad.item():
        return None
    return out[:, 0], out[:, 1], out[:, 2]


def _flat(x: Tensor, rows: int) -> Tensor:
    if x.dtype == torch.bool:
        x = x.view(torch.uint8)
    return x.contiguous().view(rows, math.prod(x.shape[1:]))


if triton is not None:

    @triton.jit(do_not_specialize=["n", "classes"])
    def _confusion(pred, target, out, bad, n, classes, BINS: tl.constexpr, BLOCK: tl.constexpr):
        case = tl.program_id(1).to(tl.int64)
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = offs < n
        p = tl.load(pred + case * n + offs, mask=valid, other=0).to(tl.int64)
        t = tl.load(target + case * n + offs, mask=valid, other=0).to(tl.int64)
        inside = (p >= 0) & (p < classes) & (t >= 0) & (t < classes)
        tl.atomic_max(bad, tl.max((valid & ~inside).to(tl.int32), axis=0))
        keep = valid & inside
        counts = tl.histogram(tl.where(keep, p * classes + t, 0).to(tl.int32), BINS, mask=keep)
        bins = tl.arange(0, BINS)
        tl.atomic_add(out + case * BINS + bins, counts.to(tl.int64), mask=counts > 0)

    @triton.jit(do_not_specialize=["n"])
    def _channel_counts(pred, target, out, bad, n, BLOCK: tl.constexpr):
        row = tl.program_id(1).to(tl.int64)
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = offs < n
        p = tl.load(pred + row * n + offs, mask=valid, other=0)
        t = tl.load(target + row * n + offs, mask=valid, other=0)
        not_binary = ((p != 0) & (p != 1)) | ((t != 0) & (t != 1))  # NaN included
        tl.atomic_max(bad, tl.max((valid & not_binary).to(tl.int32), axis=0))
        a, b = valid & (p != 0), valid & (t != 0)
        tl.atomic_add(out + row * 3, tl.sum((a & b).to(tl.int32), axis=0).to(tl.int64))
        tl.atomic_add(out + row * 3 + 1, tl.sum(a.to(tl.int32), axis=0).to(tl.int64))
        tl.atomic_add(out + row * 3 + 2, tl.sum(b.to(tl.int32), axis=0).to(tl.int64))
