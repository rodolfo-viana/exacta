"""Fused Bernoulli statistics with bounded tile partials and float64 arithmetic."""
from __future__ import annotations

import math

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = None

_BLOCK = 1024
_MAX_BINS = 32


def available() -> bool:
    return triton is not None


def statistics(pred: Tensor, target: Tensor, mask: Tensor | None, from_logits: bool,
               kind: int, bins: int) -> Tensor | None:
    batch, channels = pred.shape[:2]
    rows, n = batch * channels, math.prod(pred.shape[2:])
    p, t = (x.contiguous().reshape(rows, n) for x in (pred, target))
    m = p if mask is None else mask.contiguous()
    tiles = triton.cdiv(n, _BLOCK)
    width = 3 * bins if kind == 2 else 2
    partials = torch.empty((rows, tiles, width), dtype=torch.float64, device=pred.device)
    bad = torch.zeros(1, dtype=torch.int32, device=pred.device)
    if rows and n:
        with torch.cuda.device(pred.device):
            _statistics[(tiles, rows)](
                p, t, m, partials, bad, n, tiles, CHANNELS=channels, KIND=kind,
                HAS_MASK=mask is not None, SHARED_MASK=mask is not None and mask.shape[1] == 1,
                FROM_LOGITS=from_logits, BINS=bins, WIDTH=width, BLOCK=_BLOCK,
            )
    if bad.item():
        return None
    sums = partials.sum(1)
    return sums.reshape(batch, channels, 3, bins) if kind == 2 else sums.reshape(batch, channels, 2)


if triton is not None:

    @triton.jit
    def _statistics(pred, target, mask, out, bad, n, tiles, CHANNELS: tl.constexpr,
                    KIND: tl.constexpr, HAS_MASK: tl.constexpr, SHARED_MASK: tl.constexpr,
                    FROM_LOGITS: tl.constexpr, BINS: tl.constexpr, WIDTH: tl.constexpr,
                    BLOCK: tl.constexpr):
        row = tl.program_id(1).to(tl.int64)
        tile = tl.program_id(0).to(tl.int64)
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        present = offsets < n
        keep = present
        if HAS_MASK:
            mask_row = row // CHANNELS if SHARED_MASK else row
            keep &= tl.load(mask + mask_row * n + offsets, mask=present, other=0) != 0
        x = tl.load(pred + row * n + offsets, mask=keep, other=0).to(tl.float64)
        y = tl.load(target + row * n + offsets, mask=keep, other=0)
        valid = ((y == 0) | (y == 1)) & (tl.abs(x) < float("inf"))
        if not FROM_LOGITS:
            valid &= (x >= 0) & (x <= 1)
        tl.atomic_max(bad, tl.max((keep & ~valid).to(tl.int32), axis=0))
        keep &= valid
        x = tl.where(keep, x, 0.0)
        positive = y == 1
        base = out + (row * tiles + tile) * WIDTH
        if KIND == 1:
            if FROM_LOGITS:
                z = tl.where(positive, -x, x)
                value = tl.maximum(z, 0.0) + libdevice.log1p(libdevice.exp(-tl.abs(z)))
            else:
                value = tl.where(positive, -libdevice.log(x), -libdevice.log1p(-x))
        else:
            if FROM_LOGITS:
                exp = libdevice.exp(-tl.abs(x))
                p = tl.where(x >= 0, 1.0 / (1.0 + exp), exp / (1.0 + exp))
            else:
                p = x
            if KIND == 0:
                difference = p - positive.to(tl.float64)
                value = difference * difference
            else:
                index = tl.minimum((p * BINS).to(tl.int32), BINS - 1)
                for bin in range(BINS):
                    included = keep & (index == bin)
                    count = tl.sum(included.to(tl.int32), axis=0).to(tl.float64)
                    confidence = tl.sum(tl.where(included, p, 0.0), axis=0)
                    positives = tl.sum((included & positive).to(tl.int32), axis=0).to(tl.float64)
                    tl.store(base + bin, count)
                    tl.store(base + BINS + bin, confidence)
                    tl.store(base + 2 * BINS + bin, positives)
        if KIND != 2:
            tl.store(base, tl.sum(keep.to(tl.int32), axis=0).to(tl.float64))
            tl.store(base + 1, tl.sum(tl.where(keep, value, 0.0), axis=0))
