"""Exact squared Euclidean distance transform on GPU, in Triton.

The same algorithm as the native CPU kernel (``csrc/cpu/edt.cpp``), with one GPU lane per line:
each axis pass computes, for every line along the axis, the lower envelope of the parabolas
``(h*(q - p))^2 + f[p]`` and evaluates it, keeping every parabola that ties within rounding
error so that the result is the exact floating-point minimum over p. Where lanes diverge
(popping the envelope, scanning tied parabolas) the block loops until no lane is active. The
first axis needs no envelope: its input is 0 at features and inf elsewhere, so two sweeps find
each element's nearest feature on the line.

Size limit: the kernels index the samples of a line, and the lines of a pass, with 32-bit integers
(element offsets are 64-bit). Each axis may have at most ``2**31 - 2`` samples and the other
axes together, and the batch axes, at most ``2**31 - 32`` lines per pass; a larger input raises
``ValueError`` before anything is allocated. The CPU kernel has no such limit; the exactness
proof covers lines of up to ``2**40`` samples.

Two details keep the results bit-identical to the CPU kernel and to a brute-force minimum:
kernels are compiled with ``enable_fp_fusion=False``, since a fused multiply-add rounds
``dx*dx + f`` once instead of twice, and every float64 parameter is passed in a tensor, since
Triton passes Python floats to kernels as float32.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from exacta import _native

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only PyTorch builds ship without Triton
    triton = None

_BLOCK = 32  # lanes (lines) per program, one warp: small blocks wait less on divergent lanes
_SCRATCH_BYTES = 256 << 20  # budget for the per-lane envelope stacks

# Indices of the samples of a line (the stacks, the loop counters, the offsets into the scratch
# stacks before they are widened) and the line counter of the program loop are int32 in the
# kernels. A line may have n <= _MAX_LINE samples, so that the stack slot n and the loop bound
# n + 1 fit, and a pass may have at most _MAX_LINES lines, so that cdiv(lines, _BLOCK) * _BLOCK
# does too.
_MAX_LINE = 2**31 - 2
_MAX_LINES = 2**31 - _BLOCK


def check_index_range(shape: Sequence[int], batch_ndim: int = 0) -> None:
    """Raise ``ValueError`` if a pass over a grid of ``shape`` would overflow the kernels' int32
    indices: an axis of more than ``2**31 - 2`` samples, or more than ``2**31 - 32`` lines in a
    pass (all the other axes, batch axes included, make the lines of an axis)."""
    total = math.prod(shape)
    if total == 0:
        return  # nothing is indexed
    for n in shape[batch_ndim:]:
        if n > _MAX_LINE or total // n > _MAX_LINES:
            raise ValueError(
                f"the Triton distance transform indexes the samples of a line and the lines of "
                f"a pass with 32-bit integers, so it cannot process shape {tuple(shape)} "
                f"(at most {_MAX_LINE} samples per line and {_MAX_LINES} lines per pass); "
                f"use the CPU kernel (exacta._native.edt_sq) instead"
            )


def available() -> bool:
    return triton is not None


def edt_sq(features: Tensor, spacing: Sequence[float], batch_ndim: int = 0) -> Tensor:
    """Squared distance from every voxel to the nearest nonzero voxel of ``features`` (CUDA).

    Same contract and same values, bit for bit, as :func:`exacta._native.edt_sq`, including
    ``batch_ndim`` leading axes that index separate images. The kernels index lines and samples
    with 32-bit integers, so the input is limited to ``2**31 - 2`` samples per line and
    ``2**31 - 32`` lines per pass; :func:`check_index_range` raises ``ValueError`` above that
    before anything is allocated. The CPU kernel has no such limit.
    """
    if triton is None:
        raise RuntimeError("Triton is not installed")
    _native.check_edt_extent(features.shape[batch_ndim:], spacing)
    check_index_range(features.shape, batch_ndim)
    with torch.cuda.device(features.device):  # Triton launches on the current device
        return _transform(features, spacing, batch_ndim)


def _transform(features: Tensor, spacing: Sequence[float], batch_ndim: int) -> Tensor:
    out = torch.empty(features.shape, dtype=torch.float64, device=features.device)
    if out.numel() == 0:
        return out
    features = features.bool().contiguous()
    if batch_ndim >= out.ndim:  # nothing to transform
        return out.masked_fill_(features, 0.0).masked_fill_(~features, math.inf)
    total = out.numel()
    n = out.shape[batch_ndim]
    stride = math.prod(out.shape[batch_ndim + 1 :])
    consts = torch.tensor([float(spacing[0])], dtype=torch.float64, device=out.device)
    _first_pass[(triton.cdiv(total // n, _BLOCK),)](
        features, out, consts, n, stride, total // n, BLOCK=_BLOCK, num_warps=1, enable_fp_fusion=False
    )
    buf = torch.empty_like(out)
    for axis in range(batch_ndim + 1, out.ndim):
        h = spacing[axis - batch_ndim]
        n = out.shape[axis]
        tol = _tolerance(spacing, out.shape[batch_ndim:], axis - batch_ndim)
        stride = math.prod(out.shape[axis + 1 :])
        _pass(out, buf, float(h), tol, n, stride, total // n)
        out, buf = buf, out
    return out


def _pass(src: Tensor, dst: Tensor, h: float, tol: float, n: int, stride: int, lines: int) -> None:
    programs = min(triton.cdiv(lines, _BLOCK), max(1, _SCRATCH_BYTES // (_BLOCK * (n + 1) * 28)))
    # Scratch only for lanes that have a line: a few very long lines would otherwise get a full
    # block of stacks each. Lanes past the last line never touch their (absent) column.
    lanes = min(programs * _BLOCK, lines)
    v = torch.empty((n + 1) * lanes, dtype=torch.int32, device=src.device)
    z = torch.empty((n + 1) * lanes, dtype=torch.float64, device=src.device)
    b = torch.empty((n + 1) * lanes, dtype=torch.float64, device=src.device)
    g = torch.empty((n + 1) * lanes, dtype=torch.float64, device=src.device)
    consts = torch.tensor([h, tol], dtype=torch.float64, device=src.device)
    _edt_pass[(programs,)](
        src, dst, v, z, b, g, consts, n, stride, lines, lanes,
        BLOCK=_BLOCK, num_warps=1, enable_fp_fusion=False,
    )


def edt_lines(f: Tensor, h: float, bound: float) -> Tensor:
    """One pass over each row of ``f`` (CUDA float64, ``(lines, n)``, ``inf`` where there is no
    parabola, finite values in ``[0, bound]``), as ``exacta._native.edt_lines``, including its
    checks of the assumptions of the exactness proof and the int32 limit of this module. For
    testing."""
    if triton is None:
        raise RuntimeError("Triton is not installed")
    check_index_range(f.shape, 1)
    f = f.contiguous()
    _native.check_edt_lines(f, h, bound)
    out = torch.empty_like(f)
    lines, n = f.shape
    if out.numel():
        xmax = float(h) * float(n)
        tol = 64.0 * torch.finfo(torch.float64).eps * (float(bound) + xmax * xmax) / float(h)
        with torch.cuda.device(f.device):
            _pass(f, out, float(h), tol, n, 1, lines)
    return out


def _tolerance(spacing: Sequence[float], shape: Sequence[int], axis: int) -> float:
    """Tie tolerance of the breakpoints in the pass along spatial ``axis``, as the CPU kernel
    computes it: 64 epsilon times the largest term of a breakpoint, divided by h. The input of
    that pass is at most the sum of squared extents of the axes before it."""
    bound = 0.0
    for h, n in zip(spacing[:axis], shape[:axis]):
        extent = float(h) * float(n - 1)
        bound += extent * extent
    h = float(spacing[axis])
    xmax = h * float(shape[axis])
    return 64.0 * torch.finfo(torch.float64).eps * (bound + xmax * xmax) / h


if triton is not None:

    @triton.jit(do_not_specialize=["n", "stride", "lines"])
    def _first_pass(features, dst, consts, n, stride, lines, BLOCK: tl.constexpr):
        # Along the first transformed axis the input is 0 at features and inf elsewhere, so the
        # envelope minimum is (h * k)^2 for the distance k to the nearest feature on the line:
        # two sweeps suffice.
        index = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        valid = index < lines
        stride64 = stride.to(tl.int64)
        line = (index // stride64) * n * stride64 + index % stride64  # offset of the line's start
        h = tl.load(consts)
        inf = float("inf")
        last = tl.full([BLOCK], -1, dtype=tl.int32)  # nearest feature at or before q, -1: none
        for q in range(0, n):
            hit = tl.load(features + line + q * stride64, mask=valid, other=0) != 0
            last = tl.where(hit, q, last)
            dx = h * (q - last).to(tl.float64)
            tl.store(dst + line + q * stride64, tl.where(last >= 0, dx * dx, inf), mask=valid)
        nxt = tl.full([BLOCK], -1, dtype=tl.int32)  # nearest feature at or after q
        for r in range(0, n):
            q = n - 1 - r
            hit = tl.load(features + line + q * stride64, mask=valid, other=0) != 0
            nxt = tl.where(hit, q, nxt)
            dx = h * (q - nxt).to(tl.float64)
            ahead = tl.where(nxt >= 0, dx * dx, inf)
            behind = tl.load(dst + line + q * stride64, mask=valid, other=inf)
            tl.store(dst + line + q * stride64, tl.minimum(behind, ahead), mask=valid)

    @triton.jit
    def _slot(index, lanes, slot):
        # Scratch stacks are laid out index-major, so lanes reading the same index coalesce.
        return index.to(tl.int64) * lanes + slot

    @triton.jit
    def _parabola(src, base, stride, h, q, p, mask):
        # (h*(q - p))^2 + f[p], with h * (q - p) rounded once, as the CPU kernel and SciPy do
        dx = h * (q - p).to(tl.float64)
        f = tl.load(src + base + p.to(tl.int64) * stride, mask=mask, other=0.0)
        return dx * dx + f

    # Sizes stay runtime values: Triton would otherwise turn those equal to 1 into constants.
    @triton.jit(do_not_specialize=["n", "stride", "lines", "lanes"])
    def _edt_pass(src, dst, v_buf, z_buf, b_buf, g_buf, consts, n, stride, lines, lanes, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        slot = (pid * BLOCK + lane).to(tl.int64)  # this lane's column in the scratch stacks
        stride = stride.to(tl.int64)  # offsets can pass 2^31 on large volumes
        lanes = lanes.to(tl.int64)
        h = tl.load(consts)
        tol = tl.load(consts + 1)
        inf = float("inf")

        for block in range(pid, tl.cdiv(lines, BLOCK), tl.num_programs(0)):
            line = (block * BLOCK + lane).to(tl.int64)
            valid = line < lines
            base = (line // stride) * n * stride + line % stride

            # Lower envelope; k is the top of each lane's stack (-1: empty). b holds each entry's
            # breakpoint with the entry below it, z their running maximum (see csrc/cpu/edt.cpp).
            # "Alg. 1, line N" refers to the paper's pseudo-code of this pass, as in edt.cpp.
            k = tl.full([BLOCK], -1, dtype=tl.int32)
            for q in range(0, n):
                fq = tl.load(src + base + q * stride, mask=valid, other=inf)
                finite = fq != inf
                xq = h * q.to(tl.float64)
                gq = fq + xq * xq
                # Breakpoint num / den with the top of the stack; pops compare num with
                # (b - tol) * den (den > 0), so only the breakpoint that is kept is divided.
                num = tl.zeros([BLOCK], dtype=tl.float64)
                den = tl.full([BLOCK], 1.0, dtype=tl.float64)
                active = finite & (k >= 0)
                while tl.max(active.to(tl.int32), axis=0) > 0:
                    at = _slot(k, lanes, slot)
                    vk = tl.load(v_buf + at, mask=active, other=0)
                    bk = tl.load(b_buf + at, mask=active, other=0.0)
                    gv = tl.load(g_buf + at, mask=active, other=0.0)  # f[v] + xv * xv
                    xv = h * vk.to(tl.float64)
                    num = tl.where(active, gq - gv, num)
                    den = tl.where(active, 2.0 * (xq - xv), den)
                    # Alg. 1, lines 6-9: pop v[k] when it is neither lowest anywhere nor tied.
                    pop = active & (num <= (bk - tol) * den)
                    k = tl.where(pop, k - 1, k)
                    active = pop & (k >= 0)
                s = tl.where(k >= 0, num / den, -inf)  # line 11; -inf only for the first push
                k = tl.where(finite, k + 1, k)
                at = _slot(k, lanes, slot)
                z_prev = tl.load(z_buf + at - lanes, mask=finite & (k > 0), other=-inf)
                tl.store(v_buf + at, q + tl.zeros([BLOCK], dtype=tl.int32), mask=finite)
                tl.store(g_buf + at, gq, mask=finite)
                tl.store(b_buf + at, s, mask=finite)
                tl.store(z_buf + at, tl.where(k == 0, s, tl.maximum(s, z_prev)), mask=finite)
                tl.store(z_buf + at + lanes, tl.full([BLOCK], inf, dtype=tl.float64), mask=finite)

            # Suffix minimum of b (Alg. 1, line 16), stored back into b, bounds where entries
            # above j can be lowest.
            for r in range(1, n):
                i = n - 1 - r
                inner = valid & (i < k)
                b_up = tl.load(b_buf + _slot(i + 1 + tl.zeros([BLOCK], dtype=tl.int32), lanes, slot), mask=inner, other=0.0)
                b_i = tl.load(b_buf + _slot(i + tl.zeros([BLOCK], dtype=tl.int32), lanes, slot), mask=inner, other=0.0)
                tl.store(b_buf + _slot(i + tl.zeros([BLOCK], dtype=tl.int32), lanes, slot), tl.minimum(b_i, b_up), mask=inner)

            # Evaluate (Alg. 1, lines 17-25): the envelope's parabola and every parabola tied with
            # it near xq; the left scan uses z, the right scan the suffix minima of b.
            has = valid & (k >= 0)
            j = tl.zeros([BLOCK], dtype=tl.int32)
            for q in range(0, n):
                xq = h * q.to(tl.float64)
                advance = has & (tl.load(z_buf + _slot(j + 1, lanes, slot), mask=has, other=inf) < xq)
                while tl.max(advance.to(tl.int32), axis=0) > 0:
                    j = tl.where(advance, j + 1, j)
                    z_next = tl.load(z_buf + _slot(j + 1, lanes, slot), mask=advance, other=inf)
                    advance = advance & (z_next < xq)
                vj = tl.load(v_buf + _slot(j, lanes, slot), mask=has, other=0)
                best = _parabola(src, base, stride, h, q, vj, has)

                i = j - 1
                z_end = tl.load(z_buf + _slot(j, lanes, slot), mask=has & (i >= 0), other=-inf)
                scan = has & (i >= 0) & (z_end >= xq - tol)
                while tl.max(scan.to(tl.int32), axis=0) > 0:
                    vi = tl.load(v_buf + _slot(i, lanes, slot), mask=scan, other=0)
                    best = tl.where(scan, tl.minimum(best, _parabola(src, base, stride, h, q, vi, scan)), best)
                    i = tl.where(scan, i - 1, i)
                    z_end = tl.load(z_buf + _slot(i + 1, lanes, slot), mask=scan & (i >= 0), other=-inf)
                    scan = scan & (i >= 0) & (z_end >= xq - tol)

                i = j + 1
                b_min = tl.load(b_buf + _slot(i, lanes, slot), mask=has & (i <= k), other=inf)
                scan = has & (i <= k) & (b_min <= xq + tol)
                while tl.max(scan.to(tl.int32), axis=0) > 0:
                    vi = tl.load(v_buf + _slot(i, lanes, slot), mask=scan, other=0)
                    best = tl.where(scan, tl.minimum(best, _parabola(src, base, stride, h, q, vi, scan)), best)
                    i = tl.where(scan, i + 1, i)
                    b_min = tl.load(b_buf + _slot(i, lanes, slot), mask=scan & (i <= k), other=inf)
                    scan = scan & (i <= k) & (b_min <= xq + tol)

                tl.store(dst + base + q * stride, tl.where(has, best, inf), mask=valid)
