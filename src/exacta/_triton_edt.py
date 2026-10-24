"""Exact squared Euclidean distance transform on GPU, in Triton.

The same algorithm as the native CPU kernel (``csrc/cpu/edt.cpp``), with one GPU lane per line:
each axis pass computes, for every line along the axis, the lower envelope of the parabolas
``(h*(q - p))^2 + f[p]`` and evaluates it, keeping every parabola that ties within rounding
error so that the result is the exact floating-point minimum over p. Where lanes diverge
(popping the envelope, scanning tied parabolas) the block loops until no lane is active.

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

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only PyTorch builds ship without Triton
    triton = None

_BLOCK = 128  # lanes (lines) per program
_SCRATCH_BYTES = 256 << 20  # budget for the per-lane envelope stacks


def available() -> bool:
    return triton is not None


def edt_sq(features: Tensor, spacing: Sequence[float]) -> Tensor:
    """Squared distance from every voxel to the nearest nonzero voxel of ``features`` (CUDA).

    Same contract and same values, bit for bit, as :func:`exacta._native.edt_sq`.
    """
    if triton is None:
        raise RuntimeError("Triton is not installed")
    out = torch.zeros(features.shape, dtype=torch.float64, device=features.device)
    out.masked_fill_(~features.bool(), math.inf)
    if out.numel() == 0:
        return out
    buf = torch.empty_like(out)
    total = out.numel()
    for axis, h in enumerate(spacing):
        n = out.shape[axis]
        stride = math.prod(out.shape[axis + 1 :])
        lines = total // n
        programs = min(triton.cdiv(lines, _BLOCK), max(1, _SCRATCH_BYTES // (_BLOCK * (n + 1) * 12)))
        lanes = programs * _BLOCK
        v = torch.empty((n + 1) * lanes, dtype=torch.int32, device=out.device)
        z = torch.empty((n + 1) * lanes, dtype=torch.float64, device=out.device)
        # h, and 64 epsilon for the tie tolerance, as float64
        consts = torch.tensor(
            [float(h), 64.0 * torch.finfo(torch.float64).eps], dtype=torch.float64, device=out.device
        )
        _edt_pass[(programs,)](
            out, buf, v, z, consts, n, stride, lines, lanes, BLOCK=_BLOCK, num_warps=4, enable_fp_fusion=False
        )
        out, buf = buf, out
    return out


if triton is not None:

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
    def _edt_pass(src, dst, v_buf, z_buf, consts, n, stride, lines, lanes, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        slot = (pid * BLOCK + lane).to(tl.int64)  # this lane's column in the scratch stacks
        stride = stride.to(tl.int64)  # offsets can pass 2^31 on large volumes
        lanes = lanes.to(tl.int64)
        h = tl.load(consts)
        eps64 = tl.load(consts + 1)
        inf = float("inf")
        xmax = h * n.to(tl.float64)

        for block in range(pid, tl.cdiv(lines, BLOCK), tl.num_programs(0)):
            line = (block * BLOCK + lane).to(tl.int64)
            valid = line < lines
            base = (line // stride) * n * stride + line % stride

            fmax = tl.zeros([BLOCK], dtype=tl.float64)
            for q in range(0, n):
                fq = tl.load(src + base + q * stride, mask=valid, other=inf)
                fmax = tl.where(fq != inf, tl.maximum(fmax, fq), fmax)
            tol = eps64 * (fmax + xmax * xmax) / h

            # Lower envelope; k is the top of each lane's stack (-1: empty).
            k = tl.full([BLOCK], -1, dtype=tl.int32)
            for q in range(0, n):
                fq = tl.load(src + base + q * stride, mask=valid, other=inf)
                finite = fq != inf
                xq = h * q.to(tl.float64)
                s = tl.full([BLOCK], -inf, dtype=tl.float64)
                active = finite & (k >= 0)
                while tl.max(active.to(tl.int32), axis=0) > 0:
                    at = _slot(k, lanes, slot)
                    vk = tl.load(v_buf + at, mask=active, other=0)
                    zk = tl.load(z_buf + at, mask=active, other=0.0)
                    fv = tl.load(src + base + vk.to(tl.int64) * stride, mask=active, other=0.0)
                    xv = h * vk.to(tl.float64)
                    s_k = ((fq + xq * xq) - (fv + xv * xv)) / (2.0 * (xq - xv))
                    s = tl.where(active, s_k, s)
                    pop = active & (s_k <= zk - tol)  # v[k] neither lowest anywhere nor tied
                    k = tl.where(pop, k - 1, k)
                    active = pop & (k >= 0)
                k = tl.where(finite, k + 1, k)
                at = _slot(k, lanes, slot)
                z_prev = tl.load(z_buf + at - lanes, mask=finite & (k > 0), other=-inf)
                tl.store(v_buf + at, q + tl.zeros([BLOCK], dtype=tl.int32), mask=finite)
                tl.store(z_buf + at, tl.where(k == 0, s, tl.maximum(s, z_prev)), mask=finite)
                tl.store(z_buf + at + lanes, tl.full([BLOCK], inf, dtype=tl.float64), mask=finite)

            # Evaluate: the envelope's parabola and every parabola tied with it near xq.
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
                z_start = tl.load(z_buf + _slot(i, lanes, slot), mask=has & (i <= k), other=inf)
                scan = has & (i <= k) & (z_start <= xq + tol)
                while tl.max(scan.to(tl.int32), axis=0) > 0:
                    vi = tl.load(v_buf + _slot(i, lanes, slot), mask=scan, other=0)
                    best = tl.where(scan, tl.minimum(best, _parabola(src, base, stride, h, q, vi, scan)), best)
                    i = tl.where(scan, i + 1, i)
                    z_start = tl.load(z_buf + _slot(i, lanes, slot), mask=scan & (i <= k), other=inf)
                    scan = scan & (i <= k) & (z_start <= xq + tol)

                tl.store(dst + base + q * stride, tl.where(has, best, inf), mask=valid)
