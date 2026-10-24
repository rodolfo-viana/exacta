"""Fused Dice and Dice + cross-entropy losses on GPU, in Triton.

The PyTorch implementation in :mod:`exacta.losses.segmentation` materialises the activation, the
one-hot target and their product, and autograd keeps them for the backward pass: several times
the memory of the logits. Here the forward pass reads logits and target once and accumulates,
for every case and class, ``Σ p·g``, ``Σ p``, ``Σ g`` and the cross-entropy; the backward pass
reads them again, recomputes the activation and writes the gradient from a closed form:

    Dice:   dL/dp = -A·g + B per (case, class), with A = 2 / (E·D) and B = (2I + s) / (E·D²),
            where D = Σp + Σg + s, I = Σ p·g and E is the number of averaged entries.
    softmax CE:  dL/dz = (p·Σg - g) / (cases·voxels)
    sigmoid BCE: dL/dz = (p - g) / (cases·classes·voxels)

Each block of voxels writes its own partial sums, which are then reduced with ``torch.sum``, so
results do not depend on scheduling (atomic adds would). Half-precision logits are read as they
are and computed in float32; float64 logits are computed in float64. Class indices outside
[0, classes) make the loss NaN; the loss modules also raise on them, outside torch.compile.

The backward pass is not itself differentiable: double backward (``create_graph=True``, as in
gradient penalties) raises rather than silently dropping the second-order term. Gradients do not
flow into the target; the loss modules use the PyTorch path when the target requires them.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # CPU-only PyTorch builds ship without Triton
    triton = None

ACTIVATIONS = {"softmax": 0, "sigmoid": 1, "none": 2}


def available() -> bool:
    return triton is not None


def seg_loss(
    logits: Tensor,
    target: Tensor,
    *,
    activation: str,
    one_hot: bool,
    include_background: bool,
    batch_dice: bool,
    smooth: float,
    lambda_dice: float,
    lambda_ce: float,
    cross_entropy: bool,
) -> Tensor:
    """``lambda_dice * Dice + lambda_ce * CE`` with the semantics of the PyTorch losses."""
    config = (
        activation, one_hot, include_background, batch_dice, smooth, lambda_dice, lambda_ce, cross_entropy
    )
    return _SegLoss.apply(logits, target, config)


def _layout(logits: Tensor) -> tuple[int, int, int, int, int]:
    batch, classes = logits.shape[:2]
    voxels = logits[0, 0].numel()
    if voxels >= 2**31:
        raise ValueError("the fused loss supports fewer than 2^31 voxels per case")
    c_pad = triton.next_power_of_2(classes)
    block = max(16, min(1024, 8192 // c_pad))  # keep the (classes, voxels) tile in registers
    return batch, classes, voxels, c_pad, block


class _SegLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: Tensor, target: Tensor, config: tuple) -> Tensor:  # type: ignore[override]
        activation, one_hot, include_background, batch_dice, smooth, lambda_dice, lambda_ce, ce = config
        logits, target = logits.contiguous(), target.contiguous()
        batch, classes, voxels, c_pad, block = _layout(logits)
        compute = torch.float64 if logits.dtype == torch.float64 else torch.float32
        blocks = triton.cdiv(voxels, block)
        partial = torch.empty((batch, blocks, 5, c_pad), dtype=compute, device=logits.device)
        with torch.cuda.device(logits.device):  # Triton launches on the current device
            _forward[(blocks, batch)](
                logits, target, partial, voxels, classes, blocks,
                ACT=ACTIVATIONS[activation], ONE_HOT=one_hot, C_PAD=c_pad, BLOCK=block,
                COMPUTE=tl.float64 if compute == torch.float64 else tl.float32,
            )  # fmt: skip
        sums = partial.sum(1, dtype=torch.float64)  # (batch, 5, c_pad)
        inter, prob, truth, ce_terms = (sums[:, r, :classes] for r in range(4))
        invalid = sums[:, 4, 0].sum()  # class indices out of range

        first = 0 if include_background else 1
        inter, prob, truth = inter[:, first:], prob[:, first:], truth[:, first:]
        if batch_dice:
            inter, prob, truth = inter.sum(0), prob.sum(0), truth.sum(0)
        den = prob + truth + smooth
        num = 2 * inter + smooth
        dice = num / den
        loss = lambda_dice * (1 - dice.mean())
        ce_scale = torch.zeros((), dtype=torch.float64, device=logits.device)
        if ce:
            elements = batch * voxels * (classes if activation == "sigmoid" else 1)
            loss = loss + lambda_ce * ce_terms.sum() / elements
            ce_scale = ce_scale + lambda_ce / elements

        entries = dice.numel()
        coef = torch.zeros((batch, 2, c_pad), dtype=torch.float64, device=logits.device)
        coef[:, 0, first:classes] = lambda_dice * 2 / (entries * den)
        coef[:, 1, first:classes] = lambda_dice * num / (entries * den * den)
        ctx.save_for_backward(logits, target, coef, ce_scale)
        ctx.config = config
        loss = torch.where(invalid > 0, torch.nan, loss)
        return loss.to(compute)

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        if torch.is_grad_enabled():  # autograd enables grad mode here exactly for create_graph=True
            raise RuntimeError(
                "the fused Triton loss does not support double backward (create_graph=True); "
                "use backend='torch'"
            )
        logits, target, coef, ce_scale = ctx.saved_tensors
        activation, one_hot, *_, ce = ctx.config
        batch, classes, voxels, c_pad, block = _layout(logits)
        compute = torch.float64 if logits.dtype == torch.float64 else torch.float32
        coef = (coef * grad_out).to(compute)
        ce_scale = (ce_scale * grad_out).to(compute).reshape(1)
        grad = torch.empty_like(logits)
        with torch.cuda.device(logits.device):
            _backward[(triton.cdiv(voxels, block), batch)](
                logits, target, coef, ce_scale, grad, voxels, classes,
                ACT=ACTIVATIONS[activation], ONE_HOT=one_hot, CE=ce, C_PAD=c_pad, BLOCK=block,
                COMPUTE=tl.float64 if compute == torch.float64 else tl.float32,
            )  # fmt: skip
        return grad, None, None


if triton is not None:

    @triton.jit
    def _tile(logits, target, b, voxel0, voxels, classes, ACT, ONE_HOT, C_PAD, BLOCK, COMPUTE):
        """Activation p, target g, logits z and log-normalizer of a (classes, voxels) tile.

        For softmax, z and the log-normalizer remain shifted by the maximum logit, so their
        difference does not lose precision when all logits share a large offset.
        """
        offs = voxel0 + tl.arange(0, BLOCK)
        ch = tl.arange(0, C_PAD)
        in_voxels = offs < voxels
        in_classes = ch < classes
        mask = in_classes[:, None] & in_voxels[None, :]
        row = b * classes * voxels + ch[:, None].to(tl.int64) * voxels + offs[None, :]
        z = tl.load(logits + row, mask=mask, other=0.0).to(COMPUTE)
        if ACT == 0:  # softmax over classes
            top = tl.max(tl.where(in_classes[:, None], z, float("-inf")), axis=0)
            z = z - top[None, :]
            e = tl.exp(tl.where(in_classes[:, None], z, float("-inf")))
            total = tl.sum(e, axis=0)
            p = e / total[None, :]
            lse = tl.log(total)
        elif ACT == 1:
            p = tl.sigmoid(z)
            lse = tl.zeros([BLOCK], dtype=COMPUTE)
        else:
            p = z
            lse = tl.zeros([BLOCK], dtype=COMPUTE)
        if ONE_HOT:
            g = tl.load(target + row, mask=mask, other=0).to(COMPUTE)
        else:
            # int64 so that indices of 2^31 or more are not truncated into the class range
            t = tl.load(target + b * voxels + offs, mask=in_voxels, other=0).to(tl.int64)
            g = (ch[:, None] == t[None, :]).to(COMPUTE)
        return p, g, z, lse, mask, row, ch, offs

    @triton.jit
    def _forward(
        logits, target, partial, voxels, classes, blocks,
        ACT: tl.constexpr, ONE_HOT: tl.constexpr, C_PAD: tl.constexpr, BLOCK: tl.constexpr,
        COMPUTE: tl.constexpr,
    ):  # fmt: skip
        blk = tl.program_id(0)
        b = tl.program_id(1).to(tl.int64)
        p, g, z, lse, mask, row, ch, offs = _tile(
            logits, target, b, blk * BLOCK, voxels, classes, ACT, ONE_HOT, C_PAD, BLOCK, COMPUTE
        )
        w = mask.to(COMPUTE)
        # Terms with g = 0 are left out rather than multiplied by 0: a class masked with a -inf
        # logit would otherwise make 0 * inf = NaN.
        if ACT == 0:  # cross-entropy -Σ g log p, per class
            ce = tl.where(g != 0, g * (lse[None, :] - z), 0.0)
        else:  # binary cross-entropy with logits
            ce = tl.maximum(z, 0.0) - tl.where(g != 0, z * g, 0.0) + libdevice.log1p(tl.exp(-tl.abs(z)))
        out = partial + ((b * blocks + blk) * 5) * C_PAD + ch
        tl.store(out, tl.sum(p * g * w, axis=1))
        tl.store(out + C_PAD, tl.sum(p * w, axis=1))
        tl.store(out + 2 * C_PAD, tl.sum(g * w, axis=1))
        tl.store(out + 3 * C_PAD, tl.sum(ce * w, axis=1))
        invalid = tl.zeros([C_PAD], dtype=COMPUTE)
        if not ONE_HOT:
            t = tl.load(target + b * voxels + offs, mask=offs < voxels, other=0).to(tl.int64)
            bad = tl.sum(((offs < voxels) & ((t < 0) | (t >= classes))).to(COMPUTE), axis=0)
            invalid = tl.where(ch == 0, bad, 0.0)
        tl.store(out + 4 * C_PAD, invalid)

    @triton.jit
    def _backward(
        logits, target, coef, ce_scale, grad, voxels, classes,
        ACT: tl.constexpr, ONE_HOT: tl.constexpr, CE: tl.constexpr, C_PAD: tl.constexpr, BLOCK: tl.constexpr,
        COMPUTE: tl.constexpr,
    ):  # fmt: skip
        blk = tl.program_id(0)
        b = tl.program_id(1).to(tl.int64)
        p, g, z, lse, mask, row, ch, offs = _tile(
            logits, target, b, blk * BLOCK, voxels, classes, ACT, ONE_HOT, C_PAD, BLOCK, COMPUTE
        )
        a = tl.load(coef + (b * 2) * C_PAD + ch)
        c = tl.load(coef + (b * 2 + 1) * C_PAD + ch)
        d_p = c[:, None] - a[:, None] * g  # dLdice/dp
        if ACT == 0:
            d_z = p * (d_p - tl.sum(p * d_p, axis=0)[None, :])
            if CE:
                d_z += tl.load(ce_scale) * (p * tl.sum(g, axis=0)[None, :] - g)
        elif ACT == 1:
            d_z = d_p * p * (1.0 - p)
            if CE:
                d_z += tl.load(ce_scale) * (p - g)
        else:
            d_z = d_p
        tl.store(grad + row, d_z.to(grad.dtype.element_ty), mask=mask)
