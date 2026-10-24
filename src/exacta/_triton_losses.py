"""Fused overlap and focal losses on GPU, in Triton.

The PyTorch implementation in :mod:`exacta.losses.segmentation` materialises the activation, the
one-hot target and their product, and autograd keeps them for the backward pass: several times
the memory of the logits. Here the forward pass reads logits and target once and accumulates,
for every case and class, ``Σ p·g``, ``Σ p``, ``Σ g`` and the cross-entropy; the backward pass
reads them again, recomputes the activation and writes the gradient from a closed form:

    Dice:   dL/dp = -A·g + B per (case, class), with A = 2 / (E·D) and B = (2I + s) / (E·D²),
            where D = Σp + Σg + s, I = Σ p·g and E is the number of averaged entries.
    Tversky: A = [D - (1-alpha-beta)·N] / (E·D²), B = alpha·N / (E·D²),
             where N=I+s and D=I+alpha·FP+beta·FN+s. FP/FN are accumulated directly.
    Generalized Dice: A_c = 2·w_c / (E·D), B_c = w_c·N / (E·D²), where the numerator
                      N and denominator D are class-weighted sums and E counts cases.
    softmax CE:  dL/dz = (p·Σg - g) / (cases·voxels)
    sigmoid BCE: dL/dz = (p - g) / (cases·classes·voxels)

Each block of voxels writes its own partial sums, which are then reduced with ``torch.sum``, so
results do not depend on scheduling (atomic adds would). Half-precision logits are read as they
are and computed in float32; float64 logits are computed in float64. Class indices outside
[0, classes) make the loss NaN; the loss modules also raise on them, outside torch.compile.

The backward pass is not itself differentiable: double backward (``create_graph=True``, as in
gradient penalties) raises rather than silently dropping the second-order term. Gradients do not
flow into the target; the loss modules use the PyTorch path when the target requires them.

Focal loss uses a separate pointwise reduction, sharing the activation/target loader. Its
softmax derivative uses q=dL/d(log p), then dL/dz=q-p·Σq; sigmoid differentiates the positive
and negative focal terms separately. Stable log probabilities avoid log(0) for finite logits.
"""

from __future__ import annotations

import math
import torch
from torch import Tensor

from exacta._loss_reductions import EmptyTarget, WeightType, generalized_dice_terms, masked_overlap_terms
from exacta._distributed import sum_ranks

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
    tversky: tuple[float, float] | None = None,
    generalized: tuple[WeightType, EmptyTarget] | None = None,
    mask: Tensor | None = None,
    ignore_index: int | None = None,
    class_weight: Tensor | None = None,
    focal: tuple[float, float | None] | None = None,
    distributed: bool = False,
) -> Tensor:
    """Overlap losses share volume passes; only their small final reductions differ."""
    config = (
        activation, one_hot, include_background, batch_dice, smooth, lambda_dice, lambda_ce,
        cross_entropy, tversky, generalized, ignore_index, focal, distributed,
    )
    return _SegLoss.apply(logits, target, mask, class_weight, config)


def _layout(logits: Tensor) -> tuple[int, int, int, int, int]:
    batch, classes = logits.shape[:2]
    voxels = math.prod(logits.shape[2:])
    if voxels >= 2**31:
        raise ValueError("the fused loss supports fewer than 2^31 voxels per case")
    c_pad = triton.next_power_of_2(classes)
    block = max(16, min(1024, 8192 // c_pad))  # keep the (classes, voxels) tile in registers
    return batch, classes, voxels, c_pad, block


class _SegLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: Tensor, target: Tensor, mask: Tensor | None, class_weight: Tensor | None,
                config: tuple) -> Tensor:  # type: ignore[override]
        (activation, one_hot, include_background, batch_dice, smooth, lambda_dice, lambda_ce,
         ce, tversky, generalized, ignore_index, focal, distributed) = config
        logits, target = logits.contiguous(), target.contiguous()
        mask = None if mask is None else mask.contiguous()
        masked = mask is not None or ignore_index is not None or distributed
        batch, classes, voxels, c_pad, block = _layout(logits)
        compute = torch.float64 if logits.dtype == torch.float64 else torch.float32
        blocks = triton.cdiv(voxels, block)
        rows = 6 if masked else 5
        partial = torch.empty((batch, blocks, rows, c_pad), dtype=compute, device=logits.device)
        with torch.cuda.device(logits.device):  # Triton launches on the current device
            if batch and blocks:
                _forward[(blocks, batch)](
                    logits, target, mask if mask is not None else target,
                    class_weight if class_weight is not None else target, partial, voxels, classes, blocks,
                    ACT=ACTIVATIONS[activation], ONE_HOT=one_hot, C_PAD=c_pad, BLOCK=block,
                    TVERSKY=tversky is not None, CE=ce, HAS_MASK=mask is not None, IGNORE_INDEX=ignore_index,
                    WEIGHTED=class_weight is not None, FOCAL=focal is not None,
                    GAMMA=focal[0] if focal else 0.0, ALPHA=focal[1] if focal else None,
                    COUNTS=masked,
                    COMPUTE=tl.float64 if compute == torch.float64 else tl.float32,
                )  # fmt: skip
        sums = partial.sum(1, dtype=torch.float64)
        if distributed:
            sums = sum_ranks(sums.sum(0)).unsqueeze(0)
        inter, prob, truth, ce_terms = (sums[:, r, :classes] for r in range(4))
        invalid = sums[:, 4, 0].sum()  # class indices out of range
        if masked and not one_hot and not torch.compiler.is_compiling():
            _raise_invalid_indices(invalid, classes)
        valid_counts = sums[:, 5, 0] if masked else None
        weighted_count = ((truth * class_weight).sum() if class_weight is not None
                          and activation == "softmax" and not one_hot else None)

        first = 0 if include_background else 1
        inter, prob, truth = inter[:, first:], prob[:, first:], truth[:, first:]
        if batch_dice:
            inter, prob, truth = inter.sum(0), prob.sum(0), truth.sum(0)
        if generalized is not None:
            num, den, weights = generalized_dice_terms(
                inter, prob, truth, weight_type=generalized[0], empty_target=generalized[1],
                smooth=smooth,
            )
        elif tversky is None:
            den = prob + truth + smooth
            num = 2 * inter + smooth
        else:
            alpha, beta = tversky
            # Rows named prob/truth hold FP/FN for Tversky, accumulated directly to avoid
            # cancellation from subtracting almost equal sums near a perfect prediction.
            den = inter + alpha * prob + beta * truth + smooth
            num = inter + smooth
        if lambda_dice == 0:
            num, den = torch.zeros_like(num), torch.ones_like(den)
            scale = 0.0
            loss = torch.zeros((), dtype=torch.float64, device=logits.device)
        elif valid_counts is not None:
            num, den, scale = masked_overlap_terms(
                num, den, valid_counts.sum() if batch_dice else valid_counts,
            )
            loss = lambda_dice * ((1 - num / den) * scale).sum()
        else:
            scale = 1 / num.numel()
            loss = lambda_dice * (1 - (num / den).mean())
        ce_scale = torch.zeros((), dtype=torch.float64, device=logits.device)
        if ce:
            elements = (batch * voxels * (classes if activation == "sigmoid" else 1)
                        if valid_counts is None else
                        (valid_counts.sum() * (classes if activation == "sigmoid" else 1)).clamp_min(1))
            if weighted_count is not None:
                elements = torch.where(weighted_count > 0, weighted_count, 1)
            loss = loss + lambda_ce * ce_terms.sum() / elements
            ce_scale = ce_scale + lambda_ce / elements

        coef = torch.zeros((batch, 2, c_pad), dtype=torch.float64, device=logits.device)
        if generalized is not None:
            coef[:, 0, first:classes] = lambda_dice * 2 * weights * scale / den
            coef[:, 1, first:classes] = lambda_dice * weights * scale * num / (den * den)
        elif tversky is None:
            coef[:, 0, first:classes] = lambda_dice * 2 * scale / den
            coef[:, 1, first:classes] = lambda_dice * scale * num / (den * den)
        else:
            coef[:, 0, first:classes] = scale * (den - (1 - alpha - beta) * num) / (den * den)
            coef[:, 1, first:classes] = scale * alpha * num / (den * den)
        ctx.save_for_backward(logits, target, coef, ce_scale, mask, class_weight)
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
        logits, target, coef, ce_scale, mask, class_weight = ctx.saved_tensors
        activation, one_hot = ctx.config[:2]
        ce = ctx.config[7]
        ignore_index, focal, distributed = ctx.config[-3:]
        if distributed:
            grad_out = sum_ranks(grad_out)
        batch, classes, voxels, c_pad, block = _layout(logits)
        compute = torch.float64 if logits.dtype == torch.float64 else torch.float32
        coef = (coef * grad_out).to(compute)
        ce_scale = (ce_scale * grad_out).to(compute).reshape(1)
        grad = torch.empty_like(logits)
        if not logits.numel():
            return grad, None, None, None, None
        with torch.cuda.device(logits.device):
            _backward[(triton.cdiv(voxels, block), batch)](
                logits, target, mask if mask is not None else target,
                class_weight if class_weight is not None else target, coef, ce_scale, grad, voxels, classes,
                ACT=ACTIVATIONS[activation], ONE_HOT=one_hot, CE=ce, C_PAD=c_pad, BLOCK=block,
                HAS_MASK=mask is not None, IGNORE_INDEX=ignore_index,
                WEIGHTED=class_weight is not None, FOCAL=focal is not None,
                GAMMA=focal[0] if focal else 0.0, ALPHA=focal[1] if focal else None,
                COMPUTE=tl.float64 if compute == torch.float64 else tl.float32,
            )  # fmt: skip
        return grad, None, None, None, None


def focal_loss(
    logits: Tensor, target: Tensor, *, activation: str, one_hot: bool,
    gamma: float, alpha: float | None,
    mask: Tensor | None = None, ignore_index: int | None = None,
    distributed: bool = False,
) -> Tensor:
    """Mean focal CE/BCE, recomputing activations in backward instead of saving them."""
    return _FocalLoss.apply(logits, target, mask, (activation, one_hot, gamma, alpha, ignore_index, distributed))


class _FocalLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: Tensor, target: Tensor, mask: Tensor | None, config: tuple) -> Tensor:
        activation, one_hot, gamma, alpha, ignore_index, distributed = config
        logits, target = logits.contiguous(), target.contiguous()
        mask = None if mask is None else mask.contiguous()
        masked = mask is not None or ignore_index is not None or distributed
        batch, classes, voxels, c_pad, block = _layout(logits)
        compute = torch.float64 if logits.dtype == torch.float64 else torch.float32
        blocks = triton.cdiv(voxels, block)
        partial = torch.empty((batch, 3 if masked else 1, blocks), device=logits.device, dtype=compute)
        elements = batch * voxels * (classes if activation == "sigmoid" else 1)
        with torch.cuda.device(logits.device):
            if batch and blocks:
                _focal_kernel[(blocks, batch)](
                    logits, target, mask if mask is not None else target, partial, partial, partial, partial,
                    voxels, classes, blocks,
                    ACT=ACTIVATIONS[activation], ONE_HOT=one_hot, GAMMA=gamma, ALPHA=alpha,
                    HAS_MASK=mask is not None, IGNORE_INDEX=ignore_index,
                    COUNTS=masked,
                    BACKWARD=False, ELEMENTS=elements, C_PAD=c_pad, BLOCK=block,
                    COMPUTE=tl.float64 if compute == torch.float64 else tl.float32,
                )
        normalizer = None
        sums = partial.sum((0, 2), dtype=torch.float64)
        if distributed:
            sums = sum_ranks(sums)
        if masked:
            normalizer = (sums[1] * (classes if activation == "sigmoid" else 1)).clamp_min(1)
            if not one_hot and not torch.compiler.is_compiling():
                _raise_invalid_indices(sums[2], classes)
        ctx.save_for_backward(logits, target, mask, normalizer)
        ctx.config = config
        return (sums[0]
                / (elements if normalizer is None else normalizer)).to(compute)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "the fused Triton loss does not support double backward (create_graph=True); "
                "use backend='torch'"
            )
        logits, target, mask, normalizer = ctx.saved_tensors
        activation, one_hot, gamma, alpha, ignore_index, distributed = ctx.config
        if distributed:
            grad_out = sum_ranks(grad_out)
        batch, classes, voxels, c_pad, block = _layout(logits)
        compute = torch.float64 if logits.dtype == torch.float64 else torch.float32
        blocks = triton.cdiv(voxels, block)
        elements = batch * voxels * (classes if activation == "sigmoid" else 1)
        grad = torch.empty_like(logits)
        if not logits.numel():
            return grad, None, None, None
        with torch.cuda.device(logits.device):
            _focal_kernel[(blocks, batch)](
                logits, target, mask if mask is not None else target, grad, grad, grad_out,
                normalizer if normalizer is not None else grad_out, voxels, classes, blocks,
                ACT=ACTIVATIONS[activation], ONE_HOT=one_hot, GAMMA=gamma, ALPHA=alpha,
                HAS_MASK=mask is not None, IGNORE_INDEX=ignore_index,
                COUNTS=normalizer is not None,
                BACKWARD=True, ELEMENTS=elements, C_PAD=c_pad, BLOCK=block,
                COMPUTE=tl.float64 if compute == torch.float64 else tl.float32,
            )
        return grad, None, None, None


def _raise_invalid_indices(invalid: Tensor, classes: int) -> None:
    """Validate the fused scan's result without allocating masked copies of the labels."""
    if invalid.item() > 0:
        raise ValueError(f"unmasked class indices must lie in [0, {classes})")


if triton is not None:

    @triton.jit
    def _tile(logits, target, voxel_mask, b, voxel0, voxels, classes, ACT, ONE_HOT,
              HAS_MASK, IGNORE_INDEX, C_PAD, BLOCK, COMPUTE):
        """Activation p, target g, logits z and log-normalizer of a (classes, voxels) tile.

        For softmax, z and the log-normalizer remain shifted by the maximum logit, so their
        difference does not lose precision when all logits share a large offset.
        """
        offs = voxel0 + tl.arange(0, BLOCK)
        ch = tl.arange(0, C_PAD)
        in_voxels = offs < voxels
        in_classes = ch < classes
        mask = in_classes[:, None] & in_voxels[None, :]
        valid = in_voxels
        if HAS_MASK:
            valid = valid & tl.load(voxel_mask + b * voxels + offs, mask=in_voxels, other=False).to(tl.int1)
        bad = tl.full((BLOCK,), False, tl.int1)
        if not ONE_HOT:
            t = tl.load(target + b * voxels + offs, mask=in_voxels, other=0).to(tl.int64)
            if IGNORE_INDEX is not None:
                valid = valid & (t != IGNORE_INDEX)
            bad = valid & ((t < 0) | (t >= classes))
        row = b * classes * voxels + ch[:, None].to(tl.int64) * voxels + offs[None, :]
        z = tl.load(logits + row, mask=mask, other=0.0).to(COMPUTE)
        if HAS_MASK or IGNORE_INDEX is not None:
            z = tl.where(valid[None, :], z, 0.0)
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
            g = (ch[:, None] == t[None, :]).to(COMPUTE)
        if HAS_MASK or IGNORE_INDEX is not None:
            g = tl.where(valid[None, :], g, 0.0)
        return p, g, z, lse, mask, row, ch, valid, bad

    @triton.jit
    def _forward(
        logits, target, voxel_mask, class_weight, partial, voxels, classes, blocks,
        ACT: tl.constexpr, ONE_HOT: tl.constexpr, C_PAD: tl.constexpr, BLOCK: tl.constexpr,
        TVERSKY: tl.constexpr, CE: tl.constexpr,
        WEIGHTED: tl.constexpr, FOCAL: tl.constexpr, GAMMA: tl.constexpr, ALPHA: tl.constexpr,
        COUNTS: tl.constexpr,
        HAS_MASK: tl.constexpr, IGNORE_INDEX: tl.constexpr,
        COMPUTE: tl.constexpr,
    ):  # fmt: skip
        blk = tl.program_id(0)
        b = tl.program_id(1).to(tl.int64)
        p, g, z, lse, mask, row, ch, valid, bad = _tile(
            logits, target, voxel_mask, b, blk * BLOCK, voxels, classes, ACT, ONE_HOT,
            HAS_MASK, IGNORE_INDEX, C_PAD, BLOCK, COMPUTE
        )
        w = (mask & valid[None, :]).to(COMPUTE)
        # Terms with g = 0 are left out rather than multiplied by 0: a class masked with a -inf
        # logit would otherwise make 0 * inf = NaN.
        if not CE:
            ce = tl.full((C_PAD, BLOCK), 0.0, COMPUTE)
        elif FOCAL:
            ce = _focal_terms(p, g, z, lse, ch, mask, ACT, GAMMA, ALPHA, False, C_PAD, BLOCK, COMPUTE)
        elif ACT == 0:  # cross-entropy -Σ g log p, per class
            ce = tl.where(g != 0, g * (lse[None, :] - z), 0.0)
        else:  # binary cross-entropy with logits
            ce = tl.maximum(z, 0.0) - tl.where(g != 0, z * g, 0.0) + libdevice.log1p(tl.exp(-tl.abs(z)))
        if CE and WEIGHTED:
            ce *= tl.load(class_weight + ch, mask=ch < classes, other=0).to(COMPUTE)[:, None]
        rows: tl.constexpr = 6 if COUNTS else 5
        out = partial + ((b * blocks + blk) * rows) * C_PAD + ch
        tl.store(out, tl.sum(p * g * w, axis=1))
        if TVERSKY:
            tl.store(out + C_PAD, tl.sum(p * (1.0 - g) * w, axis=1))
            tl.store(out + 2 * C_PAD, tl.sum((1.0 - p) * g * w, axis=1))
        else:
            tl.store(out + C_PAD, tl.sum(p * w, axis=1))
            tl.store(out + 2 * C_PAD, tl.sum(g * w, axis=1))
        tl.store(out + 3 * C_PAD, tl.sum(ce * w, axis=1))
        invalid = tl.zeros([C_PAD], dtype=COMPUTE)
        if not ONE_HOT:
            invalid = tl.where(ch == 0, tl.sum(bad.to(COMPUTE), axis=0), 0.0)
        tl.store(out + 4 * C_PAD, invalid)
        if COUNTS:
            tl.store(out + 5 * C_PAD, tl.where(ch == 0, tl.sum(valid.to(COMPUTE), axis=0), 0.0))

    @triton.jit
    def _backward(
        logits, target, voxel_mask, class_weight, coef, ce_scale, grad, voxels, classes,
        ACT: tl.constexpr, ONE_HOT: tl.constexpr, CE: tl.constexpr, C_PAD: tl.constexpr, BLOCK: tl.constexpr,
        HAS_MASK: tl.constexpr, IGNORE_INDEX: tl.constexpr,
        WEIGHTED: tl.constexpr, FOCAL: tl.constexpr, GAMMA: tl.constexpr, ALPHA: tl.constexpr,
        COMPUTE: tl.constexpr,
    ):  # fmt: skip
        blk = tl.program_id(0)
        b = tl.program_id(1).to(tl.int64)
        p, g, z, lse, mask, row, ch, valid, bad = _tile(
            logits, target, voxel_mask, b, blk * BLOCK, voxels, classes, ACT, ONE_HOT,
            HAS_MASK, IGNORE_INDEX, C_PAD, BLOCK, COMPUTE
        )
        a = tl.load(coef + (b * 2) * C_PAD + ch)
        c = tl.load(coef + (b * 2 + 1) * C_PAD + ch)
        d_p = c[:, None] - a[:, None] * g  # dLdice/dp
        weight = tl.full((C_PAD, 1), 1.0, COMPUTE)
        if WEIGHTED:
            weight = tl.load(class_weight + ch, mask=ch < classes, other=0).to(COMPUTE)[:, None]
        if ACT == 0:
            d_z = p * (d_p - tl.sum(p * d_p, axis=0)[None, :])
            if CE and not FOCAL:
                wg = weight * g
                d_z += tl.load(ce_scale) * (p * tl.sum(wg, axis=0)[None, :] - wg)
        elif ACT == 1:
            d_z = d_p * p * (1.0 - p)
            if CE and not FOCAL:
                d_z += tl.load(ce_scale) * weight * (p - g)
        else:
            d_z = d_p
        if CE and FOCAL:
            d_z += tl.load(ce_scale) * _focal_terms(
                p, g, z, lse, ch, mask, ACT, GAMMA, ALPHA, True, C_PAD, BLOCK, COMPUTE,
            )
        tl.store(grad + row, tl.where(valid[None, :], d_z, 0).to(grad.dtype.element_ty), mask=mask)

    @triton.jit
    def _focal_terms(p, g, z, lse, ch, mask, ACT: tl.constexpr, GAMMA: tl.constexpr,
                     ALPHA: tl.constexpr, BACKWARD: tl.constexpr, C_PAD: tl.constexpr,
                     BLOCK: tl.constexpr, COMPUTE: tl.constexpr):
        if ACT == 0:
            logp = z - lse[None, :]
            # expm1 retains 1-p near p=1. Use exp(logp) consistently in its derivative.
            p = tl.exp(logp)
            u = -libdevice.expm1(logp)
            if GAMMA == 0:
                modulation = tl.full((C_PAD, BLOCK), 1.0, COMPUTE)
            else:
                modulation = tl.where(u > 0, tl.exp(GAMMA * tl.log(tl.where(u > 0, u, 1.0))), 0.0)
            weight = tl.full((C_PAD, 1), 1.0, COMPUTE)
            if ALPHA is not None:
                weight = tl.where(
                    ch[:, None] == 0, tl.full((C_PAD, 1), 1.0 - ALPHA, COMPUTE),
                    tl.full((C_PAD, 1), ALPHA, COMPUTE),
                )
            if BACKWARD:
                if GAMMA == 0:
                    q = -weight * g
                else:
                    q = weight * g * modulation * (GAMMA * p * logp / tl.where(u > 0, u, 1.0) - 1.0)
                q = tl.where(mask, q, 0.0)
                dz = q - p * tl.sum(q, axis=0)[None, :]
            else:
                terms = -weight * g * modulation * logp
        else:
            # Stable log(sigmoid(z)) and log(sigmoid(-z)); neither forms log(0).
            tail = libdevice.log1p(tl.exp(-tl.abs(z)))
            pos_ce = tl.maximum(-z, 0.0) + tail
            neg_ce = tl.maximum(z, 0.0) + tail
            pos_mod = tl.exp(-GAMMA * neg_ce)
            neg_mod = tl.exp(-GAMMA * pos_ce)
            pos_weight = g
            neg_weight = 1.0 - g
            if ALPHA is not None:
                pos_weight *= ALPHA
                neg_weight *= 1.0 - ALPHA
            if BACKWARD:
                u = tl.sigmoid(-z)
                dz = (-pos_weight * pos_mod * (u + GAMMA * p * pos_ce)
                      + neg_weight * neg_mod * (p + GAMMA * u * neg_ce))
            else:
                terms = pos_weight * pos_mod * pos_ce + neg_weight * neg_mod * neg_ce
        if BACKWARD:
            return dz
        else:
            return terms

    @triton.jit
    def _focal_kernel(
        logits, target, voxel_mask, partial, grad, grad_out, normalizer, voxels, classes, blocks,
        ACT: tl.constexpr, ONE_HOT: tl.constexpr, GAMMA: tl.constexpr, ALPHA: tl.constexpr,
        BACKWARD: tl.constexpr, ELEMENTS: tl.constexpr, C_PAD: tl.constexpr, BLOCK: tl.constexpr,
        HAS_MASK: tl.constexpr, IGNORE_INDEX: tl.constexpr,
        COUNTS: tl.constexpr,
        COMPUTE: tl.constexpr,
    ):
        blk = tl.program_id(0)
        b = tl.program_id(1).to(tl.int64)
        p, g, z, lse, mask, row, ch, valid, bad = _tile(
            logits, target, voxel_mask, b, blk * BLOCK, voxels, classes, ACT, ONE_HOT,
            HAS_MASK, IGNORE_INDEX, C_PAD, BLOCK, COMPUTE
        )
        result = _focal_terms(p, g, z, lse, ch, mask, ACT, GAMMA, ALPHA, BACKWARD, C_PAD, BLOCK, COMPUTE)
        if BACKWARD:
            dz = result
        else:
            terms = result
        if BACKWARD:
            if COUNTS:
                dz *= tl.load(grad_out).to(COMPUTE) / tl.load(normalizer).to(COMPUTE)
            else:
                dz *= tl.load(grad_out).to(COMPUTE) / ELEMENTS
            dz = tl.where(valid[None, :], dz, 0.0)
            tl.store(grad + row, dz.to(grad.dtype.element_ty), mask=mask)
        else:
            loss = tl.sum(tl.sum(tl.where(mask & valid[None, :], terms, 0.0), axis=0), axis=0)
            invalid = tl.sum(bad.to(tl.int32), axis=0)
            if not ONE_HOT:
                loss = tl.where(invalid > 0, float("nan"), loss)
            if COUNTS:
                out = partial + b * 3 * blocks + blk
                tl.store(out, loss)
                tl.store(out + blocks, tl.sum(valid.to(COMPUTE), axis=0))
                tl.store(out + 2 * blocks, invalid)
            else:
                tl.store(partial + b * blocks + blk, loss)
