"""Fused overlap and focal losses on GPU, in Triton.

The PyTorch implementation in :mod:`exacta.losses.segmentation` materialises the activation, the
one-hot target and their product, and autograd keeps them for the backward pass: several times
the memory of the logits. Here the forward pass reads logits and target once and accumulates,
for every case and class, ``TP = Σ p·g``, ``FP = Σ p·(1-g)``, ``FN = Σ q·g``, ``Σ g`` and the
cross-entropy, with ``q = 1 - p`` formed without cancellation (see below); the backward pass reads
them again, recomputes the activation and writes the gradient from a closed form. Every overlap loss
is a ratio of non-negative sums, ``L = E / D`` with ``D = N + E``, so that nothing subtracts nearly
equal quantities (``1 - N/D`` has only absolute accuracy):

    Dice:    N = 2·TP + s, E = FP + FN (squared: E = Σ (p - g)²),
    Tversky: N = TP + s,   E = alpha·FP + beta·FN,
    Generalized Dice: N = 2 Σ_c w_c TP_c + s, E = Σ_c w_c (FP_c + FN_c),

and, with ``dL/dE = N/D²`` and ``dL/dN = -E/D²``,

    dL/dp = [(1-g)·A - g·B] / D² for per-class constants A, B, summed over the E averaged entries:
        Dice: A = N, B = N + 2E; Tversky: A = alpha·N, B = beta·N + E; weighted by w_c for generalized Dice;
        squared Dice: dL/dp = 2 [(1-g)·N·p - g·(N·q + E)] / D², since p - g = (1-g)·p - g·q.
    softmax CE:  dL/dz_i = -wg_i·q_i + p_i·Σ_{j≠i} wg_j      (wg = class weight · g), over cases·voxels
    sigmoid BCE: dL/dz = ((1-g)·p - g·q) / (cases·classes·voxels), with p = sigmoid(z), q = sigmoid(-z)

The softmax is differentiated as ``dz_i = q_i·v_i - p_i·Σ_{j≠i} v_j`` with ``v_j = p_j dL/dp_j + r_j``,
``r_j = dL/d(log p_j)`` of the cross-entropy or focal terms, which never forms ``1 - p_i`` or ``p_i - 1``.
The complement ``q_i`` and the sum over ``j != i`` use the sum over the classes other than the one above
one half (at most one class exceeds it) instead of subtracting from one or from the total, and the
log-normalizer is ``log1p`` of the sum of ``exp(z_j - max)`` over the classes other than one maximal
class, so that ``-log p_c`` is a sum of two non-negative terms (see :mod:`exacta._softmax`).

Each block of voxels writes its own partial sums, which are then reduced with ``torch.sum``, so
results do not depend on scheduling (atomic adds would). Half-precision logits are read as they
are and computed in float32; float64 logits are computed in float64. Class indices outside
[0, classes) make the loss NaN; the loss modules also raise on them, outside torch.compile.

The backward pass is not itself differentiable: double backward (``create_graph=True``, as in
gradient penalties) raises rather than silently dropping the second-order term. Gradients do not
flow into the target; the loss modules use the PyTorch path when the target requires them.

Focal loss uses a separate pointwise reduction, sharing the activation/target loader. Its
softmax derivative uses r = dL/d(log p), then the softmax product above; sigmoid differentiates the
positive and negative focal terms separately. Stable log probabilities avoid log(0) for finite logits.

The forms never subtract from one: where p rounds to 1, 1-p and p·(1-p) are 0 although q = 1 - p
(the sum of the other softmax probabilities, or sigmoid(-z)) and p·q are not (4.2e-18 at z = 40).
The derivative of the sigmoid is p·q, the gradient of the cross-entropy is (1-g)·p - g·q (sigmoid) or
-wg_i·q_i + p_i·Σ_{j≠i} wg_j (softmax), the cross-entropies are sums of non-negative terms and the
overlap losses are ratios of non-negative sums, so their relative error stays a few roundoffs at any
finite logit, in float32 and float64.
"""

from __future__ import annotations

import math
import torch
from torch import Tensor

from exacta._loss_reductions import EmptyTarget, WeightType, generalized_dice_terms, masked_overlap_terms, overlap_power
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
    overlap_power: float = 1.0,
    squared_pred: bool = False,
) -> Tensor:
    """Overlap losses share volume passes; only their small final reductions differ."""
    config = (
        activation, one_hot, include_background, batch_dice, smooth, lambda_dice, lambda_ce,
        cross_entropy, tversky, generalized, overlap_power, squared_pred, ignore_index, focal, distributed,
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
         ce, tversky, generalized, power, squared_pred, ignore_index, focal, distributed) = config
        logits, target = logits.contiguous(), target.contiguous()
        mask = None if mask is None else mask.contiguous()
        masked = mask is not None or ignore_index is not None or distributed
        batch, classes, voxels, c_pad, block = _layout(logits)
        compute = torch.float64 if logits.dtype == torch.float64 else torch.float32
        blocks = triton.cdiv(voxels, block)
        rows = 7 if masked else 6
        partial = torch.empty((batch, blocks, rows, c_pad), dtype=compute, device=logits.device)
        with torch.cuda.device(logits.device):  # Triton launches on the current device
            if batch and blocks:
                _forward[(blocks, batch)](
                    logits, target, mask if mask is not None else target,
                    class_weight if class_weight is not None else target, partial, voxels, classes, blocks,
                    ACT=ACTIVATIONS[activation], ONE_HOT=one_hot, C_PAD=c_pad, BLOCK=block,
                    CE=ce, HAS_MASK=mask is not None, IGNORE_INDEX=ignore_index,
                    WEIGHTED=class_weight is not None, FOCAL=focal is not None,
                    GAMMA=focal[0] if focal else 0.0, ALPHA=focal[1] if focal else None,
                    COUNTS=masked, SQUARED=squared_pred,
                    COMPUTE=tl.float64 if compute == torch.float64 else tl.float32,
                )  # fmt: skip
        sums = partial.sum(1, dtype=torch.float64)
        if distributed:
            sums = sum_ranks(sums.sum(0)).unsqueeze(0)
        inter, fp, fn, truth, ce_terms = (sums[:, r, :classes] for r in range(5))
        invalid = sums[:, 5, 0].sum()  # class indices out of range
        if masked and not one_hot and not torch.compiler.is_compiling():
            _raise_invalid_indices(invalid, classes)
        valid_counts = sums[:, 6, 0] if masked else None
        weighted_count = ((truth * class_weight).sum() if class_weight is not None
                          and activation == "softmax" and not one_hot else None)

        first = 0 if include_background or lambda_dice == 0 else 1
        inter, fp, fn, truth = inter[:, first:], fp[:, first:], fn[:, first:], truth[:, first:]
        if batch_dice:
            inter, fp, fn, truth = inter.sum(0), fp.sum(0), fn.sum(0), truth.sum(0)
        # The loss of every (case, class) is err / (num + err), a ratio of sums of non-negative terms.
        if generalized is not None:
            err, num, weights = generalized_dice_terms(
                inter, fp + fn, truth, weight_type=generalized[0], empty_target=generalized[1],
                smooth=smooth,
            )
        elif tversky is None:
            err = fp + fn  # squared: fp holds Σ (p - g)² and fn is zero
            num = 2 * inter + smooth
        else:
            alpha, beta = tversky
            err = alpha * fp + beta * fn
            num = inter + smooth
        if lambda_dice == 0:
            err, num = torch.zeros_like(num), torch.ones_like(num)
            scale = 0.0
            loss = torch.zeros((), dtype=torch.float64, device=logits.device)
        elif valid_counts is not None:
            err, num, scale = masked_overlap_terms(
                err, num, valid_counts.sum() if batch_dice else valid_counts,
            )
            loss = lambda_dice * (overlap_power(err / (num + err), power) * scale).sum()
        else:
            scale = 1 / num.numel()
            loss = lambda_dice * overlap_power(err / (num + err), power).mean()
        if power != 1:
            error = err / (num + err)
            factor = power * torch.where(error == 0, 1, error).pow(power - 1)
            scale = scale * torch.where(error == 0, 0, factor)
        ce_scale = torch.zeros((), dtype=torch.float64, device=logits.device)
        if ce:
            elements = (batch * voxels * (classes if activation == "sigmoid" else 1)
                        if valid_counts is None else
                        (valid_counts.sum() * (classes if activation == "sigmoid" else 1)).clamp_min(1))
            if weighted_count is not None:
                elements = torch.where(weighted_count > 0, weighted_count, 1)
            loss = loss + lambda_ce * ce_terms.sum() / elements
            ce_scale = ce_scale + lambda_ce / elements

        # dL/dp = [(1-g)·coef0 - g·coef1] with coef0 = N/D² and coef1 = (N + 2E)/D² for Dice (alpha·N/D² and
        # (beta·N + E)/D² for Tversky), times the weight of the class and the averaging scale; the squared
        # Dice has dL/dp = (1-g)·p·coef0 - g·(q·coef0 + coef1) with coef0 = 2N/D² and coef1 = 2E/D².
        coef = torch.zeros((batch, 2, c_pad), dtype=torch.float64, device=logits.device)
        den = num + err
        lead = lambda_dice * scale / (den * den)
        if generalized is not None:
            lead = lead * weights
        if tversky is not None:
            coef0, coef1 = tversky[0] * num, tversky[1] * num + err
        elif squared_pred:
            coef0, coef1 = 2 * num, 2 * err
        else:
            coef0, coef1 = num, num + 2 * err
        coef[:, 0, first:classes] = lead * coef0
        coef[:, 1, first:classes] = lead * coef1
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
                SQUARED=ctx.config[11],
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
        """Activation p, its complement q = 1 - p, target g, logits z and log-normalizer of a tile.

        q is the sum of the other probabilities where p exceeds one half (and 1 - p otherwise, where p <= 1/2),
        sigmoid(-z) or 1 - p for the three activations, so that it does not vanish where p rounds to one.
        For softmax, z remains shifted by the maximum logit, so that z and the log-normalizer, whose
        difference is -log p, do not lose precision when all logits share a large offset; the log-normalizer
        is log1p of the sum of exp(z_j) over the classes other than one maximal class.
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
            p = e / tl.sum(e, axis=0)[None, :]
            big = p > 0.5
            q = tl.where(big, tl.sum(tl.where(big, 0.0, p), axis=0)[None, :], 1.0 - p)
            below = tl.sum(tl.where(in_classes[:, None] & (z < 0.0), e, 0.0), axis=0)
            ties = tl.sum(tl.where(in_classes[:, None] & (z == 0.0), 1.0, 0.0), axis=0)
            lse = libdevice.log1p(below + (ties - 1.0))
        elif ACT == 1:
            p = tl.sigmoid(z)
            q = tl.sigmoid(-z)
            lse = tl.zeros([BLOCK], dtype=COMPUTE)
        else:
            p = z
            q = 1.0 - z
            lse = tl.zeros([BLOCK], dtype=COMPUTE)
        if ONE_HOT:
            g = tl.load(target + row, mask=mask, other=0).to(COMPUTE)
        else:
            # int64 so that indices of 2^31 or more are not truncated into the class range
            g = (ch[:, None] == t[None, :]).to(COMPUTE)
        if HAS_MASK or IGNORE_INDEX is not None:
            g = tl.where(valid[None, :], g, 0.0)
        return p, q, g, z, lse, mask, row, ch, valid, bad

    @triton.jit
    def _softmax_vjp(p, q, v):
        """dz_i = q_i v_i - p_i sum_{j != i} v_j, which equals v_i - p_i sum_j v_j without forming 1 - p_i."""
        big = p > 0.5
        below = tl.sum(tl.where(big, 0.0, v), axis=0)[None, :]
        others = tl.where(big, below, tl.sum(v, axis=0)[None, :] - v)
        return q * v - p * others

    @triton.jit
    def _forward(
        logits, target, voxel_mask, class_weight, partial, voxels, classes, blocks,
        ACT: tl.constexpr, ONE_HOT: tl.constexpr, C_PAD: tl.constexpr, BLOCK: tl.constexpr,
        CE: tl.constexpr,
        WEIGHTED: tl.constexpr, FOCAL: tl.constexpr, GAMMA: tl.constexpr, ALPHA: tl.constexpr,
        COUNTS: tl.constexpr, SQUARED: tl.constexpr,
        HAS_MASK: tl.constexpr, IGNORE_INDEX: tl.constexpr,
        COMPUTE: tl.constexpr,
    ):  # fmt: skip
        blk = tl.program_id(0)
        b = tl.program_id(1).to(tl.int64)
        p, q, g, z, lse, mask, row, ch, valid, bad = _tile(
            logits, target, voxel_mask, b, blk * BLOCK, voxels, classes, ACT, ONE_HOT,
            HAS_MASK, IGNORE_INDEX, C_PAD, BLOCK, COMPUTE
        )
        w = (mask & valid[None, :]).to(COMPUTE)
        # Terms with g = 0 are left out rather than multiplied by 0: a class masked with a -inf
        # logit would otherwise make 0 * inf = NaN.
        if not CE:
            ce = tl.full((C_PAD, BLOCK), 0.0, COMPUTE)
        elif FOCAL:
            ce = _focal_terms(p, q, g, z, lse, ch, mask, ACT, GAMMA, ALPHA, False, C_PAD, BLOCK, COMPUTE)
        elif ACT == 0:  # cross-entropy -Σ g log p = Σ g (log-normalizer - z), both terms non-negative
            ce = tl.where(g != 0, g * (lse[None, :] - z), 0.0)
        else:  # binary cross-entropy with logits: log1p(exp(-|z|)) plus (1-g)·z for z >= 0, g·|z| for z < 0
            linear = tl.where(z >= 0, tl.where(g != 1, (1.0 - g) * z, 0.0), tl.where(g != 0, -g * z, 0.0))
            ce = linear + libdevice.log1p(tl.exp(-tl.abs(z)))
        if CE and WEIGHTED:
            ce *= tl.load(class_weight + ch, mask=ch < classes, other=0).to(COMPUTE)[:, None]
        rows: tl.constexpr = 7 if COUNTS else 6
        out = partial + ((b * blocks + blk) * rows) * C_PAD + ch
        tl.store(out, tl.sum(p * g * w, axis=1))
        if SQUARED:  # Σ (p - g)² with p - g = (1-g)·p - g·q, and no separate FN
            gap = (1.0 - g) * p - g * q
            tl.store(out + C_PAD, tl.sum(gap * gap * w, axis=1))
            tl.store(out + 2 * C_PAD, tl.zeros([C_PAD], dtype=COMPUTE))
        else:  # FP and FN, each a sum of non-negative terms
            tl.store(out + C_PAD, tl.sum(p * (1.0 - g) * w, axis=1))
            tl.store(out + 2 * C_PAD, tl.sum(q * g * w, axis=1))
        tl.store(out + 3 * C_PAD, tl.sum(g * w, axis=1))
        tl.store(out + 4 * C_PAD, tl.sum(ce * w, axis=1))
        invalid = tl.zeros([C_PAD], dtype=COMPUTE)
        if not ONE_HOT:
            invalid = tl.where(ch == 0, tl.sum(bad.to(COMPUTE), axis=0), 0.0)
        tl.store(out + 5 * C_PAD, invalid)
        if COUNTS:
            tl.store(out + 6 * C_PAD, tl.where(ch == 0, tl.sum(valid.to(COMPUTE), axis=0), 0.0))

    @triton.jit
    def _backward(
        logits, target, voxel_mask, class_weight, coef, ce_scale, grad, voxels, classes,
        ACT: tl.constexpr, ONE_HOT: tl.constexpr, CE: tl.constexpr, C_PAD: tl.constexpr, BLOCK: tl.constexpr,
        HAS_MASK: tl.constexpr, IGNORE_INDEX: tl.constexpr,
        WEIGHTED: tl.constexpr, FOCAL: tl.constexpr, GAMMA: tl.constexpr, ALPHA: tl.constexpr,
        COMPUTE: tl.constexpr, SQUARED: tl.constexpr,
    ):  # fmt: skip
        blk = tl.program_id(0)
        b = tl.program_id(1).to(tl.int64)
        p, q, g, z, lse, mask, row, ch, valid, bad = _tile(
            logits, target, voxel_mask, b, blk * BLOCK, voxels, classes, ACT, ONE_HOT,
            HAS_MASK, IGNORE_INDEX, C_PAD, BLOCK, COMPUTE
        )
        a = tl.load(coef + (b * 2) * C_PAD + ch)[:, None]
        c = tl.load(coef + (b * 2 + 1) * C_PAD + ch)[:, None]
        # Every term is non-negative: the Dice derivative is (1-g)·a - g·c, and for the squared denominator
        # (1-g)·a·p - g·(a·q + c), where p - g = (1-g)·p - g·q does not vanish as p rounds to g.
        if SQUARED:
            d_p = a * (1.0 - g) * p - g * (a * q + c)
        else:
            d_p = a * (1.0 - g) - c * g
        weight = tl.full((C_PAD, 1), 1.0, COMPUTE)
        if WEIGHTED:
            weight = tl.load(class_weight + ch, mask=ch < classes, other=0).to(COMPUTE)[:, None]
        if ACT == 0:
            v = p * d_p
            if CE and not FOCAL:
                v -= tl.load(ce_scale) * weight * g  # cross-entropy: d/d(log p) = -weight·g
            if CE and FOCAL:
                v += tl.load(ce_scale) * _focal_terms(
                    p, q, g, z, lse, ch, mask, ACT, GAMMA, ALPHA, True, C_PAD, BLOCK, COMPUTE,
                )
            d_z = _softmax_vjp(p, q, v)
        elif ACT == 1:
            d_z = d_p * p * q  # 1 - p is 0 where p rounds to 1
            if CE and not FOCAL:
                d_z += tl.load(ce_scale) * weight * ((1.0 - g) * p - g * q)
            if CE and FOCAL:
                d_z += tl.load(ce_scale) * _focal_terms(
                    p, q, g, z, lse, ch, mask, ACT, GAMMA, ALPHA, True, C_PAD, BLOCK, COMPUTE,
                )
        else:
            d_z = d_p
        tl.store(grad + row, tl.where(valid[None, :], d_z, 0).to(grad.dtype.element_ty), mask=mask)

    @triton.jit
    def _focal_terms(p, q, g, z, lse, ch, mask, ACT: tl.constexpr, GAMMA: tl.constexpr,
                     ALPHA: tl.constexpr, BACKWARD: tl.constexpr, C_PAD: tl.constexpr,
                     BLOCK: tl.constexpr, COMPUTE: tl.constexpr):
        """Focal terms (forward) or their derivative (backward): d/d(log p) for softmax, d/dz for sigmoid."""
        if ACT == 0:
            nll = lse[None, :] - z  # -log p, a sum of two non-negative terms
            weight = tl.full((C_PAD, 1), 1.0, COMPUTE)
            if ALPHA is not None:
                weight = tl.where(
                    ch[:, None] == 0, tl.full((C_PAD, 1), 1.0 - ALPHA, COMPUTE),
                    tl.full((C_PAD, 1), ALPHA, COMPUTE),
                )
            # (1-p)**gamma from q = 1 - p, accurate where p rounds to 1 (exp(gamma * log) would lose
            # gamma * |log| * u of relative accuracy). The power of an underflowed q is zero, with zero gradient.
            live = q > 0
            safe = tl.where(live, q, 1.0)
            if GAMMA == 0:
                modulation = tl.full((C_PAD, BLOCK), 1.0, COMPUTE)
            else:
                modulation = tl.where(live, libdevice.pow(safe, tl.full((C_PAD, BLOCK), GAMMA, COMPUTE)), 0.0)
            if BACKWARD:
                # d/d(log p) of weight·g·q**gamma·nll, with dq/d(log p) = -p and dnll/d(log p) = -1.
                if GAMMA == 0:
                    r = -weight * g
                else:  # q**(gamma-1) nll as q**gamma (nll / q): the first overflows for gamma < 1 and tiny q
                    ratio = tl.where(live, nll / safe, 0.0)
                    r = -weight * g * modulation * (1.0 + GAMMA * p * ratio)
                result = tl.where(mask, r, 0.0)
            else:
                result = tl.where(g != 0, weight * g * modulation * nll, 0.0)
        else:
            # Stable log(sigmoid(z)) and log(sigmoid(-z)); neither forms log(0).
            tail = libdevice.log1p(tl.exp(-tl.abs(z)))
            pos_ce = tl.maximum(-z, 0.0) + tail
            neg_ce = tl.maximum(z, 0.0) + tail
            u = q
            # (1-p)**gamma and p**gamma from the accurate probabilities: exp(-gamma * ce) would lose
            # gamma * |log| * u of relative accuracy.
            pos_mod = libdevice.pow(u, tl.full((C_PAD, BLOCK), GAMMA, COMPUTE))
            neg_mod = libdevice.pow(p, tl.full((C_PAD, BLOCK), GAMMA, COMPUTE))
            pos_weight = g
            neg_weight = 1.0 - g
            if ALPHA is not None:
                pos_weight *= ALPHA
                neg_weight *= 1.0 - ALPHA
            if BACKWARD:
                result = (-pos_weight * pos_mod * (u + GAMMA * p * pos_ce)
                          + neg_weight * neg_mod * (p + GAMMA * u * neg_ce))
            else:
                result = pos_weight * pos_mod * pos_ce + neg_weight * neg_mod * neg_ce
        return result

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
        p, q, g, z, lse, mask, row, ch, valid, bad = _tile(
            logits, target, voxel_mask, b, blk * BLOCK, voxels, classes, ACT, ONE_HOT,
            HAS_MASK, IGNORE_INDEX, C_PAD, BLOCK, COMPUTE
        )
        result = _focal_terms(p, q, g, z, lse, ch, mask, ACT, GAMMA, ALPHA, BACKWARD, C_PAD, BLOCK, COMPUTE)
        if BACKWARD:
            if ACT == 0:
                dz = _softmax_vjp(p, q, result)
            else:
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
