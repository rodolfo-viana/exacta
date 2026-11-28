"""Softmax and cross-entropy that stay accurate where the dominant probability rounds to one.

When one logit exceeds the others by 37 (binary64; 17 in binary32), the dominant ``p = softmax(z)`` rounds
to 1 although the others are representable (``exp(-40) = 4.2e-18``), and every expression that subtracts
from one loses all its digits:

* ``torch.softmax`` differentiates as ``p_i (d_i - sum_j p_j d_j)``, which is exactly 0 for the dominant
  class, where the true value is of the order of ``(1 - p)`` times the largest term;
* ``-log p_t`` is ``-log(1)`` = 0 and the gradient ``p - onehot`` is 0, for the dominant target;
* ``1 - p_i``, which every overlap and focal loss needs, is 0.

Here ``q_i = 1 - p_i`` is formed as ``1 - p_i`` for every class but the one with the largest probability
(``p_i <= 1/2`` there, so that the subtraction is exact enough), and as the sum of the other probabilities
for that class. The negative log-probability ``-log p_c = log1p(sum_{j != top} exp(z_j - top)) + (top - z_c)``
is a sum of two non-negative terms for every class, so it keeps its relative accuracy whether or not ``z_c`` is
the maximum; ``sum_{j != top} exp(z_j - top)`` is ``q_top / p_top``. The vector-Jacobian product of all three
outputs has the form ``dz_i = q_i v_i - p_i sum_{j != i} v_j``, with ``v_j`` a signed sum of the upstream
gradients, so that no ``1 - p_i`` and no ``p_i - 1`` is ever formed for the dominant class: the sum over
``j != i`` is taken over the other classes for it, and as the total minus ``v_i`` for every other class,
where that is exact enough.

The backward passes are built from differentiable operations on the saved probabilities, so the PyTorch
backend of the losses keeps supporting double backward (gradient penalties) and gradients into a soft target.

Under ``torch.compile`` the public functions use ordinary differentiable operations instead of these Functions
(dynamo in PyTorch 2.4 cannot trace Functions with non-tensor arguments, None outputs or in-place writes): the
values are the same sums of non-negative terms, the gradient is autograd's.

For integer labels (the usual case) the overlap sums and the cross-entropy come from the label indices
directly (:class:`_IndexOverlap`, :class:`_CrossEntropyIndex`): no one-hot tensor, no dense ``1 - p``, and one
softmax shared by the Dice and cross-entropy terms of ``DiceCELoss``. Their backward passes are written
twice, in place when no graph is being built and out of place when one is (double backward), and tested
against each other and against the one-hot composition.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


# CUDA: a mask and a sum beat scatter/gather kernels (several times slower at this size) from about half a million
# voxels per batch on; below that the extra launches of the mask form cost more than they save.
_DENSE_VOXELS = 2**19


def _dense(index: Tensor) -> bool:
    return index.device.type == "cuda" and index.numel() >= _DENSE_VOXELS


def _positions(index: Tensor, classes: int) -> Tensor | None:
    """Where the mask form is used (large CUDA tensors): mask ``(B, C, *)`` of the entries at the class ``index``
    ``(B, 1, *)`` of every voxel, to pass as ``hit`` to the helpers below that would otherwise rebuild it.
    ``None`` elsewhere."""
    if not _dense(index):
        return None
    return torch.arange(classes, device=index.device).reshape(1, -1, *([1] * (index.ndim - 2))) == index


def _gather(x: Tensor, index: Tensor, hit: Tensor | None = None) -> Tensor:
    """``x`` ``(B, C, V)`` at the class ``index`` of every voxel, ``(B, 1, V)``. CUDA's scatter/gather kernels are
    several times slower than a mask and a sum over the few classes (the sum of one entry and zeros is exact)."""
    if _dense(index):
        hit = _positions(index, x.shape[1]) if hit is None else hit
        return torch.where(hit, x, 0).sum(1, keepdim=True)
    return x.gather(1, index)


def _lookup(table: Tensor, index: Tensor, hit: Tensor | None = None) -> Tensor:
    """``table`` ``(B, C)`` at the class ``index`` ``(B, 1, V)`` of every voxel, ``(B, 1, V)``."""
    if _dense(index):
        hit = _positions(index, table.shape[1]) if hit is None else hit
        return torch.where(hit, table.reshape(*table.shape, *([1] * (index.ndim - 2))), 0).sum(1, keepdim=True)
    return table.gather(1, index[:, 0])[:, None]


def _put(x: Tensor, index: Tensor, value: Tensor, hit: Tensor | None = None) -> Tensor:
    """``x`` with ``x[index] = value`` along the class axis: in place on the CPU unless a graph is being built."""
    if _dense(index):
        hit = _positions(index, x.shape[1]) if hit is None else hit
        return torch.where(hit, value, x)
    return x.scatter(1, index, value) if torch.is_grad_enabled() else x.scatter_(1, index, value)


def _exclusive(x: Tensor, index: Tensor, *, owned: bool = False, kept: Tensor | None = None,
               hit: Tensor | None = None) -> Tensor:
    """Sum over the classes other than ``index``, formed without subtracting from the total.

    ``owned``: on the CPU ``x`` may be modified in place (and is restored), which avoids a copy; it is never so
    while a graph is being built. ``kept`` is ``x`` at ``index`` when the caller has it already.
    """
    if _dense(index):
        hit = _positions(index, x.shape[1]) if hit is None else hit
        return torch.where(hit, 0, x).sum(1, keepdim=True)
    if owned and not torch.is_grad_enabled():
        kept = x.gather(1, index) if kept is None else kept
        x.scatter_(1, index, 0)
        rest = x.sum(1, keepdim=True)
        x.scatter_(1, index, kept)
        return rest
    return x.scatter(1, index, 0).sum(1, keepdim=True)


def _neg_log(z: Tensor, top: Tensor, p_top: Tensor, q_top: Tensor) -> Tensor:
    """``-log softmax(z)`` as ``log1p(q_top / p_top) + (z_top - z)``, non-negative terms."""
    return torch.log1p(q_top / p_top) + (_gather(z, top) - z)


class _Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: Tensor, complement: bool, nll: bool):  # type: ignore[override]
        z = z.detach()  # eager forward runs without grad, but a tracer (dynamo) sees the input's requires_grad
        p = torch.softmax(z, 1)
        top = p.max(1, keepdim=True).indices
        q_top = _exclusive(p, top, owned=True)
        q = _put(1 - p, top, q_top) if complement else None
        out_nll = _neg_log(z, top, p.amax(1, keepdim=True), q_top) if nll else None
        ctx.save_for_backward(p, top, q_top, *(() if q is None else (q,)))
        ctx.set_materialize_grads(False)
        return p, q, out_nll

    @staticmethod
    def backward(ctx, grad_p, grad_q, grad_nll):  # type: ignore[override]
        p, top, q_top, *saved = ctx.saved_tensors
        if grad_p is None and grad_q is None and grad_nll is None:
            return None, None, None
        graph = torch.is_grad_enabled()  # double backward: differentiate through everything
        if graph:
            q_top = _exclusive(p, top)
        q = saved[0] if saved else _put(1 - p, top, q_top)
        # d p_j / d z_i = p_j (delta_ij - p_i), d q_j = -d p_j and d nll_j / d z_i = p_i - delta_ij, so that
        # dz_i = v_i - p_i sum_j v_j for v = p * grad_p - p * grad_q - grad_nll.
        if graph:
            v = None if grad_p is None else p * grad_p
            if grad_q is not None:
                v = -p * grad_q if v is None else v - p * grad_q
            if grad_nll is not None:
                v = -grad_nll if v is None else v - grad_nll
            total = v.sum(1, keepdim=True)
            # v_i - p_i sum_j v_j = (1 - p_i) v_i - p_i sum_{j != i} v_j: for the class top, with p_top near one,
            # the sum over the other classes is formed directly; elsewhere the total minus v_i is exact enough.
            dz = q * v - p * (total - v)
        else:
            if grad_p is None and grad_q is None:
                v = grad_nll.neg()
            else:
                v = torch.mul(p, grad_p if grad_q is None else grad_p - grad_q if grad_p is not None else -grad_q)
                if grad_nll is not None:
                    v.sub_(grad_nll)
            total = v.sum(1, keepdim=True)
            dz = torch.sub(v, total).mul_(p).addcmul_(q, v)
        hit = _positions(top, p.shape[1])
        v_top = _gather(v, top, hit)
        p_top = _gather(p, top, hit) if graph else p.amax(1, keepdim=True)  # (amax would split a tie's gradient)
        dominant = _gather(q, top, hit) * v_top - p_top * _exclusive(v, top, owned=True, kept=v_top, hit=hit)
        return _put(dz, top, dominant, hit), None, None


# Under torch.compile the public functions below use ordinary differentiable operations instead of the custom
# autograd Functions: dynamo in PyTorch 2.4 cannot trace Functions with non-tensor arguments, None outputs or in-place
# writes. The values keep their accuracy (the same sums of non-negative terms); the gradient is that of autograd,
# which cancels where a probability rounds to one, as before this module existed.


def _plain_complement(p: Tensor) -> Tensor:
    """``1 - p`` along the class axis as the sum of the other probabilities where ``p`` exceeds one half."""
    big = p > 0.5
    return torch.where(big, torch.where(big, 0, p).sum(1, keepdim=True), 1 - p)


def _plain_neg_log(z: Tensor) -> Tensor:
    """``-log softmax(z)`` along the class axis: ``(max - z) + log(sum_j exp(z_j - max))``, the logarithm as
    ``log1p(rest / p_max)`` when one probability exceeds a half (``rest`` the others) and ``-log p_max`` otherwise."""
    p = torch.softmax(z, 1)
    p_max = p.amax(1, keepdim=True)
    rest = torch.where(p > 0.5, 0, p).sum(1, keepdim=True)
    log_norm = torch.where(p_max > 0.5, torch.log1p(rest / p_max), -torch.log(p_max))
    return log_norm + (z.amax(1, keepdim=True) - z)


def _plain_index_overlap(z: Tensor, target: Tensor, valid: Tensor | None, squared: bool, with_nll: bool):
    """:class:`_IndexOverlap` as a composition of ordinary operations (one-hot mask, products, sums)."""
    classes = z.shape[1]
    zf = z.flatten(2)  # (B, C, V), no math.prod or -1 (dynamo 2.4)
    index = target.flatten(1).unsqueeze(1)
    hit = torch.arange(classes, device=z.device).reshape(1, -1, 1) == index
    p = torch.softmax(zf, 1)
    q = _plain_complement(p)
    g = hit.to(z.dtype)
    if valid is not None:
        ok = valid.flatten(1).unsqueeze(1)
        p, q, g = torch.where(ok, p, 0), torch.where(ok, q, 0), torch.where(ok, g, 0)
    inter, mass = (p * g).sum(2), g.sum(2)
    if squared:
        first, third = ((1 - g) * p - g * q).square().sum(2), torch.zeros_like(inter)
    else:
        first, third = (p * (1 - g)).sum(2), (q * g).sum(2)
    nll = torch.where(hit, _plain_neg_log(zf), 0).sum(1).reshape(target.shape) if with_nll else None
    return inter, first, third, mass, nll


def softmax(z: Tensor) -> Tensor:
    """``torch.softmax(z, 1)`` with a derivative that does not lose the dominant class at saturation."""
    if torch.compiler.is_compiling():
        return torch.softmax(z, 1)
    return _Softmax.apply(z, False, False)[0]


def softmax_pair(z: Tensor) -> tuple[Tensor, Tensor]:
    """``(p, q)``: ``softmax(z, 1)`` and ``1 - p``, formed as the sum of the other probabilities."""
    if torch.compiler.is_compiling():
        p = torch.softmax(z, 1)
        return p, _plain_complement(p)
    p, q, _ = _Softmax.apply(z, True, False)
    return p, q


def softmax_stats(z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``(p, q, nll)``: ``softmax(z, 1)``, ``1 - p`` and ``-log p``, each accurate at any finite logits."""
    if torch.compiler.is_compiling():
        p = torch.softmax(z, 1)
        return p, _plain_complement(p), _plain_neg_log(z)
    p, q, nll = _Softmax.apply(z, True, True)
    return p, q, nll


def neg_log_softmax(z: Tensor) -> Tensor:
    """``-log_softmax(z, 1)``: non-negative, relatively accurate for every class, with the stable gradient."""
    if torch.compiler.is_compiling():
        return _plain_neg_log(z)
    return _Softmax.apply(z, False, True)[2]


def _label_nll(zf: Tensor, p: Tensor, index: Tensor, q_target: Tensor, hit: Tensor | None = None) -> Tensor:
    """``-log p_t`` of the label ``t`` of every voxel, ``(B, 1, V)``, from ``q_t = 1 - p_t`` formed as the sum of
    the other probabilities: ``-log1p(-q_t)`` for a label of largest logit, otherwise ``(max - z_t) - log p_max``
    (``p_max = 1 / sum_j exp(z_j - max)``), a sum of non-negative terms."""
    z_target = _gather(zf, index, hit)
    top = zf.amax(1, keepdim=True)
    return torch.where(z_target >= top, -torch.log1p(-q_target), (top - z_target) - torch.log(p.amax(1, keepdim=True)))


def _check_labels(target: Tensor, classes: int) -> None:
    """The errors ``F.one_hot`` raises for labels outside ``[0, classes)``, which the index-based sums below
    would otherwise miscount (or, on CUDA, hit a device-side assert). One host synchronisation."""
    if target.numel() and not torch.compiler.is_compiling():
        low, high = torch.aminmax(target)
        if bool((low < 0) | (high >= classes)):
            raise RuntimeError("Class values must be non-negative." if int(low) < 0
                               else "Class values must be smaller than num_classes.")


def _per_class_counts(index: Tensor, invalid: Tensor | None, classes: int, dtype: torch.dtype,
                      hit: Tensor | None = None) -> Tensor:
    """Exact number of valid voxels of every class in every case, ``(B, C)``, from int64 counts."""
    if _dense(index):  # atomic adds into a few bins are slow; a mask and a sum are not
        hit = _positions(index, classes) if hit is None else hit
        return (hit if invalid is None else hit & ~invalid).sum(2).to(dtype)
    labels = index[:, 0] if invalid is None else torch.where(invalid[:, 0], classes, index[:, 0])  # extra bin: excluded
    counts = torch.zeros(index.shape[0], classes + 1, dtype=torch.long, device=index.device)
    return counts.scatter_add_(1, labels, torch.ones_like(labels))[:, :classes].to(dtype)


class _IndexOverlap(torch.autograd.Function):
    """Per-case, per-class ``(TP, FP, FN, mass)`` of the softmax overlap losses for integer class labels.

    With ``q_t = sum_{j != t} p_j`` the complement of the probability of the label ``t`` of a voxel (the sum of
    the other probabilities), ``TP_c = sum_{t = c} p_c``, ``FP_c = sum_{t != c} p_c``, ``FN_c = sum_{t = c} q_t``
    and the mass ``sum_{t = c} 1``: the sums of ``p * g``, ``p * (1 - g)`` and ``(1 - p) * g`` for the one-hot
    ``g`` of the label, each term a non-negative number. They come from one buffer: ``p`` with the label entries
    zeroed (``FP`` and ``q_t``), then ``p`` at the label entries only (``p - buffer``, exact elementwise), then
    ``q_t`` at the label entries, so that no one-hot tensor and no ``(1 - g)`` or ``q`` tensor is formed. With
    ``squared`` the second slot holds ``sum (p - g)**2 = sum_{t != c} p_c**2 + sum_{t = c} q_t**2`` and the third
    is zero. The backward pass differentiates through the simplex, ``d_c = d FP_c`` for ``c != t`` and
    ``d_t = d TP_t - d FN_t`` as the effective gradient with respect to ``p`` (``2 p_c d M_c`` and
    ``d TP_t - 2 q_t d M_t`` when squared), then applies the softmax product ``dz_i = (1 - p_i) v_i - p_i
    sum_{j != i} v_j`` with ``v = p d``; the class of largest probability takes the sum over the others
    directly. While a graph is being built (double backward) everything is recomputed from the logits with
    differentiable operations.
    """

    @staticmethod
    def forward(ctx, z: Tensor, target: Tensor, valid: Tensor | None, squared: bool, with_nll: bool):  # type: ignore[override]
        saved_z, z = z, z.detach()  # the in-place writes below must not see a tensor that requires grad (tracers)
        batch, classes = z.shape[:2]
        voxels = math.prod(z.shape[2:])
        _check_labels(target, classes)
        zf = z.reshape(batch, classes, voxels)
        index = target.reshape(batch, 1, voxels)
        invalid = None if valid is None else ~valid.reshape(batch, 1, voxels)
        p = torch.softmax(zf, 1)
        work = p.scatter(1, index, 0)
        q_target = work.sum(1, keepdim=True)
        hit = _positions(index, classes)
        nll = _label_nll(zf, p, index, q_target, hit).reshape(target.shape) if with_nll else None
        if invalid is not None:
            work.masked_fill_(invalid, 0)
            q_target.masked_fill_(invalid, 0)
        first = work.square().sum(2) if squared else work.sum(2)
        work.neg_().add_(p)  # p - work, exact elementwise: p at the label entries, 0 elsewhere
        if invalid is not None:
            work.masked_fill_(invalid, 0)
        inter = work.sum(2)
        work.scatter_(1, index, q_target)
        third = work.square().sum(2) if squared else work.sum(2)
        if squared:
            first, third = first + third, torch.zeros_like(third)
        mass = _per_class_counts(index, invalid, classes, z.dtype, hit)
        ctx.save_for_backward(saved_z, target, p, q_target, *(() if valid is None else (valid,)))
        ctx.squared = squared
        ctx.mark_non_differentiable(mass)
        ctx.set_materialize_grads(False)
        return inter, first, third, mass, nll

    @staticmethod
    def backward(ctx, grad_inter, grad_first, grad_third, grad_mass, grad_nll):  # type: ignore[override]
        z, target, p, q_target, *rest = ctx.saved_tensors
        squared = ctx.squared
        batch, classes = z.shape[:2]
        voxels = math.prod(z.shape[2:])
        graph = torch.is_grad_enabled()  # double backward: differentiate through everything
        index = target.reshape(batch, 1, voxels)
        invalid = ~rest[0].reshape(batch, 1, voxels) if rest else None
        zero = z.new_zeros(batch, classes)
        grad_inter = zero if grad_inter is None else grad_inter
        grad_first = zero if grad_first is None else grad_first
        grad_third = zero if grad_third is None else grad_third
        hit = _positions(index, classes)
        if graph:
            p = softmax(z.reshape(batch, classes, voxels))
            q_target = _exclusive(p, index, hit=hit)
        # Effective gradient d with respect to p, per voxel: a class constant for the other classes, and
        # d_t for the label t; v = p d.
        if squared:
            v = p.square() * (2 * grad_first)[:, :, None]
            label = _lookup(grad_inter, index, hit) - 2 * _lookup(grad_first, index, hit) * q_target
        else:
            v = p * grad_first[:, :, None]
            label = _lookup(grad_inter - grad_third, index, hit)
        label = _gather(p, index, hit) * label
        if invalid is not None:  # excluded voxels contribute nothing to the sums (but may to the -log p_t)
            v = v.masked_fill(invalid, 0) if graph else v.masked_fill_(invalid, 0)
            label = label.masked_fill(invalid, 0)
        if grad_nll is not None:  # -log p_t of the label: v_t - grad
            label = label - grad_nll.reshape(batch, 1, voxels)
        v = _put(v, index, label, hit)
        total = v.sum(1, keepdim=True)
        p_top, top = p.max(1, keepdim=True)
        # p is a saved tensor, so its sum over the other classes is formed on a copy; v is ours.
        top_hit = _positions(top, classes)
        q_top = _exclusive(p, top, hit=top_hit)
        v_top = _gather(v, top, top_hit)
        dominant = q_top * v_top - p_top * _exclusive(v, top, owned=True, kept=v_top, hit=top_hit)
        dz = v - p * total if graph else v.addcmul_(p, total, value=-1)
        return _put(dz, top, dominant, top_hit).reshape(z.shape), None, None, None, None


def index_overlap_statistics(
    z: Tensor, target: Tensor, valid: Tensor | None = None, *, squared: bool = False, with_nll: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    """``(TP, FP, FN, mass, nll)`` of the softmax of ``z`` against integer labels ``target`` (``(B, *spatial)``):
    the first four of shape ``(B, C)``, summed over the voxels where ``valid`` (or all of them), and with
    ``with_nll`` the per-voxel ``-log p_t`` (``target.shape``), which shares the softmax, the sum of the other
    probabilities and the backward pass; see :class:`_IndexOverlap`. The gradient flows to ``z`` only, with no
    one-hot tensor and no cancellation at saturated logits."""
    if torch.compiler.is_compiling():
        return _plain_index_overlap(z, target.long(), valid, squared, with_nll)
    return _IndexOverlap.apply(z, target.long(), valid, squared, with_nll)


class _CrossEntropyIndex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: Tensor, target: Tensor) -> Tensor:  # type: ignore[override]
        saved_z, z = z, z.detach()
        batch, classes = z.shape[:2]
        voxels = math.prod(z.shape[2:])
        _check_labels(target, classes)
        zf = z.reshape(batch, classes, voxels)
        index = target.reshape(batch, 1, voxels)
        p = torch.softmax(zf, 1)
        hit = _positions(index, classes)
        q_target = _exclusive(p, index, owned=True, hit=hit)  # the other probabilities: 1 - p_t, without cancellation
        nll = _label_nll(zf, p, index, q_target, hit)
        ctx.save_for_backward(saved_z, target, p, q_target)
        return nll.reshape(target.shape)

    @staticmethod
    def backward(ctx, grad: Tensor):  # type: ignore[override]
        z, target, p, q_target = ctx.saved_tensors
        batch, classes = z.shape[:2]
        voxels = math.prod(z.shape[2:])
        index = target.reshape(batch, 1, voxels)
        if torch.is_grad_enabled():  # double backward: differentiate through the softmax
            p = softmax(z.reshape(batch, classes, voxels))
            q_target = _exclusive(p, index)
        weight = grad.reshape(batch, 1, voxels)
        # dz_i = grad (p_i - delta_it), the entry of the label being -grad (1 - p_t) = -grad q_t.
        return _put(p * weight, index, -weight * q_target).reshape(z.shape), None


def cross_entropy_index(z: Tensor, target: Tensor) -> Tensor:
    """Per-voxel ``-log softmax(z, 1)[target]`` for integer class indices, with the stable gradient
    ``p - onehot`` (shape ``target.shape``). ``torch.nn.functional.cross_entropy(reduction="none")`` returns 0
    for the dominant class at saturation and a zero gradient."""
    if torch.compiler.is_compiling():
        index = target.long().flatten(1).unsqueeze(1)
        return _plain_neg_log(z.flatten(2)).gather(1, index).reshape(target.shape)
    return _CrossEntropyIndex.apply(z, target.long())
