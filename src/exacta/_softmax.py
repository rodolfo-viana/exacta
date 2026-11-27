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
"""

from __future__ import annotations

import torch
from torch import Tensor


def _put(x: Tensor, top: Tensor, value: Tensor) -> Tensor:
    """``x`` with ``x[top] = value`` along the class axis: in place unless a graph is being built."""
    return x.scatter(1, top, value) if torch.is_grad_enabled() else x.scatter_(1, top, value)


def _exclusive(x: Tensor, top: Tensor, *, owned: bool = False) -> Tensor:
    """Sum over the classes other than ``top``, formed without subtracting from the total.

    ``owned``: ``x`` may be modified in place (and is restored), which avoids a copy; it is never so while
    a graph is being built.
    """
    if owned and not torch.is_grad_enabled():
        kept = x.gather(1, top)
        x.scatter_(1, top, 0)
        rest = x.sum(1, keepdim=True)
        x.scatter_(1, top, kept)
        return rest
    return x.scatter(1, top, 0).sum(1, keepdim=True)


def _neg_log(z: Tensor, top: Tensor, p_top: Tensor, q_top: Tensor) -> Tensor:
    """``-log softmax(z)`` as ``log1p(q_top / p_top) + (z_top - z)``, non-negative terms."""
    return torch.log1p(q_top / p_top) + (z.gather(1, top) - z)


class _Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: Tensor, complement: bool, nll: bool):  # type: ignore[override]
        p = torch.softmax(z, 1)
        top = p.max(1, keepdim=True).indices
        q_top = _exclusive(p, top, owned=True)
        q = _put(1 - p, top, q_top) if complement else None
        out_nll = _neg_log(z, top, p.gather(1, top), q_top) if nll else None
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
        dominant = q.gather(1, top) * v.gather(1, top) - p.gather(1, top) * _exclusive(v, top, owned=True)
        return _put(dz, top, dominant), None, None


def softmax(z: Tensor) -> Tensor:
    """``torch.softmax(z, 1)`` with a derivative that does not lose the dominant class at saturation."""
    return _Softmax.apply(z, False, False)[0]


def softmax_pair(z: Tensor) -> tuple[Tensor, Tensor]:
    """``(p, q)``: ``softmax(z, 1)`` and ``1 - p``, formed as the sum of the other probabilities."""
    p, q, _ = _Softmax.apply(z, True, False)
    return p, q


def softmax_stats(z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``(p, q, nll)``: ``softmax(z, 1)``, ``1 - p`` and ``-log p``, each accurate at any finite logits."""
    p, q, nll = _Softmax.apply(z, True, True)
    return p, q, nll


def neg_log_softmax(z: Tensor) -> Tensor:
    """``-log_softmax(z, 1)``: non-negative, relatively accurate for every class, with the stable gradient."""
    return _Softmax.apply(z, False, True)[2]


class _CrossEntropyIndex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: Tensor, target: Tensor) -> Tensor:  # type: ignore[override]
        p = torch.softmax(z, 1)
        top = p.max(1, keepdim=True).indices
        q_top = _exclusive(p, top, owned=True)
        ctx.save_for_backward(z, p, top, q_top, target)
        nll = torch.log1p(q_top / p.gather(1, top)) + (z.gather(1, top) - z.gather(1, target.unsqueeze(1)))
        return nll.squeeze(1)

    @staticmethod
    def backward(ctx, grad: Tensor):  # type: ignore[override]
        z, p, top, q_top, target = ctx.saved_tensors
        if torch.is_grad_enabled():  # double backward: differentiate through the softmax
            p = softmax(z)
            q_top = _exclusive(p, top)
        index = target.unsqueeze(1)
        # dz_i = grad (p_i - delta_it), with the entry of the target -grad (1 - p_t), formed as the sum of the
        # other probabilities where t is the class of largest probability.
        q_target = torch.where(index == top, q_top, 1 - p.gather(1, index))
        weight = grad.unsqueeze(1)
        return _put(weight * p, index, -weight * q_target), None


def cross_entropy_index(z: Tensor, target: Tensor) -> Tensor:
    """Per-voxel ``-log softmax(z, 1)[target]`` for integer class indices, with the stable gradient
    ``p - onehot`` (shape ``target.shape``). ``torch.nn.functional.cross_entropy(reduction="none")`` returns 0
    for the dominant class at saturation and a zero gradient."""
    return _CrossEntropyIndex.apply(z, target.long())
