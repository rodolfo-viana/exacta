"""Class pooling, averaging over present classes and smoothing placement in Dice and Tversky losses.

The three arguments (``average``, ``skip_absent``, ``smooth_numerator``) reproduce Kornia's
``dice_loss`` and ``TverskyLoss`` (0.9.2). The references below are plain per-case, per-class
transcriptions of Kornia's documented definitions, in binary64, so the compositions are checked
without Kornia installed; the same compositions are also compared with the installed Kornia.
"""

import pytest
import torch
import torch.nn.functional as F

from exacta import _triton_losses
from exacta.losses import DiceCELoss, DiceLoss, FocalTverskyLoss, GeneralizedDiceLoss, TverskyLoss

GPU = torch.cuda.is_available() and _triton_losses.available()
needs_gpu = pytest.mark.skipif(not GPU, reason="needs CUDA and Triton")

IGNORE = -100


def _inputs(shape=(4, 5, 12, 14), seed=0, absent=True, dtype=torch.float64):
    """Random logits and labels; with ``absent``, case 0 lacks class 3 and case 1 is all background."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(shape, generator=g, dtype=dtype)
    target = torch.randint(0, shape[1], (shape[0], *shape[2:]), generator=g)
    if absent:
        target[0][target[0] == 3] = 1
        target[1] = 0
    return logits, target


def _statistics(probs, target, ignore_index):
    """Per case and class: TP, FP, FN and the target mass, over the pixels that are not ignored."""
    classes = probs.shape[0]
    keep = torch.ones_like(target, dtype=torch.bool) if ignore_index is None else target != ignore_index
    onehot = torch.stack([(target == c) & keep for c in range(classes)]).to(probs.dtype)
    p = probs * keep
    return (p * onehot).sum((1, 2)), (p * (1 - onehot)).sum((1, 2)), ((1 - p) * onehot).sum((1, 2)), onehot.sum((1, 2))


def _kornia_dice(logits, target, average, eps, ignore_index=None):
    """Kornia's dice_loss without class weights: 2 I / (P + G + eps), eps in the denominator only."""
    probs = logits.softmax(1)
    losses = []
    for b in range(logits.shape[0]):
        tp, fp, fn, truth = _statistics(probs[b], target[b], ignore_index)
        pred = tp + fp
        if average == "micro":
            ratio = 2 * tp.sum() / (pred.sum() + truth.sum() + eps)
            losses.append(1 - ratio)
        else:
            present = truth > 0
            if not present.any():
                losses.append(logits.new_tensor(1.0))
                continue
            ratio = 2 * tp[present] / (pred[present] + truth[present] + eps)
            losses.append((1 - ratio).sum() / present.sum())
    return torch.stack(losses).mean()


def _kornia_tversky(logits, target, alpha, beta, eps, ignore_index=None):
    """Kornia's tversky_loss: TP / (TP + alpha FP + beta FN + eps) averaged over the present classes."""
    probs = logits.softmax(1)
    scores = []
    for b in range(logits.shape[0]):
        tp, fp, fn, truth = _statistics(probs[b], target[b], ignore_index)
        present = truth > 0
        if not present.any():
            scores.append(logits.new_tensor(0.0))
            continue
        scores.append((tp[present] / (tp[present] + alpha * fp[present] + beta * fn[present] + eps)).mean())
    return 1 - torch.stack(scores).mean()


def _value_and_grad(fn, logits, target):
    x = logits.clone().requires_grad_(True)
    value = fn(x, target)
    value.backward()
    return value.detach(), x.grad


def _assert_same(ours, theirs):
    """Value and gradient to a few ulps of binary64."""
    (value, grad), (expected, expected_grad) = ours, theirs
    torch.testing.assert_close(value, expected, rtol=1e-13, atol=1e-15)
    torch.testing.assert_close(grad, expected_grad, rtol=0, atol=1e-13 * expected_grad.abs().max().item())


KORNIA_DICE = {
    "micro": dict(average="micro"),
    "macro": dict(skip_absent=True),
}


@pytest.mark.parametrize("eps", [0.0, 1e-8, 1e-3])
@pytest.mark.parametrize("average", ["micro", "macro"])
@pytest.mark.parametrize("shape", [(4, 5, 12, 14), (3, 2, 7, 5)])
@pytest.mark.parametrize("absent", [False, True])
def test_dice_composition_matches_kornia_arithmetic(average, eps, shape, absent):
    logits, target = _inputs(shape, seed=3, absent=absent and shape[1] > 3)
    ours = _value_and_grad(
        lambda x, t: DiceLoss(smooth=eps, smooth_numerator=False, **KORNIA_DICE[average])(x, t), logits, target)
    reference = _value_and_grad(lambda x, t: _kornia_dice(x, t, average, eps), logits, target)
    _assert_same(ours, reference)


@pytest.mark.parametrize("eps", [0.0, 1e-8, 1e-3])
@pytest.mark.parametrize("alpha, beta", [(0.5, 0.5), (0.3, 0.7), (1.0, 1.0), (0.0, 2.0)])
@pytest.mark.parametrize("absent", [False, True])
def test_tversky_composition_matches_kornia_arithmetic(alpha, beta, eps, absent):
    logits, target = _inputs(seed=4, absent=absent)
    ours = _value_and_grad(
        lambda x, t: TverskyLoss(alpha=alpha, beta=beta, smooth=eps, smooth_numerator=False,
                                 skip_absent=True)(x, t), logits, target)
    reference = _value_and_grad(lambda x, t: _kornia_tversky(x, t, alpha, beta, eps), logits, target)
    _assert_same(ours, reference)


@pytest.mark.parametrize("average", ["micro", "macro"])
def test_dice_composition_ignores_the_same_pixels_as_kornia(average):
    logits, target = _inputs(seed=5)
    target[:, :3] = IGNORE  # a band of every case; no case is entirely ignored
    target[2, 5:] = IGNORE  # a third of case 2 remains
    eps = 1e-8
    ours = _value_and_grad(
        lambda x, t: DiceLoss(smooth=eps, smooth_numerator=False, ignore_index=IGNORE, **KORNIA_DICE[average])(x, t),
        logits, target)
    reference = _value_and_grad(lambda x, t: _kornia_dice(x, t, average, eps, IGNORE), logits, target)
    _assert_same(ours, reference)


def test_tversky_composition_ignores_the_same_pixels_as_kornia():
    logits, target = _inputs(seed=6)
    target[:, :3] = IGNORE
    target[2, 5:] = IGNORE
    ours = _value_and_grad(
        lambda x, t: TverskyLoss(alpha=0.3, beta=0.7, smooth=1e-8, smooth_numerator=False, skip_absent=True,
                                 ignore_index=IGNORE)(x, t), logits, target)
    reference = _value_and_grad(lambda x, t: _kornia_tversky(x, t, 0.3, 0.7, 1e-8, IGNORE), logits, target)
    _assert_same(ours, reference)


def _kornia_losses():
    return pytest.importorskip("kornia.losses")


@pytest.mark.parametrize("shape", [(4, 5, 12, 14), (2, 5, 64, 64)])  # the second takes Kornia's CPU path
@pytest.mark.parametrize("eps", [0.0, 1e-8])
@pytest.mark.parametrize("average", ["micro", "macro"])
def test_dice_composition_matches_installed_kornia(average, eps, shape):
    kornia = _kornia_losses()
    logits, target = _inputs(shape, seed=7)
    ours = _value_and_grad(
        lambda x, t: DiceLoss(smooth=eps, smooth_numerator=False, **KORNIA_DICE[average])(x, t), logits, target)
    theirs = _value_and_grad(
        lambda x, t: kornia.dice_loss(x, t, average=average, eps=eps, ignore_index=None), logits, target)
    _assert_same(ours, theirs)
    module = _value_and_grad(
        lambda x, t: kornia.DiceLoss(average=average, eps=eps, ignore_index=None)(x, t), logits, target)
    _assert_same(ours, module)


@pytest.mark.parametrize("eps", [0.0, 1e-8])
@pytest.mark.parametrize("alpha, beta", [(0.5, 0.5), (0.3, 0.7)])
def test_tversky_composition_matches_installed_kornia(alpha, beta, eps):
    kornia = _kornia_losses()
    logits, target = _inputs(seed=8)
    ours = _value_and_grad(
        lambda x, t: TverskyLoss(alpha=alpha, beta=beta, smooth=eps, smooth_numerator=False,
                                 skip_absent=True)(x, t), logits, target)
    theirs = _value_and_grad(
        lambda x, t: kornia.TverskyLoss(alpha, beta, eps, ignore_index=None)(x, t), logits, target)
    _assert_same(ours, theirs)


def test_composition_matches_installed_kornia_with_ignore_index():
    kornia = _kornia_losses()
    logits, target = _inputs(seed=9)
    target[:, :3] = IGNORE
    for average in ("micro", "macro"):
        ours = _value_and_grad(
            lambda x, t: DiceLoss(smooth=1e-8, smooth_numerator=False, ignore_index=IGNORE,
                                  **KORNIA_DICE[average])(x, t), logits, target)
        theirs = _value_and_grad(
            lambda x, t: kornia.dice_loss(x, t, average=average, eps=1e-8, ignore_index=IGNORE), logits, target)
        _assert_same(ours, theirs)
    ours = _value_and_grad(
        lambda x, t: TverskyLoss(alpha=0.3, beta=0.7, smooth=1e-8, smooth_numerator=False, skip_absent=True,
                                 ignore_index=IGNORE)(x, t), logits, target)
    theirs = _value_and_grad(
        lambda x, t: kornia.tversky_loss(x, t, 0.3, 0.7, eps=1e-8, ignore_index=IGNORE), logits, target)
    _assert_same(ours, theirs)


def test_smoothing_the_numerator_too_is_a_different_loss_from_kornias():
    """With eps > 0, smoothing the numerator too is a different loss; smooth_numerator=False removes it."""
    logits, target = _inputs(seed=10)
    eps = 1e-3
    reference = _kornia_dice(logits, target, "macro", eps)
    both = DiceLoss(smooth=eps, skip_absent=True)(logits, target)
    denominator = DiceLoss(smooth=eps, skip_absent=True, smooth_numerator=False)(logits, target)
    assert abs(both - reference) > 1e-6
    torch.testing.assert_close(denominator, reference, rtol=1e-13, atol=1e-15)


# --- defaults --------------------------------------------------------------------------------------


@pytest.mark.parametrize("loss_cls", [DiceLoss, TverskyLoss])
def test_defaults_are_the_previous_conventions(loss_cls):
    loss = loss_cls()
    assert (loss.average, loss.skip_absent, loss.smooth_numerator) == ("macro", False, True)
    logits, target = _inputs(seed=11)
    explicit = loss_cls(average="macro", skip_absent=False, smooth_numerator=True)
    assert torch.equal(loss(logits, target), explicit(logits, target))
    probs = logits.softmax(1)
    onehot = F.one_hot(target, 5).movedim(-1, 1).double()
    dims = (2, 3)
    tp = (probs * onehot).sum(dims)
    if loss_cls is DiceLoss:
        ratio = (2 * tp + 1e-5) / (probs.sum(dims) + onehot.sum(dims) + 1e-5)
    else:
        fp, fn = (probs * (1 - onehot)).sum(dims), ((1 - probs) * onehot).sum(dims)
        ratio = (tp + 1e-5) / (tp + 0.5 * fp + 0.5 * fn + 1e-5)
    torch.testing.assert_close(loss(logits, target), 1 - ratio.mean(), rtol=1e-13, atol=0)


def test_tversky_with_half_weights_is_dice_with_twice_the_smoothing_in_either_placement():
    logits, target = _inputs(seed=12)
    for kwargs in (dict(), dict(smooth_numerator=False), dict(skip_absent=True), dict(average="micro")):
        tversky = TverskyLoss(smooth=1e-3, **kwargs)(logits, target)
        dice = DiceLoss(smooth=2e-3, **kwargs)(logits, target)
        torch.testing.assert_close(tversky, dice, rtol=1e-13, atol=1e-15)


# --- average="micro" ---------------------------------------------------------------------------------


@pytest.mark.parametrize("batch_dice", [False, True])
@pytest.mark.parametrize("include_background", [True, False])
@pytest.mark.parametrize("smooth_numerator", [True, False])
def test_micro_dice_pools_classes_into_one_ratio(batch_dice, include_background, smooth_numerator):
    logits, target = _inputs(seed=13)
    smooth = 1e-3
    probs = logits.softmax(1)
    onehot = F.one_hot(target, 5).movedim(-1, 1).double()
    if not include_background:
        probs, onehot = probs[:, 1:], onehot[:, 1:]
    dims = (0, 1, 2, 3) if batch_dice else (1, 2, 3)
    numerator = 2 * (probs * onehot).sum(dims) + (smooth if smooth_numerator else 0)
    expected = 1 - (numerator / (probs.sum(dims) + onehot.sum(dims) + smooth)).mean()
    loss = DiceLoss(average="micro", batch_dice=batch_dice, include_background=include_background, smooth=smooth,
                    smooth_numerator=smooth_numerator)
    torch.testing.assert_close(loss(logits, target), expected, rtol=1e-13, atol=1e-15)


def test_micro_tversky_pools_tp_fp_and_fn_over_classes():
    logits, target = _inputs(seed=14)
    probs = logits.softmax(1)
    onehot = F.one_hot(target, 5).movedim(-1, 1).double()
    dims = (1, 2, 3)
    tp = (probs * onehot).sum(dims)
    fp, fn = (probs * (1 - onehot)).sum(dims), ((1 - probs) * onehot).sum(dims)
    expected = 1 - ((tp + 1e-4) / (tp + 0.3 * fp + 0.7 * fn + 1e-4)).mean()
    loss = TverskyLoss(alpha=0.3, beta=0.7, average="micro", smooth=1e-4)
    torch.testing.assert_close(loss(logits, target), expected, rtol=1e-13, atol=1e-15)


def test_micro_with_one_class_is_macro():
    logits, target = _inputs(seed=15, absent=False)
    logits, target = logits[:, :2], target.clamp(max=1)
    for cls, kwargs in ((DiceLoss, {}), (TverskyLoss, dict(alpha=0.3, beta=0.7))):
        macro = cls(include_background=False, **kwargs)(logits, target)
        micro = cls(include_background=False, average="micro", **kwargs)(logits, target)
        torch.testing.assert_close(micro, macro, rtol=1e-13, atol=0)


def test_micro_composes_with_zero_absent_on_the_pooled_target():
    logits, target = _inputs(seed=16, absent=False)
    target[1] = 0  # case 1 has no foreground
    loss = DiceLoss(average="micro", include_background=False, zero_absent=True, smooth=1e-3)
    x = logits.clone().requires_grad_(True)
    per_case = loss(x[1:2], target[1:2])
    per_case.backward()
    assert per_case.item() == 0.0 and not x.grad.any()  # pooled foreground target empty: zero loss


@pytest.mark.parametrize("cls", [DiceLoss, TverskyLoss])
def test_micro_pooled_gradient_matches_finite_differences(cls):
    logits, target = _inputs((2, 3, 4, 5), seed=17, absent=False)
    loss = cls(average="micro", smooth=1e-3, smooth_numerator=False)
    x = logits.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda z: loss(z, target), (x,))


# --- skip_absent ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [DiceLoss, TverskyLoss])
def test_skip_absent_averages_over_present_classes_and_ignores_absent_predictions(cls):
    probs = torch.rand(2, 4, 6, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    probs = probs / probs.sum(1, keepdim=True)
    target = torch.randint(0, 2, (2, 6, 6), generator=torch.Generator().manual_seed(2))  # classes 2, 3 absent
    kwargs = dict(activation="none", smooth=1e-3)
    skipped = cls(skip_absent=True, **kwargs)
    x = probs.clone().requires_grad_(True)
    loss = skipped(x, target)
    loss.backward()
    assert not x.grad[:, 2:].any()  # the absent classes have no gradient at all
    # The mean over the present classes is the loss computed on those channels alone.
    present_only = cls(**kwargs)(probs[:, :2], target)
    torch.testing.assert_close(loss, present_only, rtol=1e-13, atol=0)
    assert cls(**kwargs)(probs, target) > loss  # the default also penalizes the absent classes
    zero = cls(zero_absent=True, **kwargs)(probs, target)
    torch.testing.assert_close(zero, present_only / 2, rtol=1e-13, atol=0)  # still two of four in the mean


@pytest.mark.parametrize("cls", [DiceLoss, TverskyLoss])
def test_skip_absent_is_finite_without_smoothing(cls):
    logits, target = _inputs(seed=18)
    value, grad = _value_and_grad(lambda x, t: cls(skip_absent=True, smooth=0.0)(x, t), logits, target)
    assert torch.isfinite(value) and torch.isfinite(grad).all()


@pytest.mark.parametrize("cls", [DiceLoss, TverskyLoss])
def test_skip_absent_with_batch_dice_takes_presence_over_the_batch(cls):
    logits, target = _inputs(seed=19, absent=False)
    probs = logits.softmax(1)  # activation="none" below, so that dropping a channel does not change the others
    target[target == 4] = 0  # class 4 absent from the batch, class 3 present in every case
    kwargs = dict(activation="none", batch_dice=True, smooth=1e-3)
    loss = cls(skip_absent=True, **kwargs)(probs, target)
    torch.testing.assert_close(loss, cls(**kwargs)(probs[:, :4], target), rtol=1e-13, atol=0)
    assert not torch.equal(loss, cls(**kwargs)(probs, target))


@pytest.mark.parametrize("cls", [DiceLoss, TverskyLoss])
def test_all_background_case_is_scored_on_the_background_alone(cls):
    logits, target = _inputs(seed=20)
    target[:] = 0
    loss = cls(skip_absent=True, smooth=1e-3, smooth_numerator=False)
    probs = logits.softmax(1)
    background = probs[:, 0].sum((1, 2))
    pixels = probs.shape[2] * probs.shape[3]
    if cls is DiceLoss:
        ratio = 2 * background / (background + pixels + 1e-3)
    else:
        ratio = background / (background + 0.5 * (pixels - background) + 1e-3)  # FP = 0, FN = pixels - TP
    torch.testing.assert_close(loss(logits, target), 1 - ratio.mean(), rtol=1e-13, atol=0)


@pytest.mark.parametrize("cls", [DiceLoss, TverskyLoss])
def test_case_without_present_class_has_loss_one_and_zero_gradient(cls):
    logits, target = (x[:2] for x in _inputs(seed=21, absent=False))
    target[1] = 0  # background only, and the background is excluded
    loss = cls(skip_absent=True, include_background=False, smooth=1e-3)
    x = logits.clone().requires_grad_(True)
    both = loss(x, target)
    both.backward()
    alone = loss(logits[0:1], target[0:1])
    torch.testing.assert_close(both, (alone + 1) / 2, rtol=1e-13, atol=0)
    assert not x.grad[1].any()  # the empty case adds a constant, no gradient
    only = x.detach().clone().requires_grad_(True)
    constant = loss(only[1:2], target[1:2])
    constant.backward()
    assert constant.item() == 1.0 and not only.grad.any()


@pytest.mark.parametrize("cls", [DiceLoss, TverskyLoss])
def test_entirely_ignored_case_is_omitted_not_scored_one(cls):
    """Unlike Kornia, which keeps it in the mean with loss one: Exacta omits entirely ignored cases."""
    logits, target = _inputs(seed=22, absent=False)
    target[1] = IGNORE
    loss = cls(skip_absent=True, smooth=1e-3, ignore_index=IGNORE)
    x = logits.clone().requires_grad_(True)
    value = loss(x, target)
    value.backward()
    keep = [0, 2, 3]
    torch.testing.assert_close(value, loss(logits[keep], target[keep]), rtol=1e-13, atol=0)
    assert not x.grad[1].any()
    everything = loss(logits, torch.full_like(target, IGNORE))
    assert everything.item() == 0.0  # a fully ignored batch is zero, as in every overlap loss


def test_skip_absent_with_soft_targets_counts_positive_mass_as_present():
    generator = torch.Generator().manual_seed(23)
    probs = torch.rand(2, 3, 5, 5, dtype=torch.float64, generator=generator)
    probs = probs / probs.sum(1, keepdim=True)
    soft = torch.zeros_like(probs)
    soft[:, 0] = 0.9
    soft[0, 2, 0, 0] = 1e-3  # a trace of class 2 in case 0; class 1 has no mass anywhere
    kwargs = dict(activation="none", target_format="one-hot", smooth=1e-3)
    loss = DiceLoss(skip_absent=True, **kwargs)
    x = probs.clone().requires_grad_(True)
    value = loss(x, soft)
    value.backward()
    assert not x.grad[:, 1].any() and x.grad[0, 2].any() and not x.grad[1, 2].any()
    # Case 0 averages classes 0 and 2, case 1 class 0 alone.
    case0 = DiceLoss(**kwargs)(probs[0:1, [0, 2]], soft[0:1, [0, 2]])
    case1 = DiceLoss(**kwargs)(probs[1:2, [0]], soft[1:2, [0]])
    torch.testing.assert_close(value, (case0 + case1) / 2, rtol=1e-13, atol=0)


# --- smooth_numerator ----------------------------------------------------------------------------------


def test_smooth_numerator_false_puts_smoothing_in_the_denominator_only():
    probs = torch.tensor([[[[0.7, 0.2], [0.1, 0.4]]]], dtype=torch.float64)
    probs = torch.cat([probs, 1 - probs], 1)  # (1, 2, 2, 2): two classes
    target = torch.tensor([[[1, 0], [0, 1]]])
    p0, p1 = probs[0, 0], probs[0, 1]
    g0, g1 = (target[0] == 0).double(), (target[0] == 1).double()
    smooth = 0.25
    dice = [(2 * (p * g).sum()) / (p.sum() + g.sum() + smooth) for p, g in ((p0, g0), (p1, g1))]
    torch.testing.assert_close(
        DiceLoss(activation="none", smooth=smooth, smooth_numerator=False)(probs, target),
        1 - torch.stack(dice).mean(), rtol=1e-14, atol=0)
    tversky = []
    for p, g in ((p0, g0), (p1, g1)):
        tp, fp, fn = (p * g).sum(), (p * (1 - g)).sum(), ((1 - p) * g).sum()
        tversky.append(tp / (tp + 0.4 * fp + 0.6 * fn + smooth))
    torch.testing.assert_close(
        TverskyLoss(alpha=0.4, beta=0.6, activation="none", smooth=smooth, smooth_numerator=False)(probs, target),
        1 - torch.stack(tversky).mean(), rtol=1e-14, atol=0)


def test_smooth_numerator_false_without_smoothing_is_the_unsmoothed_loss():
    logits, target = _inputs(seed=24, absent=False)
    for cls in (DiceLoss, TverskyLoss):
        both = cls(smooth=0.0)(logits, target)
        denominator = cls(smooth=0.0, smooth_numerator=False)(logits, target)
        assert torch.equal(both, denominator)


# --- validation and backends -----------------------------------------------------------------------------


def test_arguments_are_validated():
    for cls in (DiceLoss, TverskyLoss):
        with pytest.raises(ValueError, match="average"):
            cls(average="weighted")
        with pytest.raises(TypeError):
            cls(skip_absent=1)
        with pytest.raises(TypeError):
            cls(smooth_numerator=0)
        with pytest.raises(ValueError, match="zero_absent"):
            cls(skip_absent=True, zero_absent=True)
        with pytest.raises(ValueError, match="micro"):
            cls(skip_absent=True, average="micro")
        for kwargs in (dict(average="micro"), dict(skip_absent=True), dict(smooth_numerator=False)):
            with pytest.raises(NotImplementedError):
                cls(backend="triton", **kwargs)


def test_other_overlap_losses_keep_their_signatures():
    for cls in (GeneralizedDiceLoss, FocalTverskyLoss, DiceCELoss):
        with pytest.raises(TypeError):
            cls(average="micro")
        with pytest.raises(TypeError):
            cls(skip_absent=True)
    assert (GeneralizedDiceLoss().average, FocalTverskyLoss().skip_absent) == ("macro", False)


@pytest.mark.parametrize("cls", [DiceLoss, TverskyLoss])
@pytest.mark.parametrize("kwargs", [dict(average="micro"), dict(skip_absent=True, smooth_numerator=False)])
@pytest.mark.parametrize("masked", [False, True])
def test_new_conventions_compile_without_graph_breaks(cls, kwargs, masked):
    torch._dynamo.reset()
    logits, target = _inputs(seed=26)
    if masked:
        target[:, :3] = IGNORE
        target[2] = IGNORE
    fn = cls(smooth=1e-4, ignore_index=IGNORE if masked else None, **kwargs)
    compiled_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")  # fullgraph: fail on any break
    eager, compiled = (logits.clone().requires_grad_() for _ in range(2))
    expected, actual = fn(eager, target), compiled_fn(compiled, target)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(compiled.grad, eager.grad, rtol=1e-10, atol=1e-14)


@needs_gpu
@pytest.mark.parametrize("cls", [DiceLoss, TverskyLoss])
@pytest.mark.parametrize("kwargs", [dict(average="micro"), dict(skip_absent=True), dict(smooth_numerator=False)])
def test_auto_backend_leaves_the_fused_kernels_for_the_new_conventions(cls, kwargs, monkeypatch):
    logits, target = _inputs(seed=25)
    cpu = cls(**kwargs)(logits, target)

    def refuse(*args, **kw):
        raise AssertionError("the fused kernel must not run for this convention")

    monkeypatch.setattr(_triton_losses, "seg_loss", refuse)
    logits_gpu, target_gpu = logits.cuda(), target.cuda()
    assert not cls(**kwargs)._fusable(logits_gpu, target_gpu)
    assert cls()._fusable(logits_gpu, target_gpu)  # the default still fuses
    torch.testing.assert_close(cls(**kwargs)(logits_gpu, target_gpu).cpu(), cpu, rtol=1e-12, atol=1e-14)
