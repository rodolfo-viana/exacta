# Classification

These APIs are **unreleased**, introduced in `0.0.1a6.dev0`. They evaluate independent
samples across a dataset, separate from segmentation's per-image scores. Functional
metrics live in `exacta.functional.classification`; predictions always come first.
PyTorch, NumPy, CuPy and JAX inputs retain their array library in functional outputs.
Statistical results are float64. Metrics detach inputs; use losses for optimization.

## Inputs and decisions

Choose the task explicitly. Labels are zero-based and each multiclass observation has
one target class. Independent multilabel targets can contain several positive labels.

| Task | Predictions | Targets |
|---|---|---|
| `binary` | `(N,)` hard labels, probabilities or logits | `(N,)` binary values |
| `multiclass` | `(N,)` integer labels or `(N,C)` floating scores | `(N,)` integer labels |
| `multilabel` | `(N,C)` hard labels, probabilities or logits | `(N,C)` binary values |

Multiclass calls require `num_classes=C`, fixing the label universe even when classes
are absent. Binary uses positive class 1. Floating inputs are probabilities by default;
set `from_logits=True` for logits. Multiclass probabilities must sum to one per row.
Binary/multilabel decisions use probability **greater than or equal to** `threshold=0.5`.
Multiclass argmax ties favor the lower class index. `top_k_accuracy` takes finite `(N,C)`
scores and uses the same tie order, without requiring normalized probabilities.

Boolean `mask` excludes complete samples `(N,)` or individual multilabel targets `(N,C)`.
For two-dimensional targets, `(N,1)` also denotes a shared sample mask. `ignore_index`
excludes matching target entries. Hard-decision metrics accept finite, nonnegative
`sample_weight` of shape `(N,)`; a zero weight excludes the sample from validation and
counting. Values at excluded entries can contain sentinels or NaNs. Probability/ranking
APIs accept masks and ignored labels, but currently do not accept sample weights.

## Rates and confusion counts

`evaluate` computes shared counts once and returns a dictionary. Standalone functions
are `accuracy`, `precision`, `recall`, `specificity`, `f1`, `fbeta`, `balanced_accuracy`,
`mcc`, `cohen_kappa`, `confusion_matrix` and `top_k_accuracy`. Kappa is unweighted Cohen
kappa and requires binary or multiclass labels. It is opt-in through `evaluate(metrics=...)`.

For class $c$, let $a_c$ be true-positive weight, $p_c$ predicted-positive weight,
$t_c$ target-positive weight and $n_c$ total observed weight. Specificity uses
$\mathrm{TN}_c=n_c-p_c-t_c+a_c$. The positive-class rates are

$$
\mathrm{Precision}_c=\frac{a_c}{p_c},\quad
\mathrm{Recall}_c=\frac{a_c}{t_c},\quad
F_{\beta,c}=\frac{(1+\beta^2)a_c}{\beta^2t_c+p_c}.
$$

Rates support `average="none"` (one value per label), `micro` (pool counts first),
`macro` (equal label weights, default) and `weighted` (target-support weights).
Binary rates always concern class 1; `none` returns a length-one vector.
Undefined denominators use `zero_division`, NaN by default, optionally 0 or 1.
Macro means include NaNs. Support-weighted means omit labels with zero target support,
and return NaN if all supports are zero. This differs from references that drop NaNs
from macro averages or automatically infer a smaller label universe.

Accuracy is a scalar: correct-sample fraction for binary/multiclass, correct-label-decision
fraction for multilabel. Multilabel accuracy is not exact-match/subset accuracy.
Multiclass balanced accuracy is mean recall over **all declared classes**; absent classes
follow `zero_division`. Binary/multilabel balanced accuracy averages sensitivity and
specificity. Multiclass accuracy, balanced accuracy, MCC and kappa are scalar regardless
of the `average` argument. For multiclass, with total weight $n$ and correct weight $s$,

$$
\mathrm{MCC}=\frac{sn-\sum_c p_ct_c}
{\sqrt{\sum_c p_c(n-p_c)}\sqrt{\sum_c t_c(n-t_c)}},\qquad
\kappa=\frac{sn-\sum_c p_ct_c}{n^2-\sum_c p_ct_c}.
$$

Confusion matrices use **target rows and predicted columns**. Shapes are `(2,2)` for
binary, `(C,C)` for multiclass and `(C,2,2)` for multilabel. Counts are int64 without
weights, float64 with weights; the function does not normalize them.

## Probability quality and ranking

`brier_score`, `negative_log_likelihood`, `expected_calibration_error`, `average_precision`
and `auroc` reuse Exacta's validated probability and ranking engines on a pooled sample
domain. Binary/multiclass Brier, NLL and top-label ECE return scalars. Multilabel scores
return `(C,)`, as do all AP/AUROC calls (binary uses `C=1`). Multiclass ranking is
one-versus-rest. Take an explicit mean of per-class scores when desired.

Binary Brier is mean squared probability error (range 0–1); multiclass Brier sums
squared errors over the class vector before averaging samples (range 0–2).
Multilabel NLL uses independent Bernoulli outcomes. NLL is infinite for an observed
impossible outcome. ECE uses equal-width probability bins (`bins=15` by default), with
top-label confidence for binary/multiclass and positive-probability bins for multilabel.
Binary top-label ECE treats a probability of exactly 0.5 as class 0, consistent with the
existing two-channel calibration convention; hard-decision rates use `>= threshold`.

AP integrates precision at recall increments, not trapezoidal PR area. No positives gives
NaN AP; all positives gives 1. AUROC needs both outcomes. `bins=None` ranks exactly;
integer `bins` approximates by quantizing probabilities. Tied positives/negatives receive
half credit in AUROC. Empty observed outputs are NaN.

## Streaming evaluation and losses

`ClassificationMetrics` retains O(C) sufficient counts and computes scores after pooling
updates. This avoids averaging batch F1 or batch MCC. Compute also pools distributed
ranks; all ranks must participate with identical options, including empty ranks.
Set each rank's current CUDA device before NCCL and avoid evaluation samplers that pad
by duplicating samples. Two-GPU NCCL validation is still limited by available hardware.

```python
import torch
from exacta.metrics.classification import ClassificationMetrics
from exacta.losses.classification import CrossEntropyLoss

logits = torch.tensor([[2., 0., -1.], [0., 1., 2.]], requires_grad=True)
target = torch.tensor([0, 1])
CrossEntropyLoss()(logits, target).backward()
metric = ClassificationMetrics(task="multiclass", num_classes=3,
                               from_logits=True, zero_division=0)
metric.update(logits, target, sample_weight=torch.tensor([1., 2.]))
print(metric.compute())
metric.reset()
```

`ClassificationRanking` similarly pools AP/AUROC and PR/ROC curves. Exact mode stores
observed scores; binned mode stores O(C*bins) counts. `compute()` returns a `RankingResult`
with score arrays `(1,C)` and curves indexed by `[0][class]`, where the first axis is the
pooled dataset. Repeated compute calls do not accumulate the same data again.

`exacta.losses.classification` exports the original PyTorch `CrossEntropyLoss`,
`BCEWithLogitsLoss`, `BCELoss`, `NLLLoss` and `KLDivLoss` classes. Their input conventions,
reductions and weighting are exactly PyTorch's; no additional kernel/wrapper is inserted.

Definitions can be compared with the [scikit-learn metric reference](https://scikit-learn.org/stable/modules/model_evaluation.html),
accounting for the averaging and undefined-value conventions above. Standard loss contracts
are in [PyTorch's loss documentation](https://docs.pytorch.org/docs/stable/nn.html#loss-functions).
