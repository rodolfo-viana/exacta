# Metrics

Overlap and surface functions in `exacta.functional` return per-case, per-selected-class
arrays of shape `(B, K)`. Corresponding classes in `exacta.metrics` accumulate results across
batches. Probability and dataset-level metrics use the reductions described below.
Choose complementary metrics according to the segmentation task; a high overlap score alone
does not describe lesion detection or boundary errors.

| Functional name | Stateful class | Definition |
|---|---|---|
| `dice` | `Dice` | $2\mathrm{TP}/(2\mathrm{TP}+\mathrm{FP}+\mathrm{FN})$ |
| `iou` | `IoU` | $\mathrm{TP}/(\mathrm{TP}+\mathrm{FP}+\mathrm{FN})$ |
| `sensitivity` | `Sensitivity` | $\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$ |
| `precision` | `Precision` | $\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$ |
| `specificity` | `Specificity` | $\mathrm{TN}/(\mathrm{TN}+\mathrm{FP})$ |
| `volume_similarity` | `VolumeSimilarity` | Default: $1-\lvert V_P-V_T\rvert/(V_P+V_T)$ |
| `relative_volume_difference` | `RelativeVolumeDifference` | $(V_P-V_T)/V_T$; `absolute=True` removes the sign |
| `hausdorff_distance` | `HausdorffDistance` | Percentile boundary distance; default percentile 100 |
| `hd95` | `HD95` | 95th-percentile boundary distance |
| `assd` | `ASSD` | Average symmetric boundary distance |
| `nsd` | `NSD` | Fraction of both surfaces within the supplied tolerance |

For each case and class, TP, FP, FN and TN count true-positive, false-positive,
false-negative and true-negative voxels. $V_P$ and $V_T$ are predicted and target foreground
voxel counts. The formulas apply when their denominators are nonzero; empty-mask and
zero-denominator policies are specified in
[Conventions](https://github.com/rodolfo-viana/exacta/wiki/Conventions).

Inputs use class indices `(B, *spatial)` with `num_classes`, or binary masks `(B, C, *spatial)`
with `input_format="one-hot"`. `include_background=False` omits channel/class zero.
Stateless functions accept PyTorch, NumPy, CuPy and JAX arrays and return the prediction's
array type. Stateful metrics store PyTorch results.

## Selected voxels and ignored labels (unreleased)

The next development version adds `mask` and `ignore_index` to the seven overlap/volume
metrics listed above. These options are not available in the published 0.0.1a4 overlap APIs.
For index inputs, `ignore_index` excludes voxels whose **target** equals that label. It does
not remove a class from the output. A boolean `mask` includes True voxels and excludes False
voxels; its shape is `(B, *spatial)` or `(B, 1, *spatial)` on the input device. For one-hot or
multilabel masks, the same voxel selection applies to every channel; use `mask` instead of
`ignore_index`. When both options are supplied, a voxel must pass both conditions.

Let $V$ be the included voxel set. All four counts are restricted to $V$:

$$
\mathrm{TP}=\lvert P\cap T\cap V\rvert,\quad
\mathrm{FP}=\lvert(P\setminus T)\cap V\rvert,\quad
\mathrm{FN}=\lvert(T\setminus P)\cap V\rvert,\quad
\mathrm{TN}=\lvert V\setminus(P\cup T)\rvert.
$$

Excluded values in either input are not validated or counted. An entirely excluded case has
zero counts and follows `both_empty` or `zero_division`, depending on the metric. The default
NaN is skipped by case averaging; choosing a finite fill explicitly includes it in that average.
Masks are consumed per call and are not retained by stateful metrics.

For example, evaluate synthetic partially annotated labels in the development version:

```python
import torch
from exacta import functional as EF
from exacta.metrics import MetricCollection

target = torch.tensor([[[1, 1], [0, -1]]])
prediction = torch.tensor([[[1, 0], [1, 999]]])
roi = torch.tensor([[[True, True], [True, True]]])
scores = EF.dice(prediction, target, num_classes=2, ignore_index=-1, mask=roi)
assert torch.allclose(scores, torch.tensor([[0., .5]], dtype=torch.float64))
metrics = MetricCollection(metrics=["dice", "specificity"], num_classes=2, ignore_index=-1)
metrics.update(prediction, target, mask=roi)
results = metrics.compute()
```

The CPU C++ and CUDA Triton paths exclude voxels inside the count pass, without one-hot
expansion of index maps. A checked PyTorch fallback covers other devices and larger class
histograms. Collections with surface or clDice metrics reject `mask` and `ignore_index`:
turning excluded regions into background would create artificial boundaries or alter topology.

## Spacing and tolerances

Distance metrics require spacing in the tensor's spatial-axis order. Supply one spatial
tuple for the batch or a `(B, ndim)` array for per-case spacing. The distance unit is the
spacing unit. NSD requires `tolerance`, which may be scalar or per selected class.

For an example using synthetic hard labels:

```python
import numpy as np
import exacta.functional as EF

target = np.zeros((1, 8, 8, 8), dtype=np.int64)
target[:, 2:5, 2:5, 2:5] = 1
prediction = np.roll(target, 1, axis=1)
scores = EF.nsd(prediction, target, num_classes=2, include_background=False,
                spacing=(1.5, 0.8, 0.8), tolerance=1.0, surface="surfels")
assert scores.shape == (1, 1)
```

## Aggregation and distributed evaluation

`compute()` returns `per_case`, `per_class` and `mean`. Aggregation averages cases with NaNs
excluded, then averages the class means. It does not pool all voxels across patients.
In distributed evaluation every rank must call `compute()`, including ranks with no local
updates. With NCCL, set each rank's current CUDA device. Avoid samplers that pad by duplicating
cases. If no rank has any cases, `compute()` raises `RuntimeError`.

See [Conventions](https://github.com/rodolfo-viana/exacta/wiki/Conventions) for empty masks,
surface representations and reference-library differences. Excluding a region from a surface
metric needs its own definition; overlap masking does not define partial surface evaluation.

## Shared metric evaluation

`functional.evaluate` shares confusion counts across overlap/area/volume metrics and
surface distances across compatible boundary metrics. It returns a dictionary of `(B, K)`
arrays. Choose names as a sequence, or map names to their individual options:

```python
selection = {"dice": {}, "hd95": {"combine": "pooled"}, "assd": {},
             "nsd": {"tolerance": 1.0, "surface": "voxels"}}
scores = EF.evaluate(prediction, target, metrics=selection, num_classes=2,
                     include_background=False, spacing=(1.5, 0.8, 0.8))
```

Each metric keeps its standalone defaults and empty-mask rules. Input layout, class selection
and spacing are shared; different surface representations form separate calculation groups.
Distances are reduced immediately per chunk, and no volume cache survives the call.

`MetricCollection` provides the same sharing during stateful evaluation, retaining only
detached scores and gathering them in one distributed payload:

```python
from exacta.metrics import MetricCollection

metrics = MetricCollection(metrics=selection, num_classes=2, include_background=False)
metrics.update(prediction, target, spacing=(1.5, 0.8, 0.8))
results = metrics.compute()
mean_hd95 = results["hd95"].mean
metrics.reset()
```

All ranks must use identical metric order and configuration, and must call `compute()`,
including ranks without updates. The usual sampler and aggregation rules above still apply.

## Centreline, probability and detection metrics

Version 0.0.1a4 adds the following evaluation APIs. The
[advanced segmentation guide](https://github.com/rodolfo-viana/exacta/wiki/Advanced-segmentation)
defines their empty-input behavior, probability conventions and matching protocol.

| Functional API | Stateful API | Quantity |
|---|---|---|
| `cldice` | `ClDice` | Harmonic mean of hard-skeleton precision and sensitivity; CPU Lee thinning |
| `brier_score` | `BrierScore` | Mean squared probability error; binary and multiclass scales differ |
| `negative_log_likelihood` | `NegativeLogLikelihood` | Mean true-class negative log probability, in nats |
| `expected_calibration_error` | `ExpectedCalibrationError` | Top-label or classwise, equal-width reliability bins |
| `reliability_diagram` | `Calibration` | Per-case summaries versus pooled voxel bin accumulation |
| `detection.lesion_detection` | `detection.LesionDetection` | Dataset lesion AP and FROC for fixed candidates ranked by confidence |

In 0.0.1a4, probability metrics accept index targets, optional masks and ignored labels. C=1 is binary
sigmoid; C>=2 is mutually exclusive multiclass. Independent multilabel probability channels
are not supported. Stateful probability scores average cases; `Calibration` pools voxels.
All ranks must participate in distributed computation.

## Independent probability channels (unreleased)

The development version adds `multilabel=True` to Brier, NLL, ECE, reliability diagrams and
their stateful wrappers. Prediction and binary target both have shape `(B,C,H,W)` or
`(B,C,D,H,W)`. Channels may overlap; `from_logits=True` applies independent sigmoids.
Masks accept `(B,*spatial)`, `(B,1,*spatial)` or `(B,C,*spatial)`. Excluded entries are not
validated. Use masks rather than `ignore_index` for independent channels.

For observed voxels $V_c$ of label $c$, the per-label scores are:

$$
\mathrm{Brier}_c=\frac{1}{|V_c|}\sum_{i\in V_c}(p_{ic}-y_{ic})^2,\qquad
\mathrm{NLL}_c=-\frac{1}{|V_c|}\sum_{i\in V_c}
\left[y_{ic}\log p_{ic}+(1-y_{ic})\log(1-p_{ic})\right].
$$

Correct endpoint probabilities contribute zero; impossible outcomes have infinite NLL.
Logit input uses stable softplus arithmetic. These are probability quality scores, not
pure calibration measures. Classwise reliability compares positive probability with
positive-label frequency; top-label calibration is rejected for multilabel inputs.

The `average` option controls within-case reduction. The following definitions also apply
when each label has a different observation mask:

| Option | Output | Reduction |
|---|---|---|
| `"none"` (default) | `(B,C)` | One score per label; unobserved labels are NaN |
| `"macro"` | `(B,1)` | Equal mean of observed label scores |
| `"micro"` | `(B,1)` | Pool observed label-voxel pairs before calculating the score |

For micro ECE, bin counts, probability sums and positive counts are pooled across labels
**before** taking absolute calibration gaps. Opposing label errors can cancel; macro ECE
does not permit this cancellation. Fully excluded cases are NaN. Stateful score classes
then average cases; `Calibration` pools voxels across cases while retaining separate labels.
C++/OpenMP and supported CUDA Triton paths fuse validation and statistics in float64;
other paths use checked PyTorch.

```python
import torch
from exacta import functional as EF
from exacta.metrics import Calibration

generator = torch.Generator().manual_seed(5)
logits = torch.randn(2, 3, 8, 8, generator=generator)
labels = torch.rand(2, 3, 8, 8, generator=generator) > 0.7
per_label = EF.brier_score(logits, labels, from_logits=True, multilabel=True)
calibration = Calibration(multilabel=True, from_logits=True, bins=15)
calibration.update(logits, labels)
diagram = calibration.compute()
```

## Physical area and error (unreleased)

The 2D APIs `predicted_area`, `target_area` and `area_error` measure occupied pixel-cell areas.
Their stateful counterparts are `PredictedArea`, `TargetArea` and `AreaError`. With pixel
spacing $(s_H,s_W)$ in millimetres, $a=s_Hs_W$ and

$$
A_P=a|P|,\qquad A_T=a|T|,\qquad E=A_P-A_T.
$$

The default `unit="mm2"` returns mm²; `unit="cm2"` divides by 100. `absolute=True` on the
error reports $|E|$. Supply scalar, shared two-axis or per-case `(B,2)` spacing. Areas require
2D inputs. Mask/ignore policies match physical volume: absent observed classes return zero;
unobserved domains return NaN. Overlapping channels are measured independently.

`evaluate` and `MetricCollection` share counts across areas and overlap scores. See the
[2D medical-segmentation guide](https://github.com/rodolfo-viana/exacta/wiki/Medical-segmentation-2D)
for runnable examples and the distinction between slice and patient averages.

## Physical volume and error (unreleased)

The development APIs `predicted_volume`, `target_volume` and `volume_error` take hard 3D
segmentations and spacing in millimetres. Their stateful counterparts are `PredictedVolume`,
`TargetVolume` and `VolumeError`. For axis spacings $s_1,s_2,s_3$, the voxel volume is
$v=s_1s_2s_3$ mm³. The reported quantities are:

$$
V_P=(\mathrm{TP}+\mathrm{FP})v,\qquad V_T=(\mathrm{TP}+\mathrm{FN})v,\qquad
E=(\mathrm{FP}-\mathrm{FN})v.
$$

`absolute=True` reports $|E|$. `unit="ml"` divides by 1000; the default is `"mm3"`.
Spacing accepts a scalar, three shared axis spacings or `(B,3)` per-case spacings.
Two-dimensional areas are rejected. Masking and ignored labels restrict the measured
domain: zero volume means an absent class in an observed domain, while an entirely
unobserved domain returns NaN. These metrics do not infer the volume of unannotated tissue.

`evaluate` and `MetricCollection` share one count pass with ordinary overlap metrics:

```python
pred = torch.zeros(2, 8, 8, 8, dtype=torch.int64)
target = pred.clone()
target[:, 2:5, 2:5, 2:5] = 1
values = EF.evaluate(pred, target, num_classes=2, include_background=False,
                     spacing=[[1, 1, 2], [0.8, 0.8, 3]],
                     metrics={"dice": {}, "predicted_volume": {"unit": "ml"},
                              "target_volume": {"unit": "ml"},
                              "volume_error": {"unit": "ml", "absolute": True}})
```

## Voxel PR, AP and AUROC (unreleased)

`voxel_ranking` returns per-case/class AP and AUROC arrays plus PR/ROC curves indexed by
`result.curves[case][class]`. `average_precision` and `auroc` return only the respective
score arrays. C=1 is binary foreground; C>=2 is categorical one-versus-rest unless
`multilabel=True` selects independent channels with same-shaped binary targets. The input
and exclusion conventions match the probability APIs above. Categorical targets are indices.

Exact mode (`bins=None`) sorts scores on the input device and groups equal scores. With
decreasing thresholds, let $r_j$ and $q_j$ be recall and precision after adding tie group $j$:

$$
\mathrm{AP}=\sum_j(r_j-r_{j-1})q_j.
$$

This is the [non-interpolated AP convention](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html),
not trapezoidal area under the PR curve. AUROC integrates the ROC curve trapezoidally,
equivalently giving half credit to tied positive-negative score pairs. AP is NaN without
positives and 1 for an observed all-positive target. AUROC is NaN unless both outcomes occur.
Entirely excluded inputs give NaN scores. These are voxel measures, distinct from lesion AP.

Curves start with threshold `+inf`, precision 1 and recall 0 when positives exist. Thresholds
decrease and inclusion means `score >= threshold`. Exact thresholds use the supplied
probabilities; logit input uses raw binary/multilabel logits or categorical log-softmax
scores. Keeping log scores avoids sigmoid/softmax saturation creating artificial ties.
Floating-point normalization can still affect almost-equal categorical scores.

`bins=256`, for example, quantizes probabilities into 256 equal-width bins; 1 belongs to
the last bin. This is approximate ranking: scores in a bin become tied. Curves use bin
lower boundaries as probability thresholds. More bins reduce quantization but do not
guarantee a monotonic error decrease or a fixed error bound.

`AveragePrecision` and `AUROC` average case scores. `VoxelRanking` instead pools observed
voxels across batches and ranks, so larger images contribute more observations:

```python
from exacta.metrics import VoxelRanking

ranking = VoxelRanking(from_logits=True, multilabel=True, bins=256)
ranking.update(logits, labels)
pooled = ranking.compute()  # AP/AUROC have shape (1,C), one pooled dataset
precision = pooled.curves[0][0].precision
recall = pooled.curves[0][0].recall
ranking.reset()
```

Exact streaming retains selected scores and labels and gathers each class on every rank.
Binned streaming retains `2*C*bins` int64 counts and sums those counts across ranks;
temporary update buffers depend on the current batch. Every rank, including empty ranks,
must call `compute` with identical configuration. Set the current CUDA device for NCCL.
Repeated `compute` does not add observations or mutate local state.
