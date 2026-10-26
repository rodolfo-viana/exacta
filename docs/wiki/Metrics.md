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

`functional.evaluate` shares confusion counts across overlap/volume metrics and
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

Probability metrics accept index targets, optional masks and ignored labels. C=1 is binary
sigmoid; C>=2 is mutually exclusive multiclass. Independent multilabel probability channels
are not supported. Stateful probability scores average cases; `Calibration` pools voxels.
All ranks must participate in distributed computation.
