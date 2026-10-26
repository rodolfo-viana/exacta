# exacta

Metrics and differentiable losses for **2D and 3D medical-image segmentation**, on CPU and GPU.
Exacta provides reusable implementations of Dice, HD95, surface distances, lesion scores,
probability metrics and training objectives, with explicit mathematical conventions.
The development version also adds general **classification and regression** APIs.

CPU operations use C++ kernels where implemented; CUDA paths use Triton or native PyTorch.
Definitions are checked against independent references such as SciPy, MONAI, MedPy,
DeepMind's surface-distance, MeshMetrics, SimpleITK, Panoptica and the BraTS 2023 evaluation
code. Where conventions differ, parameters select the intended definition.

**Published version: 0.0.1a5 (alpha). Development: 0.0.1a6.dev0.**
[Documentation](https://github.com/rodolfo-viana/exacta/wiki) ·
[Source](https://github.com/rodolfo-viana/exacta) ·
[Issues](https://github.com/rodolfo-viana/exacta/issues)

## Installation

Exacta requires **Python 3.10+**, **PyTorch 2.4+** and **NumPy 1.22+**. Install PyTorch for
your CPU or CUDA environment, then install Exacta:

```bash
python -m pip install exacta==0.0.1a5
```

Binary wheels target Linux x86-64 with glibc 2.28 or newer: Python 3.10 and 3.11 wheels,
plus a Python 3.12 stable-ABI wheel for Python 3.12 and newer. Other platforms install
from source and require a C++17 compiler and Python development headers; OpenMP is optional.
SciPy is an optional dependency for Hungarian lesion matching.

For distributed CPU evaluation with Gloo, use **PyTorch 2.8+** to avoid an older
process-group teardown deadlock. Multi-GPU NCCL support remains experimental.

## Quickstart

Train with logits and evaluate hard labels. This synthetic 3D example runs on the available
CPU or CUDA device; it demonstrates the API, not model accuracy:

```python
import torch
import exacta.functional as EF
from exacta.losses import DiceCELoss

rng = torch.Generator().manual_seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logits = torch.randn((2, 3, 16, 24, 24), generator=rng).to(device).requires_grad_()
target = torch.zeros((2, 16, 24, 24), dtype=torch.long, device=device)
target[:, 3:8, 4:12, 5:13] = 1
target[:, 9:13, 12:20, 12:20] = 2

loss = DiceCELoss(include_background=False)(logits, target)
loss.backward()

prediction = logits.detach().argmax(dim=1)
scores = EF.evaluate(
    prediction, target, num_classes=3, include_background=False,
    spacing=(1.5, 0.8, 0.8),  # spatial-axis order, in mm
    metrics={"dice": {}, "hd95": {}, "nsd": {"tolerance": 1.0}},
)
print("Loss:", loss.item())
print("Dice per case/class:", scores["dice"])  # shape (2, 2)
print("HD95 in mm:", scores["hd95"])
```

For 2D images, use logits `(B,C,H,W)`, integer targets `(B,H,W)` and two-axis spacing.
For independent overlapping labels, supported losses use `activation="sigmoid"` and
`target_format="one-hot"` with same-shaped binary targets. The
[2D medical-segmentation guide](https://github.com/rodolfo-viana/exacta/wiki/Medical-segmentation-2D)
includes binary, multiclass and multilabel examples.

## Metrics

Stateless functions live in `exacta.functional`. Their stateful counterparts in
`exacta.metrics` accumulate across batches and support distributed evaluation.

| Category | Metrics |
|---|---|
| Overlap | Dice / voxel F1; IoU / Jaccard |
| Classification rates | Sensitivity / recall; precision; specificity |
| Relative size | Volume similarity; signed or absolute relative volume difference |
| Physical size | Predicted/target **3D volume** and error in mm³ or mL; **2D area** and error in mm² or cm² |
| Boundaries | Hausdorff distance with configurable percentile; HD95; ASSD; normalized surface Dice (NSD) |
| Centreline | Hard-skeleton clDice |
| Probability quality | Brier score; negative log-likelihood (NLL) |
| Calibration | Expected calibration error (ECE); reliability diagrams |
| Voxel ranking | Average precision (AP); AUROC; precision–recall and ROC curves, exact or approximately binned |
| Lesion matching | Lesion TP/FP/FN, precision, recall, F1, segmentation quality and panoptic quality |
| Lesion segmentation | Lesion-wise Dice and HD95; BraTS 2023 evaluation conventions |
| Confidence-ranked detection | Lesion AP and FROC curves |

Use `exacta.lesions.evaluate` for lesion matching and `exacta.detection.LesionDetection`
for confidence-ranked detection. Connected-component labeling, largest-component selection
and small-component removal are available in `exacta.components`.

`functional.evaluate` and `MetricCollection` share confusion counts and compatible surface
distances when evaluating multiple metrics. For accumulation across validation batches:

```python
from exacta.metrics import MetricCollection

metrics = MetricCollection(
    metrics={"dice": {}, "hd95": {}}, num_classes=3, include_background=False,
)
metrics.update(prediction, target, spacing=(1.5, 0.8, 0.8))
result = metrics.compute()["dice"]
print(result.per_case, result.per_class, result.mean)
metrics.reset()
```

Functional APIs accept PyTorch, NumPy, CuPy and JAX arrays and return the prediction's array
type. Stateful APIs use PyTorch. See [Metrics](https://github.com/rodolfo-viana/exacta/wiki/Metrics)
and [Lesion evaluation](https://github.com/rodolfo-viana/exacta/wiki/Lesion-evaluation) for
arguments, formulas and matching protocols.

## Training losses

Losses are PyTorch modules in `exacta.losses`. Choose the objective and target representation
explicitly; different loss definitions produce different gradients.

| Category | Classes |
|---|---|
| Overlap | `DiceLoss`, `GeneralizedDiceLoss`, `TverskyLoss`, `FocalTverskyLoss` |
| Class-confusion costs | `GeneralizedWassersteinDiceLoss` |
| Voxel classification | `FocalLoss`, `TopKCELoss` |
| Combined objectives | `DiceCELoss`, `DiceFocalLoss`, `GeneralizedDiceCELoss`, `DiceTopKCELoss` |
| IoU surrogates | `LovaszSoftmaxLoss`, `LovaszHingeLoss` |
| Boundary objectives | `BoundaryLoss`, `HausdorffDTLoss` |
| Soft centreline objective | `SoftClDiceLoss` |

`DeepSupervisionLoss` combines a loss across explicitly paired prediction/target resolutions.
`DiceLoss(squared_pred=True)` uses a squared denominator; Dice–CE, Dice–focal and Dice–top-k CE
support the same option. Generalized Dice supports inverse-square, inverse-volume and uniform
class weights. Standard cross-entropy and binary cross-entropy remain available in PyTorch.

Wasserstein Dice accepts integer labels and a symmetric, nonnegative class-cost matrix with
zero diagonal and a positive maximum. The matrix is normalized by that maximum. Default
weighting requires maximal background/foreground costs; `weighting_mode="GDL"` uses inverse
observed class counts. These costs describe label confusion, not spatial distance.

`HausdorffDTLoss` is a distance-transform training surrogate, not HD95. Soft-clDice uses a
finite-iteration soft skeleton and differs from the hard clDice evaluation metric. Geometry
and topology objectives require complete target geometry; combine them with regional
supervision where appropriate. See [Losses](https://github.com/rodolfo-viana/exacta/wiki/Losses)
for equations, smoothing, reductions, target formats and derivative support.

## Classification and regression (unreleased)

The following APIs require **0.0.1a6.dev0** and are not included in the published
`0.0.1a5` package. They evaluate samples across a dataset, with separate namespaces
from the per-image segmentation metrics.

| Namespace | Metrics or losses |
|---|---|
| `functional.classification` | Accuracy, top-k accuracy, confusion matrices, precision, recall, specificity, F1/F-beta, balanced accuracy, MCC, Cohen kappa, Brier, NLL, ECE, AP and AUROC |
| `functional.regression` | MAE, MSE, RMSE, R², explained variance, Pearson and concordance correlation |
| `metrics.classification` | `ClassificationMetrics` for pooled counts; `ClassificationRanking` for pooled AP/AUROC and PR/ROC curves |
| `metrics.regression` | `RegressionMetrics` for pooled weighted errors and centered moments |
| `losses.classification` | Native PyTorch `CrossEntropyLoss`, `BCEWithLogitsLoss`, `BCELoss`, `NLLLoss`, `KLDivLoss` |
| `losses.regression` | `QuantileLoss`, `LogCoshLoss`; native PyTorch `L1Loss`, `MSELoss`, `HuberLoss`, `SmoothL1Loss`, `GaussianNLLLoss`, `PoissonNLLLoss` |

Classification explicitly selects binary, multiclass or multilabel targets. Regression
accepts `(N,)` or `(N,D)` arrays. Masks and sample weights are available for hard-decision
classification and regression metrics; probability metrics currently accept masks but
not sample weights. Stateful metrics pool sufficient statistics before computing scores.
They do not average batch F1, R² or RMSE. CPU regression uses fused C++ statistics;
CUDA regression uses native PyTorch. Standard losses are direct PyTorch exports.

```python
import torch
from exacta.functional import classification as C, regression as R
from exacta.metrics.regression import RegressionMetrics

labels = torch.tensor([0, 1, 2, 1])
class_target = torch.tensor([0, 2, 2, 1])
print(C.evaluate(labels, class_target, task="multiclass", num_classes=3))

regression_pred = torch.tensor([1., 2., 4., 5.])
regression_target = torch.tensor([1., 3., 5., 4.])
print(R.evaluate(regression_pred, regression_target))
regression_metrics = RegressionMetrics(metrics=("mae", "rmse", "r2"))
regression_metrics.update(regression_pred[:2], regression_target[:2])
regression_metrics.update(regression_pred[2:], regression_target[2:])
print(regression_metrics.compute())
```

Undefined classification rates default to NaN; `zero_division=0` or `1` is explicit.
Regression R² preserves undefined constant-target cases unless `force_finite=True`.
See [classification](https://github.com/rodolfo-viana/exacta/blob/main/docs/wiki/Classification.md)
and [regression](https://github.com/rodolfo-viana/exacta/blob/main/docs/wiki/Regression.md)
for formulas, averaging, empty-input behavior and examples.

## Partial annotations and probability channels

Overlap, area and volume metrics accept shared boolean voxel masks and index-target
`ignore_index`. Excluded voxels contribute to none of TP, FP, FN or TN:

```python
partial_target = target.clone()
partial_target[:, :2] = -1
observed_dice = EF.dice(
    prediction, partial_target, num_classes=3, include_background=False, ignore_index=-1,
)
```

Surface and hard-clDice metrics require complete segmentations: removing voxels would change
boundaries or topology. Loss masking follows each objective's documented rules.

Brier, NLL, ECE, reliability diagrams and voxel rankings support `multilabel=True` for
independent probability channels with same-shaped binary targets. Logit inputs then use
independent sigmoids. Shared or per-label masks and explicit macro/micro reductions are
available where documented.

## Mathematical conventions

Specify the definition as part of your evaluation protocol. Empty inputs, aggregation and
surface representations can change reported values even for identical segmentations.

| Choice | Default and options |
|---|---|
| Hard labels | Index maps with `num_classes`, or binary channels with `input_format="one-hot"`; overlapping channels are allowed. |
| Background | Channel/class zero is included unless `include_background=False`. |
| Spacing | Spatial-axis order of the array; physical area/volume inputs use millimetres. |
| Surfaces | Boundary voxel centres by default; `surface="surfels"` uses physical length/area weights, and `"mesh"` uses contour/triangle geometry. |
| HD95 | Maximum of the two directed 95th percentiles; `combine="pooled"` instead pools distances before the percentile. |
| ASSD | Pooled surface-element mean; `combine="mean"` averages the two directed means. |
| Empty masks | Dice/IoU return NaN when both are empty; distance metrics default to infinity when exactly one is empty, except NSD, which returns zero. Parameters make these policies explicit. |
| Case aggregation | Stateful per-case metrics skip NaNs, average cases per class, then average class means. |
| Pooled evaluation | `Calibration` pools voxel bins; `VoxelRanking` pools voxel rankings. Case averages and pooled scores are different quantities. |
| AP | Non-interpolated sum of recall increments times precision, not trapezoidal PR area. No-positive AP and single-outcome AUROC are NaN. |

For distributed evaluation, every rank must participate in `compute()`, including ranks with
no local cases. Avoid samplers that duplicate cases through padding. See
[Conventions](https://github.com/rodolfo-viana/exacta/wiki/Conventions) for full definitions,
reference-library variants and empty-input behavior.

## Performance

Backend choice depends on shape, class count, dtype and hardware. Supported metric passes use
C++/OpenMP on CPU and Triton on CUDA. Training losses use native PyTorch and, for supported
objectives, fused Triton forward/backward. `torch.compile` can improve native PyTorch paths;
measure complete forward/backward cost and account for compilation startup.

Mesh distances and hard skeletonization run on CPU, including transfers for CUDA inputs.
Lesion matching uses CPU lesion tables. The CPU extension is independent of the PyTorch
binary version. See [Performance](https://github.com/rodolfo-viana/exacta/wiki/Performance)
for measured workloads, memory behavior and backend limitations.

## License

Apache License 2.0; see the [license](https://github.com/rodolfo-viana/exacta/blob/main/LICENSE),
[notices](https://github.com/rodolfo-viana/exacta/blob/main/NOTICE) and
[third-party licenses](https://github.com/rodolfo-viana/exacta/blob/main/THIRD_PARTY_LICENSES).
