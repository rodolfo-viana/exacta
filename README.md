# exacta

Fast, reference-validated losses and metrics for deep learning, on CPU and GPU.

Every project ends up re-implementing Dice, HD95 and friends, and the implementations quietly
disagree: [Podobnik & Vrtovec (2024)](https://arxiv.org/abs/2410.02630) compared 11 open-source
tools and found that distance-based metrics such as HD95 and NSD can differ substantially across
tools for the same segmentation pair. exacta aims to be the one implementation you can trust and
cite: every metric states its exact definition and is tested against independent reference
implementations (SciPy, MONAI, MedPy, DeepMind's surface-distance, MeshMetrics, SimpleITK,
Panoptica and the BraTS 2023 evaluation code), and where they disagree with each other, exacta
can reproduce each of them through an explicit argument.

Status: alpha (v0.0.1a4). Current releases target 3D medical image segmentation.

Documentation: [GitHub Wiki](https://github.com/rodolfo-viana/exacta/wiki).
The guides cover installation, a complete quickstart, metrics, losses, mathematical conventions,
lesion evaluation and performance for this release.

## Installation

Exacta requires Python 3.10+, PyTorch 2.4+ and NumPy 1.22+. Install the alpha from PyPI:

```bash
python -m pip install exacta==0.0.1a4
```

This release includes **Linux x86-64 wheels** for glibc 2.28 or newer: Python 3.10 and 3.11
wheels, plus a Python 3.12 stable-ABI wheel also tested on Python 3.13 and 3.14.
Other platforms install from source and require a C++17 compiler and Python development
headers; OpenMP is optional. On Debian/Ubuntu, install `build-essential` and the development
headers matching your Python version before building from source.

## Usage

```python
import exacta.functional as EF
from exacta.losses import DiceCELoss
from exacta.metrics import HD95, NSD, Dice

# Training
loss_fn = DiceCELoss(include_background=False)        # logits (B, C, D, H, W), labels (B, D, H, W);
                                                      # fused Triton kernels on GPU
loss = loss_fn(model(x), y)

# Validation: accumulate per-case values across batches (and processes under torch.distributed)
dice = Dice(num_classes=4, include_background=False)
hd95 = HD95(num_classes=4, include_background=False)
nsd = NSD(tolerance=[1.0, 2.0, 2.0], num_classes=4, include_background=False)  # mm, per class
for x, y, spacing in val_loader:
    pred = model(x).argmax(1)
    dice.update(pred, y)
    hd95.update(pred, y, spacing=spacing)             # per-case voxel size in mm, (B, 3)
    nsd.update(pred, y, spacing=spacing)

result = hd95.compute()
result.per_case    # (N, C)
result.per_class   # (C,)
result.mean        # scalar

# Stateless functions accept PyTorch, NumPy, CuPy or JAX arrays and return the same type
EF.hd95(pred_np, target_np, spacing=(1.5, 0.8, 0.8), num_classes=4)  # -> np.ndarray (B, C)

# Volume and detection rates: sensitivity, precision, specificity, volume similarity, RVD
EF.sensitivity(pred, target, num_classes=4, zero_division=0.0)          # 0 where the target is empty
EF.volume_similarity(pred, target, num_classes=4, convention="itk")      # SimpleITK's signed variant

# Match another tool's definition explicitly
s = (1.5, 0.8, 0.8)                                                  # voxel size in mm
EF.hd95(pred, target, spacing=s, num_classes=4, combine="pooled")    # MedPy's hd95
EF.assd(pred, target, spacing=s, num_classes=4, combine="mean")      # MedPy 0.4's assd
EF.nsd(pred, target, spacing=s, num_classes=4, tolerance=1.0, surface="surfels")  # DeepMind's NSD
EF.hd95(pred, target, spacing=s, num_classes=4, surface="mesh")                  # MeshMetrics' HD95

# Lesion-wise (instance) evaluation: choose how lesions are matched
from exacta import lesions
scores = lesions.evaluate(pred, target, num_classes=4, include_background=False, connectivity=3,
                          matching="greedy", score="iou", threshold=0.5)   # Panoptica's default
scores.f1, scores.sq_iou, scores.pq_iou, scores.lesion_dice                # each (B, C)
brats = lesions.brats2023(lesions.brats_regions(pred), lesions.brats_regions(target),
                          challenge="GLI", spacing=(1.0, 1.0, 1.0))      # BraTS 2023, WT/TC/ET

# Connected components of one binary image (no batch or channel axis)
from exacta import components
labels, count = components.label(mask, connectivity=3)        # numbered exactly as scipy.ndimage.label
kept = components.largest(mask, connectivity=3)                # e.g. nnU-Net-style post-processing
cleaned = components.remove_small(mask, 50, connectivity=1)    # drop components under 50 voxels
```

## Centreline, probability and lesion detection scores

`functional.cldice` and `metrics.ClDice` evaluate hard-skeleton overlap. Lee thinning runs
in C++ on CPU, including transfers for CUDA inputs. `functional.skeletonize` exposes the
same binary-mask operation, and precomputed skeletons can be reused for clDice.

Probability evaluation includes `brier_score`, `negative_log_likelihood`,
`expected_calibration_error` and `reliability_diagram`. These accept probabilities or
`from_logits=True`, integer targets, masks and ignored labels. One channel means binary
sigmoid; multiple channels mean mutually exclusive softmax classes. Stateful scores average
cases, while `metrics.Calibration` pools voxel reliability bins.

`exacta.detection.LesionDetection` accumulates confidence-ranked lesion AP and FROC across
cases. Connected components form fixed candidates before the confidence sweep. Report the
connectivity, component confidence rule and IoU matching threshold alongside the scores.
See the [advanced segmentation guide](https://github.com/rodolfo-viana/exacta/wiki/Advanced-segmentation)
for definitions, empty-input conventions and examples.

### Partially annotated overlap evaluation (unreleased)

In the next development version, overlap and volume metrics accept `ignore_index` for index
targets and a boolean `mask` of shape `(B, *spatial)` or `(B, 1, *spatial)`. The mask is shared
across channels. Both options exclude voxels from validation and from TP, FP, FN **and TN**:

```python
scores = EF.dice(pred, target, num_classes=4, ignore_index=-1, mask=annotated_voxels)
metric = Dice(num_classes=4, ignore_index=-1)
metric.update(pred, target, mask=annotated_voxels)
```

Fully excluded cases use the metric's existing zero-denominator policy: NaN by default,
which stateful averages skip. `functional.evaluate` and `MetricCollection` support the same
options when selecting only overlap/volume metrics. Surface and clDice metrics require
complete segmentations. These overlap options are not part of the published 0.0.1a4 API.

### Additional segmentation APIs (unreleased)

The development version adds independent-label Brier, Bernoulli NLL and calibration with
`multilabel=True`. Targets match `(B,C,*spatial)`; logits use independent sigmoids. Per-label,
macro and micro reductions are explicit, and masks can select different voxels per label.

For 3D hard labels, `predicted_volume`, `target_volume` and `volume_error` report mm³ or mL
from spacing in millimetres. They share counts through `evaluate` and `MetricCollection`.
Signed volume error is prediction minus target; `absolute=True` gives absolute physical error.

`average_precision`, `auroc` and `voxel_ranking` evaluate voxel-level rankings for binary,
multiclass and multilabel segmentations. Exact mode groups tied scores; `bins=256` selects
an approximate probability histogram. `AveragePrecision` and `AUROC` average case scores;
`VoxelRanking` pools voxels across batches/ranks, retaining fixed-size counts in binned mode.
AP without positive targets and AUROC without both outcomes are NaN.

Training additions are `DiceLoss(squared_pred=True)` and the same option in Dice-CE,
Dice-focal and Dice-top-k CE, plus `LovaszHingeLoss` for binary logits or independent labels.
Probability/ranking APIs and both loss additions support 2D and 3D inputs. Physical volume
APIs require 3D. These additions are not included in the published 0.0.1a4 package.

## Conventions

For hard segmentation metrics, each convention is an explicit argument, because this is
where libraries disagree. Probability and confidence-ranked metrics use the definitions above.

| Topic | exacta |
|---|---|
| Input layout | `input_format="index"`: class indices `(B, *spatial)` + `num_classes`. `"one-hot"`: binary masks `(B, C, *spatial)`. Metrics take hard labels; losses take logits. |
| Spacing | Required for distance metrics, in the axis order of the tensor. A SimpleITK array is `(z, y, x)` while `GetSpacing()` returns `(x, y, z)`. |
| Surface | `surface="voxels"` (default): boundary voxels, i.e. mask voxels with a face-adjacent neighbour outside the mask, each with equal weight, as in MONAI and MedPy. `"surfels"`: DeepMind's surface elements at voxel corners, weighted by their marching-cubes area (MONAI's `use_subvoxels=True`); percentiles become area-weighted, as in DeepMind. `"mesh"`: MeshMetrics' VTK marching-cubes mesh, with distances from each triangle's centre to the nearest point of the other mesh, weighted by area (2-D: contour segments). Outside the image counts as background. |
| HD95 | `combine="max"` (default): `max(P95(d(P→T)), P95(d(T→P)))`, as in MONAI and DeepMind; linear-interpolated percentile for voxels. `"pooled"`: one percentile over both directions, as in MedPy. |
| ASSD | `combine="pooled"` (default): mean over the surface elements of both masks, as in MONAI and MedPy 0.5.2. `"mean"`: mean of the two directed means, as in MedPy 0.4 (DeepMind returns the two directed means). |
| NSD | Fraction of the surface of both masks within `tolerance` of the other surface, `d ≤ tolerance`, as in MONAI and DeepMind. Each distance is the exact floating-point minimum of the expression SciPy evaluates, so distances that tie with the tolerance land on the same side as in MedPy and DeepMind. MONAI compares in float32. |
| Class absent from both | `both_empty`, default NaN (skipped when averaging). |
| Class absent from one | Dice/IoU give 0. Distance metrics give `one_empty`: default `inf` for HD/ASSD, 0 for NSD. |
| Zero denominators | `zero_division` for sensitivity, precision, specificity, volume similarity and relative volume difference; default NaN, as in MONAI. MedPy 0.5.2 returns 0 with NumPy 1.x but NaN with NumPy 2 (its `ZeroDivisionError` handler no longer fires). |
| Volume similarity | `convention="taha"` (default): `1 - \|\|P\| - \|T\|\| / (\|P\| + \|T\|)`, in [0, 1] (Taha & Hanbury 2015). `"itk"`: `2(\|P\| - \|T\|) / (\|P\| + \|T\|)`, signed, as SimpleITK's `GetVolumeSimilarity`. |
| Relative volume difference | `(\|P\| - \|T\|) / \|T\|`, as MedPy's `ravd`; `absolute=True` for the unsigned form. |
| Aggregation | Per case, then the NaN-ignoring mean over cases, then over classes. |
| Lesion matching | `matching="greedy"`: one-to-one, best-scoring pairs first, above `threshold` (Panoptica's `NaiveThresholdMatching`). `"hungarian"`: optimal one-to-one (`MaxBipartiteMatching`). `"merge"`: a lesion predicted in pieces is merged when the union scores higher (`MaximizeMergeMatching`). `"overlap"`: every prediction touching a target lesion grown by `dilation` steps, many-to-many (BraTS 2023). F1/SQ/PQ are Panoptica's RQ/SQ/PQ; `lesion_dice`/`lesion_hd95` are BraTS's lesion-wise scores. Panoptica labels 3-D images with 26-connectivity but 2-D ones with 4-connectivity; exacta's `connectivity` is explicit. |
| Connectivity | Required for connected components: 1 = faces (6 neighbours in 3-D, SciPy's default), 2 = and edges (18), 3 = and corners (26, MONAI's default). |

Spacing components must lie in `[1e-100, 1e100]` for voxel surfaces and `[1e-50, 1e50]` for
surfels and meshes. The latter need a narrower range because area calculations and triangle
predicates involve fourth powers of physical coordinates. Unsupported values raise `ValueError`,
including for empty masks.

Under `torch.distributed`, every rank must call `compute()`, including ranks without any local
updates. All ranks receive the gathered scores; if no rank has cases, all raise `RuntimeError`.
Use an evaluation sampler without padding so repeated cases do not bias the average. With NCCL,
set each rank's current CUDA device before computing.

## Training losses

Losses are PyTorch modules taking floating-point logits `(B, C, *spatial)` and integer class
indices `(B, *spatial)`. Use `target_format="one-hot"` for masks or soft labels with the logits'
shape. Softmax targets should sum to one over classes; sigmoid targets should lie in `[0, 1]`.
Target probabilities are used as supplied, without normalization or clamping. For binary
segmentation with a single output channel, use sigmoid and a same-shaped target.

```python
from exacta.losses import DiceLoss, DiceCELoss, TverskyLoss, FocalLoss

dice_ce = DiceCELoss(include_background=False)
tversky = TverskyLoss(alpha=0.3, beta=0.7, include_background=False)
focal = FocalLoss(gamma=2.0)  # softmax, no class weighting
binary_focal = FocalLoss(activation="sigmoid", target_format="one-hot", alpha=0.25)
loss = tversky(logits, labels)
loss.backward()
```

For probabilities `p` and targets `g`, the definitions and reductions are explicit:

| Loss | Definition and reduction |
|---|---|
| `DiceLoss` | `1 - (2 Σ p·g + s) / (Σ p + Σ g + s)`, averaged over cases and classes. |
| `DiceCELoss` | `lambda_dice * DiceLoss + lambda_ce * CE`; CE is ordinary cross-entropy for softmax, BCE for sigmoid, and always includes every channel. |
| `DiceFocalLoss` | `lambda_dice * DiceLoss + lambda_focal * FocalLoss`. |
| `GeneralizedDiceCELoss` | `lambda_dice * GeneralizedDiceLoss + lambda_ce * CE`; sigmoid uses BCE. |
| `TverskyLoss` | `1 - (TP+s)/(TP+alpha*FP+beta*FN+s)`, with `TP=Σ p·g`, `FP=Σ p·(1-g)`, `FN=Σ (1-p)·g`. Average over cases and classes. |
| `GeneralizedDiceLoss` | `1 - (2 Σ_c w_c Σ p_c·g_c + s)/(Σ_c w_c Σ (p_c+g_c) + s)`, averaged over cases; weights derive from target class volumes. |
| `FocalLoss`, softmax | `-Σ_c w_c*g_c*(1-p_c)^gamma*log(p_c)`, averaged over cases and voxels. |
| `FocalLoss`, sigmoid | `-w_pos*g*(1-p)^gamma*log(p) - w_neg*(1-g)*p^gamma*log(1-p)`, averaged over cases, channels and voxels. |

Dice and Tversky sum over spatial axes, or batch and spatial axes with `batch_dice=True`.
`include_background=False` omits channel 0 after activation. Their `smooth` defaults to `1e-5`.
Tversky's `alpha` weights false positives and `beta` false negatives; both default to `0.5` and
need not sum to one. At those defaults, Tversky equals Dice with **twice** the smoothing
constant. The asymmetric overlap objective follows
[Salehi et al. (2017)](https://arxiv.org/abs/1706.05721).

`GeneralizedDiceLoss` follows [Sudre et al. (2017)](https://arxiv.org/abs/1707.03237) with
`weight_type="square"` (inverse squared target volume), `"simple"` (inverse volume), or
`"uniform"` (unit weights). It computes a ratio of weighted sums over classes, rather than
DiceLoss's mean of classwise ratios. With `batch_dice=True`, volumes and weights cover the
whole local batch. Smoothing is added **after** weighting and can materially affect the result
when inverse-volume weights are small.

Absent classes receive the largest weight among present selected classes. For inverse-volume
weighting, `empty_target="zero"` follows MONAI: cases with no selected target class contribute
zero loss, including when predictions contain false positives. `empty_target="uniform"`
instead assigns unit weights to these cases; use positive smoothing for a useful gradient on
an entirely empty target. Zero-weight cases return zero even with `smooth=0`; other zero
denominators remain NaN. Soft targets use the product-based formula above. No background
channel is added implicitly for single-channel sigmoid inputs.

```python
from exacta.losses import GeneralizedDiceLoss

loss_fn = GeneralizedDiceLoss(include_background=False, empty_target="uniform")
loss = loss_fn(logits, labels)
```

Focal loss downweights easy examples using `gamma` (default `2`), following
[Lin et al. (2017)](https://arxiv.org/abs/1708.02002). `alpha=None` gives unit weights. A scalar
`alpha` in `[0, 1]` assigns sigmoid positive/negative weights `alpha`/`1-alpha`; for softmax,
channel 0 gets `1-alpha` and every other class gets `alpha`. With `gamma=0, alpha=None`, focal
loss equals CE/BCE, including its reduction. Softmax focal loss is therefore `C` times MONAI's
default mean for the same hard targets. The formulas above define soft targets as weighted
sums of hard-label losses; sigmoid soft-target behavior differs from MONAI's modulation of a
single soft-label BCE term. Focal loss expects finite logits and includes every channel.

The overlap and focal losses compute half/bfloat16 inputs in float32 and retain float64 precision.
Their `backend="auto"` uses fused Triton on CUDA when available and PyTorch elsewhere. Use
`backend="torch"` for higher-order derivatives; the fused backward explicitly rejects them.
Targets requiring gradients automatically select PyTorch. Eager execution rejects out-of-range
indices; compiled Triton execution returns NaN for them without a host synchronization.

For repeated CPU training steps, `torch.compile` can reduce allocations and fuse native
PyTorch operations. Compilation adds startup cost; benchmark the warmed-up loss on your shapes:

```python
import torch

loss_fn = torch.compile(DiceCELoss(backend="torch"), fullgraph=True)
```

### Ignored labels and voxel masks

Overlap, CE and focal losses accept constructor argument `ignore_index` for integer targets
and a keyword-only boolean `mask=` in forward:

```python
loss_fn = DiceCELoss(ignore_index=-100)
loss = loss_fn(logits, target, mask=annotated_voxels)
```

The mask has shape `(B, *spatial)` or `(B, 1, *spatial)` on the input device. True includes
a voxel; ignored labels and False entries exclude it from all classes. CE/focal average
only included voxels; overlap averages omit entirely ignored cases. An entirely ignored
batch returns zero loss and gradients. Annotated background remains valid. GPU kernels
apply exclusions during their existing passes without full-volume masked intermediates.
See the [loss guide](https://github.com/rodolfo-viana/exacta/wiki/Losses) for a complete example.

### Boundary, topology and hard-example losses

Version 0.0.1a4 adds these objectives, with explicit choices for exponents, reductions and
geometry. The [advanced segmentation guide](https://github.com/rodolfo-viana/exacta/wiki/Advanced-segmentation)
documents their reference conventions and backend behavior.

| Loss | Definition and requirements |
|---|---|
| `FocalTverskyLoss` | Mean of `(1-Tversky)**gamma`; default actual exponent `0.75`, FP weight `0.3`, FN weight `0.7`. Supports masks, ignored labels and global distributed reduction. |
| `TopKCELoss` | Mean CE over the hardest `ceil(fraction*N_valid)` voxels; default fraction `0.1`. Integer targets and softmax. |
| `DiceTopKCELoss` | Weighted Dice plus top-k CE, with explicit coefficients and the same hard-example selection. |
| `LovaszSoftmaxLoss` | Multiclass Jaccard surrogate using stable error sorting; present/all classes and per-image/local-batch options. |
| `BoundaryLoss` | Mean probability times signed target distance in physical units; may be negative. |
| `HausdorffDTLoss` | Mean squared probability error weighted by detached prediction/target distance transforms. This surrogate differs from HD95. |
| `SoftClDiceLoss` | Differentiable finite-iteration skeleton overlap; C++ morphology on eager CPU and PyTorch pooling on CUDA. |

Geometry losses require complete binary target geometry. Their optional masks affect reduction
only; they have no `ignore_index`. Soft-clDice accepts neither ignored labels nor masks because
removing voxels changes connectivity. Cache distance maps or soft skeletons only while target
geometry and spacing remain unchanged. Uniform masks can yield zero Hausdorff-DT distances;
combine geometry/topology terms with regional supervision. Apart from Focal Tversky, these new
objectives reduce each rank's local batch rather than pooling statistics globally.

### Shared evaluation and training composition

Version `0.0.1a3` adds these APIs:

| API | Purpose |
|---|---|
| `functional.evaluate`, `metrics.MetricCollection` | Share counts and surface distances across several requested metrics |
| `losses.DeepSupervisionLoss` | Weighted loss over explicitly paired prediction/target resolutions and optional masks |
| `losses.DiceFocalLoss`, `losses.GeneralizedDiceCELoss` | Combinations with shared GPU passes and explicit term coefficients |
| `class_weight` in Dice–CE and generalized Dice–CE | CE class weights with explicit index/soft-target normalization |
| `distributed=True` in segmentation losses | Global statistics and DDP-correct parameter gradients; overlap also requires `batch_dice=True` |

See the Wiki's [metrics](https://github.com/rodolfo-viana/exacta/wiki/Metrics) and
[losses](https://github.com/rodolfo-viana/exacta/wiki/Losses) guides for examples, reduction
definitions and distributed participation requirements. CPU and single-GPU CUDA validation
includes two Gloo ranks.
**The multi-GPU NCCL path is experimental and unverified in this release:** the two-GPU
tests have not been run on suitable hardware. Gloo checks do not establish NCCL correctness
or multi-GPU training throughput.

## Backends and performance

Exacta uses compiled C++ CPU kernels for metric operations and Triton for supported CUDA
paths. Training losses use native PyTorch on CPU and fused Triton kernels on supported CUDA
inputs. The CPU extension is independent of the PyTorch binary version. GPU kernels compile
at runtime; their first invocation can include compilation overhead.

Mesh distances run on CPU, including for CUDA inputs. Measure device transfers and startup
costs when choosing a backend for your workload. Performance depends on volume size, class
count, dtype and hardware. The Wiki's
[performance guide](https://github.com/rodolfo-viana/exacta/wiki/Performance) contains measured
CPU/GPU results, memory usage and backend guidance.

## License

Apache License 2.0; see the [license](https://github.com/rodolfo-viana/exacta/blob/main/LICENSE),
[notices](https://github.com/rodolfo-viana/exacta/blob/main/NOTICE) and
[third-party licenses](https://github.com/rodolfo-viana/exacta/blob/main/THIRD_PARTY_LICENSES).
