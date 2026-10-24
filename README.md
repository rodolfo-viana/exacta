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

Status: early development (v0.0.1a2). Current releases target 3D medical image segmentation.

Documentation: [GitHub Wiki](https://github.com/rodolfo-viana/exacta/wiki) ·
[versioned sources](https://github.com/rodolfo-viana/exacta/tree/dev/docs/wiki).
The guides cover installation, a complete quickstart, metrics, losses, mathematical conventions,
lesion evaluation and performance. Unreleased features are labeled separately from the PyPI release.

## Installation

Install the alpha from PyPI with Python 3.10 or newer:

```bash
python -m pip install exacta==0.0.1a2
```

This release is a **source distribution**. Installation builds the native extension locally
and requires a C++17 compiler and Python development headers; OpenMP is optional. On
Debian/Ubuntu, install `build-essential` and the development headers matching your Python
version first. Portable Linux x86-64 wheel builds are available in development CI; they
will be published with the next release.

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

## Conventions

Each convention is an explicit argument, because this is where libraries disagree.

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

All losses compute half/bfloat16 inputs in float32 and retain float64 precision.
`backend="auto"` uses fused Triton on CUDA when available and PyTorch elsewhere. Use
`backend="torch"` for higher-order derivatives; the fused backward explicitly rejects them.
Targets requiring gradients automatically select PyTorch. Eager execution rejects out-of-range
indices; compiled Triton execution returns NaN for them without a host synchronization.

For repeated CPU training steps, `torch.compile` can reduce allocations and fuse native
PyTorch operations. Compilation adds startup cost; benchmark the warmed-up loss on your shapes:

```python
import torch

loss_fn = torch.compile(DiceCELoss(backend="torch"), fullgraph=True)
```

## Architecture

- Compiled CPU kernels (`csrc/`), multithreaded with OpenMP, for every pass over the voxels that
  dominates a metric: the exact Euclidean distance transform behind HD95, ASSD and NSD; surface
  extraction (boundary voxels, neighbour codes, the marching-cubes mesh) and point-to-mesh
  distances; the voxel counts of the overlap metrics, in one pass that also checks the values;
  connected-component labelling and the label statistics behind `largest`, `remove_small` and
  lesion matching; dilation; and the sort behind area-weighted percentiles. They sit behind a
  plain C interface and receive arrays through DLPack, so the extension never links against
  PyTorch and one wheel works across PyTorch versions. They are built without fused
  multiply-adds, which would round reference expressions differently.
- GPU kernels in Triton, compiled at run time without the CUDA toolkit.
  `backend="auto"` (the default everywhere) picks them on GPU; `backend="torch"`
  remains as a fallback.
  - The distance transform runs the same algorithm as the CPU kernel and gives the same values,
    bit for bit. Labelling gives the same labels as the CPU kernel and SciPy.
  - The overlap metrics count voxels in per-block histograms, checking the values as they go.
  - Dice, Dice-plus-CE, Tversky, generalized Dice and focal losses fuse voxel operations in forward and backward,
    retaining the inputs and small reduction statistics instead of full-volume activations or
    one-hot intermediates. They work under `torch.compile` without graph breaks. CPU losses
    use PyTorch's compiled operators and autograd; `torch.compile` can fuse them further.
- PyTorch for the glue between kernels, the CPU path of the losses, and the fallbacks.

## Benchmarks

AMD Ryzen 9 5900XT (16 cores), NVIDIA RTX 5070, PyTorch 2.14.1, CUDA 13.0. Times are in
milliseconds. Metric timings use the median of 5 runs (one run for anything slower than 5 s).
Reproduce them with `python benchmarks/bench_metrics.py --sizes 128 256` (add `--only exacta`
to skip the slow references). Loss measurements and commands are described below.

### Metrics

One synthetic 3-D prediction/target pair with 4 classes, spacing 1.5 × 0.8 × 0.8 mm. The
references run on CPU (MONAI's GPU path needs cuCIM, which is not installed here).

| Metric | Volume | exacta CPU | exacta GPU | MONAI | Other reference |
|---|---|---:|---:|---:|---|
| Dice | 128³ | 0.5 | 0.3 | | NumPy 10.0 |
| | 256³ | 7.0 | 0.7 | | NumPy 84 |
| HD95 | 128³ | 46 | 12 | 711 | SciPy EDT 2,607 |
| | 256³ | 529 | 86 | 9,600 | SciPy EDT 18,486 |
| NSD, voxels | 128³ | 43 | 11 | 697 | |
| | 256³ | 525 | 87 | 9,548 | |
| NSD, surfels | 128³ | 63 | 13 | 1,547 | DeepMind 1,600 |
| | 256³ | 653 | 96 | 29,869 | DeepMind 12,580 |
| HD95, mesh | 128³ | 253 | | | MeshMetrics 17,484 |
| | 256³ | 1,954 | | | MeshMetrics 190,634 |
| Connected components, 26-conn. | 128³ | 0.5 | 0.5 | | SciPy 14.8 |
| | 256³ | 6.0 | 3.4 | | SciPy 109 |
| Largest component, 26-conn. | 128³ | 1.5 | 0.9 | 21.2 | |
| | 256³ | 11.0 | 5.2 | 159 | |
| BraTS 2023 lesion-wise, 3 regions | 240×240×155 | 24 | 31 | | BraTS-2023-Metrics 7,700 |

"SciPy EDT" is exacta's test reference (`tests/reference.py`): the textbook implementation on
`scipy.ndimage.distance_transform_edt`. exacta's plain-PyTorch GPU fallback (`backend="torch"`)
takes 3,090 ms for HD95 at 256³. Mesh distances run on the CPU, also for CUDA inputs. The fastest
CPU kernels spend much of their time writing fresh output memory, so their times depend on how
quickly the operating system supplies it: labelling at 256³ takes 6-13 ms from run to run. The BraTS
row scores the regions WT, TC and ET of one synthetic case from the lesion test fixtures' generator
(`tests/fixtures/make_lesion_cases.py`); BraTS-2023-Metrics needs an environment of its own (see
below) and was timed there, on the same case.

### Losses

The current benchmark uses identical seeded inputs for eager PyTorch, `torch.compile` over
`backend="torch"`, and fused Triton. Compilation and 5 warm-up steps are excluded; results are
medians of 20 forward-plus-backward steps with synchronization. Default loss arguments,
softmax activation, integer targets, float32 logits of shape `2×4×128³`, RTX 5070:

| Loss | PyTorch (ms) | Compiled PyTorch (ms) | Triton (ms) | Extra MiB: PyTorch / compiled / Triton |
|---|---:|---:|---:|---:|
| Dice + CE | 5.643 | 3.105 | 0.847 | 320 / 144 / <0.1 |
| Tversky | 6.980 | 1.918 | 0.818 | 256 / 80 / <0.1 |
| Focal | 8.628 | 2.056 | 0.614 | 560 / 128 / <0.1 |
| Generalized Dice | 4.184 | 3.521 | 0.938 | 256 / 128 / <0.1 |

Extra memory is the peak allocation during the step beyond its inputs and logits' gradient;
small scratch buffers can fit below that baseline. These are local synthetic measurements,
not end-to-end training speedups. CPU measurements on the same system, with 16 threads and
`2×4×64³` float32 logits, use the same seed, warm-up and repeat count:

| Loss | PyTorch (ms) | Compiled PyTorch (ms) |
|---|---:|---:|
| Dice + CE | 15.746 | 10.248 |
| Tversky | 13.157 | 9.574 |
| Focal | 30.743 | 6.378 |
| Generalized Dice | 10.541 | 4.985 |

Reproduce these tables with:

```bash
python benchmarks/bench_losses.py --classes 4 --sizes 128 --dtypes float32 --repeat 20 --warmup 5 --seed 0
python benchmarks/bench_losses.py --device cpu --classes 4 --sizes 64 --dtypes float32 \
    --backends torch compiled --threads 16 --repeat 20 --warmup 5 --seed 0
```

The earlier Dice-plus-CE scaling measurements below use the median of 10 warmed-up GPU steps,
integer targets and the same peak-memory definition. The benchmark also supports bfloat16,
float64, sigmoid activation and other batch, class and volume sizes.

| Logits | dtype | PyTorch time | PyTorch extra memory | Fused time | Fused extra memory |
|---|---|---:|---:|---:|---:|
| 2×4×128³ | float32 | 5.6 | 320 MiB | 0.8 | 0 |
| 2×4×160³ | float32 | 11.0 | 629 MiB | 1.2 | 1 MiB |
| 2×14×128³ | float32 | 21.3 | 1,120 MiB | 1.6 | 0 |
| 2×14×160³ | float32 | 41.9 | 2,191 MiB | 3.0 | 1 MiB |
| 2×4×128³ | bfloat16 | 6.1 | 320 MiB | 0.7 | 0 |
| 2×4×160³ | bfloat16 | 12.0 | 629 MiB | 0.9 | 0 |
| 2×14×128³ | bfloat16 | 23.1 | 1,120 MiB | 1.2 | 0 |
| 2×14×160³ | bfloat16 | 45.4 | 2,190 MiB | 2.1 | 0 |

## Development

Feature development takes place on `dev`; `main` is reserved for releases. The current
development version is `0.0.1a3.dev0`. See the
[segmentation roadmap](docs/segmentation-roadmap.md) for priorities.

```bash
python -m venv ~/.venvs/exacta && source ~/.venvs/exacta/bin/activate
pip install torch scikit-build-core nanobind cmake ninja
pip install --no-build-isolation -e ".[test]"   # re-run after editing csrc/
pytest
python benchmarks/bench_metrics.py
python benchmarks/bench_losses.py
```

Reference tests against MONAI, MedPy, DeepMind's surface-distance, MeshMetrics, VTK and
SimpleITK are skipped unless those packages are installed:

```bash
pip install monai medpy scikit-image vtk SimpleITK \
    "git+https://github.com/google-deepmind/surface-distance.git" \
    "git+https://github.com/gasperpodobnik/MeshMetrics.git"
```

Lesion-wise evaluation is tested against fixtures produced by Panoptica 2.1.7 and by the BraTS
2023 evaluation code, which need environments of their own (old NumPy and pandas).
`tests/fixtures/regenerate.sh` rebuilds those environments with `uv`, reruns both tools and
reports whether their results changed (set `PANOPTICA_VERSION` or `BRATS_COMMIT` to try other
releases).

Array-library round-trip tests run for NumPy, and for CuPy and JAX when installed:
`python -m pytest tests/test_inputs.py -k round_trip`. Distributed tests exercise ranks without
updates using Gloo; their NCCL variants require two GPUs and are skipped otherwise.

Loss checks include independent NumPy formulas, MONAI comparisons with explicit reduction
adjustments, float64 finite-difference gradient checks, higher-order PyTorch gradients,
CPU/GPU agreement, mixed precision, extreme logits and full-graph compilation:
`python -m pytest tests/test_losses.py tests/test_weighted_losses.py tests/test_generalized_dice.py tests/test_masked_losses.py`.

### Unreleased: ignored labels and voxel masks

On `dev`, all segmentation losses accept constructor argument `ignore_index` for integer targets
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

### Unreleased: shared evaluation and training composition

The `dev` branch also provides these APIs, pending the next PyPI release:

| API | Purpose |
|---|---|
| `functional.evaluate`, `metrics.MetricCollection` | Share counts and surface distances across several requested metrics |
| `losses.DeepSupervisionLoss` | Weighted loss over explicitly paired prediction/target resolutions and optional masks |
| `losses.DiceFocalLoss`, `losses.GeneralizedDiceCELoss` | Combinations with shared GPU passes and explicit term coefficients |
| `class_weight` in Dice–CE and generalized Dice–CE | CE class weights with explicit index/soft-target normalization |
| `distributed=True` in segmentation losses | Global statistics and DDP-correct parameter gradients; overlap also requires `batch_dice=True` |

See the Wiki's [metrics](https://github.com/rodolfo-viana/exacta/wiki/Metrics) and
[losses](https://github.com/rodolfo-viana/exacta/wiki/Losses) guides for examples, reduction
definitions and distributed participation requirements. Run
`python -m pytest tests/test_collection.py tests/test_deep_supervision.py tests/test_composed_losses.py tests/test_distributed_losses.py`
to exercise these additions; CUDA/NCCL tests require suitable hardware.

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
