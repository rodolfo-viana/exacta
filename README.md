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

Status: early development (v0.0.1). The first release targets 3D medical image segmentation.

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
- GPU kernels in Triton, compiled at run time, so installing exacta needs no CUDA toolkit or host
  compiler. `backend="auto"` (the default everywhere) picks them on GPU; `backend="torch"`
  remains as a fallback.
  - The distance transform runs the same algorithm as the CPU kernel and gives the same values,
    bit for bit. Labelling gives the same labels as the CPU kernel and SciPy.
  - The overlap metrics count voxels in per-block histograms, checking the values as they go.
  - `DiceLoss` and `DiceCELoss` run as two fused kernels, one forward and one backward, that
    keep nothing for the backward pass but the logits: 7-22x faster than the PyTorch path on 3-D
    batches, without its 0.3-2.1 GiB of intermediates (`benchmarks/bench_losses.py`). They
    work under `torch.compile` without graph breaks.
- PyTorch for the glue between kernels, the CPU path of the losses, and the fallbacks.

## Benchmarks

AMD Ryzen 9 5900XT (16 cores), NVIDIA RTX 5070, PyTorch 2.14.1, CUDA 13.0. Times in milliseconds,
median of 5 runs (one run for anything slower than 5 s). Reproduce with
`python benchmarks/bench_metrics.py --sizes 128 256` (add `--only exacta` to skip the slow
references) and `python benchmarks/bench_losses.py`.

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

`DiceCELoss`, forward plus backward on GPU, index targets. Extra memory is the peak allocation
beyond the logits and their gradient.

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

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
