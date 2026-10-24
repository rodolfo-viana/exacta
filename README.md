# exacta

Fast, reference-validated losses and metrics for deep learning, on CPU and GPU.

Every project ends up re-implementing Dice, HD95 and friends, and the implementations quietly
disagree: [Podobnik & Vrtovec (2024)](https://arxiv.org/abs/2410.02630) compared 11 open-source
tools and found that distance-based metrics such as HD95 and NSD can differ substantially across
tools for the same segmentation pair. exacta aims to be the one implementation you can trust and
cite: every metric states its exact definition and is tested against independent reference
implementations.

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

# Match another tool's definition explicitly
EF.hd95(pred, target, spacing=s, num_classes=4, combine="pooled")    # MedPy's hd95
EF.assd(pred, target, spacing=s, num_classes=4, combine="mean")      # MedPy 0.4's assd
EF.nsd(pred, target, spacing=s, num_classes=4, tolerance=1.0, surface="surfels")  # DeepMind's NSD

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
| Surface | `surface="voxels"` (default): boundary voxels, i.e. mask voxels with a face-adjacent neighbour outside the mask, each with equal weight, as in MONAI and MedPy. `"surfels"`: DeepMind's surface elements at voxel corners, weighted by their marching-cubes area (MONAI's `use_subvoxels=True`); percentiles become area-weighted, as in DeepMind. Outside the image counts as background. |
| HD95 | `combine="max"` (default): `max(P95(d(P→T)), P95(d(T→P)))`, as in MONAI and DeepMind; linear-interpolated percentile for voxels. `"pooled"`: one percentile over both directions, as in MedPy. |
| ASSD | `combine="pooled"` (default): mean over the surface elements of both masks, as in MONAI and MedPy 0.5.2. `"mean"`: mean of the two directed means, as in MedPy 0.4 (DeepMind returns the two directed means). |
| NSD | Fraction of the surface of both masks within `tolerance` of the other surface, `d ≤ tolerance`, as in MONAI and DeepMind. Each distance is the exact floating-point minimum of the expression SciPy evaluates, so distances that tie with the tolerance land on the same side as in MedPy and DeepMind. MONAI compares in float32. |
| Class absent from both | `both_empty`, default NaN (skipped when averaging). |
| Class absent from one | Dice/IoU give 0. Distance metrics give `one_empty`: default `inf` for HD/ASSD, 0 for NSD. |
| Aggregation | Per case, then the NaN-ignoring mean over cases, then over classes. |
| Connectivity | Required for connected components: 1 = faces (6 neighbours in 3-D, SciPy's default), 2 = and edges (18), 3 = and corners (26, MONAI's default). |

## Architecture

- Pure PyTorch for anything that reduces to tensor operations: overlap metrics (a single
  confusion-matrix pass on label maps, so memory does not grow with the number of classes) and
  the CPU path of the losses.
- Compiled CPU kernels (`csrc/`) for algorithms that don't: the exact Euclidean distance
  transform behind HD95, ASSD and NSD, and run-based connected-component labelling. They sit
  behind a plain C interface and receive arrays through DLPack, so the extension never links
  against PyTorch and one wheel works across PyTorch versions.
- GPU kernels in Triton, compiled at run time, so installing exacta needs no CUDA toolkit or host
  compiler. `backend="auto"` (the default everywhere) picks them on GPU; `backend="torch"`
  remains as a fallback.
  - The distance transform runs the same algorithm as the CPU kernel and gives the same values,
    bit for bit.
  - `DiceLoss` and `DiceCELoss` run as two fused kernels, one forward and one backward, that
    keep nothing for the backward pass but the logits: 7-13x faster than the PyTorch path on 3-D
    batches, without its 0.3-1.1 GiB of intermediates (`benchmarks/bench_losses.py`).

## Development

```bash
python -m venv ~/.venvs/exacta && source ~/.venvs/exacta/bin/activate
pip install torch scikit-build-core nanobind cmake ninja
pip install --no-build-isolation -e ".[test]"   # re-run after editing csrc/
pytest
python benchmarks/bench_metrics.py
python benchmarks/bench_losses.py
```

Reference tests against MONAI, MedPy and DeepMind's surface-distance are skipped unless those
packages are installed:

```bash
pip install monai medpy "git+https://github.com/google-deepmind/surface-distance.git"
```

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
