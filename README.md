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
from exacta.metrics import HD95, Dice

# Training
loss_fn = DiceCELoss(include_background=False)        # logits (B, C, D, H, W), labels (B, D, H, W)
loss = loss_fn(model(x), y)

# Validation: accumulate per-case values across batches (and processes under torch.distributed)
dice = Dice(num_classes=4, include_background=False)
hd95 = HD95(num_classes=4, include_background=False)
for x, y, spacing in val_loader:
    pred = model(x).argmax(1)
    dice.update(pred, y)
    hd95.update(pred, y, spacing=spacing)             # per-case voxel size in mm, (B, 3)

result = hd95.compute()
result.per_case    # (N, C)
result.per_class   # (C,)
result.mean        # scalar

# Stateless functions accept PyTorch, NumPy, CuPy or JAX arrays and return the same type
EF.hd95(pred_np, target_np, spacing=(1.5, 0.8, 0.8), num_classes=4)  # -> np.ndarray (B, C)
```

## Conventions

Each convention is an explicit argument, because this is where libraries disagree.

| Topic | exacta |
|---|---|
| Input layout | `input_format="index"`: class indices `(B, *spatial)` + `num_classes`. `"one-hot"`: binary masks `(B, C, *spatial)`. Metrics take hard labels; losses take logits. |
| Spacing | Required for distance metrics, in the axis order of the tensor. A SimpleITK array is `(z, y, x)` while `GetSpacing()` returns `(x, y, z)`. |
| Boundary | Mask voxels with a face-adjacent neighbour outside the mask; outside the image counts as background. |
| HD95 | `max(P95(d(P→T)), P95(d(T→P)))`, linear-interpolated percentile, as in MONAI. MedPy pools both directions instead. |
| Class absent from both | `both_empty`, default NaN (skipped when averaging). |
| Class absent from one | Dice/IoU give 0. Distance metrics give `one_empty`, default `inf`. |
| Aggregation | Per case, then the NaN-ignoring mean over cases, then over classes. |

## Architecture

- Pure PyTorch for anything that reduces to tensor operations (most losses and overlap metrics).
  Overlap metrics on label maps use a single confusion-matrix pass, so memory does not grow
  with the number of classes.
- Compiled kernels (`csrc/`) for algorithms that don't, such as the exact Euclidean distance
  transform behind HD/HD95. Kernels sit behind a plain C interface and receive arrays through
  DLPack, so the compiled extension never links against PyTorch and one wheel works across
  PyTorch versions.
- On GPU, distance metrics currently use an exact distance transform written in PyTorch
  (separable, O(n) work per voxel per axis). CUDA kernels and Triton-fused losses come next.

## Development

```bash
python -m venv ~/.venvs/exacta && source ~/.venvs/exacta/bin/activate
pip install torch scikit-build-core nanobind cmake ninja
pip install --no-build-isolation -e ".[test]"   # re-run after editing csrc/
pytest
python benchmarks/bench_metrics.py
```
