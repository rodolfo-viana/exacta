# Metrics

Stateless functions in `exacta.functional` return per-case, per-selected-class arrays of shape
`(B, K)`. Corresponding classes in `exacta.metrics` accumulate results across batches.
Choose complementary metrics according to the segmentation task; a high overlap score alone
does not describe lesion detection or boundary errors.

| Functional name | Stateful class | Definition |
|---|---|---|
| `dice` | `Dice` | `2 TP / (2 TP + FP + FN)` |
| `iou` | `IoU` | `TP / (TP + FP + FN)` |
| `sensitivity` | `Sensitivity` | `TP / (TP + FN)` |
| `precision` | `Precision` | `TP / (TP + FP)` |
| `specificity` | `Specificity` | `TN / (TN + FP)` |
| `volume_similarity` | `VolumeSimilarity` | Default: `1 - abs(Vp - Vt)/(Vp + Vt)` |
| `relative_volume_difference` | `RelativeVolumeDifference` | `(Vp - Vt)/Vt`; `absolute=True` removes the sign |
| `hausdorff_distance` | `HausdorffDistance` | Percentile boundary distance; default percentile 100 |
| `hd95` | `HD95` | 95th-percentile boundary distance |
| `assd` | `ASSD` | Average symmetric boundary distance |
| `nsd` | `NSD` | Fraction of both surfaces within the supplied tolerance |

Inputs use class indices `(B, *spatial)` with `num_classes`, or binary masks `(B, C, *spatial)`
with `input_format="one-hot"`. `include_background=False` omits channel/class zero.
Stateless functions accept PyTorch, NumPy, CuPy and JAX arrays and return the prediction's
array type. Stateful metrics store PyTorch results.

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
surface representations and reference-library differences. Loss masking on `dev` does not
add masking to these metrics; excluding a region from a surface metric needs its own definition.
