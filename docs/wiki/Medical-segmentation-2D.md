# 2D medical-image segmentation

Exacta's segmentation losses and overlap, boundary, topology and probability metrics accept
2D images as well as 3D volumes. This guide uses synthetic images to demonstrate the input
contracts; it does not measure model quality. Physical area and independent-label probability
APIs described here are **unreleased**, following PyPI 0.0.1a4.

## Choose the target representation

Use exclusive class indices for mutually exclusive structures. Use independent binary
channels when structures overlap or when a single channel represents foreground only.

| Task | Logits | Loss target | Hard metric prediction/target |
|---|---|---|---|
| Binary, one foreground channel | `(B, 1, H, W)` | Binary `(B, 1, H, W)` with `target_format="one-hot"` | Thresholded binary channels, `input_format="one-hot"` |
| Mutually exclusive classes | `(B, C, H, W)` | Integer `(B, H, W)` with `target_format="index"` | Integer maps and explicit `num_classes=C` |
| Overlapping labels | `(B, C, H, W)` | Independent binary `(B, C, H, W)` with `target_format="one-hot"` | Independent binary channels, `input_format="one-hot"` |

The name `one-hot` also accepts overlapping binary channels; it does not require their sum
to equal one. Keep `include_background=True` for foreground-only channels: setting it to
`False` drops channel zero. Thresholds are part of the evaluation protocol; select them on
validation data and keep them fixed for the held-out test set.

## Exclusive structures: train and evaluate

This example computes Dice plus cross-entropy, then shares hard counts and boundary work.
Pixel spacing follows the tensor's spatial axis order `(H, W)`, in millimetres. Obtain it
from image metadata after resampling; Exacta does not read image headers or infer spacing.

```python
import torch
import exacta.functional as EF
from exacta.losses import DiceCELoss

rng = torch.Generator().manual_seed(24)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logits = torch.randn((2, 3, 32, 40), generator=rng).to(device).requires_grad_()
target = torch.randint(3, (2, 32, 40), generator=rng).to(device)
loss = DiceCELoss(include_background=False)(logits, target)
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()

prediction = logits.detach().argmax(1)
scores = EF.evaluate(
    prediction, target, num_classes=3, include_background=False, spacing=(0.5, 0.8),
    metrics={"dice": {}, "hd95": {}, "nsd": {"tolerance": 1.0},
             "predicted_area": {}, "target_area": {}, "area_error": {"absolute": True}},
)
assert all(value.shape == (2, 2) for value in scores.values())
# Dice and NSD are dimensionless; HD95 is in mm; areas and area error are in mm².
```

The default boundary representation uses boundary-pixel centres. `surface="surfels"` weights
2D contour elements by their physical length; `surface="mesh"` uses contour-segment geometry
in 2D. These representations have different definitions, so record the choice alongside
HD95's direction-combination convention. See [Metrics](https://github.com/rodolfo-viana/exacta/wiki/Metrics).

## Binary foreground or overlapping structures

Use sigmoid logits for independent labels. Dice plus binary cross-entropy is available through
the same `DiceCELoss` class. The example below uses two potentially overlapping structures;
set `channels=1` for a single foreground mask.

```python
channels = 2
logits = torch.randn((2, channels, 32, 40), generator=rng).to(device).requires_grad_()
target = (torch.rand((2, channels, 32, 40), generator=rng) > 0.7).to(device)
loss = DiceCELoss(activation="sigmoid", target_format="one-hot")(logits, target)
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()

prediction = logits.detach() >= 0  # equivalent to sigmoid >= 0.5
areas = EF.predicted_area(prediction, target, input_format="one-hot", spacing=(0.5, 0.8))
brier = EF.brier_score(logits.detach(), target, from_logits=True, multilabel=True)
assert areas.shape == brier.shape == (2, channels)
```

Existing focal/Tversky and generalized Dice variants, Lovasz losses, boundary/distance-transform
surrogates, soft clDice and deep supervision also accept 2D inputs under their documented
contracts. Choose them according to the error you need to penalize, and measure their forward
and backward cost for your image size. See [Losses](https://github.com/rodolfo-viana/exacta/wiki/Losses)
and [Advanced segmentation](https://github.com/rodolfo-viana/exacta/wiki/Advanced-segmentation).

## Physical area and partial annotation

For a rectangular pixel grid with spacing $(s_H,s_W)$, one pixel has area $a=s_Hs_W$ mm².
For observed-domain prediction $P$ and target $T$, Exacta reports

$$
A_P=a|P|,\qquad A_T=a|T|,\qquad E=A_P-A_T.
$$

`absolute=True` reports $|E|$; `unit="cm2"` divides mm² by 100. These are occupied pixel-cell
areas, independent of the contour representation used for boundary metrics. Overlapping
channels have independent areas; adding them double-counts their intersections. Spacing can
be a scalar, a shared pair or a `(B, 2)` array. The area APIs reject 3D inputs; use physical
volume APIs with all three axis spacings for reconstructed volumes.

`ignore_index` and shared boolean masks exclude pixels from overlap and area evaluation.
An absent class in an observed image has zero area. An entirely unobserved image returns
NaN. These results describe the annotated domain, not the missing tissue.

```python
partial_target = torch.tensor([[[0, 1, -1], [0, 1, -1]]], device=device)
partial_prediction = torch.tensor([[[0, 1, 99], [1, 1, 99]]], device=device)
partial_scores = EF.evaluate(
    partial_prediction, partial_target, num_classes=2, include_background=False,
    ignore_index=-1, spacing=(0.5, 0.8), metrics=["dice", "predicted_area", "target_area"],
)
assert torch.allclose(partial_scores["predicted_area"], torch.tensor([[1.2]], device=device, dtype=torch.float64))
```

Boundary and clDice metrics require complete masks: treating unannotated tissue as background
creates artificial boundaries and can alter connectivity. Their masked evaluation is rejected.

## Choose the unit of analysis

Exacta treats each batch item as one case. If items are slices, stateful `Dice`, `HD95`,
`PredictedArea` and `MetricCollection` average slice scores equally, skipping NaNs per class.
They do not know which slices belong to the same patient. For a patient-level report, keep
patient identifiers outside the metric and specify the aggregation in advance:

- **Mean patient slice score:** average each patient's slice scores first, then average
  patients equally. Report the number of observed slices and patients for each class.
- **Patient volume score:** reconstruct each patient's prediction and target in their original
  3D geometry, including slice order and spacing, then evaluate one volume per patient.
- **Pooled pixel score:** combine pixel statistics. This weights larger images and patients
  with more pixels differently. `VoxelRanking` pools pixels; `AveragePrecision` and `AUROC`
  average per-case scores instead.

These estimands are different. For example, patients with slice Dice scores `[1, 0]` and `[1]`
have a pooled slice mean of $2/3$, but an equal-patient mean of $3/4$. Neither mean is generally
3D Dice. A mean of slice HD95 values is likewise not a 3D HD95. Summing areas alone is not a
physical volume: through-plane geometry is also required.

Keep all slices from one patient in the same data split. When reporting uncertainty across
patients, resample patients as groups; do not treat correlated slices as independent patients.
Record absent-class and NaN handling so differences in annotation coverage remain visible.
