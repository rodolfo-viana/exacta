# Losses

Losses are PyTorch modules accepting floating-point logits `(B, C, *spatial)` and integer
targets `(B, *spatial)`. Use `target_format="one-hot"` for same-shaped hard masks or soft labels.
Softmax targets should sum to one over classes; sigmoid targets should lie in `[0, 1]`.
Exacta does not silently normalize or clamp target probabilities.

Let `p` be activated predictions, `g` the target and `s` smoothing. The released losses are:

| Loss | Definition and reduction |
|---|---|
| `DiceLoss` | `1 - (2 sum(p*g)+s)/(sum(p)+sum(g)+s)`, averaged across cases and selected classes |
| `DiceCELoss` | `lambda_dice * DiceLoss + lambda_ce * CE`; sigmoid uses BCE |
| `TverskyLoss` | `1 - (TP+s)/(TP+alpha*FP+beta*FN+s)`, averaged across cases and selected classes |
| `GeneralizedDiceLoss` | Ratio of class-weighted overlap sums; class weights derive from target volumes |
| `FocalLoss` | Focal CE/BCE with exponent `gamma`; softmax sums classes then averages voxels, sigmoid averages all entries |

Overlap losses support `activation="softmax"`, `"sigmoid"` or `"none"`. Dice-CE and focal
require logits and use softmax or sigmoid. `batch_dice=True` pools overlap statistics across
the local batch before division; it does not communicate between distributed training ranks.
`include_background=False` excludes channel zero from overlap only. CE/BCE and focal include
all channels.

## Important parameters

Definitions differ across libraries, so select parameters intentionally:

- Dice/Tversky smoothing defaults to `1e-5`. Tversky's default `alpha=beta=0.5` equals Dice
  with twice the smoothing constant; alpha weights false positives and beta false negatives.
- Generalized Dice supports `weight_type="square"`, `"simple"` and `"uniform"`. Smoothing
  is added after class weighting. Absent classes receive the largest present-class weight.
  With inverse-volume weighting, `empty_target="zero"` gives zero for entirely absent selected
  targets; `"uniform"` uses unit weights there. Positive smoothing is needed for a useful
  gradient on an entirely empty target. This differs from an entirely ignored case.
- Focal uses `gamma=2` and `alpha=None` by default. Scalar alpha weights sigmoid positives by
  alpha and negatives by `1-alpha`; softmax assigns `1-alpha` to class zero and alpha to other
  classes. `gamma=0, alpha=None` reproduces CE/BCE. Softmax reduction differs from MONAI's mean
  over channels by a factor of the channel count.

## Backends and derivatives

`backend="auto"` selects fused Triton for supported CUDA inputs and PyTorch otherwise.
Half/bfloat16 inputs compute in float32; float64 inputs retain their precision. Fused losses
support first derivatives. Use `backend="torch"` for higher-order derivatives; `auto` also
selects PyTorch when targets require gradients. Full-graph `torch.compile` is supported.

## Unreleased: ignored labels and voxel masks

This section requires the `dev` branch (`0.0.1a3.dev0`), not PyPI 0.0.1a2. Every loss accepts
constructor argument `ignore_index` for index targets and keyword-only `mask=` in forward.
Masks must be boolean, on the input device, with shape `(B, *spatial)` or `(B, 1, *spatial)`.
True includes a voxel; False excludes it from every class. Both exclusions combine.

```python
import torch
from exacta.losses import DiceCELoss

generator = torch.Generator().manual_seed(0)
logits = torch.randn(2, 3, 4, 4, 4, generator=generator, requires_grad=True)
target = torch.randint(3, (2, 4, 4, 4), generator=generator)
target[:, 0] = -100
mask = torch.ones_like(target, dtype=torch.bool)
mask[:, :, 0] = False
loss = DiceCELoss(ignore_index=-100)(logits, target, mask=mask)
loss.backward()
assert torch.count_nonzero(logits.grad[:, :, 0]) == 0
```

CE/focal normalize over included voxels (and channels for sigmoid). Overlap averages exclude
entirely ignored cases; a fully ignored batch returns zero loss and gradients even with zero
smoothing. Annotated background voxels remain valid observations. Ignored invalid labels and
nonfinite values are excluded before activation/reduction. Valid out-of-range labels still
raise eagerly; compiled Triton returns NaN. Soft/one-hot targets use `mask`, not `ignore_index`.

The GPU kernels read exclusion information directly and count valid voxels in their existing
passes. They do not create full-volume masked probability or one-hot buffers.
