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
| `DiceFocalLoss` | `lambda_dice * DiceLoss + lambda_focal * FocalLoss` |
| `GeneralizedDiceCELoss` | `lambda_dice * GeneralizedDiceLoss + lambda_ce * CE`; sigmoid uses BCE |
| `TverskyLoss` | `1 - (TP+s)/(TP+alpha*FP+beta*FN+s)`, averaged across cases and selected classes |
| `GeneralizedDiceLoss` | Ratio of class-weighted overlap sums; class weights derive from target volumes |
| `FocalLoss` | Focal CE/BCE with exponent `gamma`; softmax sums classes then averages voxels, sigmoid averages all entries |

Overlap losses support `activation="softmax"`, `"sigmoid"` or `"none"`. Dice-CE and focal
require logits and use softmax or sigmoid. `batch_dice=True` pools overlap statistics across
the local batch before division. The explicit global option is documented below.
`include_background=False` excludes channel zero from overlap only. CE/BCE and focal include
all channels.

## Overlap equations

For one case and selected class, let $p_i$ be the activated prediction and $g_i$ the target
at included voxel $i$. With smoothing $s$, the soft overlap statistics are:

$$
\mathrm{TP}=\sum_i p_i g_i,\qquad
\mathrm{FP}=\sum_i p_i(1-g_i),\qquad
\mathrm{FN}=\sum_i (1-p_i)g_i.
$$

Dice and Tversky use these ratios before averaging over cases and selected classes:

$$
\mathcal{L}_{\mathrm{Dice}}
=1-\frac{2\sum_i p_i g_i+s}{\sum_i p_i+\sum_i g_i+s},\qquad
\mathcal{L}_{\mathrm{Tversky}}
=1-\frac{\mathrm{TP}+s}{\mathrm{TP}+\alpha\mathrm{FP}+\beta\mathrm{FN}+s}.
$$

Generalized Dice instead takes a single class-weighted ratio per case, where $c$ ranges
over selected classes:

$$
\mathcal{L}_{\mathrm{GDice}}
=1-\frac{2\sum_c w_c\sum_i p_{ic}g_{ic}+s}
{\sum_c w_c\sum_i(p_{ic}+g_{ic})+s}.
$$

For present target classes, `weight_type="square"` sets $w_c=(\sum_i g_{ic})^{-2}$;
`"simple"` uses the inverse volume and `"uniform"` uses one. The absent-class rules below
handle zero volumes. `batch_dice=True` includes the batch axis in the sums before taking
the ratio. Smoothing is added once per ratio, after weighting for generalized Dice.

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
selects PyTorch when targets require gradients. Full-graph `torch.compile` is tested for
local losses. Distributed losses are tested in eager execution; compiled collectives are
not part of the current compatibility guarantee.

## Ignored labels and voxel masks

Every loss accepts constructor argument `ignore_index` for index targets and keyword-only `mask=` in forward.
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

## Loss combinations and CE weights

`DiceFocalLoss` adds Dice and focal terms; `GeneralizedDiceCELoss` adds generalized Dice and
CE/BCE. Their GPU implementations share one activation/target pass forward and one backward.
The corresponding native PyTorch compositions provide a reference and higher-order gradients.
Coefficients must be finite and nonnegative, with at least one positive term. A coefficient
of zero skips that term, including otherwise undefined zero-smoothing overlap ratios.

```python
from exacta.losses import DiceFocalLoss, GeneralizedDiceCELoss

dice_focal = DiceFocalLoss(include_background=False, gamma=2, alpha=0.25,
                          lambda_dice=1, lambda_focal=0.5)
gdl_ce = GeneralizedDiceCELoss(include_background=False, empty_target="uniform",
                              class_weight=[0.2, 1.0, 2.0])
```

`DiceCELoss` and `GeneralizedDiceCELoss` accept `class_weight`, one nonnegative finite value
per input channel, stored as a module buffer. These weights apply only to CE/BCE. Normalization
matches [PyTorch's cross-entropy definition](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)
and depends on target format:

| Pointwise term | Weighted numerator | Denominator |
|---|---|---|
| Softmax, index target | Sum of `-w[y] log p[y]` over included voxels | Sum of included `w[y]` |
| Softmax, soft/one-hot target | Sum of `-w[c] g[c] log p[c]` | Number of included voxels |
| Sigmoid BCE | Class weight multiplies both positive and negative terms | Included voxels × channels |

Zero total target weight returns zero CE and zero CE gradient. Exact one-hot and index labels
therefore produce different weighted CE scales, as in PyTorch. `include_background=False`
still affects only the overlap term. Use focal's `alpha` for its documented positive/negative
or background/foreground weighting.

## Deep supervision

`DeepSupervisionLoss` applies one scalar loss to explicitly paired resolutions. Weights are
used as supplied, without normalization; zero-weight levels are not evaluated and receive
no gradient. Targets and optional masks must already match each prediction's spatial shape.
The wrapper does not resize labels or choose downsampling conventions.

```python
from exacta.losses import DeepSupervisionLoss, DiceCELoss

loss_fn = DeepSupervisionLoss(DiceCELoss(ignore_index=-100), weights=[1.0, 0.5, 0.25])
# Sequences of three tensors, ordered consistently from fine to coarse:
loss = loss_fn(predictions, targets, masks=voxel_masks)
```

## Global distributed losses

Set `distributed=True` to pool statistics over the default `torch.distributed` process group.
Overlap losses and their combinations also require `batch_dice=True`. This computes the
objective for the concatenated global batch, including generalized-Dice class weights and
weighted CE denominators. It differs from averaging independently computed rank-local Dice.

```python
# After initializing the process group and wrapping the model in ordinary DDP:
loss_fn = GeneralizedDiceCELoss(batch_dice=True, distributed=True,
                               include_background=False, ignore_index=-100)
loss = loss_fn(ddp_model(images), labels, mask=annotated)
loss.backward()
```

Every rank must use identical loss configuration and backend, and call forward and backward
in the same order, including ranks with zero cases or entirely ignored data. An entirely
ignored or empty global batch returns zero. Do not multiply the loss by local batch size or
divide by world size: the differentiable statistics sum and DDP's gradient averaging together
produce concatenated-batch **parameter gradients**; see
[DDP's gradient-averaging convention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html).
Logit/target gradients before DDP averaging
carry a world-size factor; custom gradient communication hooks need their own scaling analysis.
Gradient accumulation across microbatches averages microbatch objectives; it does not pool
their nonlinear Dice ratios into one larger-batch ratio.

Only small statistics communicate. NCCL keeps these on the rank's CUDA device. Gloo uses CPU
buffers, also allowing two CUDA workers on a single GPU for correctness tests. Local tests
cover CPU and fused CUDA with two Gloo ranks, uneven batches, empty ranks, ignored ranks and
all-ignored batches. **The multi-GPU NCCL path is experimental and unverified in 0.0.1a3:**
the two-GPU tests have not been run on suitable hardware. Gloo checks do not establish NCCL
correctness or multi-GPU training throughput. Loss configuration and valid input shapes
must agree across ranks; as with ordinary distributed training, a rank-local error can interrupt its peers.

## Unreleased additions

Development on `feature/advanced-segmentation` adds Focal Tversky, top-k CE, Dice/top-k CE,
Lovasz-Softmax, boundary, Hausdorff-DT and soft-clDice objectives. These are not in PyPI
0.0.1a3. The loss docstrings in the
[feature source](https://github.com/rodolfo-viana/exacta/tree/feature/advanced-segmentation/src/exacta/losses)
describe their APIs, masking restrictions and empty-mask definitions.
