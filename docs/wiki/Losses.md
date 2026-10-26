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

The following options describe the released objectives. Development-only squared Dice and
Lovasz-hinge and Wasserstein Dice additions are described at the end of this page.

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

Dice, Tversky, generalized Dice, focal and their regional combinations accept constructor
argument `ignore_index` for index targets and keyword-only `mask=` in forward. Focal Tversky,
top-k CE and Lovasz-Softmax also support exclusions. Geometry/topology losses have different
requirements, described below.
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
all-ignored batches. **The multi-GPU NCCL path is experimental and unverified in 0.0.1a4:**
the two-GPU tests have not been run on suitable hardware. Gloo checks do not establish NCCL
correctness or multi-GPU training throughput. Loss configuration and valid input shapes
must agree across ranks; as with ordinary distributed training, a rank-local error can interrupt its peers.

## Boundary, topology and hard-example objectives

Version 0.0.1a4 adds `FocalTverskyLoss`, `TopKCELoss`, `DiceTopKCELoss`, `LovaszSoftmaxLoss`,
`BoundaryLoss`, `HausdorffDTLoss` and `SoftClDiceLoss`. Their formulas, reductions, defaults,
reference conventions and CPU/GPU paths are documented in the
[advanced segmentation guide](https://github.com/rodolfo-viana/exacta/wiki/Advanced-segmentation).

Focal Tversky supports the global distributed reduction above. Top-k/Lovasz selection and
geometry/topology losses operate on each rank's local batch. Geometry masks affect reduction
only; complete target geometry is required. Geometry/topology losses have no `ignore_index`,
and soft-clDice has no mask argument. Distance-transform and skeleton objectives should be
combined with regional supervision according to the task.

## Squared Dice denominator (unreleased)

The development version adds `squared_pred=True` to `DiceLoss`, `DiceCELoss`, `DiceFocalLoss`
and `DiceTopKCELoss`. Both prediction and target terms in the denominator are squared:

$$
\mathcal{L}_{\mathrm{squared\ Dice}}
=1-\frac{2\sum_i p_i g_i+s}{\sum_i p_i^2+\sum_i g_i^2+s}.
$$

The numerator retains the product overlap, including for soft targets. With binary targets,
$g_i^2=g_i$. The ordinary denominator remains the default. Smoothing, background exclusion,
masking and case/batch reduction follow ordinary Dice. The option is named consistently with
[MONAI's squared-prediction option](https://monai.readthedocs.io/en/stable/losses.html#diceloss);
the equation above defines Exacta's soft-target convention explicitly.

```python
from exacta.losses import DiceCELoss

loss_fn = DiceCELoss(squared_pred=True, include_background=False)
```

PyTorch supports differentiation through predictions and soft targets. The fused CUDA path
includes the squared denominator in both forward statistics and analytic backward, and
supports global distributed Dice/CE reductions. Fused higher-order derivatives remain
unsupported; use `backend="torch"` for those. Dice-top-k still selects hard examples locally.

## Binary and multilabel Lovasz-hinge (unreleased)

`LovaszHingeLoss` follows the binary real-valued-score formulation in the
[authors' implementation](https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py).
Given binary labels $y_i$ and raw logits $z_i$, margin errors are
$e_i=1-z_i(2y_i-1)$. Sort the errors decreasingly, and let $\Delta J_j$ be successive
increments of the Jaccard error for that ordering. The objective is:

$$
\mathcal{L}_{\mathrm{Lovasz\ hinge}}=\sum_j\max(e_{(j)},0)\,\Delta J_j.
$$

Binary input has shape `(B,1,*spatial)` with target `(B,*spatial)`. Independent overlapping
labels use `multilabel=True` and a same-shaped binary target. No sigmoid is applied and
soft targets are rejected. Masks can be shared or per-label; `ignore_index` applies only
to integer binary targets. Excluded invalid logits/targets do not affect the result.

`per_image=True` (default) averages observed case/label losses. False pools the local batch
per label before averaging labels. An absent positive class still penalizes false positives;
unobserved labels are omitted, and entirely excluded inputs give differentiable zero.
Stable native PyTorch sorting uses flattened voxel order for ties. The subgradient at a
zero hinge margin is zero. This is a local objective under DDP, without global sorting.

```python
from exacta.losses import LovaszHingeLoss

binary_loss = LovaszHingeLoss(ignore_index=-1)
region_loss = LovaszHingeLoss(multilabel=True, per_image=True)
```

These loss additions support both 2D and 3D medical segmentation. They are alternative
objectives with different gradients, not a claim of improved accuracy for every task.

## Generalized Wasserstein Dice (unreleased)

`GeneralizedWassersteinDiceLoss` adds explicit costs for confusing mutually exclusive labels.
It follows the crisp-target formulation of [Fidon et al. (2017)](https://arxiv.org/abs/1707.00478),
with the alternative `GDL` weighting described by
[MONAI](https://monai.readthedocs.io/en/stable/losses.html#generalizedwassersteindiceloss).
It supports 2D images and 3D volumes. Costs describe relationships between labels; voxel
spacing and physical boundary distances do not enter this objective.

Let $M$ be the supplied class-cost matrix divided by its positive maximum, $y_i$ the integer
target at observed voxel $i$, and $p_{ic}$ the softmax probability for class $c$. The expected
cost against that crisp target is

$$
w_i=\sum_c M_{y_i,c}\,p_{ic}.
$$

For each case, define $N_c=\sum_i\mathbf{1}[y_i=c]$ over the observed domain and
$T=\sum_i\alpha_{y_i}(1-w_i)$. The two modes use different weights and error sums:

| `weighting_mode` | Class weights $\alpha_c$ | Error $E$ |
|---|---|---|
| `"default"` | $\alpha_0=0$; $\alpha_c=1$ for $c>0$ | $\sum_i w_i$ |
| `"GDL"` | $1/(N_c+1)$, including label zero | $\sum_i\alpha_{y_i}w_i$ |

Both return the following loss, with the same smoothing constant $s$ in both terms:

$$
\mathcal{L}=1-\frac{2T+s}{2T+E+s}.
$$

The constructor requires a finite, nonnegative, symmetric square matrix with zero diagonal
and a positive maximum. `default` additionally requires maximal costs between background
(label zero) and every foreground class, matching the assumption behind the simplified
foreground formula. `GDL` permits other background costs. Triangle inequalities are not
checked; a zero off-diagonal cost makes that label confusion cost-free. Specify these costs
from the task's label relationships, and assess any tuning using training/validation data.

The following synthetic example gives confusion between foreground labels a smaller cost
than confusion with background. The same loss accepts `(B,C,H,W)` or `(B,C,D,H,W)` logits.

```python
import torch
from exacta.losses import GeneralizedWassersteinDiceLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = torch.Generator().manual_seed(879)
logits = torch.randn((2, 3, 16, 20), generator=rng).to(device).requires_grad_()
target = torch.randint(3, (2, 16, 20), generator=rng).to(device)
target[0, :2] = -1

loss_fn = GeneralizedWassersteinDiceLoss(
    [[0, 1, 1], [1, 0, 0.25], [1, 0.25, 0]],
    weighting_mode="default", ignore_index=-1,
).to(device=device, dtype=logits.dtype)
loss = loss_fn(logits, target)
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
```

The default `reduction="mean"` averages observed cases equally. `"sum"` sums case losses;
`"none"` returns `(B,)`. Fully excluded cases and empty inputs return differentiable zero,
and are omitted from the mean. A zero denominator with `smooth=0` also returns zero.
With default weights and `smooth=0`, an observed background-only case with any nonzero error
has loss one and no useful foreground Dice gradient; positive smoothing or the `GDL` mode
changes that behavior. This differs from an entirely unobserved case.

This API takes integer targets and logits; it has no sigmoid/multilabel, soft-target or
cross-rank pooling option. It supports shared boolean masks, ignored indices, half/bfloat16
computation in float32, float64, higher derivatives and full-graph `torch.compile` on CPU/CUDA.
Eager calls reject out-of-range observed labels. Compiled calls return NaN for invalid labels
without unsafe indexing. Excluded labels/nonfinite logits are sanitized before activation.

The implementation uses native PyTorch gather/reduction kernels and an integer class histogram,
without expanding targets to one-hot. Move the loss to the logits' device/dtype once and
benchmark `torch.compile(loss_fn)` with forward **and backward** on your workload. The cost
matrix is a detached normalized buffer, included in `state_dict`; it is not a learnable parameter.
