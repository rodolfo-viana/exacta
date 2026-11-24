# Advanced segmentation objectives and evaluation

Exacta includes boundary-sensitive training, hard-example selection, tubular
structure objectives, probability quality and confidence-ranked lesion detection. Every function
states its aggregation and empty-input convention. No loss is claimed to improve MRI
segmentation quality without a task-specific experiment.

## Training objectives

Inputs are floating network outputs `(B,C,D,H,W)` or `(B,C,H,W)`. Integer targets are
`(B,*spatial)`; geometry/topology losses also accept binary channel masks with
`target_format="one-hot"`, including single-channel sigmoid segmentation.

| Class in `exacta.losses` | Definition and important choices |
|---|---|
| `FocalTverskyLoss` | Mean of `(1-TI)**gamma`, **before** averaging cases/classes. Defaults: FP weight .3, FN weight .7, actual exponent .75. Gamma is not its reciprocal. Exact-zero endpoint uses zero gradient for gamma != 1; gamma 1 retains the ordinary Tversky derivative. Shares the existing fused GPU overlap kernels, masks, ignored labels and opt-in global distributed reduction. |
| `TopKCELoss` | Mean softmax CE over the hardest `ceil(fraction*N_valid)` voxels in the local batch. Default fraction .1; fraction 1 is ordinary unweighted CE. Index targets only; masks/ignored labels excluded before selection. |
| `DiceTopKCELoss` | Unnormalized `lambda_dice*Dice + lambda_ce*TopKCE`. Index targets/softmax. Zero coefficients skip terms. Dice uses the existing backend dispatcher; selection uses native PyTorch. |
| `LovaszSoftmaxLoss` | Lovasz extension of multiclass Jaccard error. Stable error sorting; present classes by default, optional all classes; optionally per-image rather than local-batch sorting. Supports masks and ignored index labels. |
| `BoundaryLoss` | Mean `p * signed_distance(target)`. Can be negative. Signed distances use physical voxel-centre distances, negative inside, without a half-voxel correction or the original code's subtraction of one inside. Combine with a regional loss. |
| `HausdorffDTLoss` | Mean `(p-y)^2*(d_p**alpha+d_y**alpha)`, default alpha 2. Distances use detached per-channel masks `p > .5`, including softmax outputs. This is the distance-transform surrogate, not HD or HD95. |
| `SoftClDiceLoss` | Soft-skeleton precision/sensitivity harmonic mean, averaged over cases/classes. Cross-shaped erosion, full-neighbourhood dilation, explicit iteration count (10 by default), positive smoothing (1 by default). |

Focal Tversky follows the exponent convention in the
[authors' implementation](https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py).
Top-k selection follows the hard-voxel idea used by
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/loss/robust_ce_loss.py),
with an explicit fraction and ceil rule. Lovasz follows
[Berman et al.](https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html).
These choices do not imply identical defaults to every reference library.

Boundary loss follows the signed-distance objective of
[Kervadec et al.](https://proceedings.mlr.press/v102/kervadec19a.html), with the voxel-centre
convention above. An empty target receives the positive physical image diagonal; the
image exterior is background for inside distances, so full targets remain finite.
`HausdorffDTLoss` follows the surrogate of
[Karimi and Salcudean](https://arxiv.org/abs/1904.10030), with
[MONAI's uniform-mask convention](https://github.com/Project-MONAI/MONAI/blob/dev/monai/losses/hausdorff_loss.py):
entirely foreground/background masks have distance zero. Opposite uniform masks can
therefore produce zero surrogate loss; combine it with Dice or CE.

Cache geometry only when the target and spacing have not changed. Cropping, resampling,
elastic deformation and other geometric augmentations can invalidate a cache. There is
no hidden cache inside a loss object. For example, with known `logits` and `labels`:

```python
import torch
from exacta.losses import BoundaryLoss, DiceCELoss, distance_maps

channels = logits.shape[1]
truth = torch.nn.functional.one_hot(labels, channels).movedim(-1, 1)
signed = distance_maps(truth[:, 1:], spacing=(1.0, 1.0, 2.0), signed=True)
loss = DiceCELoss(include_background=False)(logits, labels)
loss = loss + 0.01 * BoundaryLoss()(logits, labels, distance_map=signed)
loss.backward()
```

The coefficient .01 is an example, not a tuned recommendation: physical units change the
scale of geometry losses. Forward `spacing` accepts a scalar, a shared spatial vector or
`(B,ndim)`. `distance_maps` returns float64 on the input device using C++ CPU or exact Triton
CUDA EDT; losses cast maps to their computation dtype. Half/bfloat16 inputs compute in float32.

Geometry loss masks affect **reduction only**; complete target geometry is still required.
Hausdorff prediction geometry also uses the full image, so all logits must be finite.
There is no `ignore_index` for geometry/topology losses. Removing unannotated voxels would
create boundaries or alter connectivity. Soft-clDice has no mask option. All-ignored batches
return differentiable zero for the new losses that support ignored labels.

`SoftClDiceLoss` follows the finite-iteration pooling surrogate of
[Shit et al.](https://arxiv.org/abs/2003.07311). Pooling excludes the image exterior, matching
the authors' code. Thick structures may need more iterations; full masks may have empty
soft skeletons. This objective is intended to complement regional supervision. A cached
soft target skeleton can be supplied as `target_skeleton`; use `soft_skeletonize` with the
same iteration count and selected channels.

Only Focal Tversky adds global `distributed=True` reductions. Top-k/Lovasz selection and
geometry/topology objectives operate on each rank's local batch. DDP does not turn local
hard-example selection into a global sort. Native Torch paths support autograd; the fused
Focal Tversky path has the same double-backward restriction as existing overlap losses.

## Hard centreline Dice

`functional.cldice` and `metrics.ClDice` measure hard skeleton overlap using **Lee thinning**.
`functional.evaluate(..., metrics=["dice", "cldice"])` and `MetricCollection` also accept it.
The C++ implementation matches scikit-image's `skeletonize(method="lee")`; its tables and
attribution are included in `THIRD_PARTY_LICENSES`. This is separate from soft-clDice.

Skeleton precision is `|S_pred ∩ target|/|S_pred|`, sensitivity exchanges prediction/target,
and clDice is their harmonic mean. One empty mask gives 0; both empty default to NaN (configurable).
Lee thinning can erase some even-width objects: a nonempty mask with an empty skeleton is
assigned 0, including an identical-mask comparison in that case. This choice avoids reporting
perfect topology when no centreline survives. Counts are in voxel space; anisotropy is not
silently corrected by resampling.

`functional.skeletonize(mask)` accepts one binary 2D/3D mask. Lee's sequential deletion
recheck runs in C++ on CPU. CUDA inputs transfer to CPU and back; it is not GPU-native hard
thinning. Precomputed `pred_skeleton`/`target_skeleton` for clDice have shape
`(B,selected_C,*spatial)` and must be binary subsets of the masks. Soft training skeletons
stay on the input device. Eager CPU soft skeletons use fused C++ morphology plus native
PyTorch scatter-add for autograd; CUDA and `backend="torch"` use PyTorch pooling. At ties,
the C++ path selects the first raster neighbour, while PyTorch can split erosion gradients
between equal minima. These are valid, potentially different subgradients. `torch.compile`
with the default backend uses the PyTorch path so its tensor kernels can be compiled.

**Unreleased (`0.0.1a6.dev0`): thinning method.** The clDice authors compute their metric with
scikit-image's `skeletonize`, which uses Zhang-Suen thinning in 2D and Lee thinning in 3D.
`skeletonize(mask, method=None)` now follows that default, and `cldice`, `ClDice` and the
collection option accept `method="zhang"` (2D only) or `"lee"`. The new C++ Zhang kernel
reproduces scikit-image 0.26's lookup table and two-subiteration schedule exactly; that table
differs from the textbook Zhang-Suen conditions in 25 of its 256 entries, so it is copied, with
attribution, rather than derived. In 2D the two methods give different skeletons, and therefore
different clDice: pass `method="lee"` to reproduce `0.0.1a5` values.

## Skeleton Recall (unreleased, `0.0.1a6.dev0`)

`SkeletonRecallLoss` implements the loss of
[Kirchhoff et al. (ECCV 2024)](https://doi.org/10.1007/978-3-031-72980-5_13). With $s_c$ the
tubed target skeleton of class $c$ and $p_c$ the predicted probabilities,

$$
\mathcal{L}_{\mathrm{SR}}=1-\frac{1}{\lvert\mathcal{C}\rvert}\sum_{c\in\mathcal{C}}
\frac{\sum_i p_{c,i}s_{c,i}+s}{\sum_i s_{c,i}+s},
$$

averaged over cases unless `batch_dice=True` sums the batch first. The authors' code returns the
negative mean recall, which differs by the constant 1. Background is excluded by default, as
the authors require for softmax outputs. A class without skeleton voxels has recall 1 and no
gradient; `mask` and `ignore_index` exclude voxels from both sums, and `distributed=True`
reduces the global batch. The loss rewards covering thin structures end to end. It does not
thin predictions, so its GPU cost is one product and two sums. Add it to a regional loss:
the authors use Dice + CE + Skeleton Recall with unit weights.

`skeleton_recall_target(target)` builds $s$ as the authors' nnU-Net transform does. The whole
foreground (labels above 0) is thinned as one mask, the skeleton is dilated by `tube=2` steps
of the face neighbourhood (in voxel steps, regardless of spacing), and every tube voxel keeps
its target label; tube voxels on background become 0. `union=False` thins each class on its
own, which differs where classes touch. One-hot targets default to separate channels, as
overlapping labels require. Thinning runs on the CPU, so compute targets in the data loader after
augmentation and pass them as `target_skeleton`; otherwise `forward` thins every target. The
targets are tested against the authors' transform written with scikit-image, in 2D (Zhang) and
3D (Lee).

```python
from exacta.losses import DiceCELoss, SkeletonRecallLoss, skeleton_recall_target

skeleton = skeleton_recall_target(target)          # in the data loader, on CPU
regional = DiceCELoss(include_background=False)
recall = SkeletonRecallLoss()
loss = regional(logits, target) + recall(logits, target, target_skeleton=skeleton.to(target.device))
```

## Betti numbers and Betti error (unreleased, `0.0.1a6.dev0`)

`functional.betti_numbers(seg, connectivity=...)` returns $(\beta_0,\beta_1)$ in 2D (components,
holes) and $(\beta_0,\beta_1,\beta_2)$ in 3D (components, tunnels, cavities) for every case and
class. `connectivity` is required: 1 (4/6 neighbours) or `ndim` (8/26). The background takes the
complementary neighbourhood, the pairing under which digital and continuous topology agree;
18-connectivity has no such partner and is rejected. The image exterior is background.

$\beta_0$ counts foreground components. The last Betti number counts background components of
the zero-padded image other than the exterior. In 3D, $\beta_1=\beta_0+\beta_2-\chi$, where the
Euler characteristic $\chi$ is counted on the cell complex that matches the connectivity:
closed voxels for full connectivity, open voxels for faces only. The results match
scikit-image's `label` and `euler_number` on random masks, and give $(1,1,0)$ for a solid torus
and $(1,0,1)$ for a hollow sphere.

`functional.betti_error(pred, target, connectivity=..., dimension=k)` and `metrics.BettiError`
report $\lvert\beta_k(P)-\beta_k(T)\rvert$ as `(B, K)` float64; `dimension=0` is the Betti-0
error of the TopCoW challenges. `functional.evaluate` and `MetricCollection` accept
`"betti_error"` with its options; masks are rejected, because removing voxels changes topology.
Betti errors ignore location: a missed component and a spurious one cancel. Report them
next to an overlap or centreline metric.

## Probability quality and calibration

`functional.brier_score`, `negative_log_likelihood`, `expected_calibration_error` and
`reliability_diagram` accept probabilities by default, or logits with `from_logits=True`.
Targets are integer labels. Masks and ignored labels are supported. C=1 means sigmoid/binary
(targets 0/1); C>=2 means softmax/multiclass (probabilities must sum to one).

| Quantity | Reduction |
|---|---|
| Brier | Multiclass: mean of sum of squared class errors, range [0,2]. One-channel binary: mean squared error, range [0,1]. Equivalent one-/two-channel binary forms therefore differ by two. |
| NLL | Mean true-class negative log probability in nats. Logits use stable operations; probability zero for the true class yields +inf, without clipping. |
| Top-label ECE | Bin maximum confidence against prediction correctness; binary confidence .5 predicts class 0, matching two-channel argmax ties. Returns `(B,1)`. |
| Classwise ECE | Bin each class probability against binary class membership. Returns `(B,C)`; C=1 means positive-class probability. |
| Reliability summary | Counts, mean confidence, mean correctness/target membership, bin edges and ECE. Equal-width bins `[k/n,(k+1)/n)`, final bin includes 1. Empty bin means and empty-case scores are NaN. |

Brier and NLL are proper probability scores, not pure calibration measures. ECE depends on
binning; a large background can hide foreground errors. These distinctions matter in medical
segmentation; see [Mehrtash et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7704933/).

`metrics.BrierScore`, `NegativeLogLikelihood` and `ExpectedCalibrationError` average **cases**
through the existing per-case accumulator. `metrics.Calibration` instead streams pooled
bin sums and weights **voxels** equally. Pooled ECE is not mean per-case ECE. Both support
cross-rank computation; all ranks must participate, including ranks without updates.

```python
from exacta.metrics import Calibration

calibration = Calibration(bins=15, mode="classwise", from_logits=True)
for logits, labels in validation_batches:
    calibration.update(logits, labels)
diagram = calibration.compute()
# diagram.counts/confidence/accuracy: (1,C,15); diagram.ece: (1,C)
```

## Confidence-ranked lesion detection

`exacta.detection.LesionDetection` streams lesion events; `lesion_detection` evaluates one
batch as a dataset. These complement the existing hard-mask `lesions.evaluate` API.
Inputs are hard segmentations plus a same-shape floating confidence map in [0,1]. Each
connected component is a candidate scored by its maximum (default) or mean confidence.

Within each case/class, candidates are processed in descending confidence. Each matches the
highest-IoU unused target at or above `iou_threshold` (default .1). Ties use component raster
order. All confidence ties form one curve step. Candidate extraction is fixed before the
confidence sweep: this API does not re-segment a probability map at every threshold.

AP is `sum(delta_recall * precision)` without a precision envelope. Missed ground-truth lesions
remain in the recall denominator. FROC uses false positives per case and lesion recall;
`sensitivity_at` returns the best achieved recall within each FP/case budget, without
interpolation. Include lesion-free scans in the dataset denominator. No targets yields NaN
AP/recall; targets with no predictions yields AP=0. These rules are explicit and are **not**
a claim of PI-CAI/challenge-protocol equivalence.

```python
from exacta.detection import LesionDetection

evaluator = LesionDetection(num_classes=2, connectivity=3, iou_threshold=0.1)
for pred_labels, target_labels, confidence in validation_batches:
    evaluator.update(pred_labels, target_labels, confidence)
result = evaluator.compute()
curve = result.curves[0]  # foreground class 1
print(curve.average_precision, curve.sensitivity_at([0.5, 1.0, 2.0]))
```

Connected components/overlaps reuse C++ CPU and Triton CUDA kernels. Small lesion tables move
to CPU for deterministic matching/sorting. The accumulator keeps scores, matches and counts,
not volumes, and gathers those tables across distributed ranks. Confidence-map extraction,
connected-component connectivity and overlap threshold are part of an evaluation protocol
and must be reported alongside AP/FROC.

## Performance guidance

Cache target distance maps or soft skeletons only while geometry and spacing remain unchanged.
Benchmark cache construction separately from reuse. The
[performance guide](https://github.com/rodolfo-viana/exacta/wiki/Performance) covers measured
CPU/GPU behavior. Results depend on volume dimensions, class count, precision and hardware;
these implementation checks do not establish segmentation quality on an MRI dataset.
