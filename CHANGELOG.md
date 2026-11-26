# Changelog

Release notes for Exacta.

## Unreleased

Changes for the next release will be recorded here.

## 0.1.1 — 2026-11-25

Version 0.1.1 fixes a rare rounding defect in the exact squared Euclidean distance transform
used by every voxel surface metric.

- **Fix:** the squared Euclidean distance transform (CPU and Triton) could return a value one or
  two ulps above the exact floating-point minimum when several parabolas of a pass tie within
  rounding error far apart on a line. The envelope now removes a parabola only when one of its two
  neighbours lies below it everywhere by more than any rounding error, and evaluates every kept
  parabola whose breakpoints come within the tie tolerance of the sample. The failing lines were
  constructed adversarially from the worst case of the exactness argument; random lines, random
  masks and all label volumes of three Medical Segmentation Decathlon tasks gave the exact minimum
  with 0.1.0 as well. Surface distances built on the transform change only in such cases.

## 0.1.0 — 2026-11-25

Version 0.1.0 is the first release outside the alpha series. NSD now counts
distances that equal the tolerance in exact arithmetic, reporting signatures name the full
definition behind a score, and Dice and Tversky losses can reproduce
segmentation_models_pytorch's treatment of absent classes.

- Add `zero_absent=True` to `DiceLoss`, `TverskyLoss` and `FocalTverskyLoss`: classes whose target
  is empty within the reduction get zero loss and gradient but still count in the mean, as in
  segmentation_models_pytorch (its Dice and Tversky losses are reproduced with `batch_dice=True`,
  `smooth=0`). By default, as in MONAI, an empty target still penalizes predicted probability.
- Add reporting signatures: `exacta.conventions.signature("hd95")` returns
  `hd95[both_empty=nan;combine=max;one_empty=inf;surface=voxels]@exacta-<version>`, naming every
  convention behind a score with defaults filled in. `parse_signature` reads it back into options
  for `functional.evaluate`, which then recomputes the same score bit for bit.
  `signatures(selection)` and `MetricCollection.signatures()` cover several metrics and record
  dataset pooling.
- **Behavior change:** NSD now counts a distance as within the tolerance when
  `d <= tolerance * (1 + tie_rtol)`, with a new `tie_rtol` argument (default `1e-6`) on
  `functional.nsd`, `metrics.NSD` and the `nsd` options of `evaluate`/`MetricCollection`.
  Distances equal to the tolerance in exact arithmetic previously could fall outside it after
  rounding: with spacing 0.4 and tolerance 1.2, `fl(3 * 0.4) = 1.2000000000000002` excluded every
  boundary voxel lying exactly three voxels away, changing NSD by up to 0.04 on test volumes.
  `tie_rtol=0` restores the strict binary64 comparison used by SciPy-based tools such as MedPy
  and DeepMind's surface-distance; MONAI's float32 comparison counts such ties, as the new
  default does.

## 0.0.1a6 — 2026-11-24

This release broadens Exacta beyond per-image segmentation scores: dataset-level classification
and regression, instance label maps, Betti errors and dataset-pooled overlap. Training gains
Skeleton Recall and soft-label-compatible overlap losses. 2D hard clDice changes its default
thinning to match the clDice authors' metric.

- Add general classification and regression namespaces, separate from per-image
  segmentation semantics. Classification includes hard-decision rates, confusion
  matrices, top-k accuracy, MCC, Cohen kappa, probability scores and AP/AUROC.
- Add MAE, MSE, RMSE, R², explained variance, Pearson and concordance correlation,
  with multioutput reductions, boolean observation masks and sample weights.
- Add dataset-level streaming counts, rankings and centered regression moments,
  including distributed aggregation and ranks without updates. Scores are computed
  after pooling, not averaged across batches.
- Reuse C++/Triton classification counts, retain O(C) counts for large label spaces,
  and fuse CPU regression validation/errors/moments in C++ with float64 statistics.
  Native PyTorch supplies GPU regression and weighted classification reductions.
- Add differentiable masked quantile and stable log-cosh losses. Expose standard
  classification/regression losses directly from PyTorch without extra wrappers.
- Document numerical, tie-breaking, averaging and undefined-score conventions,
  with independent references, CPU/CUDA tests and a reproducible tabular benchmark.
- Add `aggregate="pooled"` to count-based segmentation metrics and `MetricCollection`: O(C)
  dataset totals of TP/FP/FN/TN, summed across ranks including empty ones, giving global Dice,
  per-class IoU and mIoU with mmsegmentation's absent-class convention. Expose per-case
  `functional.confusion_counts`, and add per-case and pooled F-beta (`fbeta`, `FBeta`).
- Add `exacta.instances` for instance label maps (touching instances, arbitrary values): IoU
  matching with unique strict matching above 0.5 and greedy or per-group Hungarian matching
  below it, precision/recall/F1/accuracy/SQ/PQ, AJI and AJI+ following HoVer-Net's
  procedures, the Cell Tracking Challenge SEG measure, the Data Science Bowl 2018 score, and
  per-case or pooled `InstanceMetrics` with distributed accumulation.
- Add Betti numbers and Betti errors (`betti_numbers`, `betti_error`, `BettiError`, collection
  support) for 2D/3D with paired 4/8 and 6/26 connectivities; 3D tunnels follow from an Euler
  characteristic counted on the matching closed or open cell complex.
- Add C++ Zhang thinning that reproduces scikit-image's 2D `skeletonize` exactly, and a
  `method` option for `skeletonize`, `cldice` and `ClDice`. **Behavior change:** 2D hard
  clDice now defaults to Zhang thinning, as scikit-image and the clDice authors' metric do;
  `method="lee"` reproduces 0.0.1a5 values. 3D is unchanged.
- Add `SkeletonRecallLoss` and `skeleton_recall_target` (Kirchhoff et al., ECCV 2024), matching
  the authors' tubed-skeleton transform and loss up to a constant, with masks, ignored labels,
  batch and distributed reductions.
- Add `soft_label=True` to Dice, Tversky, focal Tversky, generalized Dice and their CE/focal
  combinations: Dice semimetric statistics (Wang et al., MICCAI 2023), identical for hard
  labels and minimal at the target for soft labels, matching MONAI's `soft_label`.

## 0.0.1a5 — 2026-10-26

This release extends 2D and 3D segmentation evaluation with selected voxels, independent
probability channels, physical areas/volumes and voxel rankings. Training gains squared
Dice, Lovasz-hinge and class-distance Wasserstein Dice.

- Add boolean voxel masks and index-target `ignore_index` to Dice, IoU, sensitivity,
  precision, specificity, volume similarity and relative volume difference, including
  stateful metrics and shared collections.
- Exclude masked/ignored voxels from both label validation and all confusion counts,
  including true negatives. Fully excluded cases use each metric's zero-denominator policy
  (NaN by default). Surface and clDice collections reject these options because partial
  boundary/topology evaluation needs a separate definition.
- Fuse exclusion and counting in C++/OpenMP and Triton without expanding index targets or
  broadcasting masks into channel-sized buffers; retain a checked PyTorch fallback.
- Add explicit multilabel Brier, Bernoulli NLL, ECE and reliability bins, with per-label
  masks, macro/micro reductions and fused C++/CUDA statistics. Micro ECE pools bins before
  absolute gaps; existing binary/multiclass defaults retain their definitions.
- Add predicted/target 3D physical volume and signed/absolute error in mm³ or mL, including
  per-case spacing, stateful wrappers and shared collections. Unobserved domains give NaN.
- Add predicted/target 2D physical area and signed/absolute error in mm² or cm², reusing
  accelerated counts with masks, per-case spacing, stateful wrappers and shared collections.
- Preserve float64 results and int64 labels when converting to JAX under its default 32-bit
  configuration, without changing the caller's configuration. Validate shared spacing even
  for empty batches.
- Document binary, categorical and overlapping-label 2D medical segmentation, physical pixel
  spacing, partial annotation and patient-versus-slice aggregation.
- Refresh the README and Wiki for this release, including the complete API catalogue and
  executable CPU/CUDA quickstart examples.
- Add voxel PR/ROC curves, AP and AUROC with grouped ties and exact or approximate binned
  evaluation. Separate case-averaged metrics from dataset-pooled `VoxelRanking`; binned
  streaming retains fixed-size int64 histograms, including distributed accumulation.
- Add `GeneralizedWassersteinDiceLoss` for exclusive 2D/3D labels with validated, normalized
  class-distance matrices, foreground or GDL weighting, shared masks/ignored labels and
  per-case reductions. Native PyTorch supports mixed precision, higher derivatives and
  compilation without one-hot target expansion. Default weighting requires maximal
  background/foreground costs; unobserved cases contribute zero and are omitted from means.
- Add `squared_pred=True` to Dice and Dice-CE/focal/top-k CE, including fused CUDA derivatives,
  soft targets, masked observations and global distributed Dice/CE statistics.
- Add binary/multilabel `LovaszHingeLoss` with stable native PyTorch sorting, explicit
  per-image/local-batch reductions, ignored labels and per-label observation masks.
- Prevent the older PyTorch Gloo teardown deadlock in distributed tests by preserving the
  backend through outer group destruction; add a repeated-collective regression. Document
  PyTorch 2.8+ for distributed Gloo use; metric execution gains no synchronization overhead.
- Report test skip reasons and generate only applicable test parameter combinations;
  retain hardware and optional-reference skips as explicit coverage limitations.

## 0.0.1a4 — 2026-10-24

This release adds boundary and topology objectives, probability metrics and confidence-ranked
lesion evaluation, with explicit mathematical conventions and CPU/GPU backend choices.

- Restrict distributions to package/build files and licenses; keep documentation drafts,
  plans and measurement outputs local, and limit the README to the PyPI package description.
  Keep Wiki pages, the changelog, benchmark scripts and JSON reference fixtures in the
  repository; development and test files are excluded from package distributions.
- Add Focal Tversky (including fused CUDA and global distributed reduction), top-k CE,
  Dice/top-k CE and Lovasz-Softmax losses with explicit exponents, selection and masking.
- Add boundary and Hausdorff distance-transform surrogate losses using existing C++/Triton
  EDT kernels, physical spacing and optional precomputed target maps.
- Add hard clDice with C++ Lee thinning, precomputed skeleton support and collection
  integration; add differentiable soft-clDice with fused C++ CPU morphology and native PyTorch CUDA pooling.
- Add Brier, NLL, top-label/classwise ECE and reliability summaries, with case-averaged
  metrics and pooled calibration bins, including distributed accumulation.
- Add confidence-ranked lesion AP/FROC with one-to-one matching, grouped confidence
  ties, missed-lesion accounting and streaming/distributed dataset accumulation.
- Add independent reference/gradient tests, frozen Lee skeleton fixtures, reproducible
  CPU/GPU benchmark scripts and Wiki documentation covering definitions and backend tradeoffs.
- Run source-distribution tests and type checks on all branch pushes and pull requests into
  `main`. Build and test the portable-wheel matrix only for published GitHub Releases.
- Publish to PyPI only when a GitHub Release is published; branch merges and tag pushes do
  not publish packages. Verify the release tag, source version and main ancestry first.
- Group generated release notes by PR labels and document the feature-to-release cycle.
- Update SHA-pinned artifact upload/download actions to Node 24 releases.

## 0.0.1a3 — 2026-10-24

This release adds shared metric evaluation, masked and composed losses, deep supervision,
global distributed reductions and portable Linux wheels.

- Add integer `ignore_index` and boolean voxel masks to every segmentation loss, including
  the fused GPU paths. Normalize CE/focal over included voxels and omit entirely ignored
  cases from overlap averages; fully ignored batches return zero loss and gradients.
- Exclude masked-out invalid labels and nonfinite values before activation or reduction.
- Add reproducible masked/unmasked loss benchmarks and record CPU/GPU latency and GPU memory.
- Prepare the GitHub Wiki with installation, runnable examples, metric/loss conventions,
  lesion evaluation, performance guidance and a versioned source in `docs/wiki`.
- Add `functional.evaluate` and `MetricCollection`, sharing counts and compatible surface
  distances across requested metrics. Stateful collections gather scores in one payload.
- Add `DeepSupervisionLoss` with explicit per-resolution targets, masks and weights.
- Add Dice–focal and generalized Dice–CE combinations, sharing GPU forward/backward passes,
  and explicit CE class weights with PyTorch-compatible index/soft-target normalization.
  Zero coefficients skip disabled terms; zero target-weight sums return zero CE.
- Add opt-in distributed loss reductions over the global batch, with DDP-correct gradients,
  including uneven batches and empty or entirely ignored ranks.
- Add automatic development CI and installed-artifact tests for portable Linux x86-64
  wheels for Python 3.10/3.11 and the Python 3.12 stable ABI, tested through Python 3.14.
  Publish these wheels alongside the source distribution.
- Trigger the guarded publishing workflow when a release-version change reaches `main`;
  retain manual dispatch, installed-artifact tests and the `pypi` environment controls.

Multi-GPU NCCL execution is experimental and unverified in this release. CPU and single-GPU
CUDA tests cover two Gloo ranks; the separate two-GPU NCCL tests still need suitable hardware.

## 0.0.1a2

This release adds generalized Dice and separates maintainer instructions from the public
package description.

- Add `GeneralizedDiceLoss` with inverse-square, inverse-volume and uniform class weights,
  explicit empty-target policies, native PyTorch operations and fused Triton forward/backward.
- Validate values and gradients against independent NumPy formulas, MONAI and finite
  differences, including mixed precision, soft targets, absent classes and compilation.
- Add generalized Dice to the reproducible CPU/GPU loss benchmarks and document measured
  latency and memory use.
- Move publishing procedures to `docs/releasing.md`, add user installation instructions,
  and document the remaining segmentation priorities in `docs/segmentation-roadmap.md`.

## 0.0.1a1

Initial source release with segmentation metrics, lesion evaluation, connected components,
Dice/Dice-CE/Tversky/focal losses, compiled CPU kernels and fused GPU kernels.
