# Changelog

Release notes for Exacta's alpha versions.

## Unreleased

Segmentation evaluation now supports selected voxels, independent probability channels,
physical volumes and voxel rankings, alongside two additional training objectives.

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
- Add voxel PR/ROC curves, AP and AUROC with grouped ties and exact or approximate binned
  evaluation. Separate case-averaged metrics from dataset-pooled `VoxelRanking`; binned
  streaming retains fixed-size int64 histograms, including distributed accumulation.
- Add `squared_pred=True` to Dice and Dice-CE/focal/top-k CE, including fused CUDA derivatives,
  soft targets, masked observations and global distributed Dice/CE statistics.
- Add binary/multilabel `LovaszHingeLoss` with stable native PyTorch sorting, explicit
  per-image/local-batch reductions, ignored labels and per-label observation masks.
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
