# Changelog

Release notes for Exacta's alpha versions.

## Unreleased (`dev`)

Development now takes place on `dev`; `main` is reserved for releases.

- Add integer `ignore_index` and boolean voxel masks to every segmentation loss, including
  the fused GPU paths. Normalize CE/focal over included voxels and omit entirely ignored
  cases from overlap averages; fully ignored batches return zero loss and gradients.
- Exclude masked-out invalid labels and nonfinite values before activation or reduction.
- Add reproducible masked/unmasked loss benchmarks and record CPU/GPU latency and GPU memory.
- Prepare the GitHub Wiki with installation, runnable examples, metric/loss conventions,
  lesion evaluation, performance guidance and a versioned source in `docs/wiki`.

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
