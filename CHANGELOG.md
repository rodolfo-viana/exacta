# Changelog

Release notes for Exacta's alpha versions.

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
