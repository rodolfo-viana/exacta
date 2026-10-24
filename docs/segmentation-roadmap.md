# 3D MRI segmentation coverage

Exacta should cover reusable training and evaluation needs with explicit mathematical
definitions, independently checked results, and measured CPU/GPU performance. This assessment
assumes supervised voxel segmentation of anatomical structures or lesions; vascular topology
and probability calibration are additional, task-dependent requirements. Priorities below are
engineering judgments, not claims that one loss improves every MRI task.

## Current coverage

The core evaluation families are already represented. Choosing complementary measurements
matters more than increasing the number of names exported by the package; this follows the
problem-dependent approach in [Metrics Reloaded](https://doi.org/10.1038/s41592-023-02151-z).

| Need | Available implementation |
|---|---|
| Voxel overlap | Dice and IoU |
| Boundary agreement | HD/HD95, ASSD and NSD, with voxel, surfel and mesh conventions |
| Volume bias | Volume similarity and signed/absolute relative volume difference |
| Missed or spurious foreground | Sensitivity, precision and specificity |
| Individual lesions | Matching, lesion F1, SQ/PQ, lesion Dice/HD95 and BraTS 2023 conventions |
| Training baseline | Dice, Dice plus CE/BCE, Tversky and focal losses |
| Class-volume imbalance | Generalized Dice, with inverse-square, inverse-volume and uniform weighting |
| Partial annotations | Ignored labels and boolean voxel masks in all segmentation losses, including fused GPU paths |

Generalized Dice follows [Sudre et al. (2017)](https://arxiv.org/abs/1707.03237).
Its default absent-class weighting follows MONAI, while an explicit empty-target option
avoids silently ignoring false positives in entirely empty foreground cases. It reuses the
fused GPU overlap passes and native PyTorch operations on CPU.

Ignored labels and voxel masks are included in `0.0.1a3`. They exclude
unannotated voxels from every loss term and gradient, normalize CE/focal over included
voxels, and omit entirely ignored cases from overlap averages. Entirely ignored batches
return zero. GPU kernels read exclusions directly in their existing passes. This does not
add masking to surface metrics: cropping a mask can create artificial surfaces and needs
a separate definition.

## Added in 0.0.1a3

The following additions are included in `0.0.1a3`. They have explicit reduction definitions
and tests; see the Wiki for examples and constraints.

1. **Shared metric calculations.** `functional.evaluate` and `MetricCollection` compute counts
   once and share distances for compatible surface metrics. Each metric retains its options.
2. **Deep supervision.** `DeepSupervisionLoss` pairs predictions and targets explicitly at each
   resolution, applies declared weights without normalization and skips zero-weight levels.
   There is no implicit label interpolation, consistent with nnU-Net's
   [deep-supervision implementation](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/loss/deep_supervision.py).
3. **Loss combinations and weighting.** `DiceFocalLoss` and `GeneralizedDiceCELoss` share GPU
   passes. CE class weighting follows PyTorch's distinct index and soft-target reductions.
   Independent compositions validate values and gradients.
4. **Distributed training reductions.** `distributed=True` reduces loss statistics globally.
   Overlap losses also require `batch_dice=True`; the default still means the local batch.
   Two-rank DDP tests compare values and parameter gradients with one concatenated batch,
   including uneven, empty and fully ignored ranks using Gloo. The multi-GPU NCCL path remains
   experimental and unverified until the two-GPU tests run on suitable hardware.

Development CI tests installed sdists; release jobs build and test portable Linux wheels. Feature branches
merge into `main`; GitHub Releases select tagged snapshots for publication. GPU performance and multi-GPU transport need their own
hardware measurements; CPU CI cannot establish either.

## Task-dependent extensions

Specialized additions should have a concrete evaluation use case and a reference definition.
They are useful candidates, but are not prerequisites for ordinary tissue or tumor segmentation.

| Addition | Use case and implementation concern |
|---|---|
| Boundary loss | Boundary-sensitive or highly imbalanced lesions; define signed distance, physical spacing and empty masks. Reuse distance-transform kernels and allow caching target maps. See [Kervadec et al. (2019)](https://proceedings.mlr.press/v102/kervadec19a.html). |
| Hausdorff surrogate loss | Reduce large boundary errors; name the surrogate precisely, since it is not differentiable HD95 itself. See [Karimi and Salcudean (2019)](https://arxiv.org/abs/1904.10030). |
| clDice and soft-clDice | Vessels or other tubular structures; skeletonization and topology conventions need separate validation. See [Shit et al.](https://arxiv.org/abs/2003.07311). |
| Lesion sensitivity versus false positives per scan | Detection operating curves require confidence-bearing lesion predictions and threshold/matching rules beyond a single hard mask. |
| Brier score and calibration summaries | Probabilistic segmentation assessment; specify voxel/case/class weighting so large background regions do not dominate silently. |

## Performance acceptance

Correctness and performance are both release requirements. Language choice alone is not
evidence of speed: benchmark the complete forward/backward operation, including allocations,
reductions, synchronization and conversions, against eager and compiled PyTorch on the same
seeded inputs. Keep tests of float64 gradients, absent classes, tiny lesions, anisotropic
spacing where relevant, mixed precision, and noncontiguous inputs. Report hardware, dtype,
shape, threads, warm-up, repeat count, median latency and peak additional memory.

For each new volume pass, compare native PyTorch, C++ and Triton where practical. Reuse a
validated native pass when only the final statistics differ. Prefer implementations that
avoid full-volume intermediates, but verify that memory savings also preserve numerical
behavior. Synthetic microbenchmarks support implementation decisions; they do not establish
better segmentation quality or end-to-end training speedups.
