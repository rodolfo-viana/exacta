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

Generalized Dice follows [Sudre et al. (2017)](https://arxiv.org/abs/1707.03237).
Its default absent-class weighting follows MONAI, while an explicit empty-target option
avoids silently ignoring false positives in entirely empty foreground cases. It reuses the
fused GPU overlap passes and native PyTorch operations on CPU.

## Next implementation priorities

The largest remaining general-purpose gaps are training semantics and composition. Implement
them with exact reduction definitions and tests before adding specialized objectives.

1. **Ignored labels and voxel masks.** Exclude unannotated voxels from every overlap term,
   CE/focal normalization and gradient, with explicit behavior for all-ignored cases. An
   excluded region is different from background. nnU-Net's
   [compound losses](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/loss/compound_losses.py)
   illustrate the need. Fuse masking into volume passes rather than materializing masked
   copies on GPU. Surface metrics need a separate definition because cropping a mask creates
   artificial surfaces; do not apply voxel-loss masking conventions to them automatically.
2. **Deep supervision.** Apply a loss to explicitly paired predictions and targets at several
   resolutions, with declared weights and validation of lengths. Avoid implicit label
   interpolation. Check against nnU-Net's
   [deep-supervision implementation](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/loss/deep_supervision.py).
3. **Loss combinations and weighting.** Dice plus focal, generalized Dice plus CE, and explicit
   per-class CE weights. Use composition as a correctness baseline; fuse only where measured
   time or memory justifies it. Separate class weighting from spatial and batch reductions.
4. **Distributed training reductions.** Evaluation already gathers per-case results. Losses'
   `batch_dice=True` currently means the local batch, not all distributed ranks. Offer an
   explicit differentiable global reduction with empty-rank and uneven-batch tests.
5. **Shared metric calculations.** Compute counts once for overlap/volume metrics and surface
   distances once for HD95, ASSD and NSD requested together. Preserve each metric's conventions
   while avoiding repeated volume scans, extraction and distance transforms.

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
