# Changelog

Release notes for Exacta.

## Unreleased

- **Overlap and softmax losses at saturated logits:** three computations lost digits where a loss is below
  the roundoff or the dominant softmax probability rounds to one (a logit gap above 37 in binary64, 17 in
  binary32), on both backends. (1) The Dice, Tversky, focal Tversky and generalized Dice losses were
  `1 - ratio`, which has only absolute accuracy: with eight voxels per class and a gap of 40 (loss `4.2e-18`)
  the value was exactly 0, and on batches with gaps of 10 to 80 it was off by up to `6e-12` in binary64 and
  `1e-3` in binary32 (all digits for the squared Dice in binary32). (2) The squared Dice differentiates as
  `dL/dp = (2 p N / D - 2 g) / D`, with `N = 2 sum p g + smooth` and `D = sum p^2 + sum g^2 + smooth`: where
  `p = g = 1`, `N / D` is within roundoff of one, and the two terms, each of order `2 / D`, cancel to a
  difference of the order of `1 - p` that is formed with an absolute error of roundoff times each term, so
  the logit gradient of a saturated batch had relative errors of up to 190 in binary64. (3) `torch.softmax`
  differentiates as `p_i (d_i - sum_j p_j d_j)`, `-log p_t` is `-log 1` and the gradient `p - onehot` is 0 for
  the dominant class: every gradient element of the dominant class of the Dice, Tversky, focal Tversky,
  generalized Dice, cross-entropy and focal losses, and of their sums, had a relative error of 100%, and the
  cross-entropy and focal values were off by up to `2e-12` in binary64 and `5e-4` in binary32 on those
  batches, and 0 beyond the gap of 37 (17). Now the overlap losses are ratios of sums of non-negative terms,
  with `TP = sum p g`, `FP = sum p (1 - g)`, `FN = sum (1 - p) g`: Dice is `(FP + FN) / (2 TP + FP + FN +
  smooth)`, Tversky `(alpha FP + beta FN) / (TP + alpha FP + beta FN + smooth)`, the squared Dice
  `sum (p - g)^2 / (sum p^2 + sum g^2 + smooth)` and the generalized Dice `sum_c w_c (FP_c + FN_c)` over its
  denominator. They are algebraically the same losses (each docstring gives the identity; `smooth_numerator`,
  `average="micro"`, `skip_absent`, `zero_absent`, `soft_label`, `batch_dice`, masks, class weights and the
  focal power keep their meaning), and differentiate as `error / (numerator + error)` does,
  `numerator / denominator^2` and `-error / denominator^2`. `1 - p` is never formed from a rounded `p`: it is
  `sigmoid(-z)`, or the sum of the other softmax probabilities, and enters `FN` and `p - g = p (1 - g) -
  (1 - p) g`. The softmax is a `torch.autograd.Function` whose backward is `dz_i = (1 - p_i) v_i - p_i
  sum_{j != i} v_j`, the sum over the other classes being formed directly for the class of largest
  probability; the cross-entropy is `log1p(sum_{j != t} exp(z_j - z_t))` where `z_t` is the maximum
  (`(max - z_t) + log1p(sum_{j != max} exp(z_j - max))` otherwise, a sum of non-negative terms), with gradient
  `p_i` for `i != t` and `-(1 - p_t)` for the target; the softmax focal terms are `(1 - p)^gamma (-log p)`
  from the same two quantities. `TopKCELoss`, the softmax of `LovaszSoftmaxLoss`, `BoundaryLoss`,
  `HausdorffDTLoss` and `SkeletonRecallLoss` use the new softmax, and `SkeletonRecallLoss` and
  `GeneralizedWassersteinDiceLoss` the ratio of non-negative sums. The fused kernels use the same forms.
  Against a 160-digit mpmath reference, on saturated batches with gaps of 10 to 80 (losses of `4e-5` to
  `1e-34`), the largest relative error is `3 u` in value and `9 u` in every gradient element in binary64 on
  both backends, and `7 u` and `40 u` in binary32 on PyTorch (`3 u` and `75 u` in the fused kernels, whose
  fast exponential costs about the gap times `u`; the shifted logit `z - max` is itself rounded in binary32),
  `u` being the roundoff; relative accuracy holds while the quantity is representable (in binary32 `(1 - p)^2`
  is subnormal beyond a gap of 44). At moderate logits values change by at most `4.4 u` and gradients by
  `13 u` of their largest component, in both dtypes. Double backward on the PyTorch backend and gradients
  into a soft target are unchanged. Not changed: the Lovasz softmax still sorts the rounded errors
  `|g - p|`, the squared error of `HausdorffDTLoss` still subtracts where `p` rounds to the target, and the
  value of `BoundaryLoss` is a sum of terms of both signs.
  Under `torch.compile` the softmax, cross-entropy and overlap sums are formed with ordinary operations instead
  of the custom autograd functions (PyTorch 2.4's dynamo cannot trace them): the values keep their accuracy, the
  gradient is autograd's and loses the dominant class at saturated logits as before.
  Cost: with softmax and integer labels (or a one-hot tensor that is exactly such labels: values 0 and 1, one 1
  per included voxel, which costs the `max` over the classes that yields the labels, about 3 ms of reductions
  and one host synchronisation) the PyTorch backend forms the sums from the label indices (no one-hot tensor)
  and `DiceCELoss` shares one softmax between its two terms. Forward and backward of `DiceCELoss` on a
  `1 x 4 x 128^3` volume then take 1.07 times the previous time on the CPU for index targets in binary32 and
  binary64 (1.2 times for `DiceLoss`, 1.7 times for the cross-entropy alone; 2 threads), and 1.5 to 1.7 times
  for a contiguous hard one-hot target (2.2 to 2.6 times for `DiceLoss`: the previous composition was cheap
  there, 1.1 to 1.3 times for a channels-last one-hot tensor, where it was not). The fused kernels take 1.0 times
  in binary32 and 1.25 times in binary64 (the added per-voxel log-normalizer and complement are
  double-precision work in a kernel limited by double-precision throughput). The PyTorch backend on CUDA
  (index targets, `DiceCELoss`, `64^3` and `128^3`, two and four classes) takes 1.1 to 1.8 times the previous
  time: it forms the sums with masks above about half a million voxels, where CUDA's scatter/gather kernels are
  slower, and is launch-bound below. Sigmoid activations, soft one-hot targets and the focal loss keep the
  generic composition and take 1.2 to 3 times as long on the CPU.

- **Sigmoid losses at saturated logits:** the gradients through a sigmoid lost accuracy as `sigma(z)`
  approached one and every digit once it rounded to one (`z` above 37 in binary64, 17 in binary32), on both
  backends, and so did the value of the binary cross-entropy on the PyTorch backend. `torch.sigmoid`
  differentiates as `sigma * (1 - sigma)`, exactly 0 there (`sigma(z) * sigma(-z)` is `4.2e-18` at `z = 40`);
  the BCE gradient `sigma(z) - g` is 0 for `g = 1` and `z > 0`; and
  `F.binary_cross_entropy_with_logits` returned 0 instead of `4.2e-18` for `g = 0` and `z = -40`. Against a
  120-digit mpmath reference, the relative error of the gradient of every element with `z > 37` was 100%
  (`9e15 u` in binary64, `1.7e7 u` in binary32, `u` being the roundoff) for sigmoid Dice, Tversky, focal
  Tversky, generalized Dice, Dice+BCE and Dice+focal on the PyTorch and fused Triton backends, and for BCE
  with `g = 1`; the PyTorch BCE value was off by 100% for `g = 0` and `z < -37` (the fused value was exact).
  The sigmoid focal loss, which formed `p**gamma` as `exp(gamma * log p)`, was off by up to `23 u` in value
  and `35 u` in gradient in binary64 (`gamma * |log p| * u`, growing with `|z|`). Now the sigmoid is a small
  `torch.autograd.Function` with derivative `sigma(z) * sigma(-z)`; the BCE is `log1p(exp(-|z|))` plus
  `(1 - g) z` for `z >= 0` or `g |z|` for `z < 0` (a sum of non-negative terms), with gradient
  `(1 - g) sigma(z) - g sigma(-z)`; and the focal powers are taken from `sigma(z)` and `sigma(-z)`
  directly. The fused kernels use the same forms. For logits of magnitude 10 to 80 with 0/1 targets, and for
  batches mixing saturated and moderate logits, the largest relative error is now `3 u` in value and `7 u`
  in gradient in binary64 on both backends, and `7 u` in binary32 on PyTorch (the fused binary32 kernels
  reach `43 u` at `|z| = 40` through the fast exponential, as before). The same sigmoid serves
  `BoundaryLoss`, `HausdorffDTLoss` and `SkeletonRecallLoss` with `activation="sigmoid"` (the squared
  error `(p - y)**2` of `HausdorffDTLoss` still cancels where `p` rounds to `y`). Values at moderate
  logits change only in the last bits, and the PyTorch backend still supports double backward and
  gradients into a soft target. The value of the Dice and Tversky losses remains `1 - ratio`, accurate in
  absolute rather than relative terms when the loss is below about `1e-16`.

- **Dice and Tversky losses:** three new arguments make conventions of other libraries compositions of
  Exacta's. `average="micro"` pools the selected classes into one ratio per case (per batch with
  `batch_dice=True`); the default `"macro"` is the previous mean of per-class ratios.
  `skip_absent=True` averages only over the classes present in each case's target, where
  `zero_absent=True` keeps them in the mean; a case with no present class has loss one and zero
  gradient. `smooth_numerator=False` adds `smooth` to the denominator only; the default, in both terms,
  is unchanged. With these, Kornia 0.9.2's `dice_loss(average="micro")`, `dice_loss(average="macro")` and
  `TverskyLoss` are `DiceLoss(average="micro", ...)`, `DiceLoss(skip_absent=True, ...)` and
  `TverskyLoss(skip_absent=True, ...)`, each with `smooth=eps, smooth_numerator=False,
  ignore_index=-100`, and agree with Kornia in binary64 to about `2e-15` in value and gradient (a
  smoothed numerator left up to `7e-10` at `eps=1e-8`). Remaining differences: Kornia's class
  `weight`, and an entirely ignored sample, which Kornia keeps in the mean with loss one and Exacta
  omits. The arguments run on PyTorch (`backend="auto"` leaves the fused kernels for them,
  `backend="triton"` raises). Defaults, and so every previous result, are unchanged.

- **Lovasz losses:** `LovaszSoftmaxLoss` and `LovaszHingeLoss` now compute the increments of the
  Jaccard sequence (the weights of the sorted errors) from exact int64 counts,
  `(I[k-1] U[k] - I[k] U[k-1]) / (U[k-1] U[k])`, with one binary64 division and one cast to the dtype
  of the loss, instead of subtracting rounded Jaccard values (in the input dtype for the softmax
  loss, in binary64 for the hinge). That subtraction loses about log2(n) bits: the gradient error,
  relative to the largest component, was about `0.45 n u` (`2.7e-2` for the softmax loss in binary32
  and `5e-11` in binary64 with `1e6` voxels; `2e-13` for the hinge in binary64 with 1728 voxels per
  case). For the same sort order every component of the gradient (with respect to the probabilities
  for the softmax loss, to the logits for the hinge) now has a relative error of at most `2u`, `u`
  being the unit roundoff of the dtype (the rounding of the increment and that of the division by
  the number of classes or cases averaged), in binary32 and binary64, on CPU and GPU, whatever `n`
  (`3u` in binary64 beyond 4.7e7 voxels, where the denominator is rounded as well). Binary32 counts
  are exact beyond `2**24` voxels too. The sort, its tie breaking, the masks and the
  values of the losses do not change (values differ in the last bits); the temporary counts take
  the memory of int64 instead of the input dtype. smp's Lovasz loss (binary32) differences the
  sequence as Exacta did, so under the same sort order its gradient used to equal Exacta's bit for
  bit and now differs by smp's own error, which grows with `n`: relative difference of the gradient
  norm, median over the batches of real binary32 logits, 1.4e-4 with 3e5 voxels, 1.3e-3 with 1.2e6
  and 2.8e-3 with 1.8e6 (at most 5.3e-2). kornia's, in binary64 on 2-D slices, differs by at most
  3e-14 under the same sort order.
- **Mesh distances:** the nearest-element search behind `surface="mesh"` (a k-d tree that skips a
  node or element whose bounding box is farther than the best distance so far plus the margin
  `1e-9 * (X + 1)`, `X` being the largest coordinate magnitude) was tested, not shown, to return the
  minimum over all elements of the computed point-to-element distances: a computed point-to-triangle
  distance can fall below the exact one by an amount that grows as the triangle gets thinner, and
  nothing tied that amount to the margin. For segments (2-D) it is at most about `30 u X` (`u` the
  unit roundoff), far below the margin, so 2-D is unchanged. For triangles the kernel now computes
  shape constants of every triangle (edge lengths, area, the angle at the first vertex) when it
  builds the tree, and uses the larger of `1e-9 * (X + 1)` and a bound on those errors that holds
  while the constants and `X` stay within stated limits; where they do not (a degenerate triangle,
  or coordinates too large for the thinnest triangle), the search visits every element. The result
  is thus the minimum over all elements of the distances `point_triangle` and `point_segment`
  compute, for every input without overflow or underflow. For the marching-cubes meshes Exacta
  builds, the fixed margin is the larger one whenever the spacing ratio `R = max(s) / min(s)` is at
  most `1.5e4` and coordinates stay below `9e10 * min(s) / R`, so results there do not change: HD,
  HD95, ASSD, NSD and all 1.19e6 underlying distances on Medical Segmentation Decathlon heart and
  spleen labels are bitwise identical. The pass over the triangles adds up to about 3% to the
  distance step. `tests/test_mesh_margin.py` checks thin triangles (spacing ratios up to `1e6`),
  meshes translated up to `1e9` spacings from the origin and degenerate triangles against a search
  of every element.
- **MeshMetrics compatibility:** with `surface="mesh"` in 3-D, masks are now meshed in the box
  MeshMetrics 1.2 to 1.4.1 use for NumPy masks (the bounding box of both masks, starting one voxel
  before it), so vertex coordinates are measured from the same voxel. This removes many differences
  at NSD ties, the triangle centres whose distance to the other mesh equals the tolerance in exact
  arithmetic, which `tie_rtol=0` decides by the last bit of the distance. On 24 test cases (spacings
  1 and 0.5, tolerances 1 and 2), 13 differed from MeshMetrics by up to 0.004 in NSD before and all
  agree to 1e-12 on x86-64 now. The agreement is not general: MeshMetrics computes a coordinate as
  `origin + index * spacing` and exacta as `index * spacing`, which can differ in the last bit when
  the product is inexact (at spacing 0.3 with tolerances 0.6 and 0.9, NSD differed by several
  thousandths, up to 0.008 in the cases tried); the
  closest point of a triangle comes from different algorithms; SimpleITK inputs and
  `compute_metrics_multilabel` use other grids; and MeshMetrics' VTK rounds ties differently on
  Linux aarch64 and macOS arm64 than on x86-64 (its NSD differed from exacta's in 22 of 32 test
  cases there, where exacta's values equal those on x86-64). Where they differ, only ties do:
  in the tests MeshMetrics' NSD lies between exacta's with every tie outside and with every tie
  within the tolerance. The default tie margin counts every tie as within and does not depend on the
  tool or the platform. Mesh HD, HD95 and ASSD change only in the last bits (about 1e-14); NSD with
  the default tie margin is unchanged on the cases tested. The documentation now states which
  MeshMetrics releases use discrete marching cubes (1.0 to 1.4.1) and that the 0.2 releases used
  surface nets, and gives the range in which the default tie margin cannot count a distinct
  distance in terms of the grid unit.
- **Validation:** the exact squared distance transform (CPU and Triton) now rejects grids so
  anisotropic that a pass could overflow binary64 (squared distance over spacing above `1e300`,
  e.g. spacings `1e100` and `1e-100` on long lines), where its result would no longer be the exact
  minimum. Spacings in `[1e-100, 1e100]` were accepted before without this check; ordinary images
  are far inside the limit, and no value changes.
- **Build:** the native kernels can no longer be built with floating-point contraction on any
  compiler. CMake used to add `-ffp-contract=off` only for compilers whose ID matched `GNU` or
  `Clang`; any other compiler built silently, and compilers that fuse multiply-adds by default (Intel
  icx) would have produced kernels outside the exactness proof. Each supported compiler now gets its
  flags (GCC, Clang and Apple clang `-ffp-contract=off`; icx `-fp-model=precise -ffp-contract=off`;
  MSVC `/fp:precise`, and clang-cl or icx on Windows `/fp:precise /clang:-ffp-contract=off`), each
  flag must be accepted by the compiler, MSVC before 19.30 (Visual Studio 2022), where `/fp:precise`
  may contract, is refused, and configuration stops with an error for every other compiler. At
  configuration time a probe (`csrc/fp_contract_probe.cpp`), built with those flags and optimized,
  must round a product and a sum separately; configuration fails if it does not (it is skipped when
  cross-compiling and for universal macOS builds). The flags for MSVC, icx and clang-cl follow the
  vendors' documentation and were not run on those compilers. No change for GCC, Clang and Apple
  clang, and no numeric change.
- **Validation:** the range guard of the exact squared distance transform is now also in the
  compiled kernel. `exacta_edt_sq_f64`, the exported C interface, used to accept any grid inside
  the spacing range; it now applies the same limit as `check_edt_extent` (squared distance over
  spacing at most `1e300` in every pass, and lines of at most 2**40 samples, the bound of the proof)
  before writing anything, and returns the new status `EXACTA_OUT_OF_RANGE`, which the extension
  raises as the same `ValueError`. The Python check stays and runs first, and `check_edt_extent`
  also rejects lines of more than 2**40 samples. `exacta_edt_lines_f64` and `edt_lines` (the single
  pass used by the tests, in C++, Python and Triton) now check the other assumptions of the proof:
  every input is `inf` or lies in `[0, bound]` (NaN, negative and `-inf` values and values above
  `bound` are rejected with `ValueError`), and `(bound + (h n)^2) / h` is at most `1e300`. Valid
  inputs are unaffected.
- **Validation:** the Triton distance transform indexes the samples of a line and the lines of a
  pass with 32-bit integers (element offsets were already 64-bit), which the exactness proof does
  not assume. It now raises a `ValueError` before allocating anything for a line of more than
  `2**31 - 2` samples or a pass of more than `2**31 - 32` lines (batch axes count as lines), and the
  docstring states the limit; the CPU kernel has no such limit. Inputs below it are unaffected.
- **Documentation:** the correctly rounded sum (`exact_sum`, `csrc/cpu/sum.cpp`) now states two
  properties it already had. It runs serially on the host, so a GPU input is copied there, and its
  agreement across devices is by construction. If a partial sum overflows it returns `inf` (or
  `-inf`) even when the exact sum is finite, where `math.fsum` raises `OverflowError`; the order of
  the terms matters only in that case. A test pins this behaviour.
- **Tests:** a GPU test checks that the Triton kernels' binary64 arithmetic is what the exactness
  argument assumes: every add, subtract, multiply and divide rounded once, and no fused
  multiply-add. The C++ and Triton passes now number the steps of the tie-tolerant envelope.
- **Tests:** a Portability CI workflow builds the source distribution and runs the test suite on
  Linux x86-64 with Clang and OpenMP, Linux aarch64 and macOS arm64 (Apple clang, without OpenMP),
  and checks that a fixed set of results (`scripts/portability_digest.py`) is bit-identical on all
  of them.
- **Tests:** `tests/test_signature_roundtrip.py` checks, in 264 cases, that a reporting signature
  recomputes its score bit for bit. For each surface representation, HD, HD95 and ASSD with each
  join (`max`, `mean`, `pooled`) and NSD with the default tie margin and with `tie_rtol=0`, on 2-D
  and 3-D masks built from integer arithmetic with dyadic, decimal and anisotropic spacings, it
  computes the score, writes its signature, rebuilds the call from the signature alone
  (`parse_signature` and `functional.evaluate`) and compares the bits of the recomputed score. The
  NSD tolerances sit at distances that tie with them, so the default margin and `tie_rtol=0` give
  different scores in some cases of every representation. `scripts/portability_digest.py` runs the
  same cases and adds the signature strings and the recomputed scores (12 results) to
  what the Portability CI compares across its platforms.
- **Example driver:** `examples/driver.py` runs a short tour of the public API on
  deterministic synthetic inputs, and `tests/test_example_driver.py` compares its output with
  `examples/driver_expected.txt` (the losses, whose last bits depend on the PyTorch build, to a
  relative `1e-9`; everything else, the mesh scores included, as text).
- **Fixed arithmetic of the scores:** every surface score is now a fixed binary64 expression of the
  exact squared distances, independent of the CPU's vector instructions, the BLAS, the C library,
  the PyTorch and NumPy versions and the number of threads. Four steps were not, and a fifth is
  documented (see the `surface` module docstring for every expression):
  - **Mesh element sizes** (triangle areas, segment lengths) were computed with
    `torch.linalg.cross(...).norm(...)`, which fuses multiply-adds under PyTorch's AVX2 kernels and
    not under its default ones (`ATEN_CPU_CAPABILITY=default`) or other builds (the `aarch64` CI
    job): on random 3-D masks 2 of 100 mesh ASSD and 19 of 579 mesh NSD values changed with the kernel
    set, and 20 of 4.8 million element areas of 346 Medical Segmentation Decathlon predictions
    differed. They now come from the compiled kernel together with the vertices and centres
    (`exacta_mesh_elements_f64` has a new `sizes` output, and a changed signature), each operation
    rounded once in the documented order. The values equal those of PyTorch's default kernels, so
    results that used to differ between machines now equal the unfused ones; on the 346 predictions
    no mesh score changed, and the driver's mesh ASSD changed in its last digit.
  - **Interpolated percentile** (voxel HD95, `percentile` other than 100): Exacta evaluated
    `a + (b - a) * t` for every `t`; `numpy.percentile` (MedPy's HD95) uses
    `b - (b - a) * (1 - t)` for `t >= 0.5`. The two differ in the last bit for about 1 in 4000 random
    vectors at `q = 95` (5 of 20000 in the test). Exacta now evaluates NumPy's expression, bit for
    bit, on Python floats, so the MedPy preset reproduces MedPy exactly wherever the distances
    agree; no change on the 346 predictions.
  - **Weighted nearest-rank percentile** (surfels and meshes): the total weight came from `np.sum`,
    whose grouping depends on the NumPy version beyond 8192 terms (NumPy 2.3 release notes). The
    compiled kernels `exacta_pairwise_sum_f64` and `exacta_nearest_rank_f64` fix the grouping of
    NumPy 2.3 and later and the left-to-right running sums, so the rule is DeepMind's with those
    NumPy versions and the same everywhere. No change on the 346 predictions or on 25200 further
    percentiles of cubes, spheres and noise (a correctly rounded total gave the same values).
  - **Surfel areas and lengths** used `np.linalg.norm` (a BLAS dot product) in 3-D and `s ** 2` (the
    C library's `pow`) in 2-D. They now use explicit products and sums in the order of DeepMind's
    code (`sqrt((v0 * v0 + v1 * v1) + v2 * v2)`, `0.5 * sqrt(s0 * s0 + s1 * s1)`). In 3-D the values
    equal the previous ones where the BLAS rounds like the expression (OpenBLAS on x86-64 here: 0
    of 220200 differences); in 2-D they differ from the previous ones, and from DeepMind's on
    glibc, for spacings where `pow(s, 2)` is not the correctly rounded square (89 of 200000 random
    pairs change the half diagonal). DeepMind's own table is machine-dependent: on a GitHub x86-64
    runner its 3-D norm is fused, and 34 of the 256 entries for spacing (0.7, 0.9, 1.3) differ in
    the last bit from the unfused expression. The test against DeepMind's tables is therefore bit
    for bit where its library rounds like Exacta's expressions, and to a relative `1e-15` elsewhere.
  - **Audited and unchanged:** square roots (correctly rounded: the compiled kernel, NumPy's and
    CUDA's), the distance transform, sums (correctly rounded), the NSD bound `fl(tau * fl(1 + rho))`,
    comparisons, sorting, the point-to-mesh distances (compiled without contraction) and the means
    of two directions. `tests/test_fixed_arithmetic.py` checks the expressions against independent
    evaluations and that scores are identical with PyTorch's default kernels and with 1 and 8
    threads; `scripts/portability_digest.py` adds mesh sizes and centres, percentiles, weighted
    ranks, surfel tables and the scores of random masks to what the Portability CI compares across
    its platforms.

## 0.1.2 — 2026-11-26

Version 0.1.2 makes every surface-distance score bit-reproducible across machines.

- **Fix:** ASSD and NSD with surfel or mesh weights could differ in the last bits with the number
  of PyTorch threads and between CPU and GPU, because PyTorch's parallel sums split the terms
  differently. Means and surface fractions now use correctly rounded sums (the exact sum rounded
  once, as Python's `math.fsum`), so a score has the same bits on any machine. Values change by
  at most a few ulps.

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
