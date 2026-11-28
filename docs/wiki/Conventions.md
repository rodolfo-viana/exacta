# Mathematical conventions

Reference libraries sometimes implement different quantities under the same name. Exacta
makes these choices arguments; matching a library requires matching all relevant settings.

| Choice | Exacta behavior |
|---|---|
| HD95 directions | `combine="max"`: maximum of directional percentiles; `"pooled"`: percentile over both directions together |
| ASSD directions | `combine="pooled"`: mean over both sets of distances; `"mean"`: mean of the two directional means |
| Voxel surfaces | `surface="voxels"`: foreground boundary voxels with equal weights; face-adjacent background identifies the boundary |
| Surfels | `surface="surfels"`: marching-cubes surface elements weighted by area (contour lengths in 2D); percentiles use these weights |
| Meshes | `surface="mesh"`: triangle centers to nearest points on the other mesh, weighted by triangle area (contour segments in 2D); evaluated on CPU |
| NSD threshold | Distances at or below tolerance are included, up to a relative margin `tie_rtol` (default `1e-6`, since 0.1.0) that keeps exact ties within despite rounding; `tie_rtol=0` compares strictly |
| Sums over surface elements | Means (ASSD) and weighted NSD fractions use correctly rounded sums (since 0.1.2), so their bits do not depend on element order, thread count or device |
| Image exterior | Counts as background |
| Dice/Tversky class reduction (losses) | `average="macro"` (default): mean of per-class ratios; `"micro"`: one ratio over the pooled classes (since 0.2.0) |
| Classes absent from the target (Dice/Tversky losses) | Default (MONAI): predicted probability is penalized through the smoothed ratio; `zero_absent=True` (segmentation_models_pytorch): zero loss, class stays in the mean; `skip_absent=True` (Kornia): class leaves the mean (since 0.2.0) |
| Smoothing of Dice/Tversky losses | `smooth_numerator=True` (default, MONAI): `smooth` in numerator and denominator; `False` (Kornia): denominator only (since 0.2.0) |
| Both masks empty | `both_empty`, default NaN |
| One mask empty | Dice/IoU: zero; HD/ASSD: `one_empty`, default infinity; NSD: `one_empty`, default zero |
| Undefined ratio | `zero_division`, default NaN, for precision/sensitivity/specificity and volume ratios |

## Surface equations

For two nonempty surfaces, let $d_i^P$ denote the distance from prediction surface element
$i$ to the target surface, and $d_j^T$ the reverse distance. Distances use physical coordinates
from `spacing`. Weights $w_i^P,w_j^T$ are one for voxel surfaces and area (length in 2D) for
surfels or meshes, using the distance geometry described above.

For `combine="max"`, HD95 takes the larger directional 95th percentile:

$$
\mathrm{HD95}_{\mathrm{max}}
=\max\left(Q_{0.95}(d^P;w^P),Q_{0.95}(d^T;w^T)\right).
$$

$Q_{0.95}$ uses linear interpolation for voxel distances and an area-weighted percentile
for surfels and meshes. `combine="pooled"` takes one percentile after concatenating the two
distance/weight sequences. These two operations need not agree.

The default pooled ASSD weights all surface elements together:

$$
\mathrm{ASSD}_{\mathrm{pooled}}
=\frac{\sum_i w_i^P d_i^P+\sum_j w_j^T d_j^T}
{\sum_i w_i^P+\sum_j w_j^T}.
$$

For `combine="mean"`, take each direction's weighted mean separately, then average the two
means with equal weight. NSD at tolerance $\tau$ is the weighted fraction within tolerance,
where $\mathbf{1}[\cdot]$ is one when the condition holds and zero otherwise:

$$
\mathrm{NSD}_{\tau}
=\frac{\sum_i w_i^P\mathbf{1}[d_i^P\leq\tau]
+\sum_j w_j^T\mathbf{1}[d_j^T\leq\tau]}
{\sum_i w_i^P+\sum_j w_j^T}.
$$

**Ties with the tolerance (since 0.1.0).** A distance and a tolerance that are equal in exact
arithmetic can differ after rounding. With spacing 0.4 mm, three voxels evaluate to
$\mathrm{fl}(3\times0.4)=1.2000000000000002$ in binary64, above a tolerance of 1.2 mm, and a
spacing read from a float32 image header carries a relative error up to $6\times10^{-8}$. A strict
comparison then excludes every surface element lying exactly on the tolerance, which can move NSD
by several percentage points. Exacta therefore evaluates $\mathbf{1}[d\leq\tau(1+\rho)]$, where
$\rho$ is `tie_rtol` (default $10^{-6}$, at most $10^{-3}$). The default exceeds both rounding
errors and stays below the relative gap between distinct distances on an isotropic grid, about
$1/(2n)$ for a squared distance of $n$ voxels, for tolerances up to about 700 voxels. It never
widens a tolerance of zero. `tie_rtol=0` reproduces SciPy-based tools (MedPy, DeepMind's
surface-distance), which compare in binary64; MONAI compares float32 distances, which counts
exact ties as the default does.

**Fixed arithmetic (since 0.2.0).** After the exact squared distances, each score is a fixed
sequence of binary64 operations, every one rounded once, so the bits of a score do not depend on
the CPU's vector instructions, the BLAS or C library, the PyTorch or NumPy version, the number of
threads or the device. A distance is the correctly rounded square root of the squared distance.
For voxels, $Q_q$ with $q<100$ is `numpy.percentile(d, q)` (method `"linear"`) bit for bit: with
$p=\mathrm{fl}((n-1)\,\mathrm{fl}(q/100))$, $k=\lfloor p\rfloor$ and $t=p-k$ it is
$d_k+(d_{k+1}-d_k)\,t$ for $t<0.5$ and $d_{k+1}-(d_{k+1}-d_k)(1-t)$ otherwise. A surfel's area is
DeepMind's table with each operation written out, $\sum_n\sqrt{(v_0^2+v_1^2)+v_2^2}$ with
$v=(n_0s_1s_2,\,n_1s_0s_2,\,n_2s_0s_1)$ (in 2D the corner cut is $\tfrac12\sqrt{s_0s_0+s_1s_1}$),
and a mesh triangle's area is $\tfrac12\sqrt{(c_0^2+c_1^2)+c_2^2}$ for the cross product $c$ of two
edges ($c_0=e_{1,1}e_{2,2}-e_{1,2}e_{2,1}$ and so on), computed in the compiled kernels. The
area-weighted percentile takes the first sorted element with
$\mathrm{fl}(c_i/W)\geq\mathrm{fl}(q/100)$, where $c_i$ is the running sum of the weights from the
left and $W$ their pairwise sum, grouped as `np.sum` groups it since NumPy 2.3. Means and the
weight within the tolerance are correctly rounded sums, and the NSD bound is
$\mathrm{fl}(\tau\,\mathrm{fl}(1+\rho))$.

Empty surfaces use the explicit policies in the table rather than these ratios.

## Loss conventions

The same rule applies to losses: what differs between libraries is an argument. The overlap
losses name three choices, documented with equations on the [Losses](https://github.com/rodolfo-viana/exacta/wiki/Losses)
page: how classes are reduced (`average`), what happens to a class absent from the target
(`zero_absent`, `skip_absent`) and where the smoothing constant goes (`smooth_numerator`).
Kornia's `dice_loss` and `TverskyLoss` are
`DiceLoss(average="micro", smooth=eps, smooth_numerator=False, ignore_index=-100)`,
`DiceLoss(skip_absent=True, smooth=eps, smooth_numerator=False, ignore_index=-100)` and
`TverskyLoss(alpha=alpha, beta=beta, skip_absent=True, smooth=eps, smooth_numerator=False, ignore_index=-100)`;
MONAI's are the defaults, and segmentation_models_pytorch's use `zero_absent=True`, `batch_dice=True`
and `smooth=0`.

## Reproducing reference definitions

MedPy's HD95 pools directions; MONAI's usual voxel HD95 takes their maximum. DeepMind's
surface-distance uses surfel areas. MeshMetrics uses mesh geometry. ASSD changed between
MedPy 0.4 (mean of directional means) and 0.5.2 (pooled mean). Record library versions as well
as Exacta's arguments when reporting comparisons.

`volume_similarity(convention="taha")` returns `1 - abs(Vp-Vt)/(Vp+Vt)`.
`convention="itk"` returns signed `2*(Vp-Vt)/(Vp+Vt)`. These are different quantities.

## Physical axes and numerical limits

Spacing follows the array's axes. A SimpleITK array commonly uses `(z, y, x)`, whereas
`GetSpacing()` returns `(x, y, z)`; reverse it when those are the layouts in use.
All distances and NSD tolerances share the spacing unit.

Voxel-surface spacing components must be in `[1e-100, 1e100]`; surfels and meshes require
`[1e-50, 1e50]` because their geometric calculations include higher powers of coordinates.
Unsupported spacing raises even for empty masks. Small floating-point differences near
an NSD threshold can change membership; do not silently replace reference precision or
enable compiler transformations that change rounding.
