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
| NSD threshold | Distances at or below tolerance are included |
| Image exterior | Counts as background |
| Both masks empty | `both_empty`, default NaN |
| One mask empty | Dice/IoU: zero; HD/ASSD: `one_empty`, default infinity; NSD: `one_empty`, default zero |
| Undefined ratio | `zero_division`, default NaN, for precision/sensitivity/specificity and volume ratios |

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
