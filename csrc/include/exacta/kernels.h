/* C interface to exacta's compiled kernels.
 *
 * Every kernel takes raw pointers to C-contiguous arrays and returns an exacta_status.
 * Nothing here depends on Python or on a deep learning framework, so the same kernels can
 * be bound to PyTorch, NumPy, JAX or plain C code.
 */
#ifndef EXACTA_KERNELS_H
#define EXACTA_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum exacta_status {
  EXACTA_OK = 0,
  EXACTA_INVALID_ARGUMENT = 1,
  EXACTA_OUT_OF_MEMORY = 2,
  EXACTA_INTERNAL_ERROR = 3,
};

const char *exacta_status_string(int status);

/* Regression (N,D) -> (11,D) float64 statistics: weight, count, means p/t,
 * centered sums pp/tt/pt, mean error, centered error sum, absolute/squared error.
 * mask_columns is 0 (no mask), 1 (sample mask), or D. Weights may be null.
 * moments=0 computes only weight/count/absolute/squared error. Inputs and output
 * must not alias. All weights must be finite/nonnegative; observed p/t finite.
 */
int exacta_regression_f32(const float *p, const float *t, const uint8_t *mask, const double *weights,
    int64_t n, int64_t d, int mask_columns, int moments, double *out, int num_threads);
int exacta_regression_f64(const double *p, const double *t, const uint8_t *mask, const double *weights,
    int64_t n, int64_t d, int mask_columns, int moments, double *out, int num_threads);

/* Lee thinning of a 2-D or 3-D binary image, with zero exterior padding.
 * Writes a 0/1 skeleton into `out` (same shape); nonzero inputs are foreground.
 * Deletion order matches scikit-image's method="lee". Input and output may alias.
 */
int exacta_skeleton_u8(const uint8_t *mask, uint8_t *out, const int64_t *shape, int ndim);

/* Soft morphology on (B,C,H,W) or (B,C,D,H,W): erosion=1 uses cross min,
 * erosion=0 uses full 3x3[x3] max. The image exterior is excluded. `indices`
 * may be null; otherwise write flattened source indices for backward. Ties
 * select the first visited neighbour. Input and output must not alias.
 */
int exacta_soft_pool_f32(const float *x, float *out, int64_t *indices,
                         const int64_t *shape, int ndim, int erosion, int num_threads);
int exacta_soft_pool_f64(const double *x, double *out, int64_t *indices,
                         const int64_t *shape, int ndim, int erosion, int num_threads);

/* Exact squared Euclidean distance transform.
 *
 * The first `batch_ndim` axes of the `ndim`-dimensional array described by `shape` index
 * independent images. For every element, writes to `out` the squared distance, in the physical
 * units of `spacing` (one value in [1e-100, 1e100] per remaining axis), to the nearest nonzero
 * element of `features` in the same image; +inf if that image has none. `num_threads` <= 0 uses the OpenMP
 * default.
 */
int exacta_edt_sq_f64(const uint8_t *features, double *out, const int64_t *shape, int ndim,
                      int batch_ndim, const double *spacing, int num_threads);

/* Connected-component labelling of a binary image of 1 to 3 dimensions.
 *
 * Writes to `out` 0 for every zero element of `mask` and, for every nonzero element, the
 * number of its connected component, components being numbered 1, 2, ... in C order of their
 * first element (the numbering of scipy.ndimage.label). Two elements are neighbours when
 * their indices differ by at most 1 along each axis, along at most `connectivity` axes
 * (1 <= connectivity <= ndim). Stores the number of components in `num_labels`. Images of
 * 2^31 elements or more are rejected.
 */
int exacta_label_u8(const uint8_t *mask, int32_t *out, const int64_t *shape, int ndim,
                    int connectivity, int64_t *num_labels);

/* Elements of a label map whose label is selected: out[i] = select[labels[i]] != 0, for `n`
 * elements and labels in [0, count). Returns EXACTA_INVALID_ARGUMENT for a label out of range.
 */
int exacta_select_labels_i32(const int32_t *labels, int64_t n, const uint8_t *select, int64_t count,
                             uint8_t *out, int num_threads);

/* Bounding box of every class of a label map of 1 to 3 dimensions.
 *
 * For each class c in [0, num_classes) and axis a, writes to lo[c * ndim + a] and
 * hi[c * ndim + a] the first and last index along a of the elements equal to c; lo = shape[a]
 * and hi = -1 when c does not occur. Values outside [0, num_classes) are ignored.
 */
/* Distance from each of `n_points` points (n_points x dim, C order) to the nearest element of a
 * mesh: triangles when dim is 3, segments when dim is 2, given as n_elements x dim vertices x dim
 * coordinates. Writes the exact minimum point-to-element distance to `out` (+inf when the mesh
 * has no element).
 */
int exacta_point_mesh_distances_f64(const double *points, int64_t n_points, const double *elements,
                                    int64_t n_elements, int dim, double *out, int num_threads);

/* Bounding box of each of `count` binary masks (count x shape, C order, nonzero = inside) of 1 to 3
 * dimensions: lo[c * ndim + a] and hi[c * ndim + a] are the first and last index along axis a of
 * mask c's nonzero elements, shape[a] and -1 for an empty mask.
 */
int exacta_mask_extents_u8(const uint8_t *masks, int64_t count, const int64_t *shape, int ndim,
                           int64_t *lo, int64_t *hi, int num_threads);

/* Binary dilation of a 1-D to 3-D mask by `steps` steps (scipy.ndimage.binary_dilation with
 * `iterations=steps` and generate_binary_structure(ndim, connectivity)): neighbours differ by at
 * most 1 along each axis and along at most `connectivity` axes, and nothing lies beyond the image.
 * Writes 0/1 to `out`.
 */
int exacta_dilate_u8(const uint8_t *mask, uint8_t *out, const int64_t *shape, int ndim,
                     int connectivity, int steps, int num_threads);

/* Independent Bernoulli statistics over rows = batch * channels, each with n voxels.
 * mask_channels: 0 = no mask, 1 = shared per case, channels = separate per label.
 * kind: 0 = Brier, 1 = NLL, 2 = reliability. Scores write (rows,2) [count,sum];
 * reliability writes (rows,3*bins) [counts,confidence sums,positive counts].
 * Inputs are validated before use, only at included voxels. Accumulation is float64.
 */
int exacta_probability_f32_u8(const float *pred, const uint8_t *target,
    const uint8_t *mask, int64_t rows, int64_t n, int64_t channels, int mask_channels,
    int from_logits, int kind, int bins, double *out, int num_threads);
int exacta_probability_f32_i64(const float *pred, const int64_t *target,
    const uint8_t *mask, int64_t rows, int64_t n, int64_t channels, int mask_channels,
    int from_logits, int kind, int bins, double *out, int num_threads);
int exacta_probability_f32_f32(const float *pred, const float *target,
    const uint8_t *mask, int64_t rows, int64_t n, int64_t channels, int mask_channels,
    int from_logits, int kind, int bins, double *out, int num_threads);
int exacta_probability_f32_f64(const float *pred, const double *target,
    const uint8_t *mask, int64_t rows, int64_t n, int64_t channels, int mask_channels,
    int from_logits, int kind, int bins, double *out, int num_threads);
int exacta_probability_f64_u8(const double *pred, const uint8_t *target,
    const uint8_t *mask, int64_t rows, int64_t n, int64_t channels, int mask_channels,
    int from_logits, int kind, int bins, double *out, int num_threads);
int exacta_probability_f64_i64(const double *pred, const int64_t *target,
    const uint8_t *mask, int64_t rows, int64_t n, int64_t channels, int mask_channels,
    int from_logits, int kind, int bins, double *out, int num_threads);
int exacta_probability_f64_f32(const double *pred, const float *target,
    const uint8_t *mask, int64_t rows, int64_t n, int64_t channels, int mask_channels,
    int from_logits, int kind, int bins, double *out, int num_threads);
int exacta_probability_f64_f64(const double *pred, const double *target,
    const uint8_t *mask, int64_t rows, int64_t n, int64_t channels, int mask_channels,
    int from_logits, int kind, int bins, double *out, int num_threads);

/* Confusion matrix of two label maps of `n` elements: out[i * classes + j] (classes x classes) is
 * the number of elements labelled i in `pred` and j in `target`. Returns EXACTA_INVALID_ARGUMENT
 * if a label lies outside [0, classes).
 */
int exacta_confusion_u8(const uint8_t *pred, const uint8_t *target, int64_t n, int64_t classes,
                        int64_t *out, int num_threads);
int exacta_confusion_i32(const int32_t *pred, const int32_t *target, int64_t n, int64_t classes,
                         int64_t *out, int num_threads);
int exacta_confusion_i64(const int64_t *pred, const int64_t *target, int64_t n, int64_t classes,
                         int64_t *out, int num_threads);

/* Masked confusion: skip mask == 0 (mask may be NULL) and, when has_ignore != 0,
 * target == ignore_index before validating either label. The mask has n bytes. */
int exacta_masked_confusion_u8(const uint8_t *pred, const uint8_t *target,
    int64_t n, int64_t classes, const uint8_t *mask, int has_ignore, int64_t ignore_index,
    int64_t *out, int num_threads);
int exacta_masked_confusion_i32(const int32_t *pred, const int32_t *target,
    int64_t n, int64_t classes, const uint8_t *mask, int has_ignore, int64_t ignore_index,
    int64_t *out, int num_threads);
int exacta_masked_confusion_i64(const int64_t *pred, const int64_t *target,
    int64_t n, int64_t classes, const uint8_t *mask, int has_ignore, int64_t ignore_index,
    int64_t *out, int num_threads);

/* Masked channel counts: mask is (count / channels, n), shared across each case's channels.
 * Excluded values are not validated or counted. valid receives included elements per channel.
 * count must be divisible by channels (> 0); all four outputs have count elements. */
int exacta_masked_channel_counts_u8(const uint8_t *pred, const uint8_t *target,
    int64_t count, int64_t n, const uint8_t *mask, int64_t channels,
    int64_t *both, int64_t *in_pred, int64_t *in_target, int64_t *valid, int num_threads);
int exacta_masked_channel_counts_i64(const int64_t *pred, const int64_t *target,
    int64_t count, int64_t n, const uint8_t *mask, int64_t channels,
    int64_t *both, int64_t *in_pred, int64_t *in_target, int64_t *valid, int num_threads);
int exacta_masked_channel_counts_f32(const float *pred, const float *target,
    int64_t count, int64_t n, const uint8_t *mask, int64_t channels,
    int64_t *both, int64_t *in_pred, int64_t *in_target, int64_t *valid, int num_threads);
int exacta_masked_channel_counts_f64(const double *pred, const double *target,
    int64_t count, int64_t n, const uint8_t *mask, int64_t channels,
    int64_t *both, int64_t *in_pred, int64_t *in_target, int64_t *valid, int num_threads);

/* Number of elements of each value of a label map of `n` elements: out[v] for v in [0, bins).
 * Returns EXACTA_INVALID_ARGUMENT if a value lies outside [0, bins).
 */
int exacta_histogram_i32(const int32_t *values, int64_t n, int64_t bins, int64_t *out,
                         int num_threads);

/* Overlap of two label maps of `n` elements, `a` with labels in [0, na] and `b` in [0, nb]: the
 * number of elements of every label (`size_a`, na + 1 entries; `size_b`, nb + 1 entries) and, for
 * every pair of nonzero labels (i, j) that share elements, the number they share, as rows
 * (i, j, count) of `pairs` in increasing order of (i, j). `*num_pairs` receives the number of
 * pairs; when it exceeds `capacity` (rows available), `pairs` is left unwritten and the caller
 * calls again with enough room. Returns EXACTA_INVALID_ARGUMENT if a label lies out of range.
 */
int exacta_label_overlap_i32(const int32_t *a, const int32_t *b, int64_t n, int64_t na, int64_t nb,
                             int64_t *size_a, int64_t *size_b, int64_t *pairs, int64_t capacity,
                             int64_t *num_pairs, int num_threads);

/* For each of `count` pairs of binary masks of `n` elements (mask-major, C order), the number of
 * elements in both, in `pred` and in `target`. Returns EXACTA_INVALID_ARGUMENT if a value is not
 * 0 or 1.
 */
int exacta_channel_counts_u8(const uint8_t *pred, const uint8_t *target, int64_t count, int64_t n,
                             int64_t *both, int64_t *in_pred, int64_t *in_target, int num_threads);
int exacta_channel_counts_i64(const int64_t *pred, const int64_t *target, int64_t count, int64_t n,
                              int64_t *both, int64_t *in_pred, int64_t *in_target,
                              int num_threads);
int exacta_channel_counts_f32(const float *pred, const float *target, int64_t count, int64_t n,
                              int64_t *both, int64_t *in_pred, int64_t *in_target, int num_threads);
int exacta_channel_counts_f64(const double *pred, const double *target, int64_t count, int64_t n,
                              int64_t *both, int64_t *in_pred, int64_t *in_target, int num_threads);

int exacta_class_extents_u8(const uint8_t *labels, const int64_t *shape, int ndim,
                            int64_t num_classes, int64_t *lo, int64_t *hi, int num_threads);
int exacta_class_extents_i32(const int32_t *labels, const int64_t *shape, int ndim,
                             int64_t num_classes, int64_t *lo, int64_t *hi, int num_threads);
int exacta_class_extents_i64(const int64_t *labels, const int64_t *shape, int ndim,
                             int64_t num_classes, int64_t *lo, int64_t *hi, int num_threads);

/* Sorts `n` (key, value) pairs in place, in increasing lexicographic order: by key, then by
 * value. Returns EXACTA_INVALID_ARGUMENT, with both arrays as they were, if a key or value is NaN.
 */
int exacta_sort_pairs_f64(double *keys, double *values, int64_t n, int num_threads);

/* DeepMind's neighbour code at every corner of the voxels of a 2-D or 3-D mask (`shape`; nonzero
 * is inside): `code` has shape + 1 along each axis, and the code of corner c has one bit per
 * voxel c - 1 + o of the block around it, o in {0, 1}^ndim, set when the voxel is inside (outside
 * the image counts as outside); offset number i (last axis fastest) sets bit 2^ndim - 1 - i. With
 * `counts` (elements per code), `*num_elements` receives the number of elements of the blocks
 * whose code is neither 0 nor all ones.
 */
int exacta_neighbour_codes_u8(const uint8_t *mask, const int64_t *shape, int ndim, uint8_t *code,
                              const int64_t *counts, int64_t *num_elements, int num_threads);

/* The mesh of neighbour codes (`code`, of a mask of shape `shape`): for every block whose code is
 * neither 0 nor all ones, in C order, the counts[code] elements of `table` (codes x max_per_code
 * x ndim x ndim offsets in half-voxels from the block's first voxel), as vertices in physical
 * units, (h / 2) * spacing for half-voxel coordinate h; with their centres (VTK's parametric
 * centre of a triangle, the midpoint of a segment), blocks (flat index into `code`), codes and
 * slots among their code's elements. `num_elements` must be the total.
 */
int exacta_mesh_elements_f64(const uint8_t *code, const int64_t *shape, int ndim,
                             const int64_t *table, const int64_t *counts, int64_t max_per_code,
                             const double *spacing, int64_t num_elements, double *elements,
                             double *centres, int64_t *block, int64_t *element_code,
                             int64_t *slot, int num_threads);

/* out[i] = values[i] != 0 for `n` values, as bytes. Returns EXACTA_INVALID_ARGUMENT if a value
 * is neither 0 nor 1 (NaN included); `out` is then still written.
 */
int exacta_binary_mask_u8(const uint8_t *values, int64_t n, uint8_t *out, int num_threads);
int exacta_binary_mask_i32(const int32_t *values, int64_t n, uint8_t *out, int num_threads);
int exacta_binary_mask_i64(const int64_t *values, int64_t n, uint8_t *out, int num_threads);
int exacta_binary_mask_f32(const float *values, int64_t n, uint8_t *out, int num_threads);
int exacta_binary_mask_f64(const double *values, int64_t n, uint8_t *out, int num_threads);

/* Boundary of a 1-D to 3-D mask (nonzero is inside), with element strides `strides`: out (C order,
 * contiguous) is 1 where the mask is and a face neighbour is outside it or outside the image.
 */
int exacta_boundary_u8(const uint8_t *mask, const int64_t *shape, const int64_t *strides, int ndim,
                       uint8_t *out, int num_threads);

/* sqrt(values[i]) for each i with mask[i] nonzero, in order, into `out`, which holds `count`, the
 * number of such i. Returns EXACTA_INVALID_ARGUMENT if `count` is not that number.
 */
int exacta_masked_sqrt_f64(const double *values, const uint8_t *mask, int64_t n, double *out,
                           int64_t count, int num_threads);

#ifdef __cplusplus
}
#endif

#endif /* EXACTA_KERNELS_H */
