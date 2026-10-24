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

#ifdef __cplusplus
}
#endif

#endif /* EXACTA_KERNELS_H */
