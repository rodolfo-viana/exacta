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

#ifdef __cplusplus
}
#endif

#endif /* EXACTA_KERNELS_H */
