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
 * For every element of the `ndim`-dimensional array described by `shape`, writes to `out`
 * the squared distance, in the physical units of `spacing` (one positive value per axis),
 * to the nearest nonzero element of `features`. If `features` has no nonzero element, every
 * output is +inf. `num_threads` <= 0 uses the OpenMP default.
 */
int exacta_edt_sq_f64(const uint8_t *features, double *out, const int64_t *shape, int ndim,
                      const double *spacing, int num_threads);

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

#ifdef __cplusplus
}
#endif

#endif /* EXACTA_KERNELS_H */
