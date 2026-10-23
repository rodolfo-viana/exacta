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

#ifdef __cplusplus
}
#endif

#endif /* EXACTA_KERNELS_H */
