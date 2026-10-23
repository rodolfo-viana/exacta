// Exact squared Euclidean distance transform on a grid with anisotropic spacing.
//
// Separable algorithm of Felzenszwalb & Huttenlocher, "Distance Transforms of Sampled
// Functions", Theory of Computing 8 (2012): one lower-envelope pass per axis, O(n) overall.

#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "exacta/kernels.h"

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Evaluates, at every sample q, the lower envelope of the parabolas (h*q - h*p)^2 + f[p]
// over samples p with finite f[p]. Scratch: v holds n indices, z holds n + 1 breakpoints.
void edt_1d(const double *f, double *d, std::int64_t n, double h, std::int64_t *v, double *z) {
  std::int64_t k = -1;
  for (std::int64_t q = 0; q < n; ++q) {
    if (f[q] == kInf) continue;
    const double xq = h * static_cast<double>(q);
    double s = -kInf;
    while (k >= 0) {
      const double xv = h * static_cast<double>(v[k]);
      s = ((f[q] + xq * xq) - (f[v[k]] + xv * xv)) / (2.0 * (xq - xv));
      if (s > z[k]) break;
      --k;
    }
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = kInf;
  }
  if (k < 0) {
    for (std::int64_t q = 0; q < n; ++q) d[q] = kInf;
    return;
  }
  std::int64_t j = 0;
  for (std::int64_t q = 0; q < n; ++q) {
    const double xq = h * static_cast<double>(q);
    while (z[j + 1] < xq) ++j;
    const double dx = xq - h * static_cast<double>(v[j]);
    d[q] = dx * dx + f[v[j]];
  }
}

int edt_sq(const std::uint8_t *features, double *out, const std::int64_t *shape, int ndim,
           const double *spacing, int num_threads) {
  std::int64_t total = 1;
  std::int64_t max_n = 0;
  for (int axis = 0; axis < ndim; ++axis) {
    total *= shape[axis];
    if (shape[axis] > max_n) max_n = shape[axis];
  }
  if (total == 0) return EXACTA_OK;

  int threads = 1;
#ifdef _OPENMP
  threads = num_threads > 0 ? num_threads : omp_get_max_threads();
#else
  (void)num_threads;
#endif

#pragma omp parallel for num_threads(threads) schedule(static)
  for (std::int64_t i = 0; i < total; ++i) out[i] = features[i] ? 0.0 : kInf;

  // Per-thread line buffers, allocated once so no allocation happens inside parallel regions.
  std::vector<double> f_buf(threads * max_n), d_buf(threads * max_n), z_buf(threads * (max_n + 1));
  std::vector<std::int64_t> v_buf(threads * max_n);

  std::int64_t stride = total;
  for (int axis = 0; axis < ndim; ++axis) {
    const std::int64_t n = shape[axis];
    stride /= n;  // C order: stride of `axis` is the product of the trailing extents
    const std::int64_t lines = total / n;
    const double h = spacing[axis];

#pragma omp parallel num_threads(threads)
    {
      int tid = 0;
#ifdef _OPENMP
      tid = omp_get_thread_num();
#endif
      double *f = f_buf.data() + tid * max_n;
      double *d = d_buf.data() + tid * max_n;
      double *z = z_buf.data() + tid * (max_n + 1);
      std::int64_t *v = v_buf.data() + tid * max_n;

#pragma omp for schedule(static)
      for (std::int64_t line = 0; line < lines; ++line) {
        const std::int64_t base = (line / stride) * n * stride + line % stride;
        for (std::int64_t i = 0; i < n; ++i) f[i] = out[base + i * stride];
        edt_1d(f, d, n, h, v, z);
        for (std::int64_t i = 0; i < n; ++i) out[base + i * stride] = d[i];
      }
    }
  }
  return EXACTA_OK;
}

}  // namespace

extern "C" const char *exacta_status_string(int status) {
  switch (status) {
    case EXACTA_OK:
      return "ok";
    case EXACTA_INVALID_ARGUMENT:
      return "invalid argument";
    case EXACTA_OUT_OF_MEMORY:
      return "out of memory";
    default:
      return "internal error";
  }
}

extern "C" int exacta_edt_sq_f64(const uint8_t *features, double *out, const int64_t *shape,
                                 int ndim, const double *spacing, int num_threads) {
  if (features == nullptr || out == nullptr || shape == nullptr || spacing == nullptr || ndim < 1) {
    return EXACTA_INVALID_ARGUMENT;
  }
  for (int axis = 0; axis < ndim; ++axis) {
    if (shape[axis] < 0 || !(spacing[axis] > 0.0) || !std::isfinite(spacing[axis])) {
      return EXACTA_INVALID_ARGUMENT;
    }
  }
  try {
    return edt_sq(features, out, shape, ndim, spacing, num_threads);
  } catch (const std::bad_alloc &) {
    return EXACTA_OUT_OF_MEMORY;
  } catch (...) {
    return EXACTA_INTERNAL_ERROR;
  }
}
