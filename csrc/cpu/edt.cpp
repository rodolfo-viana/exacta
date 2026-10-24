// Exact squared Euclidean distance transform on a grid with anisotropic spacing.
//
// Separable algorithm of Felzenszwalb & Huttenlocher, "Distance Transforms of Sampled
// Functions", Theory of Computing 8 (2012): one lower-envelope pass per axis, O(n) overall.

#include <algorithm>
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

// Evaluates, at every sample q, min over samples p with finite f[p] of (h*(q - p))^2 + f[p],
// exactly as rounded in floating point (the value a brute-force minimum would give).
// fmax is the largest finite f. Scratch: v holds n indices, z holds n + 1 breakpoints.
//
// The lower envelope gives the minimising parabola, but its breakpoints carry rounding error.
// Where several parabolas meet at one point, which is common on a voxel grid, they tie
// mathematically while their rounded values differ by an ulp, and the standard algorithm may
// discard the one that rounds lowest. So breakpoints that agree to within their rounding error
// count as ties: tied parabolas are all kept, and all of them are evaluated near the breakpoint.
// Extra candidates can only lower the result towards the true minimum, never below it.
void edt_1d(const double *f, double fmax, double *d, std::int64_t n, double h, std::int64_t *v,
            double *z) {
  // A breakpoint is a difference of terms up to fmax + xmax^2, divided by at least 2h.
  const double xmax = h * static_cast<double>(n);
  const double tol = 64.0 * std::numeric_limits<double>::epsilon() * (fmax + xmax * xmax) / h;

  std::int64_t k = -1;
  for (std::int64_t q = 0; q < n; ++q) {
    if (f[q] == kInf) continue;
    const double xq = h * static_cast<double>(q);
    double s = -kInf;
    while (k >= 0) {
      const double xv = h * static_cast<double>(v[k]);
      s = ((f[q] + xq * xq) - (f[v[k]] + xv * xv)) / (2.0 * (xq - xv));
      if (s > z[k] - tol) break;  // v[k] is lowest somewhere, or ties at z[k]: keep it
      --k;
    }
    ++k;
    v[k] = q;
    z[k] = k == 0 ? s : std::max(s, z[k - 1]);  // a kept tie leaves v[k - 1] a zero-width interval
    z[k + 1] = kInf;
  }
  if (k < 0) {
    for (std::int64_t q = 0; q < n; ++q) d[q] = kInf;
    return;
  }
  // h * (q - p) rounds once, unlike h*q - h*p, and is how SciPy forms the offset, so values
  // match SciPy's bit for bit; threshold metrics (NSD) are sensitive to the last bit.
  const auto parabola = [&](std::int64_t q, std::int64_t p) {
    const double dx = h * static_cast<double>(q - p);
    return dx * dx + f[p];
  };
  std::int64_t j = 0;
  for (std::int64_t q = 0; q < n; ++q) {
    const double xq = h * static_cast<double>(q);
    while (z[j + 1] < xq) ++j;
    double best = parabola(q, v[j]);
    for (std::int64_t i = j - 1; i >= 0 && z[i + 1] >= xq - tol; --i) {
      best = std::min(best, parabola(q, v[i]));
    }
    for (std::int64_t i = j + 1; i <= k && z[i] <= xq + tol; ++i) {
      best = std::min(best, parabola(q, v[i]));
    }
    d[q] = best;
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
        double fmax = 0.0;
        for (std::int64_t i = 0; i < n; ++i) {
          f[i] = out[base + i * stride];
          if (f[i] != kInf) fmax = std::max(fmax, f[i]);
        }
        edt_1d(f, fmax, d, n, h, v, z);
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
  if (shape == nullptr || spacing == nullptr || ndim < 1) return EXACTA_INVALID_ARGUMENT;
  bool empty = false;
  for (int axis = 0; axis < ndim; ++axis) {
    if (shape[axis] < 0 || !(spacing[axis] > 0.0) || !std::isfinite(spacing[axis])) {
      return EXACTA_INVALID_ARGUMENT;
    }
    empty = empty || shape[axis] == 0;
  }
  if (empty) return EXACTA_OK;  // the data pointers of an empty array may be null
  if (features == nullptr || out == nullptr) return EXACTA_INVALID_ARGUMENT;
  try {
    return edt_sq(features, out, shape, ndim, spacing, num_threads);
  } catch (const std::bad_alloc &) {
    return EXACTA_OUT_OF_MEMORY;
  } catch (...) {
    return EXACTA_INTERNAL_ERROR;
  }
}
