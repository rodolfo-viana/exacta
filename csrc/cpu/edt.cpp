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
// tol is the tie tolerance of the breakpoints (see tolerance()). Scratch: v holds n indices,
// z and b hold n + 1 breakpoints each.
//
// The lower envelope gives the minimising parabola, but its breakpoints carry rounding error.
// Where several parabolas meet at one point, which is common on a voxel grid, they tie
// mathematically while their rounded values differ by an ulp, and the standard algorithm may
// discard the one that rounds lowest. So breakpoints that agree to within their rounding error
// count as ties: tied parabolas are all kept, and all of them are evaluated near the breakpoint.
// Extra candidates can only lower the result towards the true minimum, never below it.
//
// Each stack entry k keeps b[k], its computed breakpoint with the entry below it (the parabola
// below is lower left of b[k]), and z[k], the running maximum of b. The top is popped only when
// the new parabola's breakpoint with it lies a full tol left of b[k]: then one of the two
// neighbours is below it everywhere by more than any rounding error, so it is never the rounded
// minimum. A kept entry can be lowest only between b[k] and b[k + 1], up to rounding error; the
// scan therefore looks left while the running maximum z[i + 1] reaches xq - tol and right while
// the suffix minimum of b (stored back into b) stays below xq + tol. Testing pops against z, or
// scanning right with z, can miss the minimum by an ulp when tied parabolas are far apart on
// the line (tests/test_edt.py has such lines).
//
// The comments "Alg. 1, line N" refer to the pseudo-code of this pass in the paper that
// describes Exacta (Algorithm 1, the tie-tolerant lower envelope); its proof of exactness assumes
// exactly these binary64 operations, each rounded once (no fused multiply-add, see CMakeLists.txt).
void edt_1d(const double *f, double tol, double *d, std::int64_t n, double h, std::int64_t *v,
            double *z, double *b) {
  std::int64_t k = -1;  // Alg. 1, line 1
  for (std::int64_t q = 0; q < n; ++q) {  // line 2
    if (f[q] == kInf) continue;
    const double xq = h * static_cast<double>(q);  // line 3
    // The breakpoint with the top of the stack is num / den, den > 0. Comparing num with
    // (b[k] - tol) * den decides as comparing the quotient would, but for an ulp far inside the
    // tolerance, and leaves one division per kept breakpoint.
    double num = 0.0, den = 1.0;
    while (k >= 0) {  // line 4
      const double xv = h * static_cast<double>(v[k]);
      num = (f[q] + xq * xq) - (f[v[k]] + xv * xv);  // line 5
      den = 2.0 * (xq - xv);
      if (num > (b[k] - tol) * den) break;  // lines 6-7: v[k] may be lowest somewhere, or tie
      --k;  // line 9: v[k] lies above a neighbour everywhere
    }
    // b[0] = -inf, so the stack, once started, never empties.
    const double s = k >= 0 ? num / den : -kInf;  // line 11
    ++k;
    v[k] = q;
    b[k] = s;
    // Line 12; a kept tie leaves v[k - 1] a zero-width interval.
    z[k] = k == 0 ? s : std::max(s, z[k - 1]);
    z[k + 1] = kInf;
  }
  if (k < 0) {  // line 14
    for (std::int64_t q = 0; q < n; ++q) d[q] = kInf;
    return;
  }
  for (std::int64_t i = k - 1; i >= 0; --i) b[i] = std::min(b[i], b[i + 1]);  // line 16: minima m
  // h * (q - p) rounds once, unlike h*q - h*p, and is how SciPy forms the offset, so values
  // match SciPy's bit for bit; threshold metrics (NSD) are sensitive to the last bit.
  const auto parabola = [&](std::int64_t q, std::int64_t p) {
    const double dx = h * static_cast<double>(q - p);
    return dx * dx + f[p];
  };
  std::int64_t j = 0;  // line 17
  for (std::int64_t q = 0; q < n; ++q) {  // line 18
    const double xq = h * static_cast<double>(q);  // line 19
    while (z[j + 1] < xq) ++j;
    double best = parabola(q, v[j]);  // line 20
    for (std::int64_t i = j - 1; i >= 0 && z[i + 1] >= xq - tol; --i) {  // lines 21-22
      best = std::min(best, parabola(q, v[i]));
    }
    for (std::int64_t i = j + 1; i <= k && b[i] <= xq + tol; ++i) {  // lines 24-25
      best = std::min(best, parabola(q, v[i]));
    }
    d[q] = best;
  }
}

// The first transformed axis sees only 0 (feature) and +inf, where the lower envelope reduces to
// the nearest feature p along the axis and the value (h * (q - p))^2: the minimum over features,
// since rounding is monotonic. Two sweeps find it, row by row of the trailing axes, so the
// memory is read in order. `outer` images of n x inner elements; columns [c0, c1) of one.
void first_axis(const std::uint8_t *features, double *out, std::int64_t n, std::int64_t inner,
                std::int64_t image, std::int64_t c0, std::int64_t c1, double h) {
  const std::int64_t base = image * n * inner;
  for (std::int64_t i = 0; i < n; ++i) {  // index distance to the nearest feature above
    const std::int64_t row = base + i * inner;
    for (std::int64_t c = c0; c < c1; ++c) {
      out[row + c] = features[row + c] ? 0.0 : (i == 0 ? kInf : out[row - inner + c] + 1.0);
    }
  }
  for (std::int64_t i = n - 2; i >= 0; --i) {  // or below
    const std::int64_t row = base + i * inner;
    for (std::int64_t c = c0; c < c1; ++c) {
      out[row + c] = std::min(out[row + c], out[row + inner + c] + 1.0);
    }
  }
  for (std::int64_t i = 0; i < n; ++i) {  // h * (q - p) rounds once, as in edt_1d
    const std::int64_t row = base + i * inner;
    for (std::int64_t c = c0; c < c1; ++c) {
      const double dx = h * out[row + c];
      out[row + c] = dx * dx;
    }
  }
}

// Lines along an axis are transformed in groups of adjacent ones (consecutive offsets within the
// stride), so that gathering and scattering read and write whole cache lines.
constexpr std::int64_t kGroup = 8;

// Tie tolerance of the breakpoints along an axis of spacing h and n samples whose finite inputs
// are at most `bound`: a breakpoint is a difference of terms up to bound + xmax^2, divided by at
// least 2h, and 64 epsilon of that is far above its rounding error. A looser tolerance only keeps
// more candidates, so one bound serves every line of the pass.
double tolerance(double bound, double h, std::int64_t n) {
  const double xmax = h * static_cast<double>(n);
  return 64.0 * std::numeric_limits<double>::epsilon() * (bound + xmax * xmax) / h;
}

int edt_sq(const std::uint8_t *features, double *out, const std::int64_t *shape, int ndim,
           int batch_ndim, const double *spacing, int num_threads) {
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
  if (batch_ndim == ndim) {  // no spatial axes: every element is its own image
#pragma omp parallel for num_threads(threads) schedule(static)
    for (std::int64_t i = 0; i < total; ++i) out[i] = features[i] ? 0.0 : kInf;
    return EXACTA_OK;
  }

  // Axes are transformed in order, so squared offsets add up in axis order, as SciPy sums them.
  std::int64_t stride = total;
  for (int axis = 0; axis <= batch_ndim; ++axis) stride /= shape[axis];
  {
    const std::int64_t n = shape[batch_ndim], images = total / (n * stride);
    // Ranges of 64 columns: the three sweeps over a range then stay in the core's cache.
    const std::int64_t width = 64;
    const std::int64_t ranges = (stride + width - 1) / width;
#pragma omp parallel for num_threads(threads) schedule(static)
    for (std::int64_t task = 0; task < images * ranges; ++task) {
      const std::int64_t c0 = task % ranges * width;
      first_axis(features, out, n, stride, task / ranges, c0, std::min(stride, c0 + width),
                 spacing[0]);
    }
  }

  // The finite inputs of the pass along an axis are at most the sum of the squared extents of
  // the axes before it.
  const double first_extent = spacing[0] * static_cast<double>(shape[batch_ndim] - 1);
  double bound = first_extent * first_extent;

  // Per-thread line buffers, allocated once so no allocation happens inside parallel regions.
  std::vector<double> f_buf(threads * kGroup * max_n), d_buf(threads * max_n);
  std::vector<double> z_buf(threads * (max_n + 1)), b_buf(threads * (max_n + 1));
  std::vector<std::int64_t> v_buf(threads * max_n);

  for (int axis = batch_ndim + 1; axis < ndim; ++axis) {
    const std::int64_t n = shape[axis];
    stride /= n;  // C order: stride of `axis` is the product of the trailing extents
    const std::int64_t groups_per_slab = (stride + kGroup - 1) / kGroup;
    const std::int64_t groups = total / (n * stride) * groups_per_slab;
    const double h = spacing[axis - batch_ndim];
    const double tol = tolerance(bound, h, n);
    const double extent = h * static_cast<double>(n - 1);
    bound += extent * extent;
#pragma omp parallel num_threads(threads)
    {
      int tid = 0;
#ifdef _OPENMP
      tid = omp_get_thread_num();
#endif
      double *f = f_buf.data() + tid * kGroup * max_n;
      double *d = d_buf.data() + tid * max_n;
      double *z = z_buf.data() + tid * (max_n + 1);
      double *brk = b_buf.data() + tid * (max_n + 1);
      std::int64_t *v = v_buf.data() + tid * max_n;

#pragma omp for schedule(static)
      for (std::int64_t group = 0; group < groups; ++group) {
        const std::int64_t c0 = group % groups_per_slab * kGroup;
        const std::int64_t width = std::min(kGroup, stride - c0);
        const std::int64_t base = group / groups_per_slab * n * stride + c0;
        for (std::int64_t i = 0; i < n; ++i) {
          for (std::int64_t b = 0; b < width; ++b) f[b * n + i] = out[base + i * stride + b];
        }
        for (std::int64_t b = 0; b < width; ++b) {
          edt_1d(f + b * n, tol, d, n, h, v, z, brk);
          std::copy(d, d + n, f + b * n);
        }
        for (std::int64_t i = 0; i < n; ++i) {
          for (std::int64_t b = 0; b < width; ++b) out[base + i * stride + b] = f[b * n + i];
        }
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
                                 int ndim, int batch_ndim, const double *spacing,
                                 int num_threads) {
  if (shape == nullptr || spacing == nullptr || ndim < 1 || batch_ndim < 0 || batch_ndim > ndim) {
    return EXACTA_INVALID_ARGUMENT;
  }
  bool empty = false;
  for (int axis = 0; axis < ndim; ++axis) {
    if (shape[axis] < 0) return EXACTA_INVALID_ARGUMENT;
    if (axis >= batch_ndim) {
      // Beyond these, squared distances overflow or turn subnormal and stop being exact.
      const double h = spacing[axis - batch_ndim];
      if (!(h >= 1e-100 && h <= 1e100)) return EXACTA_INVALID_ARGUMENT;
    }
    empty = empty || shape[axis] == 0;
  }
  if (empty) return EXACTA_OK;  // the data pointers of an empty array may be null
  if (features == nullptr || out == nullptr) return EXACTA_INVALID_ARGUMENT;
  try {
    return edt_sq(features, out, shape, ndim, batch_ndim, spacing, num_threads);
  } catch (const std::bad_alloc &) {
    return EXACTA_OUT_OF_MEMORY;
  } catch (...) {
    return EXACTA_INTERNAL_ERROR;
  }
}

extern "C" int exacta_edt_lines_f64(const double *f, double *out, int64_t lines, int64_t n,
                                    double h, double bound) {
  if (lines < 0 || n < 0 || !(h >= 1e-100 && h <= 1e100) || !(bound >= 0.0 && bound < kInf)) {
    return EXACTA_INVALID_ARGUMENT;
  }
  if (lines == 0 || n == 0) return EXACTA_OK;
  if (f == nullptr || out == nullptr) return EXACTA_INVALID_ARGUMENT;
  try {
    std::vector<double> z(n + 1), b(n + 1);
    std::vector<std::int64_t> v(n);
    const double tol = tolerance(bound, h, n);
    for (std::int64_t line = 0; line < lines; ++line) {
      edt_1d(f + line * n, tol, out + line * n, n, h, v.data(), z.data(), b.data());
    }
    return EXACTA_OK;
  } catch (const std::bad_alloc &) {
    return EXACTA_OUT_OF_MEMORY;
  } catch (...) {
    return EXACTA_INTERNAL_ERROR;
  }
}
