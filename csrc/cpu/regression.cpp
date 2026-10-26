// Weighted errors and centered moments. Validation shares the first input pass.
#include <algorithm>
#include <cmath>
#include <cstdint>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "exacta/kernels.h"

namespace {
template <typename T>
int statistics(const T *p, const T *t, const std::uint8_t *mask, const double *weights,
               std::int64_t n, std::int64_t d, int mask_columns, int moments,
               double *out, int threads) {
  if (n < 0 || d < 1 || (mask_columns != 0 && mask_columns != 1 && mask_columns != d) ||
      (mask_columns && !mask) || (moments != 0 && moments != 1)) return EXACTA_INVALID_ARGUMENT;
  if (threads <= 0) {
#ifdef _OPENMP
    threads = omp_get_max_threads();
#else
    threads = 1;
#endif
  }
  threads = static_cast<int>(std::min<std::int64_t>(threads, d));
  std::fill(out, out + 11 * d, 0.);
  bool bad = false;
#pragma omp parallel for num_threads(threads) schedule(static) reduction(|| : bad)
  for (std::int64_t j = 0; j < d; ++j) {
    double total = 0, count = 0, ap = 0, at = 0, ae = 0, sp = 0, st = 0, se = 0;
    double absolute = 0, squared = 0;
    for (std::int64_t i = 0; i < n; ++i) {
      const double w = weights ? weights[i] : 1.;
      if (!std::isfinite(w) || w < 0) { bad = true; continue; }
      if (w == 0 || (mask_columns && !mask[i * mask_columns + (mask_columns == 1 ? 0 : j)])) continue;
      const double a = static_cast<double>(p[i * d + j]), b = static_cast<double>(t[i * d + j]);
      if (!std::isfinite(a) || !std::isfinite(b)) { bad = true; continue; }
      const double e = a - b;
      if (count == 0) { ap = a; at = b; ae = e; }
      total += w;
      count += 1;
      absolute += w * std::abs(e);
      squared += w * e * e;
      if (moments) { sp += w * (a - ap); st += w * (b - at); se += w * (e - ae); }
    }
    out[j] = total; out[d + j] = count;
    out[9 * d + j] = absolute; out[10 * d + j] = squared;
    if (!moments || total == 0) continue;
    sp /= total; st /= total; se /= total;
    out[2 * d + j] = ap + sp; out[3 * d + j] = at + st; out[7 * d + j] = ae + se;
    double vp = 0, vt = 0, ve = 0, cov = 0;
    for (std::int64_t i = 0; i < n; ++i) {
      const double w = weights ? weights[i] : 1.;
      if (!(w > 0) || (mask_columns && !mask[i * mask_columns + (mask_columns == 1 ? 0 : j)])) continue;
      const double a = static_cast<double>(p[i * d + j]), b = static_cast<double>(t[i * d + j]);
      const double cp = (a - ap) - sp, ct = (b - at) - st, ce = ((a - b) - ae) - se;
      vp += w * cp * cp; vt += w * ct * ct; ve += w * ce * ce; cov += w * cp * ct;
    }
    out[4 * d + j] = vp; out[5 * d + j] = vt; out[6 * d + j] = cov; out[8 * d + j] = ve;
  }
  return bad ? EXACTA_INVALID_ARGUMENT : EXACTA_OK;
}
}

extern "C" int exacta_regression_f32(const float *p, const float *t, const std::uint8_t *m,
    const double *w, std::int64_t n, std::int64_t d, int mask_columns, int moments, double *out, int threads) {
  return statistics(p, t, m, w, n, d, mask_columns, moments, out, threads);
}
extern "C" int exacta_regression_f64(const double *p, const double *t, const std::uint8_t *m,
    const double *w, std::int64_t n, std::int64_t d, int mask_columns, int moments, double *out, int threads) {
  return statistics(p, t, m, w, n, d, mask_columns, moments, out, threads);
}
