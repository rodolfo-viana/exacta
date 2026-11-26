// Correctly rounded sums, for statistics over surface elements (ASSD, weighted NSD).
//
// A plain or parallel sum depends on the order of the terms, and so on the number of threads and
// on the device; the last bits of a mean then change from one machine to another. The sum
// returned here is the exact sum of the terms rounded once to the nearest double (ties to even),
// which no order can change. The algorithm is Shewchuk's, as Python's math.fsum implements it:
// the running sum is kept as a short list of nonoverlapping partials, each new term is added to
// them exactly with two-sum, and the partials are rounded to one double at the end.
// J. R. Shewchuk, "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric
// Predicates", Discrete & Computational Geometry 18 (1997).
//
// Two properties are design choices, not consequences of the algorithm:
//  - Intermediate overflow. When a partial sum overflows, the function returns +-inf even if the
//    exact sum is finite, where math.fsum raises OverflowError: the terms DBL_MAX, DBL_MAX,
//    -DBL_MAX give inf, and in the order DBL_MAX, -DBL_MAX, DBL_MAX they give DBL_MAX. The result
//    is therefore independent of the order of the terms only while no partial sum overflows.
//    A product x[i] * w[i] that overflows is a non-finite term, added as IEEE adds inf.
//  - Serial host code. The sum runs on one host thread, in index order, so it cannot depend on a
//    thread count or a device; a GPU input is copied to the host first (exact_sum in _native.py)
//    at the cost of that transfer. The agreement between devices is therefore by construction,
//    not a property of a parallel algorithm.

#include <cmath>
#include <cstdint>
#include <new>
#include <utility>
#include <vector>

#include "exacta/kernels.h"

namespace {

// Partials stay few (usually one to three) for the sums Exacta forms, so a vector suffices.
double exact_sum(const double *x, const double *w, const std::uint8_t *select, std::int64_t n) {
  std::vector<double> p;
  p.reserve(16);
  double special = 0.0;  // sum of the non-finite terms: inf, -inf or nan, as IEEE adds them
  bool nonfinite = false;
  for (std::int64_t i = 0; i < n; ++i) {
    if (select != nullptr && !select[i]) continue;
    double t = w != nullptr ? x[i] * w[i] : x[i];
    if (!std::isfinite(t)) {
      special += t;
      nonfinite = true;
      continue;
    }
    std::size_t k = 0;
    for (std::size_t j = 0; j < p.size(); ++j) {
      double y = p[j];
      if (std::fabs(t) < std::fabs(y)) std::swap(t, y);
      const double hi = t + y;
      if (!std::isfinite(hi)) return hi;  // the partial sum overflows (|sum| near DBL_MAX)
      const double lo = y - (hi - t);
      if (lo != 0.0) p[k++] = lo;
      t = hi;
    }
    p.resize(k);
    p.push_back(t);
  }
  if (nonfinite) return special;
  // Round the partials, largest last, to the nearest double; ties go to even (math.fsum).
  std::size_t m = p.size();
  if (m == 0) return 0.0;
  double hi = p[--m];
  double lo = 0.0;
  while (m > 0) {
    const double s = hi;
    const double y = p[--m];
    hi = s + y;
    const double yr = hi - s;
    lo = y - yr;
    if (lo != 0.0) break;
  }
  if (m > 0 && ((lo < 0.0 && p[m - 1] < 0.0) || (lo > 0.0 && p[m - 1] > 0.0))) {
    const double y = lo * 2.0;
    const double s = hi + y;
    const double yr = s - hi;
    if (y == yr) hi = s;
  }
  return hi;
}

}  // namespace

extern "C" int exacta_exact_sum_f64(const double *x, const double *w, const uint8_t *select,
                                    int64_t n, double *sum) {
  if (n < 0 || sum == nullptr || (n > 0 && x == nullptr)) return EXACTA_INVALID_ARGUMENT;
  try {
    *sum = exact_sum(x, w, select, n);
    return EXACTA_OK;
  } catch (const std::bad_alloc &) {
    return EXACTA_OUT_OF_MEMORY;
  } catch (...) {
    return EXACTA_INTERNAL_ERROR;
  }
}
