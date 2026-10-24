// Lexicographic sort of (key, value) pairs of doubles, for weighted percentiles.
//
// Chunks are sorted in parallel, then merged pairwise between two buffers; each merge is split
// among the threads at points found by binary search, so the last merges run in parallel too.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <new>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "exacta/kernels.h"

namespace {

struct Pair {  // trivial, so that buffers of them start uninitialised
  double key, value;
  bool operator<(const Pair &other) const {
    return key < other.key || (key == other.key && value < other.value);
  }
};

// How many of the first k elements of merge(a[0, m), b[0, n)) come from a, as std::merge takes
// them (from a first on ties): the first i at which a[i] would not come before b[k - i - 1].
std::int64_t co_rank(std::int64_t k, const Pair *a, std::int64_t m, const Pair *b, std::int64_t n) {
  std::int64_t lo = std::max<std::int64_t>(0, k - n), hi = std::min(k, m);
  while (lo < hi) {
    const std::int64_t i = lo + (hi - lo) / 2, j = k - i;
    if (j > 0 && !(b[j - 1] < a[i])) {
      lo = i + 1;
    } else {
      hi = i;
    }
  }
  return lo;
}

int sort_pairs(double *keys, double *values, std::int64_t n, int num_threads) {
#ifdef _OPENMP
  const int threads = num_threads > 0 ? num_threads : omp_get_max_threads();
#else
  (void)num_threads;
  const int threads = 1;
#endif
  // NaN has no place in a strict weak order, and std::sort needs one.
  bool nan = false;
#pragma omp parallel for num_threads(threads) schedule(static) reduction(|| : nan)
  for (std::int64_t i = 0; i < n; ++i) nan = nan || std::isnan(keys[i]) || std::isnan(values[i]);
  if (nan) return EXACTA_INVALID_ARGUMENT;

  // Not zeroed: the threads touch the pages first, as they fill or merge into them.
  std::unique_ptr<Pair[]> first(new Pair[n]), second(new Pair[n]);
  Pair *a = first.get(), *b = second.get();
#pragma omp parallel for num_threads(threads) schedule(static)
  for (std::int64_t i = 0; i < n; ++i) a[i] = {keys[i], values[i]};
  const std::int64_t chunks = std::max<std::int64_t>(1, std::min<std::int64_t>(threads, n / 4096));
  const auto bound = [&](std::int64_t k) { return n * k / chunks; };
#pragma omp parallel for num_threads(threads) schedule(static, 1)
  for (std::int64_t k = 0; k < chunks; ++k) std::sort(a + bound(k), a + bound(k + 1));
  for (std::int64_t width = 1; width < chunks; width *= 2) {
    // Each pair of runs [lo, mid) and [mid, hi) is merged in `parts` slices of the output.
    const std::int64_t pairs = (chunks + 2 * width - 1) / (2 * width);
    const std::int64_t parts = std::max<std::int64_t>(1, threads / pairs);
#pragma omp parallel for num_threads(threads) schedule(static, 1)
    for (std::int64_t task = 0; task < pairs * parts; ++task) {
      const std::int64_t k = task / parts * 2 * width, part = task % parts;
      const std::int64_t lo = bound(k), mid = bound(std::min(k + width, chunks));
      const std::int64_t hi = bound(std::min(k + 2 * width, chunks));
      const std::int64_t m = mid - lo, size = hi - lo;
      const std::int64_t start = size * part / parts, end = size * (part + 1) / parts;
      const std::int64_t i0 = co_rank(start, a + lo, m, a + mid, size - m);
      const std::int64_t i1 = co_rank(end, a + lo, m, a + mid, size - m);
      std::merge(a + lo + i0, a + lo + i1, a + mid + (start - i0), a + mid + (end - i1),
                 b + lo + start);
    }
    std::swap(a, b);
  }
#pragma omp parallel for num_threads(threads) schedule(static)
  for (std::int64_t i = 0; i < n; ++i) {
    keys[i] = a[i].key;
    values[i] = a[i].value;
  }
  return EXACTA_OK;
}

}  // namespace

extern "C" int exacta_sort_pairs_f64(double *keys, double *values, int64_t n, int num_threads) {
  if (n < 0 || (n > 0 && (keys == nullptr || values == nullptr))) return EXACTA_INVALID_ARGUMENT;
  try {
    return sort_pairs(keys, values, n, num_threads);
  } catch (const std::bad_alloc &) {
    return EXACTA_OUT_OF_MEMORY;
  }
}
