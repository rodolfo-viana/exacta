// Bounding box of every class of a label map, in one pass.
//
// Threads scan disjoint ranges of the first axis into private tables, which are then merged, so
// the cost is one read of the volume whatever the number of classes.

#include <algorithm>
#include <cstdint>
#include <new>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "exacta/kernels.h"

namespace {

template <typename T>
int class_extents(const T *labels, const std::int64_t *shape, int ndim, std::int64_t num_classes,
                  std::int64_t *lo, std::int64_t *hi, int num_threads) {
  // Work in 3-D: a 1-D or 2-D image is a 3-D one with leading extents of 1.
  std::int64_t n[3] = {1, 1, 1};
  for (int axis = 0; axis < ndim; ++axis) n[3 - ndim + axis] = shape[axis];
  const int skip = 3 - ndim;  // padded axes, not reported

  int threads = 1;
#ifdef _OPENMP
  threads = num_threads > 0 ? num_threads : omp_get_max_threads();
#else
  (void)num_threads;
#endif
  const std::int64_t entries = num_classes * 3;
  std::vector<std::int64_t> lo_all(threads * entries), hi_all(threads * entries, -1);
  for (int t = 0; t < threads; ++t) {
    for (std::int64_t c = 0; c < num_classes; ++c) {
      for (int a = 0; a < 3; ++a) lo_all[t * entries + c * 3 + a] = n[a];
    }
  }

#pragma omp parallel num_threads(threads)
  {
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    std::int64_t *my_lo = lo_all.data() + tid * entries;
    std::int64_t *my_hi = hi_all.data() + tid * entries;
#pragma omp for schedule(static)
    for (std::int64_t x0 = 0; x0 < n[0] * n[1]; ++x0) {  // one row (line along the last axis)
      const std::int64_t i0 = x0 / n[1], i1 = x0 % n[1];
      const T *row = labels + x0 * n[2];
      for (std::int64_t x = 0; x < n[2];) {
        const std::int64_t c = static_cast<std::int64_t>(row[x]);
        const std::int64_t start = x;
        while (x < n[2] && static_cast<std::int64_t>(row[x]) == c) ++x;  // a run of one class
        if (c < 0 || c >= num_classes) continue;
        std::int64_t *l = my_lo + c * 3, *h = my_hi + c * 3;
        l[0] = std::min(l[0], i0);
        h[0] = std::max(h[0], i0);
        l[1] = std::min(l[1], i1);
        h[1] = std::max(h[1], i1);
        l[2] = std::min(l[2], start);
        h[2] = std::max(h[2], x - 1);
      }
    }
  }

  for (std::int64_t c = 0; c < num_classes; ++c) {
    for (int a = skip; a < 3; ++a) {
      std::int64_t l = n[a], h = -1;
      for (int t = 0; t < threads; ++t) {
        l = std::min(l, lo_all[t * entries + c * 3 + a]);
        h = std::max(h, hi_all[t * entries + c * 3 + a]);
      }
      lo[c * ndim + a - skip] = l;
      hi[c * ndim + a - skip] = h;
    }
  }
  return EXACTA_OK;
}

template <typename T>
int checked(const T *labels, const int64_t *shape, int ndim, int64_t num_classes, int64_t *lo,
            int64_t *hi, int num_threads) {
  if (shape == nullptr || lo == nullptr || hi == nullptr) return EXACTA_INVALID_ARGUMENT;
  if (ndim < 1 || ndim > 3 || num_classes < 1) return EXACTA_INVALID_ARGUMENT;
  bool empty = false;
  for (int axis = 0; axis < ndim; ++axis) {
    if (shape[axis] < 0) return EXACTA_INVALID_ARGUMENT;
    empty = empty || shape[axis] == 0;
  }
  if (empty) {  // no class anywhere; the data pointer of an empty array may be null
    for (int64_t c = 0; c < num_classes; ++c) {
      for (int a = 0; a < ndim; ++a) {
        lo[c * ndim + a] = shape[a];
        hi[c * ndim + a] = -1;
      }
    }
    return EXACTA_OK;
  }
  if (labels == nullptr) return EXACTA_INVALID_ARGUMENT;
  try {
    return class_extents(labels, shape, ndim, num_classes, lo, hi, num_threads);
  } catch (const std::bad_alloc &) {
    return EXACTA_OUT_OF_MEMORY;
  }
}

}  // namespace

extern "C" int exacta_class_extents_u8(const uint8_t *labels, const int64_t *shape, int ndim,
                                       int64_t num_classes, int64_t *lo, int64_t *hi,
                                       int num_threads) {
  return checked(labels, shape, ndim, num_classes, lo, hi, num_threads);
}

extern "C" int exacta_class_extents_i32(const int32_t *labels, const int64_t *shape, int ndim,
                                        int64_t num_classes, int64_t *lo, int64_t *hi,
                                        int num_threads) {
  return checked(labels, shape, ndim, num_classes, lo, hi, num_threads);
}

extern "C" int exacta_class_extents_i64(const int64_t *labels, const int64_t *shape, int ndim,
                                        int64_t num_classes, int64_t *lo, int64_t *hi,
                                        int num_threads) {
  return checked(labels, shape, ndim, num_classes, lo, hi, num_threads);
}
