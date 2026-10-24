// Binary morphology on masks: bounding boxes of several masks, and binary dilation.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "exacta/kernels.h"

namespace {

int threads_for(int num_threads) {
#ifdef _OPENMP
  return num_threads > 0 ? num_threads : omp_get_max_threads();
#else
  (void)num_threads;
  return 1;
#endif
}

int thread_id() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

// Extents in 3-D: a 1-D or 2-D image is a 3-D one with leading extents of 1.
void as_3d(const std::int64_t *shape, int ndim, std::int64_t n[3]) {
  n[0] = n[1] = n[2] = 1;
  for (int axis = 0; axis < ndim; ++axis) n[3 - ndim + axis] = shape[axis];
}

int mask_extents(const std::uint8_t *masks, std::int64_t count, const std::int64_t *shape, int ndim,
                 std::int64_t *lo, std::int64_t *hi, int num_threads) {
  std::int64_t n[3];
  as_3d(shape, ndim, n);
  const int skip = 3 - ndim;
  const std::int64_t rows = n[0] * n[1], volume = rows * n[2];
  const int threads = threads_for(num_threads);
  const std::int64_t entries = count * 3;
  std::vector<std::int64_t> lo_all(threads * entries), hi_all(threads * entries, -1);
  for (int t = 0; t < threads; ++t) {
    for (std::int64_t c = 0; c < count; ++c) {
      for (int a = 0; a < 3; ++a) lo_all[t * entries + c * 3 + a] = n[a];
    }
  }

  // One task per (mask, row); each thread keeps the first and last foreground element of the
  // rows it scans, and the tables are merged at the end.
#pragma omp parallel num_threads(threads)
  {
    std::int64_t *my_lo = lo_all.data() + thread_id() * entries;
    std::int64_t *my_hi = hi_all.data() + thread_id() * entries;
#pragma omp for schedule(static)
    for (std::int64_t task = 0; task < count * rows; ++task) {
      const std::int64_t c = task / rows, r = task % rows;
      const std::uint8_t *row = masks + c * volume + r * n[2];
      std::int64_t first = -1, last = -1;
      for (std::int64_t x = 0; x < n[2]; ++x) {
        if (row[x]) {
          first = x;
          break;
        }
      }
      if (first < 0) continue;
      for (std::int64_t x = n[2] - 1; x >= first; --x) {
        if (row[x]) {
          last = x;
          break;
        }
      }
      std::int64_t *l = my_lo + c * 3, *h = my_hi + c * 3;
      const std::int64_t i0 = r / n[1], i1 = r % n[1];
      l[0] = std::min(l[0], i0);
      h[0] = std::max(h[0], i0);
      l[1] = std::min(l[1], i1);
      h[1] = std::max(h[1], i1);
      l[2] = std::min(l[2], first);
      h[2] = std::max(h[2], last);
    }
  }

  for (std::int64_t c = 0; c < count; ++c) {
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

// One step of binary dilation: out = OR of `in` over the neighbourhood of every element.
// Neighbours differ by at most 1 along each axis and along at most `connectivity` axes; nothing
// lies beyond the image. Each output row (a line along the last axis) ORs the neighbouring input
// rows (offset (d0, d1)), shifted along the row by -1, 0, 1 where the neighbourhood allows it.
void dilate_step(const std::uint8_t *in, std::uint8_t *out, const std::int64_t n[3],
                 int connectivity, int threads) {
  const std::int64_t rows = n[0] * n[1], width = n[2];
#pragma omp parallel for num_threads(threads) schedule(static)
  for (std::int64_t r = 0; r < rows; ++r) {
    const std::int64_t i0 = r / n[1], i1 = r % n[1];
    std::uint8_t *dst = out + r * width;
    std::memset(dst, 0, static_cast<std::size_t>(width));
    for (std::int64_t d0 = -1; d0 <= 1; ++d0) {
      for (std::int64_t d1 = -1; d1 <= 1; ++d1) {
        const int across = (d0 != 0) + (d1 != 0);
        if (across > connectivity) continue;
        const std::int64_t j0 = i0 + d0, j1 = i1 + d1;
        if (j0 < 0 || j0 >= n[0] || j1 < 0 || j1 >= n[1]) continue;
        const std::uint8_t *src = in + (j0 * n[1] + j1) * width;
        for (std::int64_t x = 0; x < width; ++x) dst[x] |= src[x];
        if (across + 1 <= connectivity) {  // diagonal contact along the row too
          for (std::int64_t x = 1; x < width; ++x) dst[x] |= src[x - 1];
          for (std::int64_t x = 0; x + 1 < width; ++x) dst[x] |= src[x + 1];
        }
      }
    }
  }
}

int dilate(const std::uint8_t *mask, std::uint8_t *out, const std::int64_t *shape, int ndim,
           int connectivity, int steps, int num_threads) {
  std::int64_t n[3];
  as_3d(shape, ndim, n);
  const std::int64_t volume = n[0] * n[1] * n[2];
  const int threads = threads_for(num_threads);
  // Binary values: any nonzero input counts as 1, so that ORs stay 0/1.
#pragma omp parallel for num_threads(threads) schedule(static)
  for (std::int64_t i = 0; i < volume; ++i) out[i] = mask[i] != 0;
  if (steps == 0) return EXACTA_OK;
  std::vector<std::uint8_t> buffer(static_cast<std::size_t>(volume));
  std::uint8_t *a = out, *b = buffer.data();
  for (int step = 0; step < steps; ++step) {
    dilate_step(a, b, n, connectivity, threads);
    std::swap(a, b);
  }
  if (a != out) std::memcpy(out, a, static_cast<std::size_t>(volume));
  return EXACTA_OK;
}

bool any_empty(const int64_t *shape, int ndim) {
  for (int axis = 0; axis < ndim; ++axis) {
    if (shape[axis] == 0) return true;
  }
  return false;
}

}  // namespace

extern "C" int exacta_mask_extents_u8(const uint8_t *masks, int64_t count, const int64_t *shape,
                                      int ndim, int64_t *lo, int64_t *hi, int num_threads) {
  if (shape == nullptr || lo == nullptr || hi == nullptr || count < 0) {
    return EXACTA_INVALID_ARGUMENT;
  }
  if (ndim < 1 || ndim > 3) return EXACTA_INVALID_ARGUMENT;
  for (int axis = 0; axis < ndim; ++axis) {
    if (shape[axis] < 0) return EXACTA_INVALID_ARGUMENT;
  }
  if (count == 0) return EXACTA_OK;
  if (any_empty(shape, ndim)) {  // nothing anywhere; the data pointer may be null
    for (int64_t c = 0; c < count; ++c) {
      for (int a = 0; a < ndim; ++a) {
        lo[c * ndim + a] = shape[a];
        hi[c * ndim + a] = -1;
      }
    }
    return EXACTA_OK;
  }
  if (masks == nullptr) return EXACTA_INVALID_ARGUMENT;
  try {
    return mask_extents(masks, count, shape, ndim, lo, hi, num_threads);
  } catch (const std::bad_alloc &) {
    return EXACTA_OUT_OF_MEMORY;
  }
}

extern "C" int exacta_dilate_u8(const uint8_t *mask, uint8_t *out, const int64_t *shape, int ndim,
                                int connectivity, int steps, int num_threads) {
  if (shape == nullptr || ndim < 1 || ndim > 3 || steps < 0) return EXACTA_INVALID_ARGUMENT;
  if (connectivity < 1 || connectivity > ndim) return EXACTA_INVALID_ARGUMENT;
  for (int axis = 0; axis < ndim; ++axis) {
    if (shape[axis] < 0) return EXACTA_INVALID_ARGUMENT;
  }
  if (any_empty(shape, ndim)) return EXACTA_OK;
  if (mask == nullptr || out == nullptr) return EXACTA_INVALID_ARGUMENT;
  try {
    return dilate(mask, out, shape, ndim, connectivity, steps, num_threads);
  } catch (const std::bad_alloc &) {
    return EXACTA_OUT_OF_MEMORY;
  }
}
