// Voxel counts behind the overlap and volume metrics, in one pass over the inputs.
//
// confusion: (C, C) matrix of two label maps, cm[i][j] = voxels predicted i with target j.
// channel_counts: per channel of two stacks of binary masks, |P & T|, |P| and |T|.
// Both validate as they count (labels in [0, C), masks binary), so the caller needs no extra pass.

#include <algorithm>
#include <cstdint>
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

template <typename T>
int confusion(const T *pred, const T *target, std::int64_t n, std::int64_t classes,
              std::int64_t *out, int num_threads) {
  const std::int64_t cells = classes * classes;
  // One table per thread, as long as they fit in 64 MiB; with very many classes, fewer threads.
  const std::int64_t fit = std::max<std::int64_t>(1, (std::int64_t(64) << 20) / (cells * 8));
  const int threads = static_cast<int>(std::min<std::int64_t>(threads_for(num_threads), fit));
  std::vector<std::int64_t> tables(static_cast<std::size_t>(threads) * cells, 0);
  bool out_of_range = false;
#pragma omp parallel num_threads(threads) reduction(|| : out_of_range)
  {
    std::int64_t *table = tables.data() + thread_id() * cells;
#pragma omp for schedule(static)
    for (std::int64_t i = 0; i < n; ++i) {
      const auto p = static_cast<std::int64_t>(pred[i]), t = static_cast<std::int64_t>(target[i]);
      if (p < 0 || p >= classes || t < 0 || t >= classes) {
        out_of_range = true;
        continue;
      }
      ++table[p * classes + t];
    }
  }
  if (out_of_range) return EXACTA_INVALID_ARGUMENT;
  for (std::int64_t c = 0; c < cells; ++c) {
    std::int64_t s = 0;
    for (int t = 0; t < threads; ++t) s += tables[t * cells + c];
    out[c] = s;
  }
  return EXACTA_OK;
}

// `count` masks of `n` elements each, for prediction and target (C order, mask-major). Values
// must be 0 or 1.
template <typename T>
int channel_counts(const T *pred, const T *target, std::int64_t count, std::int64_t n,
                   std::int64_t *both, std::int64_t *in_pred, std::int64_t *in_target,
                   int num_threads) {
  const int threads = threads_for(num_threads);
  bool not_binary = false;
  for (std::int64_t c = 0; c < count; ++c) {
    const T *p = pred + c * n, *t = target + c * n;
    std::int64_t sum_both = 0, sum_p = 0, sum_t = 0;
#pragma omp parallel for num_threads(threads) schedule(static) \
    reduction(+ : sum_both, sum_p, sum_t) reduction(|| : not_binary)
    for (std::int64_t i = 0; i < n; ++i) {
      const bool a = p[i] != T(0), b = t[i] != T(0);
      if ((a && p[i] != T(1)) || (b && t[i] != T(1))) not_binary = true;
      sum_both += a && b;
      sum_p += a;
      sum_t += b;
    }
    both[c] = sum_both;
    in_pred[c] = sum_p;
    in_target[c] = sum_t;
  }
  return not_binary ? EXACTA_INVALID_ARGUMENT : EXACTA_OK;
}

}  // namespace

#define EXACTA_CONFUSION(SUFFIX, TYPE)                                                          \
  extern "C" int exacta_confusion_##SUFFIX(const TYPE *pred, const TYPE *target, int64_t n,     \
                                           int64_t classes, int64_t *out, int num_threads) {    \
    if (n < 0 || classes < 1 || out == nullptr) return EXACTA_INVALID_ARGUMENT;                 \
    if (n > 0 && (pred == nullptr || target == nullptr)) return EXACTA_INVALID_ARGUMENT;        \
    try {                                                                                       \
      return confusion(pred, target, n, classes, out, num_threads);                             \
    } catch (const std::bad_alloc &) {                                                          \
      return EXACTA_OUT_OF_MEMORY;                                                              \
    }                                                                                           \
  }

EXACTA_CONFUSION(u8, uint8_t)
EXACTA_CONFUSION(i32, int32_t)
EXACTA_CONFUSION(i64, int64_t)

#define EXACTA_CHANNEL_COUNTS(SUFFIX, TYPE)                                                       \
  extern "C" int exacta_channel_counts_##SUFFIX(const TYPE *pred, const TYPE *target,             \
                                                int64_t count, int64_t n, int64_t *both,          \
                                                int64_t *in_pred, int64_t *in_target,             \
                                                int num_threads) {                                \
    if (count < 0 || n < 0 || both == nullptr || in_pred == nullptr || in_target == nullptr ||   \
        (count * n > 0 && (pred == nullptr || target == nullptr))) {                              \
      return EXACTA_INVALID_ARGUMENT;                                                             \
    }                                                                                             \
    return channel_counts(pred, target, count, n, both, in_pred, in_target, num_threads);       \
  }

EXACTA_CHANNEL_COUNTS(u8, uint8_t)
EXACTA_CHANNEL_COUNTS(i64, int64_t)
EXACTA_CHANNEL_COUNTS(f32, float)
