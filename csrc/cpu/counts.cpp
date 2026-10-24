// Voxel counts behind the overlap and volume metrics, in one pass over the inputs.
//
// confusion: (C, C) matrix of two label maps, cm[i][j] = voxels predicted i with target j.
// channel_counts: per channel of two stacks of binary masks, |P & T|, |P| and |T|.
// label_overlap: sizes of the labels of two label maps, and counts of the label pairs that meet.
// histogram: elements of each value of a label map.
// All validate as they count (labels in range, masks binary), so the caller needs no extra pass.

#include <algorithm>
#include <cstdint>
#include <new>
#include <unordered_map>
#include <utility>
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

int team_size() {
#ifdef _OPENMP
  return omp_get_num_threads();
#else
  return 1;
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

// Labels of connected components are spatially coherent: along a row the values come in runs. Each
// thread scans a contiguous range and adds a run's length to the shared histograms (atomically) or
// to its own hash map of pairs only where the run ends, so the cost is one pass over the data.
// Pairs are keyed i * (nb + 1) + j, which orders them by (i, j).
using Pairs = std::vector<std::pair<std::uint64_t, std::int64_t>>;

int label_overlap(const std::int32_t *a, const std::int32_t *b, std::int64_t n, std::int64_t na,
                  std::int64_t nb, std::int64_t *size_a, std::int64_t *size_b, Pairs &pairs,
                  int num_threads) {
  std::fill(size_a, size_a + na + 1, 0);
  std::fill(size_b, size_b + nb + 1, 0);
  const int threads = threads_for(num_threads);
  std::vector<std::unordered_map<std::uint64_t, std::int64_t>> maps(threads);
  bool out_of_range = false, out_of_memory = false;
#pragma omp parallel num_threads(threads) reduction(|| : out_of_range, out_of_memory)
  {
    const std::int64_t id = thread_id(), count = team_size();
    const std::int64_t begin = n * id / count, end = n * (id + 1) / count;
    std::int32_t last_a = 0, last_b = 0;
    std::uint64_t last_key = 0;
    std::int64_t run_a = 0, run_b = 0, run_pair = 0;
    try {
      auto &map = maps[id];
      for (std::int64_t i = begin; i < end; ++i) {
        const std::int32_t x = a[i], y = b[i];
        if (x < 0 || x > na || y < 0 || y > nb) {
          out_of_range = true;
          break;
        }
        if (x != last_a) {
#pragma omp atomic
          size_a[last_a] += run_a;
          last_a = x;
          run_a = 0;
        }
        ++run_a;
        if (y != last_b) {
#pragma omp atomic
          size_b[last_b] += run_b;
          last_b = y;
          run_b = 0;
        }
        ++run_b;
        if (x > 0 && y > 0) {
          const std::uint64_t key = static_cast<std::uint64_t>(x) * (nb + 1) + y;
          if (key != last_key) {
            if (run_pair > 0) map[last_key] += run_pair;
            last_key = key;
            run_pair = 0;
          }
          ++run_pair;
        }
      }
      if (run_pair > 0) map[last_key] += run_pair;
    } catch (const std::bad_alloc &) {
      out_of_memory = true;
    }
#pragma omp atomic
    size_a[last_a] += run_a;
#pragma omp atomic
    size_b[last_b] += run_b;
  }
  if (out_of_memory) return EXACTA_OUT_OF_MEMORY;
  if (out_of_range) return EXACTA_INVALID_ARGUMENT;
  for (const auto &map : maps) pairs.insert(pairs.end(), map.begin(), map.end());
  std::sort(pairs.begin(), pairs.end());
  std::size_t kept = 0;  // merge the threads' counts of the same pair
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    if (kept > 0 && pairs[kept - 1].first == pairs[k].first) {
      pairs[kept - 1].second += pairs[k].second;
    } else {
      pairs[kept++] = pairs[k];
    }
  }
  pairs.resize(kept);
  return EXACTA_OK;
}

// Run-length counting as in label_overlap, for one label map.
int histogram(const std::int32_t *values, std::int64_t n, std::int64_t bins, std::int64_t *out,
              int num_threads) {
  std::fill(out, out + bins, 0);
  bool out_of_range = false;
#pragma omp parallel num_threads(threads_for(num_threads)) reduction(|| : out_of_range)
  {
    const std::int64_t id = thread_id(), count = team_size();
    const std::int64_t begin = n * id / count, end = n * (id + 1) / count;
    std::int32_t last = 0;
    std::int64_t run = 0;
    for (std::int64_t i = begin; i < end; ++i) {
      const std::int32_t v = values[i];
      if (v < 0 || v >= bins) {
        out_of_range = true;
        break;
      }
      if (v != last) {
#pragma omp atomic
        out[last] += run;
        last = v;
        run = 0;
      }
      ++run;
    }
#pragma omp atomic
    out[last] += run;
  }
  return out_of_range ? EXACTA_INVALID_ARGUMENT : EXACTA_OK;
}

}  // namespace

extern "C" int exacta_histogram_i32(const int32_t *values, int64_t n, int64_t bins, int64_t *out,
                                    int num_threads) {
  if (n < 0 || bins < 1 || out == nullptr || (n > 0 && values == nullptr)) {
    return EXACTA_INVALID_ARGUMENT;
  }
  return histogram(values, n, bins, out, num_threads);
}

extern "C" int exacta_label_overlap_i32(const int32_t *a, const int32_t *b, int64_t n,
                                        int64_t na, int64_t nb, int64_t *size_a, int64_t *size_b,
                                        int64_t *pairs, int64_t capacity, int64_t *num_pairs,
                                        int num_threads) {
  if (n < 0 || na < 0 || nb < 0 || na > INT32_MAX || nb > INT32_MAX || capacity < 0 ||
      size_a == nullptr || size_b == nullptr || num_pairs == nullptr ||
      (capacity > 0 && pairs == nullptr) || (n > 0 && (a == nullptr || b == nullptr))) {
    return EXACTA_INVALID_ARGUMENT;
  }
  try {
    Pairs found;
    const int status = label_overlap(a, b, n, na, nb, size_a, size_b, found, num_threads);
    if (status != EXACTA_OK) return status;
    *num_pairs = static_cast<int64_t>(found.size());
    if (*num_pairs > capacity) return EXACTA_OK;  // the caller asks again with enough room
    for (std::size_t k = 0; k < found.size(); ++k) {
      pairs[3 * k] = static_cast<int64_t>(found[k].first / (nb + 1));
      pairs[3 * k + 1] = static_cast<int64_t>(found[k].first % (nb + 1));
      pairs[3 * k + 2] = found[k].second;
    }
    return EXACTA_OK;
  } catch (const std::bad_alloc &) {
    return EXACTA_OUT_OF_MEMORY;
  }
}

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
