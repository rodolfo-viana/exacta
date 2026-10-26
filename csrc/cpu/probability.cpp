// Independent Bernoulli scores and reliability histograms. Validate and reduce in one pass.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <new>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "exacta/kernels.h"

namespace {

template <typename P, typename T>
int probability_stats(const P *pred, const T *target, const uint8_t *mask,
    int64_t rows, int64_t n, int64_t channels, int mask_channels, int from_logits,
    int kind, int bins, double *out, int num_threads) {
  if (rows < 0 || n < 0 || channels < 1 || rows % channels || kind < 0 || kind > 2 ||
      (kind == 2 && (bins < 1 || bins > INT32_MAX / 3)) ||
      (mask_channels != 0 && mask_channels != 1 && mask_channels != channels) ||
      (rows > 0 && !out) || (rows > 0 && n > 0 && (!pred || !target || (mask_channels && !mask))))
    return EXACTA_INVALID_ARGUMENT;
  int threads = 1;
#ifdef _OPENMP
  threads = num_threads > 0 ? num_threads : omp_get_max_threads();
#else
  (void)num_threads;
#endif
  const int width = kind == 2 ? 3 * bins : 2;
  std::vector<double> partials(static_cast<std::size_t>(threads) * width);
  bool invalid = false;
  for (int64_t row = 0; row < rows; ++row) {
    std::fill(partials.begin(), partials.end(), 0.0);
    const int64_t mask_row = mask_channels == 1 ? row / channels : row;
#pragma omp parallel num_threads(threads) reduction(|| : invalid)
    {
      int thread = 0;
#ifdef _OPENMP
      thread = omp_get_thread_num();
#endif
      double *sums = partials.data() + static_cast<std::size_t>(thread) * width;
      double count = 0, total = 0;
#pragma omp for schedule(static)
      for (int64_t i = 0; i < n; ++i) {
        if (mask_channels && !mask[mask_row * n + i]) continue;
        const T y = target[row * n + i];
        const double x = static_cast<double>(pred[row * n + i]);
        if ((y != T(0) && y != T(1)) || !std::isfinite(x) ||
            (!from_logits && (x < 0 || x > 1))) {
          invalid = true;
          continue;
        }
        const bool positive = y == T(1);
        if (kind == 1) {
          double value;
          if (from_logits) {
            const double z = positive ? -x : x;
            value = std::max(z, 0.0) + std::log1p(std::exp(-std::abs(z)));
          } else {
            value = positive ? -std::log(x) : -std::log1p(-x);
          }
          count += 1;
          total += value;
        } else {
          double p = x;
          if (from_logits) {
            const double e = std::exp(-std::abs(x));
            p = x >= 0 ? 1.0 / (1.0 + e) : e / (1.0 + e);
          }
          if (kind == 0) {
            const double error = p - static_cast<double>(positive);
            count += 1;
            total += error * error;
          } else {
            const int bin = std::min(static_cast<int>(p * bins), bins - 1);
            sums[bin] += 1;
            sums[bins + bin] += p;
            sums[2 * bins + bin] += positive;
          }
        }
      }
      if (kind != 2) { sums[0] = count; sums[1] = total; }
    }
    for (int col = 0; col < width; ++col) {
      double sum = 0;
      for (int thread = 0; thread < threads; ++thread) sum += partials[static_cast<std::size_t>(thread) * width + col];
      out[row * width + col] = sum;
    }
  }
  return invalid ? EXACTA_INVALID_ARGUMENT : EXACTA_OK;
}

}  // namespace

#define EXACTA_PROBABILITY(P_NAME, P_TYPE, T_NAME, T_TYPE)                                  \
  extern "C" int exacta_probability_##P_NAME##_##T_NAME(                                  \
      const P_TYPE *pred, const T_TYPE *target, const uint8_t *mask,                        \
      int64_t rows, int64_t n, int64_t channels, int mask_channels, int from_logits,        \
      int kind, int bins, double *out, int num_threads) {                                 \
    try {                                                                                \
      return probability_stats(pred, target, mask, rows, n, channels, mask_channels,      \
                                from_logits, kind, bins, out, num_threads);              \
    } catch (const std::bad_alloc &) { return EXACTA_OUT_OF_MEMORY; }                      \
  }

EXACTA_PROBABILITY(f32, float, u8, uint8_t)
EXACTA_PROBABILITY(f32, float, i64, int64_t)
EXACTA_PROBABILITY(f32, float, f32, float)
EXACTA_PROBABILITY(f32, float, f64, double)
EXACTA_PROBABILITY(f64, double, u8, uint8_t)
EXACTA_PROBABILITY(f64, double, i64, int64_t)
EXACTA_PROBABILITY(f64, double, f32, float)
EXACTA_PROBABILITY(f64, double, f64, double)
