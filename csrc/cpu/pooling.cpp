// Fixed-neighbourhood morphology for soft skeletons. CPU tensors, no libtorch link.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "exacta/kernels.h"

namespace {
template <typename T>
int pool(const T* input, T* output, std::int64_t* indices, const std::int64_t* shape,
         int ndim, int erosion, int num_threads) {
  if ((ndim != 4 && ndim != 5) || (erosion != 0 && erosion != 1)) return EXACTA_INVALID_ARGUMENT;
  std::int64_t n = 1;
  for (int i=0; i<ndim; ++i) {
    if (shape[i] < 0 || (shape[i] && n > std::numeric_limits<std::int64_t>::max()/shape[i]))
      return EXACTA_INVALID_ARGUMENT;
    n *= shape[i];
  }
  if (!n) return EXACTA_OK;
  const auto d = ndim == 5 ? shape[2] : 1;
  const auto h = shape[ndim-2], w = shape[ndim-1], plane = h*w;
  int threads = 1;
#ifdef _OPENMP
  threads = num_threads > 0 ? num_threads : omp_get_max_threads();
#else
  (void)num_threads;
#endif
  // Avoid creating a team for small tensors, common in gradient/reference tests.
  if (n < 32768) threads = 1;
#pragma omp parallel for schedule(static) num_threads(threads)
  for (std::int64_t i=0; i<n; ++i) {
    const auto x = i%w, y = (i/w)%h, z = (i/plane)%d;
    T best = erosion ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
    std::int64_t arg = i;
    for (int dz = (ndim == 5 ? -1 : 0); dz <= (ndim == 5 ? 1 : 0); ++dz) {
      if (z+dz < 0 || z+dz >= d) continue;
      for (int dy=-1; dy<=1; ++dy) {
        if (y+dy < 0 || y+dy >= h) continue;
        for (int dx=-1; dx<=1; ++dx) {
          if (x+dx < 0 || x+dx >= w || (erosion && std::abs(dx)+std::abs(dy)+std::abs(dz)>1)) continue;
          const auto j = i+dz*plane+dy*w+dx;
          const T value = input[j];
          if ((erosion ? value < best : value > best) || std::isnan(value)) {
            best = value; arg = j;
          }
        }
      }
    }
    output[i] = best;
    if (indices) indices[i] = arg;
  }
  return EXACTA_OK;
}
}

extern "C" int exacta_soft_pool_f32(const float* x, float* out, int64_t* indices,
                                     const int64_t* shape, int ndim, int erosion, int threads) {
  return pool(x,out,indices,shape,ndim,erosion,threads);
}
extern "C" int exacta_soft_pool_f64(const double* x, double* out, int64_t* indices,
                                     const int64_t* shape, int ndim, int erosion, int threads) {
  return pool(x,out,indices,shape,ndim,erosion,threads);
}
