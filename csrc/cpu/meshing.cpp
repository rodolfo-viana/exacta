// Surfaces of 2-D and 3-D masks from 2x2(x2) blocks of voxels.
//
// neighbour_codes: DeepMind's neighbour code at every voxel corner, one bit per voxel of the block
// around the corner (voxels outside the image count as outside).
// mesh_elements: the elements (triangles or segments) a table assigns to each code, placed at
// their blocks, as exacta.functional.surface._mesh lays them out.

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

// Corner grid of a mask of shape n: n + 1 corners per axis, as 3-D (a 2-D grid has one plane).
struct Grid {
  int ndim;
  std::int64_t n[3];  // mask extents, leading 1 in 2-D
  std::int64_t c[3];  // corner extents, leading 1 in 2-D
  std::int64_t rows() const { return c[0] * c[1]; }
};

Grid grid_of(const std::int64_t *shape, int ndim) {
  Grid g;
  g.ndim = ndim;
  g.n[0] = ndim == 3 ? shape[0] : 1;
  g.n[1] = shape[ndim - 2];
  g.n[2] = shape[ndim - 1];
  g.c[0] = ndim == 3 ? g.n[0] + 1 : 1;
  g.c[1] = g.n[1] + 1;
  g.c[2] = g.n[2] + 1;
  return g;
}

// Codes of one row of corners (r0, r1, all x). The block of corner c holds voxels c - 1 + o for
// offsets o in {0, 1}^ndim; offset number i (last axis fastest) sets bit 2^ndim - 1 - i.
void code_row(const std::uint8_t *mask, const Grid &g, std::int64_t r0, std::int64_t r1,
              std::uint8_t *out) {
  const int bits = 1 << g.ndim;
  const std::uint8_t *voxel_rows[4];  // voxel rows of offsets (a, b) in 3-D, (b) in 2-D
  int count = 0;
  for (int a = 0; a < (g.ndim == 3 ? 2 : 1); ++a) {
    for (int b = 0; b < 2; ++b) {
      const std::int64_t v0 = g.ndim == 3 ? r0 - 1 + a : 0, v1 = r1 - 1 + b;
      const bool inside = v0 >= 0 && v0 < g.n[0] && v1 >= 0 && v1 < g.n[1];
      voxel_rows[count++] = inside ? mask + (v0 * g.n[1] + v1) * g.n[2] : nullptr;
    }
  }
  for (std::int64_t x = 0; x < g.c[2]; ++x) {
    unsigned code = 0;
    for (int k = 0; k < count; ++k) {
      if (voxel_rows[k] == nullptr) continue;
      for (int o = 0; o < 2; ++o) {
        const std::int64_t v = x - 1 + o;
        if (v >= 0 && v < g.n[2] && voxel_rows[k][v]) code |= 1u << (bits - 1 - (2 * k + o));
      }
    }
    out[x] = static_cast<std::uint8_t>(code);
  }
}

// Elements of the blocks of one row of corners, given the full code (2^bits - 1).
std::int64_t row_elements(const std::uint8_t *code, std::int64_t width, unsigned full,
                          const std::int64_t *counts) {
  std::int64_t total = 0;
  for (std::int64_t x = 0; x < width; ++x) {
    if (code[x] != 0 && code[x] != full) total += counts[code[x]];
  }
  return total;
}

}  // namespace

extern "C" int exacta_neighbour_codes_u8(const uint8_t *mask, const int64_t *shape, int ndim,
                                         uint8_t *code, const int64_t *counts,
                                         int64_t *num_elements, int num_threads) {
  if (shape == nullptr || code == nullptr || (ndim != 2 && ndim != 3)) {
    return EXACTA_INVALID_ARGUMENT;
  }
  for (int axis = 0; axis < ndim; ++axis) {
    if (shape[axis] < 0) return EXACTA_INVALID_ARGUMENT;
  }
  const Grid g = grid_of(shape, ndim);
  if (mask == nullptr && g.n[0] * g.n[1] * g.n[2] > 0) return EXACTA_INVALID_ARGUMENT;
  const unsigned full = (1u << (1 << ndim)) - 1;
  std::int64_t total = 0;
#pragma omp parallel for num_threads(threads_for(num_threads)) schedule(static) reduction(+ : total)
  for (std::int64_t r = 0; r < g.rows(); ++r) {
    std::uint8_t *out = code + r * g.c[2];
    code_row(mask, g, r / g.c[1], r % g.c[1], out);
    if (counts != nullptr) total += row_elements(out, g.c[2], full, counts);
  }
  if (num_elements != nullptr) *num_elements = total;
  return EXACTA_OK;
}

extern "C" int exacta_mesh_elements_f64(const uint8_t *code, const int64_t *shape, int ndim,
                                        const int64_t *table, const int64_t *counts,
                                        int64_t max_per_code, const double *spacing,
                                        int64_t num_elements, double *elements, double *centres,
                                        int64_t *block, int64_t *element_code, int64_t *slot,
                                        int num_threads) {
  if (code == nullptr || shape == nullptr || table == nullptr || counts == nullptr ||
      spacing == nullptr || (ndim != 2 && ndim != 3) || max_per_code < 0 || num_elements < 0) {
    return EXACTA_INVALID_ARGUMENT;
  }
  if (num_elements > 0 && (elements == nullptr || centres == nullptr || block == nullptr ||
                           element_code == nullptr || slot == nullptr)) {
    return EXACTA_INVALID_ARGUMENT;
  }
  for (int axis = 0; axis < ndim; ++axis) {
    if (shape[axis] < 0) return EXACTA_INVALID_ARGUMENT;
  }
  const Grid g = grid_of(shape, ndim);
  const unsigned full = (1u << (1 << ndim)) - 1;
  const int threads = threads_for(num_threads);
  std::vector<std::int64_t> start;
  try {
    start.assign(static_cast<std::size_t>(g.rows()) + 1, 0);
  } catch (const std::bad_alloc &) {
    return EXACTA_OUT_OF_MEMORY;
  }
#pragma omp parallel for num_threads(threads) schedule(static)
  for (std::int64_t r = 0; r < g.rows(); ++r) {
    start[r + 1] = row_elements(code + r * g.c[2], g.c[2], full, counts);
  }
  for (std::int64_t r = 0; r < g.rows(); ++r) start[r + 1] += start[r];
  if (start[g.rows()] != num_elements) return EXACTA_INVALID_ARGUMENT;

  const int d = ndim, stride = d * d;
  // VTK's parametric centre of a triangle, as exacta.functional.surface evaluates it: products
  // and sums rounded one at a time, in this order (the build turns off fused multiply-adds).
  const double third = 1.0 / 3.0, first_weight = 1.0 - third - third;
#pragma omp parallel for num_threads(threads) schedule(static)
  for (std::int64_t r = 0; r < g.rows(); ++r) {
    const std::uint8_t *row = code + r * g.c[2];
    const std::int64_t corner[3] = {r / g.c[1], r % g.c[1], 0};
    std::int64_t e = start[r];
    for (std::int64_t x = 0; x < g.c[2]; ++x) {
      const unsigned value = row[x];
      if (value == 0 || value == full) continue;
      // The corner's position along the mask's axes (2-D drops the leading 1).
      const std::int64_t position[3] = {corner[0], corner[1], x};
      const std::int64_t *at = position + (3 - d);
      for (std::int64_t k = 0; k < counts[value]; ++k, ++e) {
        const std::int64_t *half = table + (value * max_per_code + k) * stride;
        double *v = elements + e * stride;
        for (int vertex = 0; vertex < d; ++vertex) {
          for (int i = 0; i < d; ++i) {
            // Vertices in half-voxels, exact integers: the table's offset plus twice the
            // block's first voxel (corner - 1).
            const std::int64_t h = half[vertex * d + i] + 2 * (at[i] - 1);
            v[vertex * d + i] = (static_cast<double>(h) / 2.0) * spacing[i];
          }
        }
        double *centre = centres + e * d;
        for (int i = 0; i < d; ++i) {
          centre[i] = d == 3 ? v[i] * first_weight + v[d + i] * third + v[2 * d + i] * third
                             : (v[i] + v[d + i]) / 2.0;
        }
        block[e] = r * g.c[2] + x;
        element_code[e] = value;
        slot[e] = k;
      }
    }
  }
  return EXACTA_OK;
}
