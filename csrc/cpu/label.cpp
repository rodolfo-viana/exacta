// Connected-component labelling of a binary image.
//
// Run-based union-find. Each row (a line along the last axis) is split into runs of nonzero
// elements; a run is joined with the overlapping runs of the earlier rows it neighbours, at most
// four in 3-D. Every run is a node, numbered by its position among all runs in raster (C) order,
// and equivalent nodes are always merged into the smaller one, so each component's root is its
// first run, and numbering roots in increasing order numbers components by first appearance: the
// same labels as scipy.ndimage.label.
//
// In parallel: the rows are split into blocks of about as many runs each, and each block joins its
// runs on its own (nodes are global, so blocks never share one). The runs that neighbour an earlier
// block are then joined across, and one sweep numbers the roots.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "exacta/kernels.h"

namespace {

struct Run {
  std::int32_t start, end;  // [start, end) along the last axis
};

// An earlier row that can hold neighbours: offset (d0, d1) and whether elements one step apart
// along the last axis still count as neighbours (diagonal contact).
struct RowOffset {
  std::int64_t d0, d1;
  std::int32_t reach;
};

std::int32_t find(std::int32_t *parent, std::int32_t x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];  // path halving
    x = parent[x];
  }
  return x;
}

void unite(std::int32_t *parent, std::int32_t a, std::int32_t b) {
  a = find(parent, a);
  b = find(parent, b);
  if (a < b) {
    parent[b] = a;
  } else if (b < a) {
    parent[a] = b;
  }
}

// Calls emit(start, end) for each run of nonzero elements of a row, in order.
template <typename Emit>
void scan_row(const std::uint8_t *row, std::int64_t width, Emit emit) {
  std::int64_t x = 0;
  while (x < width) {
    while (x + 8 <= width) {  // background, eight elements at a time
      std::uint64_t word;
      std::memcpy(&word, row + x, 8);
      if (word != 0) break;
      x += 8;
    }
    while (x < width && !row[x]) ++x;
    if (x == width) break;
    const std::int64_t start = x;
    while (x < width && row[x]) ++x;
    emit(start, x);
  }
}

// Joins every run of row r with the runs of row q it touches. Node of run i: i + 1.
void join_rows(const Run *runs, const std::int64_t *row_start, std::int64_t r, std::int64_t q,
               std::int32_t reach, std::int32_t *parent) {
  std::int64_t b = row_start[q];
  const std::int64_t b_end = row_start[q + 1];
  for (std::int64_t a = row_start[r]; a < row_start[r + 1]; ++a) {
    // Runs of row q that end before run a starts touch no later run of row r either.
    while (b < b_end && runs[b].end + reach <= runs[a].start) ++b;
    for (std::int64_t c = b; c < b_end && runs[c].start < runs[a].end + reach; ++c) {
      unite(parent, static_cast<std::int32_t>(a + 1), static_cast<std::int32_t>(c + 1));
    }
  }
}

int label(const std::uint8_t *mask, std::int32_t *out, const std::int64_t *shape, int ndim,
          int connectivity, std::int64_t *num_labels) {
  // Work in 3-D: a 1-D or 2-D image is a 3-D one with leading extents of 1.
  std::int64_t n[3] = {1, 1, 1};
  for (int axis = 0; axis < ndim; ++axis) n[3 - ndim + axis] = shape[axis];
  const std::int64_t total = n[0] * n[1] * n[2];
  if (total > std::numeric_limits<std::int32_t>::max()) return EXACTA_INVALID_ARGUMENT;
  const std::int64_t rows = n[0] * n[1], width = n[2];

  // Earlier rows in raster order: (0, -1), (-1, -1), (-1, 0), (-1, 1). A row offset with k
  // nonzero components holds neighbours only if k <= connectivity, and diagonal contact along
  // the last axis adds one more nonzero component.
  std::vector<RowOffset> earlier;
  const std::int64_t candidates[4][2] = {{0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};
  for (const auto &c : candidates) {
    const int nonzero = (c[0] != 0) + (c[1] != 0);
    if (nonzero <= connectivity) {
      earlier.push_back({c[0], c[1], nonzero + 1 <= connectivity ? 1 : 0});
    }
  }
  const auto neighbour_row = [&](std::int64_t r, const RowOffset &offset) -> std::int64_t {
    const std::int64_t q0 = r / n[1] + offset.d0, q1 = r % n[1] + offset.d1;
    return (q0 < 0 || q1 < 0 || q1 >= n[1]) ? -1 : q0 * n[1] + q1;
  };

#ifdef _OPENMP
  const int threads = omp_get_max_threads();
#else
  const int threads = 1;
#endif
  std::vector<std::int64_t> row_start(rows + 1, 0);
  std::vector<Run> runs;
  std::vector<std::int32_t> parent;
  try {
    // Runs of every row: count, place, then fill.
#pragma omp parallel for num_threads(threads) schedule(static)
    for (std::int64_t r = 0; r < rows; ++r) {
      std::int64_t count = 0;
      scan_row(mask + r * width, width, [&](std::int64_t, std::int64_t) { ++count; });
      row_start[r + 1] = count;
    }
    for (std::int64_t r = 0; r < rows; ++r) row_start[r + 1] += row_start[r];
    const std::int64_t num_runs = row_start[rows];
    runs.resize(static_cast<std::size_t>(num_runs));
    parent.resize(static_cast<std::size_t>(num_runs) + 1);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (std::int64_t r = 0; r < rows; ++r) {
      Run *next = runs.data() + row_start[r];
      scan_row(mask + r * width, width, [&](std::int64_t start, std::int64_t end) {
        *next++ = {static_cast<std::int32_t>(start), static_cast<std::int32_t>(end)};
      });
    }

    // Blocks of rows with about as many runs each; block k starts at row first[k].
    const int blocks = static_cast<int>(std::min<std::int64_t>(threads, rows));
    std::vector<std::int64_t> first(blocks + 1);
    for (int k = 0; k <= blocks; ++k) {
      const std::int64_t target = num_runs * k / blocks;
      first[k] = std::lower_bound(row_start.begin(), row_start.begin() + rows, target) -
                 row_start.begin();
    }
    first[blocks] = rows;

    std::int32_t *p = parent.data();
#pragma omp parallel for num_threads(threads) schedule(static, 1)
    for (int k = 0; k < blocks; ++k) {
      for (std::int64_t r = first[k]; r < first[k + 1]; ++r) {
        for (std::int64_t i = row_start[r]; i < row_start[r + 1]; ++i) {
          p[i + 1] = static_cast<std::int32_t>(i + 1);
        }
        for (const auto &offset : earlier) {
          const std::int64_t q = neighbour_row(r, offset);
          if (q >= first[k]) join_rows(runs.data(), row_start.data(), r, q, offset.reach, p);
        }
      }
    }
    // Across blocks: rows of a block whose neighbours lie in earlier blocks, at most one plane.
    for (int k = 1; k < blocks; ++k) {
      const std::int64_t end = std::min(first[k + 1], first[k] + n[1] + 1);
      for (std::int64_t r = first[k]; r < end; ++r) {
        for (const auto &offset : earlier) {
          const std::int64_t q = neighbour_row(r, offset);
          if (q >= 0 && q < first[k]) {
            join_rows(runs.data(), row_start.data(), r, q, offset.reach, p);
          }
        }
      }
    }

    // Every node points to a smaller one or to itself (a root), so in increasing order a node's
    // parent is numbered before it, and the numbers can replace the parents in place.
    std::int32_t count = 0;
    for (std::int64_t i = 1; i <= num_runs; ++i) {
      const std::int32_t up = p[i];
      p[i] = up == i ? ++count : p[up];
    }
    *num_labels = count;
  } catch (const std::bad_alloc &) {
    return EXACTA_OUT_OF_MEMORY;
  }

#pragma omp parallel for num_threads(threads) schedule(static)
  for (std::int64_t r = 0; r < rows; ++r) {
    std::int32_t *row = out + r * width;
    std::int64_t x = 0;
    for (std::int64_t i = row_start[r]; i < row_start[r + 1]; ++i) {
      std::fill(row + x, row + runs[i].start, 0);
      std::fill(row + runs[i].start, row + runs[i].end, parent[i + 1]);
      x = runs[i].end;
    }
    std::fill(row + x, row + width, 0);
  }
  return EXACTA_OK;
}

}  // namespace

extern "C" int exacta_select_labels_i32(const int32_t *labels, int64_t n, const uint8_t *select,
                                        int64_t count, uint8_t *out, int num_threads) {
  if (n < 0 || count < 1 || select == nullptr || (n > 0 && (labels == nullptr || out == nullptr))) {
    return EXACTA_INVALID_ARGUMENT;
  }
#ifdef _OPENMP
  const int threads = num_threads > 0 ? num_threads : omp_get_max_threads();
#else
  (void)num_threads;
  const int threads = 1;
#endif
  bool out_of_range = false;
#pragma omp parallel for num_threads(threads) schedule(static) reduction(|| : out_of_range)
  for (int64_t i = 0; i < n; ++i) {
    const int32_t l = labels[i];
    if (l < 0 || l >= count) {
      out_of_range = true;
      continue;
    }
    out[i] = select[l] != 0;
  }
  return out_of_range ? EXACTA_INVALID_ARGUMENT : EXACTA_OK;
}

extern "C" int exacta_label_u8(const uint8_t *mask, int32_t *out, const int64_t *shape, int ndim,
                               int connectivity, int64_t *num_labels) {
  if (shape == nullptr || num_labels == nullptr) return EXACTA_INVALID_ARGUMENT;
  if (ndim < 1 || ndim > 3 || connectivity < 1 || connectivity > ndim) {
    return EXACTA_INVALID_ARGUMENT;
  }
  bool empty = false;
  for (int axis = 0; axis < ndim; ++axis) {
    if (shape[axis] < 0) return EXACTA_INVALID_ARGUMENT;
    empty = empty || shape[axis] == 0;
  }
  if (empty) {  // the data pointers of an empty array may be null
    *num_labels = 0;
    return EXACTA_OK;
  }
  if (mask == nullptr || out == nullptr) return EXACTA_INVALID_ARGUMENT;
  return label(mask, out, shape, ndim, connectivity, num_labels);
}
