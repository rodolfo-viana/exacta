// Connected-component labelling of a binary image.
//
// Run-based union-find. Each row (a line along the last axis) is split into runs of nonzero
// elements; a run is joined with the overlapping runs of the earlier rows it neighbours, at most
// four in 3-D. Provisional labels are created in raster order and equivalent labels are always
// merged into the smaller one, so each component's root is the label of its first run in raster
// (C) order, and numbering roots in increasing order numbers components by first appearance:
// the same labels as scipy.ndimage.label.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

#include "exacta/kernels.h"

namespace {

std::int32_t find(std::vector<std::int32_t> &parent, std::int32_t x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];  // path halving
    x = parent[x];
  }
  return x;
}

void unite(std::vector<std::int32_t> &parent, std::int32_t a, std::int32_t b) {
  a = find(parent, a);
  b = find(parent, b);
  if (a < b) {
    parent[b] = a;
  } else if (b < a) {
    parent[a] = b;
  }
}

struct Run {
  std::int64_t start, end;  // [start, end) along the last axis
  std::int32_t label;
};

// An earlier row that can hold neighbours: offset (d0, d1) and whether elements one step apart
// along the last axis still count as neighbours (diagonal contact).
struct RowOffset {
  std::int64_t d0, d1;
  std::int64_t reach;
};

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

  std::vector<Run> runs;
  std::vector<std::int64_t> row_start(rows + 1);
  std::vector<std::int32_t> parent;
  try {
    parent.push_back(0);  // label 0 is the background
    std::vector<std::int64_t> cursor(earlier.size());
    for (std::int64_t r = 0; r < rows; ++r) {
      row_start[r] = static_cast<std::int64_t>(runs.size());
      const std::int64_t r0 = r / n[1], r1 = r % n[1];
      std::int64_t neighbour_row[4];
      for (std::size_t k = 0; k < earlier.size(); ++k) {
        const std::int64_t q0 = r0 + earlier[k].d0, q1 = r1 + earlier[k].d1;
        neighbour_row[k] = (q0 < 0 || q1 < 0 || q1 >= n[1]) ? -1 : q0 * n[1] + q1;
        cursor[k] = neighbour_row[k] < 0 ? 0 : row_start[neighbour_row[k]];
      }
      const std::uint8_t *row = mask + r * width;
      for (std::int64_t x = 0; x < width;) {
        if (!row[x]) {
          ++x;
          continue;
        }
        const std::int64_t start = x;
        while (x < width && row[x]) ++x;
        // A run adopts the label of the first earlier run it touches and is united with the
        // others. Only the first run of a component creates a label, so labels still increase
        // in raster order of each component's first run.
        std::int32_t current = 0;
        for (std::size_t k = 0; k < earlier.size(); ++k) {
          const std::int64_t q = neighbour_row[k];
          if (q < 0) continue;
          const std::int64_t reach = earlier[k].reach, q_end = row_start[q + 1];
          // Runs of the neighbouring row that end before this one starts touch no later run.
          while (cursor[k] < q_end && runs[cursor[k]].end + reach <= start) ++cursor[k];
          for (std::int64_t b = cursor[k]; b < q_end && runs[b].start < x + reach; ++b) {
            const std::int32_t other = runs[b].label;
            if (current == 0) {
              current = other;
            } else if (other != current) {
              unite(parent, current, other);
            }
          }
        }
        if (current == 0) {
          current = static_cast<std::int32_t>(parent.size());
          parent.push_back(current);
        }
        runs.push_back({start, x, current});
      }
    }
    row_start[rows] = static_cast<std::int64_t>(runs.size());
  } catch (const std::bad_alloc &) {
    return EXACTA_OUT_OF_MEMORY;
  }

  // Roots are the smallest label of their component, so one increasing sweep numbers them.
  std::vector<std::int32_t> final_label;
  try {
    final_label.assign(parent.size(), 0);
  } catch (const std::bad_alloc &) {
    return EXACTA_OUT_OF_MEMORY;
  }
  std::int32_t count = 0;
  for (std::size_t l = 1; l < parent.size(); ++l) {
    const auto provisional = static_cast<std::int32_t>(l);
    const std::int32_t root = find(parent, provisional);
    final_label[l] = root == provisional ? ++count : final_label[root];
  }

  for (std::int64_t r = 0; r < rows; ++r) {
    std::int32_t *row = out + r * width;
    std::int64_t x = 0;
    for (std::int64_t i = row_start[r]; i < row_start[r + 1]; ++i) {
      std::fill(row + x, row + runs[i].start, 0);
      std::fill(row + runs[i].start, row + runs[i].end, final_label[runs[i].label]);
      x = runs[i].end;
    }
    std::fill(row + x, row + width, 0);
  }
  *num_labels = count;
  return EXACTA_OK;
}

}  // namespace

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
