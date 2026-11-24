// Lee (1994) thinning. Euler tables and deletion order adapted from scikit-image
// v0.25.2 _skeletonize_lee_cy.pyx.in (BSD-3-Clause); see THIRD_PARTY_LICENSES.
// Connectivity is checked with a 27-bit flood fill instead of recursive octrees.
// Zhang-Suen (1984) 2-D thinning with scikit-image v0.26.0's lookup table and two-subiteration
// schedule (_skeletonize_various_cy.pyx, _fast_skeletonize; BSD-3-Clause).
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>
#include "exacta/kernels.h"

namespace {
constexpr int euler_lut[128] = {1,-1,-1,1,-3,-1,-1,1,-1,1,1,-1,3,1,1,-1,-3,-1,3,1,1,-1,3,1,-1,1,1,-1,3,1,1,-1,-3,3,-1,1,1,3,-1,1,-1,1,1,-1,3,1,1,-1,1,3,3,1,5,3,3,1,-1,1,1,-1,3,1,1,-1,-7,-1,-1,1,-3,-1,-1,1,-1,1,1,-1,3,1,1,-1,-3,-1,3,1,1,-1,3,1,-1,1,1,-1,3,1,1,-1,-3,3,-1,1,1,3,-1,1,-1,1,1,-1,3,1,1,-1,1,3,3,1,5,3,3,1,-1,1,1,-1,3,1,1,-1};
constexpr int octants[8][7] = {{2,1,11,10,5,4,14},{0,9,3,12,1,10,4},{8,7,17,16,5,4,14},{6,15,7,16,3,12,4},{20,23,19,22,11,14,10},{18,21,9,12,19,22,10},{26,23,17,14,25,22,16},{24,25,15,16,21,22,12}};
constexpr std::uint32_t adjacency[27] = {5658u,24125u,19506u,104147u,253935u,216470u,102552u,250232u,213168u,7083035u,16538175u,14173238u,57509595u,134209535u,115019190u,56692952u,132305400u,113328560u,6821376u,16014848u,13126656u,55416320u,130014720u,106646528u,39948288u,98816000u,46350336u};

bool simple(const std::array<std::uint8_t, 27>& n) {
  std::uint32_t remaining = 0;
  for (int i = 0; i < 27; ++i) if (i != 13 && n[i]) remaining |= 1u << i;
  if (!remaining) return true;
  auto frontier = remaining & (~remaining + 1u);
  remaining ^= frontier;
  while (frontier) {
    std::uint32_t neighbours = 0;
    for (int i = 0; i < 27; ++i) if (frontier & (1u << i)) neighbours |= adjacency[i];
    frontier = neighbours & remaining;
    remaining &= ~frontier;
  }
  return remaining == 0;
}

bool candidate(const std::array<std::uint8_t, 27>& n) {
  int count = 0;
  for (auto value : n) count += value;
  if (count == 2) return false;  // endpoint, including centre
  int euler = 0;
  for (const auto& octant : octants) {
    int code = 0;
    for (int j = 0; j < 7; ++j) code |= n[octant[j]] << (6-j);
    euler += euler_lut[code];
  }
  return euler == 0 && simple(n);
}
// scikit-image's table, indexed by the 8 neighbours (bit 0 north-west, then clockwise: north,
// north-east, east, south-east, south, south-west, west). 1: removable in the first
// subiteration, 2: in the second, 3: in both. It is not the textbook Zhang-Suen rule set (25
// entries differ), so it is copied rather than derived, to reproduce scikit-image exactly.
constexpr std::uint8_t zhang_lut[256] = {
    0,0,0,1,0,0,1,3,0,0,3,1,1,0,1,3,0,0,0,0,0,0,0,0,2,0,2,0,3,0,3,3,
    0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,3,0,2,2,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    2,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,3,0,0,0,0,0,0,0,3,0,0,0,3,0,2,0,
    0,0,3,1,0,0,1,3,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    2,3,1,3,0,0,1,3,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    2,3,0,1,0,0,0,1,0,0,0,0,0,0,0,0,3,3,0,1,0,0,0,0,2,2,0,0,2,0,0,0,
};
}  // namespace

extern "C" int exacta_skeleton_zhang_u8(const std::uint8_t* mask, std::uint8_t* out,
                                         const std::int64_t* shape) {
  const std::int64_t h = shape[0], w = shape[1];
  if (h < 0 || w < 0 || h > std::numeric_limits<std::int64_t>::max() - 2 ||
      w > std::numeric_limits<std::int64_t>::max() - 2)
    return EXACTA_INVALID_ARGUMENT;
  if (!h || !w) return EXACTA_OK;
  const std::int64_t row = w + 2;
  if (h + 2 > std::numeric_limits<std::int64_t>::max() / row) return EXACTA_INVALID_ARGUMENT;
  try {
    std::vector<std::uint8_t> img((h + 2) * row, 0);
    std::vector<std::int64_t> active, removed;
    for (std::int64_t y = 0; y < h; ++y)
      for (std::int64_t x = 0; x < w; ++x) {
        const auto padded = (y + 1) * row + x + 1;
        img[padded] = mask[y * w + x] != 0;
        if (img[padded]) active.push_back(padded);
      }
    const std::int64_t offsets[8] = {-row - 1, -row, -row + 1, 1, row + 1, row, row - 1, -1};
    bool changed = true;
    while (changed) {
      changed = false;
      for (int pass = 1; pass <= 2; ++pass) {
        // Every decision of a subiteration reads the image as it was before it (scikit-image
        // writes removals to a copy and swaps after the subiteration).
        removed.clear();
        for (auto index : active) {
          int code = 0;
          for (int j = 0; j < 8; ++j) code |= img[index + offsets[j]] << j;
          const int kind = zhang_lut[code];
          if (kind == 3 || kind == pass) removed.push_back(index);
        }
        for (auto index : removed) img[index] = 0;
        if (!removed.empty()) {
          changed = true;
          active.erase(std::remove_if(active.begin(), active.end(), [&](auto i) { return !img[i]; }),
                       active.end());
        }
      }
    }
    for (std::int64_t y = 0; y < h; ++y)
      for (std::int64_t x = 0; x < w; ++x) out[y * w + x] = img[(y + 1) * row + x + 1];
    return EXACTA_OK;
  } catch (const std::bad_alloc&) { return EXACTA_OUT_OF_MEMORY; }
    catch (...) { return EXACTA_INTERNAL_ERROR; }
}

extern "C" int exacta_skeleton_u8(const std::uint8_t* mask, std::uint8_t* out,
                                   const std::int64_t* shape, int ndim) {
  if (ndim != 2 && ndim != 3) return EXACTA_INVALID_ARGUMENT;
  std::int64_t count = 1;
  for (int i = 0; i < ndim; ++i) {
    if (shape[i] < 0 || shape[i] > std::numeric_limits<std::int64_t>::max() - 2)
      return EXACTA_INVALID_ARGUMENT;
    if (!shape[i]) return EXACTA_OK;
    if (count > std::numeric_limits<std::int64_t>::max() / (shape[i]+2))
      return EXACTA_INVALID_ARGUMENT;
    count *= shape[i]+2;
  }
  if (ndim == 2 && count > std::numeric_limits<std::int64_t>::max()/3)
    return EXACTA_INVALID_ARGUMENT;
  const std::int64_t d = ndim == 3 ? shape[0] : 1;
  const std::int64_t h = shape[ndim-2], w = shape[ndim-1];
  const std::int64_t row = w+2, plane = (h+2)*row;
  try {
    std::vector<std::uint8_t> img((d+2)*plane, 0);
    std::vector<std::int64_t> active, candidates;
    for (std::int64_t z=0; z<d; ++z) for (std::int64_t y=0; y<h; ++y)
      for (std::int64_t x=0; x<w; ++x) {
        const auto padded = (z+1)*plane + (y+1)*row + x+1;
        img[padded] = mask[(z*h+y)*w+x] != 0;
        if (img[padded]) active.push_back(padded);
      }
    const std::int64_t borders[] = {-row, row, 1, -1, plane, -plane};
    std::array<std::int64_t,27> offsets{};
    for (int p=-1; p<=1; ++p) for (int c=-1; c<=1; ++c) for (int r=-1; r<=1; ++r)
      offsets[(p+1)*9+(c+1)*3+r+1] = p*plane+r*row+c;
    auto neighbourhood = [&](std::int64_t index) {
      std::array<std::uint8_t,27> n{};
      for (int j=0; j<27; ++j) n[j] = img[index+offsets[j]];
      return n;
    };
    bool changed = true;
    while (changed) {
      changed = false;
      for (int direction=0; direction<(d == 1 ? 4 : 6); ++direction) {
        candidates.clear();
        for (auto index : active)
          if (img[index] && !img[index+borders[direction]] && candidate(neighbourhood(index)))
            candidates.push_back(index);
        // Sequential recheck is essential: simultaneous deletion can break topology.
        for (auto index : candidates) if (simple(neighbourhood(index))) {
          img[index] = 0;
          changed = true;
        }
      }
      active.erase(std::remove_if(active.begin(), active.end(), [&](auto i){return !img[i];}), active.end());
    }
    for (std::int64_t z=0; z<d; ++z) for (std::int64_t y=0; y<h; ++y)
      for (std::int64_t x=0; x<w; ++x)
        out[(z*h+y)*w+x] = img[(z+1)*plane+(y+1)*row+x+1];
    return EXACTA_OK;
  } catch (const std::bad_alloc&) { return EXACTA_OUT_OF_MEMORY; }
    catch (...) { return EXACTA_INTERNAL_ERROR; }
}
