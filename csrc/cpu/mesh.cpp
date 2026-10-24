// Distance from points to the nearest element of a mesh: triangles (3-D) or segments (2-D).
//
// A k-d tree over element centroids, searched depth-first. Every vertex of an element lies within
// `reach` of the element's centroid, so a node whose bounding box (of centroids) is farther from
// the query than the best distance so far plus `reach` cannot hold a closer element. The result
// is the exact minimum over elements of the point-to-element distance.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "exacta/kernels.h"

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr int kLeafSize = 8;

struct Node {
  double lo[3], hi[3];       // bounding box of the centroids below
  std::int64_t begin, end;   // range of `order`
  std::int64_t left, right;  // children, -1 for a leaf
};

double dot(const double *a, const double *b, int dim) {
  double s = 0.0;
  for (int i = 0; i < dim; ++i) s += a[i] * b[i];
  return s;
}

// Closest point on triangle abc to p (Ericson, Real-Time Collision Detection, 5.1.5); returns
// the distance.
double point_triangle(const double *p, const double *a, const double *b, const double *c) {
  double ab[3], ac[3], ap[3], bp[3], cp[3], q[3];
  for (int i = 0; i < 3; ++i) {
    ab[i] = b[i] - a[i];
    ac[i] = c[i] - a[i];
    ap[i] = p[i] - a[i];
    bp[i] = p[i] - b[i];
    cp[i] = p[i] - c[i];
  }
  const double d1 = dot(ab, ap, 3), d2 = dot(ac, ap, 3);
  const double d3 = dot(ab, bp, 3), d4 = dot(ac, bp, 3);
  const double d5 = dot(ab, cp, 3), d6 = dot(ac, cp, 3);
  const double vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
  if (d1 <= 0.0 && d2 <= 0.0) {
    for (int i = 0; i < 3; ++i) q[i] = a[i];
  } else if (d3 >= 0.0 && d4 <= d3) {
    for (int i = 0; i < 3; ++i) q[i] = b[i];
  } else if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    const double v = d1 / (d1 - d3);
    for (int i = 0; i < 3; ++i) q[i] = a[i] + v * ab[i];
  } else if (d6 >= 0.0 && d5 <= d6) {
    for (int i = 0; i < 3; ++i) q[i] = c[i];
  } else if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    const double w = d2 / (d2 - d6);
    for (int i = 0; i < 3; ++i) q[i] = a[i] + w * ac[i];
  } else if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    for (int i = 0; i < 3; ++i) q[i] = b[i] + w * (c[i] - b[i]);
  } else {
    const double denom = 1.0 / (va + vb + vc);
    const double v = vb * denom, w = vc * denom;
    for (int i = 0; i < 3; ++i) q[i] = a[i] + ab[i] * v + ac[i] * w;
  }
  double s = 0.0;
  for (int i = 0; i < 3; ++i) s += (p[i] - q[i]) * (p[i] - q[i]);
  return std::sqrt(s);
}

// Distance from p to segment ab.
double point_segment(const double *p, const double *a, const double *b, int dim) {
  double ab[3], ap[3];
  for (int i = 0; i < dim; ++i) {
    ab[i] = b[i] - a[i];
    ap[i] = p[i] - a[i];
  }
  const double length2 = dot(ab, ab, dim);
  double t = length2 > 0.0 ? dot(ap, ab, dim) / length2 : 0.0;
  t = std::min(1.0, std::max(0.0, t));
  double s = 0.0;
  for (int i = 0; i < dim; ++i) {
    const double d = p[i] - (a[i] + t * ab[i]);
    s += d * d;
  }
  return std::sqrt(s);
}

struct Tree {
  int dim;
  const double *elements;  // n x dim vertices x dim coordinates
  std::vector<double> centroid;
  std::vector<std::int64_t> order;
  std::vector<Node> nodes;
  double reach = 0.0;

  double element_distance(const double *p, std::int64_t e) const {
    const double *v = elements + e * dim * dim;
    return dim == 3 ? point_triangle(p, v, v + 3, v + 6) : point_segment(p, v, v + dim, dim);
  }

  std::int64_t build(std::int64_t begin, std::int64_t end) {
    Node node;
    node.begin = begin;
    node.end = end;
    node.left = node.right = -1;
    for (int i = 0; i < dim; ++i) {
      node.lo[i] = kInf;
      node.hi[i] = -kInf;
    }
    for (std::int64_t k = begin; k < end; ++k) {
      const double *c = centroid.data() + order[k] * dim;
      for (int i = 0; i < dim; ++i) {
        node.lo[i] = std::min(node.lo[i], c[i]);
        node.hi[i] = std::max(node.hi[i], c[i]);
      }
    }
    const auto index = static_cast<std::int64_t>(nodes.size());
    nodes.push_back(node);
    if (end - begin > kLeafSize) {
      int axis = 0;
      for (int i = 1; i < dim; ++i) {
        if (node.hi[i] - node.lo[i] > node.hi[axis] - node.lo[axis]) axis = i;
      }
      const std::int64_t mid = begin + (end - begin) / 2;
      std::nth_element(order.begin() + begin, order.begin() + mid, order.begin() + end,
                       [&](std::int64_t a, std::int64_t b) {
                         return centroid[a * dim + axis] < centroid[b * dim + axis];
                       });
      const std::int64_t left = build(begin, mid);
      const std::int64_t right = build(mid, end);
      nodes[index].left = left;
      nodes[index].right = right;
    }
    return index;
  }

  double box_distance(const Node &node, const double *p) const {
    double s = 0.0;
    for (int i = 0; i < dim; ++i) {
      const double d = std::max(0.0, std::max(node.lo[i] - p[i], p[i] - node.hi[i]));
      s += d * d;
    }
    return std::sqrt(s);
  }

  double nearest(const double *p) const {
    // Median splits keep the tree balanced (depth < 64 for any realistic mesh), and each level
    // adds at most one pending node, so a small fixed stack suffices.
    std::int64_t stack[128];
    int top = 0;
    double best = kInf;
    stack[top++] = 0;
    while (top > 0) {
      const Node &node = nodes[stack[--top]];
      if (box_distance(node, p) - reach > best) continue;
      if (node.left < 0) {
        for (std::int64_t k = node.begin; k < node.end; ++k) {
          best = std::min(best, element_distance(p, order[k]));
        }
        continue;
      }
      // Visit the nearer child first (pushed last), so `best` shrinks early.
      const double dl = box_distance(nodes[node.left], p), dr = box_distance(nodes[node.right], p);
      if (dl < dr) {
        stack[top++] = node.right;
        stack[top++] = node.left;
      } else {
        stack[top++] = node.left;
        stack[top++] = node.right;
      }
    }
    return best;
  }
};

int point_mesh_distances(const double *points, std::int64_t n_points, const double *elements,
                         std::int64_t n_elements, int dim, double *out, int num_threads) {
  if (n_elements == 0) {
    for (std::int64_t q = 0; q < n_points; ++q) out[q] = kInf;
    return EXACTA_OK;
  }
  Tree tree;
  tree.dim = dim;
  tree.elements = elements;
  tree.centroid.assign(n_elements * dim, 0.0);
  tree.order.resize(n_elements);
  for (std::int64_t e = 0; e < n_elements; ++e) {
    tree.order[e] = e;
    const double *v = elements + e * dim * dim;
    for (int i = 0; i < dim; ++i) {
      double s = 0.0;
      for (int k = 0; k < dim; ++k) s += v[k * dim + i];
      tree.centroid[e * dim + i] = s / dim;
    }
    for (int k = 0; k < dim; ++k) {
      double s = 0.0;
      for (int i = 0; i < dim; ++i) {
        const double d = v[k * dim + i] - tree.centroid[e * dim + i];
        s += d * d;
      }
      tree.reach = std::max(tree.reach, std::sqrt(s));
    }
  }
  // Rounding in the centroids and box distances: a relative margin keeps the pruning safe.
  tree.reach = tree.reach * (1.0 + 1e-9) + 1e-12;
  tree.nodes.reserve(2 * (n_elements / kLeafSize + 1));
  tree.build(0, n_elements);

  int threads = 1;
#ifdef _OPENMP
  threads = num_threads > 0 ? num_threads : omp_get_max_threads();
#else
  (void)num_threads;
#endif
#pragma omp parallel for num_threads(threads) schedule(dynamic, 256)
  for (std::int64_t q = 0; q < n_points; ++q) out[q] = tree.nearest(points + q * dim);
  return EXACTA_OK;
}

}  // namespace

extern "C" int exacta_point_mesh_distances_f64(const double *points, int64_t n_points,
                                               const double *elements, int64_t n_elements,
                                               int dim, double *out, int num_threads) {
  if ((dim != 2 && dim != 3) || n_points < 0 || n_elements < 0) return EXACTA_INVALID_ARGUMENT;
  if (n_points == 0) return EXACTA_OK;
  if (points == nullptr || out == nullptr || (n_elements > 0 && elements == nullptr)) {
    return EXACTA_INVALID_ARGUMENT;
  }
  try {
    return point_mesh_distances(points, n_points, elements, n_elements, dim, out, num_threads);
  } catch (const std::bad_alloc &) {
    return EXACTA_OUT_OF_MEMORY;
  }
}
