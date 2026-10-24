// Distance from points to the nearest element of a mesh: triangles (3-D) or segments (2-D).
//
// A k-d tree over the elements, split at the median of their centroids and searched depth-first.
// Each node keeps the bounding box of its elements' vertices, which holds the elements, so a node
// (or an element) whose box is farther from the query than the best distance so far cannot hold a
// closer element. The result is the exact minimum over elements of the point-to-element distance.
//
// The tree is stored in heap order (the children of node i are 2i + 1 and 2i + 2), so subtrees
// are built in parallel, and the elements are copied in tree order, so a leaf's are contiguous.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "exacta/kernels.h"

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr int kLeafSize = 8;
constexpr std::int64_t kTaskSize = 1 << 15;  // smaller subtrees are built by one thread

struct Node {
  double lo[3], hi[3];      // bounding box of the vertices of the elements below
  std::int64_t begin, end;  // range of the tree order
};

struct Item {
  double centroid[3];
  std::int64_t element;
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
  std::vector<Item> items;             // centroids, in tree order once built
  std::unique_ptr<double[]> elements;  // vertices (dim x dim each), in tree order
  std::vector<Node> nodes;             // heap order; unused slots are empty leaves
  double scale = 0.0;                  // largest coordinate magnitude of the vertices

  bool leaf(const Node &node) const { return node.end - node.begin <= kLeafSize; }

  double element_distance(const double *p, std::int64_t k) const {
    const double *v = elements.get() + k * dim * dim;
    return dim == 3 ? point_triangle(p, v, v + 3, v + 6) : point_segment(p, v, v + dim, dim);
  }

  // Orders items[begin, end) into the subtree of node `index`: split at the median centroid along
  // the axis where the centroids spread most.
  void build(std::int64_t index, std::int64_t begin, std::int64_t end) {
    Node &node = nodes[index];
    node.begin = begin;
    node.end = end;
    if (leaf(node)) return;
    double lo[3] = {kInf, kInf, kInf}, hi[3] = {-kInf, -kInf, -kInf};
    for (std::int64_t k = begin; k < end; ++k) {
      const double *c = items[k].centroid;
      for (int i = 0; i < dim; ++i) {
        lo[i] = std::min(lo[i], c[i]);
        hi[i] = std::max(hi[i], c[i]);
      }
    }
    int axis = 0;
    for (int i = 1; i < dim; ++i) {
      if (hi[i] - lo[i] > hi[axis] - lo[axis]) axis = i;
    }
    const std::int64_t mid = begin + (end - begin) / 2;
    std::nth_element(items.begin() + begin, items.begin() + mid, items.begin() + end,
                     [axis](const Item &a, const Item &b) {
                       return a.centroid[axis] < b.centroid[axis];
                     });
    if (end - begin > kTaskSize) {
#pragma omp task
      build(2 * index + 1, begin, mid);
#pragma omp task
      build(2 * index + 2, mid, end);
#pragma omp taskwait
    } else {
      build(2 * index + 1, begin, mid);
      build(2 * index + 2, mid, end);
    }
  }

  // The vertex box of node `index`, from its elements (a leaf) or its children.
  void fit(std::int64_t index) {
    Node &node = nodes[index];
    for (int i = 0; i < dim; ++i) {
      node.lo[i] = kInf;
      node.hi[i] = -kInf;
    }
    if (leaf(node)) {
      const double *v = elements.get() + node.begin * dim * dim;
      for (std::int64_t k = 0; k < (node.end - node.begin) * dim; ++k) {
        for (int i = 0; i < dim; ++i) {
          node.lo[i] = std::min(node.lo[i], v[k * dim + i]);
          node.hi[i] = std::max(node.hi[i], v[k * dim + i]);
        }
      }
      return;
    }
    const Node &left = nodes[2 * index + 1], &right = nodes[2 * index + 2];
    for (int i = 0; i < dim; ++i) {
      node.lo[i] = std::min(left.lo[i], right.lo[i]);
      node.hi[i] = std::max(left.hi[i], right.hi[i]);
    }
  }

  // Squared distance from p to the box [lo, hi].
  double box_squared(const double *lo, const double *hi, const double *p) const {
    double s = 0.0;
    for (int i = 0; i < dim; ++i) {
      const double d = std::max(0.0, std::max(lo[i] - p[i], p[i] - hi[i]));
      s += d * d;
    }
    return s;
  }

  double box_squared(const Node &node, const double *p) const {
    return box_squared(node.lo, node.hi, p);
  }

  // Squared distance from p to the bounding box of element k's vertices.
  double element_box_squared(const double *p, std::int64_t k) const {
    const double *v = elements.get() + k * dim * dim;
    double lo[3], hi[3];
    for (int i = 0; i < dim; ++i) lo[i] = hi[i] = v[i];
    for (int j = 1; j < dim; ++j) {
      for (int i = 0; i < dim; ++i) {
        lo[i] = std::min(lo[i], v[j * dim + i]);
        hi[i] = std::max(hi[i], v[j * dim + i]);
      }
    }
    return box_squared(lo, hi, p);
  }

  double nearest(const double *p) const {
    // The tree is balanced (depth < 64 for any realistic mesh), and each level adds at most one
    // pending node, so a small fixed stack suffices. A node or element whose box is farther than
    // the best distance so far (compared squared, with a margin far above the rounding errors of
    // either distance) cannot hold a closer element.
    double magnitude = scale;
    for (int i = 0; i < dim; ++i) magnitude = std::max(magnitude, std::abs(p[i]));
    const double margin = 1e-9 * (magnitude + 1.0);
    std::int64_t stack[128];
    int top = 0;
    double best = kInf, bound = kInf;
    stack[top++] = 0;
    while (top > 0) {
      const std::int64_t index = stack[--top];
      const Node &node = nodes[index];
      if (box_squared(node, p) > bound) continue;
      if (leaf(node)) {
        for (std::int64_t k = node.begin; k < node.end; ++k) {
          if (element_box_squared(p, k) > bound) continue;
          const double d = element_distance(p, k);
          if (d < best) {
            best = d;
            bound = (best + margin) * (best + margin);
          }
        }
        continue;
      }
      // Visit the nearer child first (pushed last), so `best` shrinks early.
      const std::int64_t left = 2 * index + 1, right = 2 * index + 2;
      if (box_squared(nodes[left], p) < box_squared(nodes[right], p)) {
        stack[top++] = right;
        stack[top++] = left;
      } else {
        stack[top++] = left;
        stack[top++] = right;
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
  int threads = 1;
#ifdef _OPENMP
  threads = num_threads > 0 ? num_threads : omp_get_max_threads();
#else
  (void)num_threads;
#endif
  const int stride = dim * dim;
  Tree tree;
  tree.dim = dim;
  tree.items.resize(n_elements);
  double scale = 0.0;
#pragma omp parallel for num_threads(threads) schedule(static) reduction(max : scale)
  for (std::int64_t e = 0; e < n_elements; ++e) {
    Item &item = tree.items[e];
    item.element = e;
    const double *v = elements + e * stride;
    for (int i = 0; i < dim; ++i) {
      double s = 0.0;
      for (int k = 0; k < dim; ++k) {
        s += v[k * dim + i];
        scale = std::max(scale, std::abs(v[k * dim + i]));
      }
      item.centroid[i] = s / dim;
    }
  }
  tree.scale = scale;

  // Node sizes at depth d are at most ceil(n / 2^d); every node at the first depth where that is
  // at most kLeafSize is a leaf.
  int depth = 0;
  while (((n_elements - 1) >> depth) + 1 > kLeafSize) ++depth;
  tree.nodes.resize((std::size_t(2) << depth) - 1);
#pragma omp parallel num_threads(threads)
#pragma omp single
  tree.build(0, 0, n_elements);

  // Not zeroed: the threads touch the pages first as they copy.
  tree.elements.reset(new double[static_cast<std::size_t>(n_elements) * stride]);
#pragma omp parallel for num_threads(threads) schedule(static)
  for (std::int64_t k = 0; k < n_elements; ++k) {
    const double *v = elements + tree.items[k].element * stride;
    std::copy(v, v + stride, tree.elements.get() + k * stride);
  }
  for (int d = depth; d >= 0; --d) {  // vertex boxes, bottom-up
    const std::int64_t first = (std::int64_t(1) << d) - 1, last = (std::int64_t(2) << d) - 1;
#pragma omp parallel for num_threads(threads) schedule(static)
    for (std::int64_t index = first; index < last; ++index) tree.fit(index);
  }

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
