// Python bindings for the C kernels. Arrays arrive through DLPack (PyTorch, NumPy, ...),
// so this layer never links against a specific framework.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "exacta/kernels.h"

namespace nb = nanobind;

namespace {

void edt_sq(nb::ndarray<const std::uint8_t, nb::device::cpu, nb::c_contig> features,
            nb::ndarray<double, nb::device::cpu, nb::c_contig> out,
            const std::vector<double> &spacing, int batch_ndim, int num_threads) {
  const int ndim = static_cast<int>(features.ndim());
  if (out.ndim() != features.ndim()) {
    throw std::invalid_argument("features and out must have the same shape");
  }
  if (batch_ndim < 0 || batch_ndim > ndim ||
      spacing.size() != static_cast<std::size_t>(ndim - batch_ndim)) {
    throw std::invalid_argument("spacing must have one value per non-batch dimension");
  }
  std::vector<std::int64_t> shape(ndim);
  for (int axis = 0; axis < ndim; ++axis) {
    if (out.shape(axis) != features.shape(axis)) {
      throw std::invalid_argument("features and out must have the same shape");
    }
    shape[axis] = static_cast<std::int64_t>(features.shape(axis));
  }

  int status;
  {
    nb::gil_scoped_release release;
    status = exacta_edt_sq_f64(features.data(), out.data(), shape.data(), ndim, batch_ndim,
                               spacing.data(), num_threads);
  }
  if (status == EXACTA_INVALID_ARGUMENT) {
    throw std::invalid_argument("edt_sq: spacing values must lie in [1e-100, 1e100]");
  }
  if (status != EXACTA_OK) throw std::runtime_error(exacta_status_string(status));
}

std::int64_t label(nb::ndarray<const std::uint8_t, nb::device::cpu, nb::c_contig> mask,
                  nb::ndarray<std::int32_t, nb::device::cpu, nb::c_contig> out, int connectivity) {
  const int ndim = static_cast<int>(mask.ndim());
  if (out.ndim() != mask.ndim()) {
    throw std::invalid_argument("mask and out must have the same shape");
  }
  std::vector<std::int64_t> shape(ndim);
  for (int axis = 0; axis < ndim; ++axis) {
    if (out.shape(axis) != mask.shape(axis)) {
      throw std::invalid_argument("mask and out must have the same shape");
    }
    shape[axis] = static_cast<std::int64_t>(mask.shape(axis));
  }
  std::int64_t num_labels = 0;
  int status;
  {
    nb::gil_scoped_release release;
    status =
        exacta_label_u8(mask.data(), out.data(), shape.data(), ndim, connectivity, &num_labels);
  }
  if (status == EXACTA_INVALID_ARGUMENT) {
    throw std::invalid_argument(
        "label: mask must have 1 to 3 dimensions and fewer than 2^31 elements, with "
        "1 <= connectivity <= ndim");
  }
  if (status != EXACTA_OK) throw std::runtime_error(exacta_status_string(status));
  return num_labels;
}

template <typename T, int (*Kernel)(const T *, const int64_t *, int, int64_t, int64_t *, int64_t *,
                                   int)>
void class_extents(nb::ndarray<const T, nb::device::cpu, nb::c_contig> labels,
                   nb::ndarray<std::int64_t, nb::device::cpu, nb::c_contig> lo,
                   nb::ndarray<std::int64_t, nb::device::cpu, nb::c_contig> hi, int num_threads) {
  const int ndim = static_cast<int>(labels.ndim());
  const auto classes = static_cast<std::int64_t>(lo.shape(0));
  if (lo.ndim() != 2 || hi.ndim() != 2 || static_cast<int>(lo.shape(1)) != ndim ||
      hi.shape(0) != lo.shape(0) || hi.shape(1) != lo.shape(1)) {
    throw std::invalid_argument("lo and hi must have shape (num_classes, labels.ndim)");
  }
  std::vector<std::int64_t> shape(ndim);
  for (int axis = 0; axis < ndim; ++axis) {
    shape[axis] = static_cast<std::int64_t>(labels.shape(axis));
  }
  int status;
  {
    nb::gil_scoped_release release;
    status = Kernel(labels.data(), shape.data(), ndim, classes, lo.data(), hi.data(), num_threads);
  }
  if (status == EXACTA_INVALID_ARGUMENT) {
    throw std::invalid_argument("class_extents: labels must have 1 to 3 dimensions");
  }
  if (status != EXACTA_OK) throw std::runtime_error(exacta_status_string(status));
}

template <typename T>
using Flat = nb::ndarray<const T, nb::ndim<1>, nb::device::cpu, nb::c_contig>;
using Counts = nb::ndarray<std::int64_t, nb::ndim<1>, nb::device::cpu, nb::c_contig>;

// Returns false when a label lies outside [0, classes).
template <typename T, int (*Kernel)(const T *, const T *, int64_t, int64_t, int64_t *, int)>
bool confusion(Flat<T> pred, Flat<T> target, nb::ndarray<std::int64_t, nb::ndim<2>, nb::device::cpu,
               nb::c_contig> out, int num_threads) {
  if (pred.shape(0) != target.shape(0) || out.shape(0) != out.shape(1)) {
    throw std::invalid_argument("pred and target (N,) and out (C, C) do not agree");
  }
  int status;
  {
    nb::gil_scoped_release release;
    status = Kernel(pred.data(), target.data(), static_cast<std::int64_t>(pred.shape(0)),
                    static_cast<std::int64_t>(out.shape(0)), out.data(), num_threads);
  }
  if (status != EXACTA_OK && status != EXACTA_INVALID_ARGUMENT) {
    throw std::runtime_error(exacta_status_string(status));
  }
  return status == EXACTA_OK;
}

// Returns false when a value is not 0 or 1.
template <typename T, int (*Kernel)(const T *, const T *, int64_t, int64_t, int64_t *, int64_t *,
                                    int64_t *, int)>
bool channel_counts(nb::ndarray<const T, nb::ndim<2>, nb::device::cpu, nb::c_contig> pred,
                    nb::ndarray<const T, nb::ndim<2>, nb::device::cpu, nb::c_contig> target,
                    Counts both, Counts in_pred, Counts in_target, int num_threads) {
  const auto count = pred.shape(0);
  if (target.shape(0) != count || target.shape(1) != pred.shape(1) || both.shape(0) != count ||
      in_pred.shape(0) != count || in_target.shape(0) != count) {
    throw std::invalid_argument("pred and target (C, N) and the counts (C,) do not agree");
  }
  int status;
  {
    nb::gil_scoped_release release;
    status = Kernel(pred.data(), target.data(), static_cast<std::int64_t>(count),
                    static_cast<std::int64_t>(pred.shape(1)), both.data(), in_pred.data(),
                    in_target.data(), num_threads);
  }
  if (status != EXACTA_OK && status != EXACTA_INVALID_ARGUMENT) {
    throw std::runtime_error(exacta_status_string(status));
  }
  return status == EXACTA_OK;
}

using Bytes = nb::ndarray<const std::uint8_t, nb::device::cpu, nb::c_contig>;
using Indices = nb::ndarray<std::int64_t, nb::ndim<2>, nb::device::cpu, nb::c_contig>;

// Returns false when a value lies outside [0, bins).
bool histogram(Flat<std::int32_t> values, Counts out, int num_threads) {
  int status;
  {
    nb::gil_scoped_release release;
    status = exacta_histogram_i32(values.data(), static_cast<std::int64_t>(values.shape(0)),
                                  static_cast<std::int64_t>(out.shape(0)), out.data(), num_threads);
  }
  if (status != EXACTA_OK && status != EXACTA_INVALID_ARGUMENT) {
    throw std::runtime_error(exacta_status_string(status));
  }
  return status == EXACTA_OK;
}

// Returns false when a label lies outside [0, len(select)).
bool select_labels(Flat<std::int32_t> labels, Flat<std::uint8_t> select,
                   nb::ndarray<std::uint8_t, nb::ndim<1>, nb::device::cpu, nb::c_contig> out,
                   int num_threads) {
  if (out.shape(0) != labels.shape(0) || select.shape(0) < 1) {
    throw std::invalid_argument("labels and out (N,) and select (C,) do not agree");
  }
  int status;
  {
    nb::gil_scoped_release release;
    status = exacta_select_labels_i32(labels.data(), static_cast<std::int64_t>(labels.shape(0)),
                                      select.data(), static_cast<std::int64_t>(select.shape(0)),
                                      out.data(), num_threads);
  }
  if (status != EXACTA_OK && status != EXACTA_INVALID_ARGUMENT) {
    throw std::runtime_error(exacta_status_string(status));
  }
  return status == EXACTA_OK;
}

// Returns the number of pairs (rows of `pairs` written only if they fit), or -1 when a label lies
// out of range.
std::int64_t label_overlap(Flat<std::int32_t> a, Flat<std::int32_t> b, Counts size_a,
                           Counts size_b, Indices pairs, int num_threads) {
  if (a.shape(0) != b.shape(0) || size_a.shape(0) < 1 || size_b.shape(0) < 1 ||
      pairs.shape(1) != 3) {
    throw std::invalid_argument("a and b (N,), size_a (na + 1,), size_b (nb + 1,) and pairs (M, 3) "
                                "do not agree");
  }
  std::int64_t num_pairs = 0;
  int status;
  {
    nb::gil_scoped_release release;
    const auto na = static_cast<std::int64_t>(size_a.shape(0)) - 1;
    const auto nb_ = static_cast<std::int64_t>(size_b.shape(0)) - 1;
    status = exacta_label_overlap_i32(a.data(), b.data(), static_cast<std::int64_t>(a.shape(0)),
                                      na, nb_, size_a.data(), size_b.data(), pairs.data(),
                                      static_cast<std::int64_t>(pairs.shape(0)), &num_pairs,
                                      num_threads);
  }
  if (status == EXACTA_INVALID_ARGUMENT) return -1;
  if (status != EXACTA_OK) throw std::runtime_error(exacta_status_string(status));
  return num_pairs;
}

void mask_extents(Bytes masks, Indices lo, Indices hi, int num_threads) {
  const int ndim = static_cast<int>(masks.ndim()) - 1;
  if (ndim < 1 || lo.shape(0) != masks.shape(0) || hi.shape(0) != masks.shape(0) ||
      static_cast<int>(lo.shape(1)) != ndim || static_cast<int>(hi.shape(1)) != ndim) {
    throw std::invalid_argument("masks (C, *spatial), lo and hi (C, ndim) do not agree");
  }
  std::vector<std::int64_t> shape(ndim);
  for (int axis = 0; axis < ndim; ++axis) {
    shape[axis] = static_cast<std::int64_t>(masks.shape(axis + 1));
  }
  int status;
  {
    nb::gil_scoped_release release;
    status = exacta_mask_extents_u8(masks.data(), static_cast<std::int64_t>(masks.shape(0)),
                                    shape.data(), ndim, lo.data(), hi.data(), num_threads);
  }
  if (status == EXACTA_INVALID_ARGUMENT) {
    throw std::invalid_argument("mask_extents: masks must have 1 to 3 spatial dimensions");
  }
  if (status != EXACTA_OK) throw std::runtime_error(exacta_status_string(status));
}

void dilate(Bytes mask, nb::ndarray<std::uint8_t, nb::device::cpu, nb::c_contig> out,
            int connectivity, int steps, int num_threads) {
  const int ndim = static_cast<int>(mask.ndim());
  if (out.ndim() != mask.ndim()) {
    throw std::invalid_argument("mask and out must have the same shape");
  }
  std::vector<std::int64_t> shape(ndim);
  for (int axis = 0; axis < ndim; ++axis) {
    if (out.shape(axis) != mask.shape(axis)) {
      throw std::invalid_argument("mask and out must have the same shape");
    }
    shape[axis] = static_cast<std::int64_t>(mask.shape(axis));
  }
  int status;
  {
    nb::gil_scoped_release release;
    status = exacta_dilate_u8(mask.data(), out.data(), shape.data(), ndim, connectivity, steps,
                              num_threads);
  }
  if (status == EXACTA_INVALID_ARGUMENT) {
    throw std::invalid_argument(
        "dilate: mask must have 1 to 3 dimensions, with 1 <= connectivity <= ndim and steps >= 0");
  }
  if (status != EXACTA_OK) throw std::runtime_error(exacta_status_string(status));
}

using Points = nb::ndarray<const double, nb::ndim<2>, nb::device::cpu, nb::c_contig>;
using Elements = nb::ndarray<const double, nb::ndim<3>, nb::device::cpu, nb::c_contig>;

void point_mesh_distances(Points points, Elements elements,
                          nb::ndarray<double, nb::ndim<1>, nb::device::cpu, nb::c_contig> out,
                          int num_threads) {
  const auto dim = static_cast<int>(points.shape(1));
  if (out.shape(0) != points.shape(0) || elements.shape(1) != points.shape(1) ||
      elements.shape(2) != points.shape(1)) {
    throw std::invalid_argument("points (N, d), elements (M, d, d) and out (N,) do not agree");
  }
  int status;
  {
    nb::gil_scoped_release release;
    status = exacta_point_mesh_distances_f64(
        points.data(), static_cast<std::int64_t>(points.shape(0)), elements.data(),
        static_cast<std::int64_t>(elements.shape(0)), dim, out.data(), num_threads);
  }
  if (status == EXACTA_INVALID_ARGUMENT) {
    throw std::invalid_argument("point_mesh_distances: dimension must be 2 or 3");
  }
  if (status != EXACTA_OK) throw std::runtime_error(exacta_status_string(status));
}

}  // namespace

NB_MODULE(_C, m) {
  m.doc() = "Compiled kernels for exacta.";
  m.def("edt_sq", &edt_sq, nb::arg("features"), nb::arg("out"), nb::arg("spacing"),
        nb::arg("batch_ndim") = 0, nb::arg("num_threads") = 0,
        "Write into `out` (float64, same shape) the squared Euclidean distance from every element "
        "to the nearest nonzero element of `features` (uint8) in the same image, with `spacing` "
        "for each axis after the first `batch_ndim` (which index images).");
  const char *extents_doc =
      "Write into `lo` and `hi` (int64, (num_classes, ndim)) the first and last index along each "
      "axis of every class of the label map `labels`; shape and -1 where a class is absent.";
  m.def("class_extents", &class_extents<std::uint8_t, exacta_class_extents_u8>, nb::arg("labels"),
        nb::arg("lo"), nb::arg("hi"), nb::arg("num_threads") = 0, extents_doc);
  m.def("class_extents", &class_extents<std::int32_t, exacta_class_extents_i32>, nb::arg("labels"),
        nb::arg("lo"), nb::arg("hi"), nb::arg("num_threads") = 0, extents_doc);
  m.def("class_extents", &class_extents<std::int64_t, exacta_class_extents_i64>, nb::arg("labels"),
        nb::arg("lo"), nb::arg("hi"), nb::arg("num_threads") = 0, extents_doc);
  const char *confusion_doc =
      "Write into `out` (int64, (C, C)) the confusion matrix of the label maps `pred` and `target` "
      "(N,); returns False if a label lies outside [0, C).";
  m.def("confusion", &confusion<std::uint8_t, exacta_confusion_u8>, nb::arg("pred"),
        nb::arg("target"), nb::arg("out"), nb::arg("num_threads") = 0, confusion_doc);
  m.def("confusion", &confusion<std::int32_t, exacta_confusion_i32>, nb::arg("pred"),
        nb::arg("target"), nb::arg("out"), nb::arg("num_threads") = 0, confusion_doc);
  m.def("confusion", &confusion<std::int64_t, exacta_confusion_i64>, nb::arg("pred"),
        nb::arg("target"), nb::arg("out"), nb::arg("num_threads") = 0, confusion_doc);
  const char *counts_doc =
      "Write into `both`, `in_pred` and `in_target` (int64, (C,)) the elements of each pair of "
      "binary masks `pred`, `target` (C, N) in both, in pred and in target; returns False if a "
      "value is not 0 or 1.";
  m.def("channel_counts", &channel_counts<std::uint8_t, exacta_channel_counts_u8>,
        nb::arg("pred"), nb::arg("target"), nb::arg("both"), nb::arg("in_pred"),
        nb::arg("in_target"), nb::arg("num_threads") = 0, counts_doc);
  m.def("channel_counts", &channel_counts<std::int64_t, exacta_channel_counts_i64>,
        nb::arg("pred"), nb::arg("target"), nb::arg("both"), nb::arg("in_pred"),
        nb::arg("in_target"), nb::arg("num_threads") = 0, counts_doc);
  m.def("channel_counts", &channel_counts<float, exacta_channel_counts_f32>, nb::arg("pred"),
        nb::arg("target"), nb::arg("both"), nb::arg("in_pred"), nb::arg("in_target"),
        nb::arg("num_threads") = 0, counts_doc);
  m.def("histogram", &histogram, nb::arg("values"), nb::arg("out"), nb::arg("num_threads") = 0,
        "Write into `out` (int64, (bins,)) the number of elements of each value of the int32 label "
        "map `values` (N,); returns False if a value lies outside [0, bins).");
  m.def("select_labels", &select_labels, nb::arg("labels"), nb::arg("select"), nb::arg("out"),
        nb::arg("num_threads") = 0,
        "Write into `out` (uint8, (N,)) select[labels[i]] != 0 for the int32 label map `labels` "
        "(N,) and the uint8 table `select` (C,); returns False if a label lies outside [0, C).");
  m.def("label_overlap", &label_overlap, nb::arg("a"), nb::arg("b"), nb::arg("size_a"),
        nb::arg("size_b"), nb::arg("pairs"), nb::arg("num_threads") = 0,
        "Write into `size_a` and `size_b` the size of every label of the int32 label maps `a` and "
        "`b` (N,), and into `pairs` (M, 3) the rows (i, j, count) of the pairs of nonzero labels "
        "that meet, if there is room; return the number of pairs, or -1 if a label is out of "
        "range.");
  m.def("mask_extents", &mask_extents, nb::arg("masks"), nb::arg("lo"), nb::arg("hi"),
        nb::arg("num_threads") = 0,
        "Write into `lo` and `hi` (int64, (C, ndim)) the bounding box of each mask of `masks` "
        "(uint8, (C, *spatial)); shape and -1 for an empty mask.");
  m.def("dilate", &dilate, nb::arg("mask"), nb::arg("out"), nb::arg("connectivity"),
        nb::arg("steps"), nb::arg("num_threads") = 0,
        "Write into `out` (uint8, same shape) the binary dilation of `mask` by `steps` steps of "
        "the `connectivity` neighbourhood, as scipy.ndimage.binary_dilation.");
  m.def("point_mesh_distances", &point_mesh_distances, nb::arg("points"), nb::arg("elements"),
        nb::arg("out"), nb::arg("num_threads") = 0,
        "Write into `out` the distance from each point (N, d) to the nearest triangle (d = 3) or "
        "segment (d = 2) of `elements` (M, d, d).");
  m.def("label", &label, nb::arg("mask"), nb::arg("out"), nb::arg("connectivity"),
        "Write into `out` (int32, same shape) the connected-component labels of `mask` (uint8), "
        "numbered as scipy.ndimage.label numbers them, and return the number of components.");
}
