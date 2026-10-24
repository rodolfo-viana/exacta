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

using Bytes = nb::ndarray<const std::uint8_t, nb::device::cpu, nb::c_contig>;
using Indices = nb::ndarray<std::int64_t, nb::ndim<2>, nb::device::cpu, nb::c_contig>;

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
