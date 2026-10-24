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
            const std::vector<double> &spacing, int num_threads) {
  const int ndim = static_cast<int>(features.ndim());
  if (out.ndim() != features.ndim()) {
    throw std::invalid_argument("features and out must have the same shape");
  }
  if (spacing.size() != features.ndim()) {
    throw std::invalid_argument("spacing must have one value per dimension");
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
    status = exacta_edt_sq_f64(features.data(), out.data(), shape.data(), ndim, spacing.data(),
                               num_threads);
  }
  if (status == EXACTA_INVALID_ARGUMENT) {
    throw std::invalid_argument("edt_sq: spacing must be positive and finite");
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

}  // namespace

NB_MODULE(_C, m) {
  m.doc() = "Compiled kernels for exacta.";
  m.def("edt_sq", &edt_sq, nb::arg("features"), nb::arg("out"), nb::arg("spacing"),
        nb::arg("num_threads") = 0,
        "Write into `out` (float64, same shape) the squared Euclidean distance from every element "
        "to the nearest nonzero element of `features` (uint8), with per-axis `spacing`.");
  m.def("label", &label, nb::arg("mask"), nb::arg("out"), nb::arg("connectivity"),
        "Write into `out` (int32, same shape) the connected-component labels of `mask` (uint8), "
        "numbered as scipy.ndimage.label numbers them, and return the number of components.");
}
