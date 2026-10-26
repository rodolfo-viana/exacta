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

// Masked bindings accept an empty mask only for index inputs (ignore_index alone).
template <typename T, auto Kernel>
bool masked_confusion(Flat<T> pred, Flat<T> target, Flat<std::uint8_t> mask,
    bool has_ignore, std::int64_t ignore_index,
    nb::ndarray<std::int64_t, nb::ndim<2>, nb::device::cpu, nb::c_contig> out, int num_threads) {
  const auto n = pred.shape(0);
  if (target.shape(0) != n || (mask.shape(0) != 0 && mask.shape(0) != n) ||
      out.shape(0) == 0 || out.shape(0) != out.shape(1)) {
    throw std::invalid_argument("pred, target and mask (N,) and out (C, C) do not agree");
  }
  int status;
  {
    nb::gil_scoped_release release;
    status = Kernel(pred.data(), target.data(), static_cast<std::int64_t>(n),
        static_cast<std::int64_t>(out.shape(0)), mask.shape(0) ? mask.data() : nullptr,
        has_ignore, ignore_index, out.data(), num_threads);
  }
  if (status != EXACTA_OK && status != EXACTA_INVALID_ARGUMENT)
    throw std::runtime_error(exacta_status_string(status));
  return status == EXACTA_OK;
}

template <typename T, auto Kernel>
bool masked_channel_counts(
    nb::ndarray<const T, nb::ndim<2>, nb::device::cpu, nb::c_contig> pred,
    nb::ndarray<const T, nb::ndim<2>, nb::device::cpu, nb::c_contig> target,
    nb::ndarray<const std::uint8_t, nb::ndim<2>, nb::device::cpu, nb::c_contig> mask,
    std::int64_t channels, Counts both, Counts in_pred, Counts in_target, Counts valid,
    int num_threads) {
  const auto count = pred.shape(0), n = pred.shape(1);
  if (channels < 1 || count % channels != 0 || target.shape(0) != count ||
      target.shape(1) != n || mask.shape(0) != count / channels || mask.shape(1) != n ||
      both.shape(0) != count || in_pred.shape(0) != count || in_target.shape(0) != count ||
      valid.shape(0) != count) {
    throw std::invalid_argument("pred, target (B*C, N), mask (B, N) and counts (B*C,) disagree");
  }
  int status;
  {
    nb::gil_scoped_release release;
    status = Kernel(pred.data(), target.data(), static_cast<std::int64_t>(count),
        static_cast<std::int64_t>(n), mask.data(), channels, both.data(), in_pred.data(),
        in_target.data(), valid.data(), num_threads);
  }
  if (status != EXACTA_OK && status != EXACTA_INVALID_ARGUMENT)
    throw std::runtime_error(exacta_status_string(status));
  return status == EXACTA_OK;
}

template <typename P, typename T, auto Kernel>
bool probability_stats(
    nb::ndarray<const P, nb::ndim<2>, nb::device::cpu, nb::c_contig> pred,
    nb::ndarray<const T, nb::ndim<2>, nb::device::cpu, nb::c_contig> target,
    nb::ndarray<const std::uint8_t, nb::ndim<2>, nb::device::cpu, nb::c_contig> mask,
    std::int64_t channels, bool from_logits, int kind, int bins,
    nb::ndarray<double, nb::ndim<2>, nb::device::cpu, nb::c_contig> out, int num_threads) {
  const auto rows = pred.shape(0), n = pred.shape(1), masks = mask.shape(0);
  if (channels < 1 || rows % channels || target.shape(0) != rows || target.shape(1) != n ||
      kind < 0 || kind > 2 || (kind == 2 && (bins < 1 || bins > INT32_MAX / 3)) ||
      out.shape(0) != rows || out.shape(1) != static_cast<std::size_t>(kind == 2 ? 3 * bins : 2) ||
      (masks && (mask.shape(1) != n || (masks != rows && masks != rows / channels)))) {
    throw std::invalid_argument("probability input, mask and statistic buffer shapes disagree");
  }
  const int mask_channels = masks == 0 ? 0 : masks == rows ? static_cast<int>(channels) : 1;
  int status;
  {
    nb::gil_scoped_release release;
    status = Kernel(pred.data(), target.data(), masks ? mask.data() : nullptr,
        static_cast<std::int64_t>(rows), static_cast<std::int64_t>(n), channels, mask_channels,
        from_logits, kind, bins, out.data(), num_threads);
  }
  if (status != EXACTA_OK && status != EXACTA_INVALID_ARGUMENT)
    throw std::runtime_error(exacta_status_string(status));
  return status == EXACTA_OK;
}

using Bytes = nb::ndarray<const std::uint8_t, nb::device::cpu, nb::c_contig>;

template <typename T, int (*Kernel)(const T*, T*, int64_t*, const int64_t*, int, int, int)>
void soft_pool(nb::ndarray<const T, nb::device::cpu, nb::c_contig> input,
               nb::ndarray<T, nb::device::cpu, nb::c_contig> output,
               nb::ndarray<int64_t, nb::device::cpu, nb::c_contig> indices, bool erosion, int threads) {
  if (input.ndim() != output.ndim()) throw std::invalid_argument("pool input/output shapes differ");
  std::vector<int64_t> shape(input.ndim());
  if (indices.size() && indices.ndim() != input.ndim()) throw std::invalid_argument("pool index shape differs");
  for (std::size_t i=0; i<input.ndim(); ++i) {
    if (input.shape(i) != output.shape(i) || (indices.size() && input.shape(i) != indices.shape(i)))
      throw std::invalid_argument("pool array shapes differ");
    shape[i] = static_cast<int64_t>(input.shape(i));
  }
  int status;
  {
    nb::gil_scoped_release release;
    status = Kernel(input.data(), output.data(), indices.size() ? indices.data() : nullptr,
                     shape.data(), static_cast<int>(shape.size()), erosion, threads);
  }
  if (status == EXACTA_INVALID_ARGUMENT) throw std::invalid_argument("pool expects (B,C,H,W) or (B,C,D,H,W)");
  if (status != EXACTA_OK) throw std::runtime_error(exacta_status_string(status));
}

void skeleton(Bytes mask, nb::ndarray<std::uint8_t, nb::device::cpu, nb::c_contig> out) {
  if (mask.ndim() != out.ndim()) throw std::invalid_argument("mask and out differ in shape");
  std::vector<std::int64_t> shape(mask.ndim());
  for (std::size_t i = 0; i < mask.ndim(); ++i) {
    if (mask.shape(i) != out.shape(i)) throw std::invalid_argument("mask and out differ in shape");
    shape[i] = static_cast<std::int64_t>(mask.shape(i));
  }
  int status;
  {
    nb::gil_scoped_release release;
    status = exacta_skeleton_u8(mask.data(), out.data(), shape.data(), static_cast<int>(shape.size()));
  }
  if (status == EXACTA_INVALID_ARGUMENT) throw std::invalid_argument("skeleton requires a 2-D or 3-D mask");
  if (status != EXACTA_OK) throw std::runtime_error(exacta_status_string(status));
}
using Indices = nb::ndarray<std::int64_t, nb::ndim<2>, nb::device::cpu, nb::c_contig>;

void boundary(nb::ndarray<const std::uint8_t, nb::device::cpu> mask,
              nb::ndarray<std::uint8_t, nb::device::cpu, nb::c_contig> out, int num_threads) {
  const int ndim = static_cast<int>(mask.ndim());
  if (out.ndim() != mask.ndim()) throw std::invalid_argument("mask and out differ in shape");
  std::vector<std::int64_t> shape(ndim), strides(ndim);
  for (int axis = 0; axis < ndim; ++axis) {
    if (out.shape(axis) != mask.shape(axis)) throw std::invalid_argument("mask and out differ");
    shape[axis] = static_cast<std::int64_t>(mask.shape(axis));
    strides[axis] = static_cast<std::int64_t>(mask.stride(axis));
  }
  int status;
  {
    nb::gil_scoped_release release;
    status = exacta_boundary_u8(mask.data(), shape.data(), strides.data(), ndim, out.data(),
                                num_threads);
  }
  if (status == EXACTA_INVALID_ARGUMENT) {
    throw std::invalid_argument("boundary: mask must have 1 to 3 dimensions");
  }
  if (status != EXACTA_OK) throw std::runtime_error(exacta_status_string(status));
}

void masked_sqrt(Flat<double> values, Flat<std::uint8_t> mask,
                 nb::ndarray<double, nb::ndim<1>, nb::device::cpu, nb::c_contig> out,
                 int num_threads) {
  if (values.shape(0) != mask.shape(0)) throw std::invalid_argument("values and mask differ");
  int status;
  {
    nb::gil_scoped_release release;
    status = exacta_masked_sqrt_f64(values.data(), mask.data(),
                                    static_cast<std::int64_t>(values.shape(0)), out.data(),
                                    static_cast<std::int64_t>(out.shape(0)), num_threads);
  }
  if (status == EXACTA_INVALID_ARGUMENT) {
    throw std::invalid_argument("masked_sqrt: out must hold one value per selected element");
  }
  if (status != EXACTA_OK) throw std::runtime_error(exacta_status_string(status));
}

// Returns false when a value is neither 0 nor 1.
template <typename T, int (*Kernel)(const T *, int64_t, uint8_t *, int)>
bool binary_mask(Flat<T> values, nb::ndarray<std::uint8_t, nb::ndim<1>, nb::device::cpu,
                                             nb::c_contig> out, int num_threads) {
  if (out.shape(0) != values.shape(0)) throw std::invalid_argument("values and out differ in size");
  int status;
  {
    nb::gil_scoped_release release;
    status = Kernel(values.data(), static_cast<std::int64_t>(values.shape(0)), out.data(),
                    num_threads);
  }
  if (status != EXACTA_OK && status != EXACTA_INVALID_ARGUMENT) {
    throw std::runtime_error(exacta_status_string(status));
  }
  return status == EXACTA_OK;
}

// Returns false when a key or value is NaN.
bool sort_pairs(nb::ndarray<double, nb::ndim<1>, nb::device::cpu, nb::c_contig> keys,
                nb::ndarray<double, nb::ndim<1>, nb::device::cpu, nb::c_contig> values,
                int num_threads) {
  if (keys.shape(0) != values.shape(0)) throw std::invalid_argument("keys and values (N,) differ");
  int status;
  {
    nb::gil_scoped_release release;
    status = exacta_sort_pairs_f64(keys.data(), values.data(),
                                   static_cast<std::int64_t>(keys.shape(0)), num_threads);
  }
  if (status != EXACTA_OK && status != EXACTA_INVALID_ARGUMENT) {
    throw std::runtime_error(exacta_status_string(status));
  }
  return status == EXACTA_OK;
}

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

using Codes = nb::ndarray<std::uint8_t, nb::device::cpu, nb::c_contig>;
template <int D>
using Table = nb::ndarray<const std::int64_t, nb::ndim<D>, nb::device::cpu, nb::c_contig>;

// Mask shape (2-D or 3-D) from the shape of its corner grid (one more along each axis).
std::vector<std::int64_t> mask_shape_of(Codes code) {
  const int ndim = static_cast<int>(code.ndim());
  if (ndim != 2 && ndim != 3) throw std::invalid_argument("code must have 2 or 3 dimensions");
  std::vector<std::int64_t> shape(ndim);
  for (int axis = 0; axis < ndim; ++axis) {
    if (code.shape(axis) < 1) throw std::invalid_argument("code has an empty axis");
    shape[axis] = static_cast<std::int64_t>(code.shape(axis)) - 1;
  }
  return shape;
}

// Returns the number of elements (0 when `counts` is empty).
std::int64_t neighbour_codes(Bytes mask, Codes code, Table<1> counts, int num_threads) {
  const std::vector<std::int64_t> shape = mask_shape_of(code);
  const int ndim = static_cast<int>(shape.size());
  bool agree = static_cast<int>(mask.ndim()) == ndim;
  for (int axis = 0; agree && axis < ndim; ++axis) {
    agree = static_cast<std::int64_t>(mask.shape(axis)) == shape[axis];
  }
  const bool with_counts = counts.shape(0) > 0;
  if (!agree || (with_counts && counts.shape(0) != (std::size_t(1) << (1 << ndim)))) {
    throw std::invalid_argument("mask (shape), code (shape + 1) and counts (2^(2^ndim)) disagree");
  }
  std::int64_t num_elements = 0;
  int status;
  {
    nb::gil_scoped_release release;
    status = exacta_neighbour_codes_u8(mask.data(), shape.data(), ndim, code.data(),
                                       with_counts ? counts.data() : nullptr, &num_elements,
                                       num_threads);
  }
  if (status != EXACTA_OK) throw std::runtime_error(exacta_status_string(status));
  return num_elements;
}

void mesh_elements(Codes code, Table<4> table, Table<1> counts, std::vector<double> spacing,
                   nb::ndarray<double, nb::ndim<3>, nb::device::cpu, nb::c_contig> elements,
                   nb::ndarray<double, nb::ndim<2>, nb::device::cpu, nb::c_contig> centres,
                   Counts block, Counts element_code, Counts slot, int num_threads) {
  const std::vector<std::int64_t> shape = mask_shape_of(code);
  const auto ndim = static_cast<std::size_t>(shape.size());
  const std::size_t m = elements.shape(0);
  if (spacing.size() != ndim || table.shape(0) != counts.shape(0) ||
      counts.shape(0) != (std::size_t(1) << (1 << ndim)) || table.shape(2) != ndim ||
      table.shape(3) != ndim || elements.shape(1) != ndim || elements.shape(2) != ndim ||
      centres.shape(0) != m || centres.shape(1) != ndim || block.shape(0) != m ||
      element_code.shape(0) != m || slot.shape(0) != m) {
    throw std::invalid_argument("mesh_elements: arguments disagree in shape");
  }
  int status;
  {
    nb::gil_scoped_release release;
    status = exacta_mesh_elements_f64(
        code.data(), shape.data(), static_cast<int>(ndim), table.data(), counts.data(),
        static_cast<std::int64_t>(table.shape(1)), spacing.data(), static_cast<std::int64_t>(m),
        elements.data(), centres.data(), block.data(), element_code.data(), slot.data(),
        num_threads);
  }
  if (status == EXACTA_INVALID_ARGUMENT) {
    throw std::invalid_argument("mesh_elements: the codes hold a different number of elements");
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
  // Never let overload resolution narrow labels or validate a converted copy of a mask.
  m.def("confusion", &confusion<std::uint8_t, exacta_confusion_u8>, nb::arg("pred").noconvert(),
        nb::arg("target").noconvert(), nb::arg("out").noconvert(),
        nb::arg("num_threads") = 0, confusion_doc);
  m.def("confusion", &confusion<std::int32_t, exacta_confusion_i32>, nb::arg("pred").noconvert(),
        nb::arg("target").noconvert(), nb::arg("out").noconvert(),
        nb::arg("num_threads") = 0, confusion_doc);
  m.def("confusion", &confusion<std::int64_t, exacta_confusion_i64>, nb::arg("pred").noconvert(),
        nb::arg("target").noconvert(), nb::arg("out").noconvert(),
        nb::arg("num_threads") = 0, confusion_doc);
  const char *counts_doc =
      "Write into `both`, `in_pred` and `in_target` (int64, (C,)) the elements of each pair of "
      "binary masks `pred`, `target` (C, N) in both, in pred and in target; returns False if a "
      "value is not 0 or 1.";
  m.def("channel_counts", &channel_counts<std::uint8_t, exacta_channel_counts_u8>,
        nb::arg("pred").noconvert(), nb::arg("target").noconvert(), nb::arg("both").noconvert(),
        nb::arg("in_pred").noconvert(), nb::arg("in_target").noconvert(),
        nb::arg("num_threads") = 0, counts_doc);
  m.def("channel_counts", &channel_counts<std::int64_t, exacta_channel_counts_i64>,
        nb::arg("pred").noconvert(), nb::arg("target").noconvert(), nb::arg("both").noconvert(),
        nb::arg("in_pred").noconvert(), nb::arg("in_target").noconvert(),
        nb::arg("num_threads") = 0, counts_doc);
  m.def("channel_counts", &channel_counts<float, exacta_channel_counts_f32>,
        nb::arg("pred").noconvert(), nb::arg("target").noconvert(), nb::arg("both").noconvert(),
        nb::arg("in_pred").noconvert(), nb::arg("in_target").noconvert(),
        nb::arg("num_threads") = 0, counts_doc);
  m.def("channel_counts", &channel_counts<double, exacta_channel_counts_f64>,
        nb::arg("pred").noconvert(), nb::arg("target").noconvert(), nb::arg("both").noconvert(),
        nb::arg("in_pred").noconvert(), nb::arg("in_target").noconvert(),
        nb::arg("num_threads") = 0, counts_doc);
  m.def("masked_confusion", &masked_confusion<std::uint8_t, exacta_masked_confusion_u8>,
        nb::arg("pred").noconvert(), nb::arg("target").noconvert(), nb::arg("mask").noconvert(),
        nb::arg("has_ignore"), nb::arg("ignore_index"), nb::arg("out").noconvert(),
        nb::arg("num_threads") = 0,
        "Confusion counts after excluding masked/ignored voxels; False for invalid kept labels.");
  m.def("masked_confusion", &masked_confusion<std::int32_t, exacta_masked_confusion_i32>,
        nb::arg("pred").noconvert(), nb::arg("target").noconvert(), nb::arg("mask").noconvert(),
        nb::arg("has_ignore"), nb::arg("ignore_index"), nb::arg("out").noconvert(),
        nb::arg("num_threads") = 0,
        "Confusion counts after excluding masked/ignored voxels; False for invalid kept labels.");
  m.def("masked_confusion", &masked_confusion<std::int64_t, exacta_masked_confusion_i64>,
        nb::arg("pred").noconvert(), nb::arg("target").noconvert(), nb::arg("mask").noconvert(),
        nb::arg("has_ignore"), nb::arg("ignore_index"), nb::arg("out").noconvert(),
        nb::arg("num_threads") = 0,
        "Confusion counts after excluding masked/ignored voxels; False for invalid kept labels.");
  m.def("masked_channel_counts", &masked_channel_counts<std::uint8_t, exacta_masked_channel_counts_u8>,
        nb::arg("pred").noconvert(), nb::arg("target").noconvert(), nb::arg("mask").noconvert(),
        nb::arg("channels"), nb::arg("both").noconvert(), nb::arg("in_pred").noconvert(),
        nb::arg("in_target").noconvert(), nb::arg("valid").noconvert(), nb::arg("num_threads") = 0,
        "Binary counts and included voxels with a shared per-case mask; False for invalid kept values.");
  m.def("masked_channel_counts", &masked_channel_counts<std::int64_t, exacta_masked_channel_counts_i64>,
        nb::arg("pred").noconvert(), nb::arg("target").noconvert(), nb::arg("mask").noconvert(),
        nb::arg("channels"), nb::arg("both").noconvert(), nb::arg("in_pred").noconvert(),
        nb::arg("in_target").noconvert(), nb::arg("valid").noconvert(), nb::arg("num_threads") = 0,
        "Binary counts and included voxels with a shared per-case mask; False for invalid kept values.");
  m.def("masked_channel_counts", &masked_channel_counts<float, exacta_masked_channel_counts_f32>,
        nb::arg("pred").noconvert(), nb::arg("target").noconvert(), nb::arg("mask").noconvert(),
        nb::arg("channels"), nb::arg("both").noconvert(), nb::arg("in_pred").noconvert(),
        nb::arg("in_target").noconvert(), nb::arg("valid").noconvert(), nb::arg("num_threads") = 0,
        "Binary counts and included voxels with a shared per-case mask; False for invalid kept values.");
  m.def("masked_channel_counts", &masked_channel_counts<double, exacta_masked_channel_counts_f64>,
        nb::arg("pred").noconvert(), nb::arg("target").noconvert(), nb::arg("mask").noconvert(),
        nb::arg("channels"), nb::arg("both").noconvert(), nb::arg("in_pred").noconvert(),
        nb::arg("in_target").noconvert(), nb::arg("valid").noconvert(), nb::arg("num_threads") = 0,
        "Binary counts and included voxels with a shared per-case mask; False for invalid kept values.");
  m.def("probability_stats", &probability_stats<float, std::uint8_t, exacta_probability_f32_u8>,
        nb::arg("pred").noconvert(), nb::arg("target").noconvert(), nb::arg("mask").noconvert(),
        nb::arg("channels"), nb::arg("from_logits"), nb::arg("kind"), nb::arg("bins"),
        nb::arg("out").noconvert(), nb::arg("num_threads") = 0,
        "Bernoulli score/bin statistics; returns False for invalid included values.");
  m.def("probability_stats", &probability_stats<float, std::int64_t, exacta_probability_f32_i64>,
        nb::arg("pred").noconvert(), nb::arg("target").noconvert(), nb::arg("mask").noconvert(),
        nb::arg("channels"), nb::arg("from_logits"), nb::arg("kind"), nb::arg("bins"),
        nb::arg("out").noconvert(), nb::arg("num_threads") = 0,
        "Bernoulli score/bin statistics; returns False for invalid included values.");
  m.def("probability_stats", &probability_stats<float, float, exacta_probability_f32_f32>,
        nb::arg("pred").noconvert(), nb::arg("target").noconvert(), nb::arg("mask").noconvert(),
        nb::arg("channels"), nb::arg("from_logits"), nb::arg("kind"), nb::arg("bins"),
        nb::arg("out").noconvert(), nb::arg("num_threads") = 0,
        "Bernoulli score/bin statistics; returns False for invalid included values.");
  m.def("probability_stats", &probability_stats<float, double, exacta_probability_f32_f64>,
        nb::arg("pred").noconvert(), nb::arg("target").noconvert(), nb::arg("mask").noconvert(),
        nb::arg("channels"), nb::arg("from_logits"), nb::arg("kind"), nb::arg("bins"),
        nb::arg("out").noconvert(), nb::arg("num_threads") = 0,
        "Bernoulli score/bin statistics; returns False for invalid included values.");
  m.def("probability_stats", &probability_stats<double, std::uint8_t, exacta_probability_f64_u8>,
        nb::arg("pred").noconvert(), nb::arg("target").noconvert(), nb::arg("mask").noconvert(),
        nb::arg("channels"), nb::arg("from_logits"), nb::arg("kind"), nb::arg("bins"),
        nb::arg("out").noconvert(), nb::arg("num_threads") = 0,
        "Bernoulli score/bin statistics; returns False for invalid included values.");
  m.def("probability_stats", &probability_stats<double, std::int64_t, exacta_probability_f64_i64>,
        nb::arg("pred").noconvert(), nb::arg("target").noconvert(), nb::arg("mask").noconvert(),
        nb::arg("channels"), nb::arg("from_logits"), nb::arg("kind"), nb::arg("bins"),
        nb::arg("out").noconvert(), nb::arg("num_threads") = 0,
        "Bernoulli score/bin statistics; returns False for invalid included values.");
  m.def("probability_stats", &probability_stats<double, float, exacta_probability_f64_f32>,
        nb::arg("pred").noconvert(), nb::arg("target").noconvert(), nb::arg("mask").noconvert(),
        nb::arg("channels"), nb::arg("from_logits"), nb::arg("kind"), nb::arg("bins"),
        nb::arg("out").noconvert(), nb::arg("num_threads") = 0,
        "Bernoulli score/bin statistics; returns False for invalid included values.");
  m.def("probability_stats", &probability_stats<double, double, exacta_probability_f64_f64>,
        nb::arg("pred").noconvert(), nb::arg("target").noconvert(), nb::arg("mask").noconvert(),
        nb::arg("channels"), nb::arg("from_logits"), nb::arg("kind"), nb::arg("bins"),
        nb::arg("out").noconvert(), nb::arg("num_threads") = 0,
        "Bernoulli score/bin statistics; returns False for invalid included values.");
  const char *binary_doc =
      "Write into `out` (uint8, (N,)) whether each of `values` (N,) is nonzero; returns False if "
      "one is neither 0 nor 1.";
  m.def("binary_mask", &binary_mask<std::uint8_t, exacta_binary_mask_u8>, nb::arg("values"),
        nb::arg("out"), nb::arg("num_threads") = 0, binary_doc);
  m.def("binary_mask", &binary_mask<std::int32_t, exacta_binary_mask_i32>, nb::arg("values"),
        nb::arg("out"), nb::arg("num_threads") = 0, binary_doc);
  m.def("binary_mask", &binary_mask<std::int64_t, exacta_binary_mask_i64>, nb::arg("values"),
        nb::arg("out"), nb::arg("num_threads") = 0, binary_doc);
  m.def("binary_mask", &binary_mask<float, exacta_binary_mask_f32>, nb::arg("values"),
        nb::arg("out"), nb::arg("num_threads") = 0, binary_doc);
  m.def("binary_mask", &binary_mask<double, exacta_binary_mask_f64>, nb::arg("values"),
        nb::arg("out"), nb::arg("num_threads") = 0, binary_doc);
  m.def("boundary", &boundary, nb::arg("mask"), nb::arg("out"), nb::arg("num_threads") = 0,
        "Write into `out` (uint8, same shape, contiguous) the boundary of the uint8 `mask` (any "
        "strides): mask elements with a face neighbour outside the mask or the image.");
  m.def("masked_sqrt", &masked_sqrt, nb::arg("values"), nb::arg("mask"), nb::arg("out"),
        nb::arg("num_threads") = 0,
        "Write into `out` the square roots of the float64 `values` where the uint8 `mask` is "
        "nonzero, in order; `out` must have one entry per such element.");
  m.def("sort_pairs", &sort_pairs, nb::arg("keys"), nb::arg("values"), nb::arg("num_threads") = 0,
        "Sort the float64 pairs (keys[i], values[i]) in place by key, then value; returns False "
        "(arrays untouched) if one is NaN.");
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
  m.def("neighbour_codes", &neighbour_codes, nb::arg("mask"), nb::arg("code"), nb::arg("counts"),
        nb::arg("num_threads") = 0,
        "Write into `code` (uint8, mask shape + 1) the neighbour code of every voxel corner of "
        "the 2-D or 3-D `mask`; with `counts` (int64, elements per code; may be empty) return the "
        "number of mesh elements.");
  m.def("mesh_elements", &mesh_elements, nb::arg("code"), nb::arg("table"), nb::arg("counts"),
        nb::arg("spacing"), nb::arg("elements"), nb::arg("centres"), nb::arg("block"),
        nb::arg("element_code"), nb::arg("slot"), nb::arg("num_threads") = 0,
        "Write the mesh elements the neighbour codes `code` select from `table` (int64, (codes, "
        "most per code, d, d), half-voxels) into `elements` (M, d, d), with their centres "
        "(M, d), blocks, codes and slots (M,).");
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
  m.def("skeleton", &skeleton, nb::arg("mask"), nb::arg("out"), "Lee thinning of a 2-D or 3-D CPU mask.");
  m.def("soft_pool", &soft_pool<float, exacta_soft_pool_f32>, nb::arg("input"), nb::arg("out"),
        nb::arg("indices"), nb::arg("erosion"), nb::arg("num_threads") = 0);
  m.def("soft_pool", &soft_pool<double, exacta_soft_pool_f64>, nb::arg("input"), nb::arg("out"),
        nb::arg("indices"), nb::arg("erosion"), nb::arg("num_threads") = 0);
}
