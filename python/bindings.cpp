#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>

#include "tensoreecs/eec.hpp"

namespace py = pybind11;
using tensoreecs::NumericType;
using tensoreecs::PhiMode;
using tensoreecs::RuntimeConfig;
using tensoreecs::TensorStorageMode;

namespace {

Eigen::VectorXd to_vector(const py::array_t<double, py::array::c_style | py::array::forcecast>& arr) {
  if (arr.ndim() != 1) {
    throw py::type_error("Expected a 1D array");
  }
  const auto n = static_cast<Eigen::Index>(arr.shape(0));
  Eigen::VectorXd out(n);
  auto r = arr.unchecked<1>();
  for (Eigen::Index i = 0; i < n; ++i) {
    out(i) = r(i);
  }
  return out;
}

Eigen::MatrixXd to_matrix(const py::array_t<double, py::array::c_style | py::array::forcecast>& arr) {
  if (arr.ndim() != 2) {
    throw py::type_error("Expected a 2D array");
  }
  const auto n0 = static_cast<Eigen::Index>(arr.shape(0));
  const auto n1 = static_cast<Eigen::Index>(arr.shape(1));
  Eigen::MatrixXd out(n0, n1);
  auto r = arr.unchecked<2>();
  for (Eigen::Index i = 0; i < n0; ++i) {
    for (Eigen::Index j = 0; j < n1; ++j) {
      out(i, j) = r(i, j);
    }
  }
  return out;
}

template <typename T>
py::array_t<T> from_eigen_vector(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) {
  py::array_t<T> out(v.size());
  auto w = out.template mutable_unchecked<1>();
  for (Eigen::Index i = 0; i < v.size(); ++i) {
    w(i) = v(i);
  }
  return out;
}

RuntimeConfig parse_runtime(
    const std::string& mode,
    const py::object& exact_epsilon,
    const std::string& storage_mode,
    const std::string& dtype) {
  RuntimeConfig cfg;
  cfg.phi_mode = tensoreecs::parse_phi_mode(mode);
  cfg.storage_mode = tensoreecs::parse_storage_mode(storage_mode);
  cfg.numeric_type = tensoreecs::parse_numeric_type(dtype);
  if (!exact_epsilon.is_none()) {
    cfg.exact_epsilon = exact_epsilon.cast<double>();
  }
  return cfg;
}

py::array compute_runtime(
    bool three_point,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& weights_xi,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& pairwise_R,
    const std::string& mode,
    const py::object& bin_edges,
    const py::object& exact_epsilon,
    const std::string& storage_mode,
    const std::string& dtype) {
  const auto xi = to_vector(weights_xi);
  const auto r = to_matrix(pairwise_R);

  Eigen::VectorXd edges;
  if (!bin_edges.is_none()) {
    edges = to_vector(bin_edges.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>());
  }

  auto cfg = parse_runtime(mode, exact_epsilon, storage_mode, dtype);

  auto out = three_point
                 ? tensoreecs::compute_eec3_runtime(xi, r, cfg, edges)
                 : tensoreecs::compute_eec2_runtime(xi, r, cfg, edges);

  if (std::holds_alternative<Eigen::VectorXf>(out)) {
    return from_eigen_vector(std::get<Eigen::VectorXf>(out));
  }
  return from_eigen_vector(std::get<Eigen::VectorXd>(out));
}

}  // namespace

PYBIND11_MODULE(_tensoreecs, m) {
  m.doc() = "TensorEECs python bindings";

  m.def(
      "compute_eec2",
      [](const py::array_t<double, py::array::c_style | py::array::forcecast>& weights_xi,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& pairwise_R,
         const std::string& mode,
         const py::object& bin_edges,
         const py::object& exact_epsilon,
         const std::string& storage_mode,
         const std::string& dtype) {
        return compute_runtime(
            false,
            weights_xi,
            pairwise_R,
            mode,
            bin_edges,
            exact_epsilon,
            storage_mode,
            dtype);
      },
      py::arg("weights_xi"),
      py::arg("pairwise_R"),
      py::kw_only(),
      py::arg("mode") = "exact",
      py::arg("bin_edges") = py::none(),
      py::arg("exact_epsilon") = py::none(),
      py::arg("storage_mode") = "sparse",
      py::arg("dtype") = "float64");

  m.def(
      "compute_eec3",
      [](const py::array_t<double, py::array::c_style | py::array::forcecast>& weights_xi,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& pairwise_R,
         const std::string& mode,
         const py::object& bin_edges,
         const py::object& exact_epsilon,
         const std::string& storage_mode,
         const std::string& dtype) {
        return compute_runtime(
            true,
            weights_xi,
            pairwise_R,
            mode,
            bin_edges,
            exact_epsilon,
            storage_mode,
            dtype);
      },
      py::arg("weights_xi"),
      py::arg("pairwise_R"),
      py::kw_only(),
      py::arg("mode") = "exact",
      py::arg("bin_edges") = py::none(),
      py::arg("exact_epsilon") = py::none(),
      py::arg("storage_mode") = "sparse",
      py::arg("dtype") = "float64");
}
