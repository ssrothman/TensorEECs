#include "tensoreecs/eec.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

template <typename T>
bool approx_equal(const Eigen::Matrix<T, Eigen::Dynamic, 1>& a, const Eigen::Matrix<T, Eigen::Dynamic, 1>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (Eigen::Index i = 0; i < a.size(); ++i) {
    const T da = std::abs(a(i) - b(i));
    const T tol = tensoreecs::constants::Tolerances<T>::kCompareAtol +
                  tensoreecs::constants::Tolerances<T>::kCompareRtol * std::abs(b(i));
    if (da > tol) {
      return false;
    }
  }
  return true;
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> brute2(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& w,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& r,
    const tensoreecs::BinningConfigT<T>& cfg) {
  auto pre = tensoreecs::build_precompute<T, tensoreecs::TensorStorageMode::SparseIndexed>(w, r, cfg);
  Eigen::Matrix<T, Eigen::Dynamic, 1> out = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(pre.n_r);
  const std::size_t n = static_cast<std::size_t>(w.size());
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const uint32_t a = pre.phi_index[tensoreecs::internal::matrix_offset(i, j, n)];
      if (a != tensoreecs::kInvalidBin && a < pre.n_r) {
        out(static_cast<Eigen::Index>(a)) += w(static_cast<Eigen::Index>(i)) * w(static_cast<Eigen::Index>(j));
      }
    }
  }
  return out;
}

template <typename T, tensoreecs::TensorStorageMode Mode>
bool run_case() {
  Eigen::Matrix<T, Eigen::Dynamic, 1> w(4);
  w << static_cast<T>(0.2), static_cast<T>(0.3), static_cast<T>(0.1), static_cast<T>(0.4);

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> r(4, 4);
  r << 0, 0.2, 0.7, 0.5,
       0.2, 0, 0.4, 0.8,
       0.7, 0.4, 0, 0.3,
       0.5, 0.8, 0.3, 0;

  tensoreecs::BinningConfigT<T> cfg;
  cfg.mode = tensoreecs::PhiMode::ExactUniqueDistances;

  const auto pre = tensoreecs::build_precompute<T, Mode>(w, r, cfg);
  const auto got = tensoreecs::compute_eec2<T, Mode>(w, pre);
  const auto ref = brute2(w, r, cfg);
  return approx_equal(got, ref);
}

}  // namespace

int main() {
  const bool ok =
      run_case<float, tensoreecs::TensorStorageMode::SparseIndexed>() &&
      run_case<float, tensoreecs::TensorStorageMode::MaterializedDense>() &&
      run_case<double, tensoreecs::TensorStorageMode::SparseIndexed>() &&
      run_case<double, tensoreecs::TensorStorageMode::MaterializedDense>();

  if (!ok) {
    std::cerr << "test_eec2 failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "test_eec2 passed\n";
  return EXIT_SUCCESS;
}
