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

}  // namespace

int main() {
  using T = double;

  Eigen::VectorXd w(3);
  w << 0.5, 0.2, 0.3;

  Eigen::MatrixXd r(3, 3);
  r << 0, 0.4, 0.9,
       0.4, 0, 0.6,
       0.9, 0.6, 0;

  Eigen::VectorXd w_perm(3);
  w_perm << w(2), w(0), w(1);

  Eigen::MatrixXd r_perm(3, 3);
  r_perm << 0, r(2, 0), r(2, 1),
            r(0, 2), 0, r(0, 1),
            r(1, 2), r(1, 0), 0;

  tensoreecs::BinningConfigT<T> cfg;
  cfg.mode = tensoreecs::PhiMode::ExactUniqueDistances;

  auto a = tensoreecs::compute_eec3_from_pairwise<T, tensoreecs::TensorStorageMode::SparseIndexed>(w, r, cfg);
  auto b = tensoreecs::compute_eec3_from_pairwise<T, tensoreecs::TensorStorageMode::SparseIndexed>(w_perm, r_perm, cfg);

  if (!approx_equal(a, b)) {
    std::cerr << "test_invariance failed\n";
    return EXIT_FAILURE;
  }

  std::cout << "test_invariance passed\n";
  return EXIT_SUCCESS;
}
