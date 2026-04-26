#include "tensoreecs/eec.hpp"

#include <Eigen/Dense>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main() {
  using T = double;

  bool caught = false;
  try {
    Eigen::VectorXd w(2);
    w << 1.0, 2.0;
    Eigen::MatrixXd bad(2, 3);
    bad.setZero();
    tensoreecs::BinningConfigT<T> cfg;
    (void)tensoreecs::compute_eec2_from_pairwise<T, tensoreecs::TensorStorageMode::SparseIndexed>(w, bad, cfg);
  } catch (const std::invalid_argument&) {
    caught = true;
  }

  if (!caught) {
    std::cerr << "test_edge_cases failed: expected invalid_argument\n";
    return EXIT_FAILURE;
  }

  std::cout << "test_edge_cases passed\n";
  return EXIT_SUCCESS;
}
