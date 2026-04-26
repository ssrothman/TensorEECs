#ifndef TENSOREECS_INTERNAL_INDEXING_HPP_
#define TENSOREECS_INTERNAL_INDEXING_HPP_

#include <cstddef>

namespace tensoreecs::internal {

constexpr std::size_t matrix_offset(std::size_t i, std::size_t j, std::size_t n) {
  return i * n + j;
}

constexpr std::size_t phi_offset(std::size_t a, std::size_t i, std::size_t j, std::size_t n) {
  return ((a * n) + i) * n + j;
}

constexpr std::size_t omega_next_offset(std::size_t a_prev, std::size_t i, std::size_t j, std::size_t n) {
  return ((a_prev * n) + i) * n + j;
}

constexpr std::size_t omega_dense_offset(
    std::size_t a_next,
    std::size_t a_prev,
    std::size_t i,
    std::size_t j,
    std::size_t n,
    std::size_t n_r) {
  return ((((a_next * n_r) + a_prev) * n) + i) * n + j;
}

constexpr std::size_t triangular_number(std::size_t n) {
  return n * (n + 1) / 2;
}

}  // namespace tensoreecs::internal

#endif  // TENSOREECS_INTERNAL_INDEXING_HPP_
