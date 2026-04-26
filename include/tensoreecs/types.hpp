#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace tensoreecs {

enum class TensorStorageMode : uint8_t {
  SparseIndexed = 0,
  MaterializedDense = 1,
};

enum class NumericType : uint8_t {
  Float32 = 0,
  Float64 = 1,
};

enum class PhiMode : uint8_t {
  ExactUniqueDistances = 0,
  BinnedEdges = 1,
};

namespace constants {

template <typename T>
struct Tolerances;

template <>
struct Tolerances<float> {
  static constexpr float kExactEpsilon = 1.0e-6f;
  static constexpr float kSymmetryAbsTol = 5.0e-6f;
  static constexpr float kDiagonalAbsTol = 5.0e-6f;
  static constexpr float kCompareRtol = 1.0e-4f;
  static constexpr float kCompareAtol = 1.0e-6f;
};

template <>
struct Tolerances<double> {
  static constexpr double kExactEpsilon = 1.0e-12;
  static constexpr double kSymmetryAbsTol = 5.0e-12;
  static constexpr double kDiagonalAbsTol = 5.0e-12;
  static constexpr double kCompareRtol = 1.0e-10;
  static constexpr double kCompareAtol = 1.0e-12;
};

}  // namespace constants

template <typename T>
using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
struct BinningConfigT {
  PhiMode mode = PhiMode::ExactUniqueDistances;
  VectorT<T> bin_edges;
  T exact_epsilon = constants::Tolerances<T>::kExactEpsilon;
};

struct RuntimeConfig {
  TensorStorageMode storage_mode = TensorStorageMode::SparseIndexed;
  NumericType numeric_type = NumericType::Float64;
  PhiMode phi_mode = PhiMode::ExactUniqueDistances;
  double exact_epsilon = constants::Tolerances<double>::kExactEpsilon;
};

template <typename T, TensorStorageMode StorageMode>
struct PrecomputeResultT;

template <typename T>
struct PrecomputeResultT<T, TensorStorageMode::SparseIndexed> {
  VectorT<T> rho;
  std::vector<uint32_t> phi_index;
  std::vector<uint32_t> omega_next;
  uint32_t n_particles = 0;
  uint32_t n_r = 0;
};

template <typename T>
struct PrecomputeResultT<T, TensorStorageMode::MaterializedDense> {
  VectorT<T> rho;
  std::vector<uint8_t> phi_dense;
  std::vector<uint8_t> omega_dense;
  std::vector<uint32_t> phi_index_lookup;
  std::vector<uint32_t> omega_next_lookup;
  uint32_t n_particles = 0;
  uint32_t n_r = 0;
};

constexpr uint32_t kInvalidBin = 0xFFFFFFFFu;

inline constexpr const char* to_string(TensorStorageMode mode) {
  return (mode == TensorStorageMode::SparseIndexed) ? "sparse" : "dense";
}

inline constexpr const char* to_string(NumericType type) {
  return (type == NumericType::Float32) ? "float32" : "float64";
}

inline constexpr const char* to_string(PhiMode mode) {
  return (mode == PhiMode::ExactUniqueDistances) ? "exact" : "binned";
}

inline TensorStorageMode parse_storage_mode(const std::string& value) {
  if (value == "sparse") {
    return TensorStorageMode::SparseIndexed;
  }
  if (value == "dense") {
    return TensorStorageMode::MaterializedDense;
  }
  throw std::invalid_argument("storage_mode must be 'sparse' or 'dense'");
}

inline NumericType parse_numeric_type(const std::string& value) {
  if (value == "float32") {
    return NumericType::Float32;
  }
  if (value == "float64") {
    return NumericType::Float64;
  }
  throw std::invalid_argument("dtype must be 'float32' or 'float64'");
}

inline PhiMode parse_phi_mode(const std::string& value) {
  if (value == "exact") {
    return PhiMode::ExactUniqueDistances;
  }
  if (value == "binned") {
    return PhiMode::BinnedEdges;
  }
  throw std::invalid_argument("mode must be 'exact' or 'binned'");
}

template <typename T>
constexpr bool is_supported_scalar_v = std::is_same_v<T, float> || std::is_same_v<T, double>;

}  // namespace tensoreecs
