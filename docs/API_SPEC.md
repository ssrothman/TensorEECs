# TensorEECs API Specification

This document defines the canonical API and behavior contract for the TensorEECs v1 implementation.

## Scope

- Implement projected 2-point and 3-point EECs from pairwise angular distances.
- Support both sparse-indexed and explicitly materialized dense tensor representations.
- Support both 32-bit and 64-bit floating point computation.
- Provide C++ and Python APIs with matching semantics.

## Build and Platform

- Language baseline: C++20
- Primary target: Linux
- Local Windows development: run configure/build/tests through WSL

## Enumerations

```cpp
namespace tensoreecs {

enum class TensorStorageMode : uint8_t {
  SparseIndexed = 0,
  MaterializedDense = 1
};

enum class NumericType : uint8_t {
  Float32 = 0,
  Float64 = 1
};

enum class PhiMode : uint8_t {
  ExactUniqueDistances = 0,
  BinnedEdges = 1
};

}  // namespace tensoreecs
```

## Defaults

- `TensorStorageMode::SparseIndexed`
- `NumericType::Float64`
- `PhiMode::ExactUniqueDistances`

## Tolerances

```cpp
namespace tensoreecs::constants {

template <typename T>
struct Tolerances;

template <>
struct Tolerances<float> {
  static constexpr float kExactEpsilon = 1e-6f;
  static constexpr float kSymmetryAbsTol = 5e-6f;
  static constexpr float kDiagonalAbsTol = 5e-6f;
  static constexpr float kCompareRtol = 1e-4f;
  static constexpr float kCompareAtol = 1e-6f;
};

template <>
struct Tolerances<double> {
  static constexpr double kExactEpsilon = 1e-12;
  static constexpr double kSymmetryAbsTol = 5e-12;
  static constexpr double kDiagonalAbsTol = 5e-12;
  static constexpr double kCompareRtol = 1e-10;
  static constexpr double kCompareAtol = 1e-12;
};

}  // namespace tensoreecs::constants
```

## Exact Mode Grouping Rule

In exact mode, near-equal pairwise distances are grouped using epsilon and collapse to the first representative in sorted order.

Algorithm:
1. Collect candidate off-diagonal distances.
2. Stable-sort ascending.
3. Start `rho` with the first value.
4. For each next value `d`, if `abs(d - rho.back()) <= exact_epsilon`, map to current bin.
5. Otherwise, append a new representative `d` to `rho`.

## Core Types

```cpp
namespace tensoreecs {

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

}  // namespace tensoreecs
```

## Function Contracts

```cpp
namespace tensoreecs {

template <typename T, TensorStorageMode StorageMode>
PrecomputeResultT<T, StorageMode> build_precompute(
    const Eigen::Ref<const VectorT<T>>& weights_xi,
    const Eigen::Ref<const MatrixT<T>>& pairwise_R,
    const BinningConfigT<T>& cfg);

template <typename T, TensorStorageMode StorageMode>
VectorT<T> compute_eec2(
    const Eigen::Ref<const VectorT<T>>& weights_xi,
    const PrecomputeResultT<T, StorageMode>& pre);

template <typename T, TensorStorageMode StorageMode>
VectorT<T> compute_eec3(
    const Eigen::Ref<const VectorT<T>>& weights_xi,
    const PrecomputeResultT<T, StorageMode>& pre);

template <typename T, TensorStorageMode StorageMode>
VectorT<T> compute_eec2_from_pairwise(
    const Eigen::Ref<const VectorT<T>>& weights_xi,
    const Eigen::Ref<const MatrixT<T>>& pairwise_R,
    const BinningConfigT<T>& cfg);

template <typename T, TensorStorageMode StorageMode>
VectorT<T> compute_eec3_from_pairwise(
    const Eigen::Ref<const VectorT<T>>& weights_xi,
    const Eigen::Ref<const MatrixT<T>>& pairwise_R,
    const BinningConfigT<T>& cfg);

}  // namespace tensoreecs
```

Input validation requirements:
- `weights_xi.size() == pairwise_R.rows() == pairwise_R.cols()`
- `pairwise_R` symmetric within dtype tolerance
- diagonal approximately zero within dtype tolerance
- binned mode requires strictly increasing `bin_edges` with at least 2 values
- invalid inputs throw `std::invalid_argument`

## Runtime Dispatch API

The library also exposes runtime dispatch wrappers that map runtime options to pre-instantiated template implementations.

## Python API

```python
build_precompute(weights_xi, pairwise_R, *, mode="exact", bin_edges=None,
                 exact_epsilon=None, storage_mode="sparse", dtype="float64") -> dict

compute_eec2(weights_xi, pairwise_R, *, mode="exact", bin_edges=None,
             exact_epsilon=None, storage_mode="sparse", dtype="float64") -> np.ndarray

compute_eec3(weights_xi, pairwise_R, *, mode="exact", bin_edges=None,
             exact_epsilon=None, storage_mode="sparse", dtype="float64") -> np.ndarray
```

Python error contract:
- wrong rank/dtype: `TypeError`
- invalid shape/config values: `ValueError`

## Constexpr Policy

Mark as `constexpr` whenever possible:
- flattening/index helper formulas
- triangular-number formulas
- compile-time constants and default parameters
- enum parse helpers and trait checks

Runtime kernels that iterate over dynamic event data are not `constexpr`.

## Test Matrix

Run correctness tests across:
- sparse + float32
- sparse + float64
- dense + float32
- dense + float64

For each, verify parity vs brute-force references for 2-point and 3-point on small synthetic events.
