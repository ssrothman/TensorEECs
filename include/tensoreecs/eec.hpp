#pragma once

#include "tensoreecs/precompute.hpp"

#include <variant>

namespace tensoreecs {

template <typename T, TensorStorageMode StorageMode>
VectorT<T> compute_eec2(
    const Eigen::Ref<const VectorT<T>>& weights_xi,
    const PrecomputeResultT<T, StorageMode>& pre) {
  const std::size_t n = static_cast<std::size_t>(weights_xi.size());
  if (pre.n_particles != n) {
    throw std::invalid_argument("weights_xi size must match precompute n_particles");
  }

  VectorT<T> out = VectorT<T>::Zero(static_cast<Eigen::Index>(pre.n_r));
  const std::size_t n_r = static_cast<std::size_t>(pre.n_r);
  const T* w = weights_xi.data();
  T* out_ptr = out.data();

  if constexpr (StorageMode == TensorStorageMode::SparseIndexed) {
    for (std::size_t i = 0; i < n; ++i) {
      const T wi = w[i];
      for (std::size_t j = 0; j < n; ++j) {
        const uint32_t a = pre.phi_index[internal::matrix_offset(i, j, n)];
        if (a == kInvalidBin || a >= n_r) {
          continue;
        }
        out_ptr[a] += wi * w[j];
      }
    }
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      const T wi = w[i];
      for (std::size_t j = 0; j < n; ++j) {
        const uint32_t a = pre.phi_index_lookup[internal::matrix_offset(i, j, n)];
        if (a == kInvalidBin || a >= n_r) {
          continue;
        }
        out_ptr[a] += wi * w[j];
      }
    }
  }

  return out;
}

template <typename T, TensorStorageMode StorageMode>
VectorT<T> compute_eec3(
    const Eigen::Ref<const VectorT<T>>& weights_xi,
    const PrecomputeResultT<T, StorageMode>& pre) {
  const std::size_t n = static_cast<std::size_t>(weights_xi.size());
  if (pre.n_particles != n) {
    throw std::invalid_argument("weights_xi size must match precompute n_particles");
  }

  VectorT<T> out = VectorT<T>::Zero(static_cast<Eigen::Index>(pre.n_r));
  const std::size_t n_r = static_cast<std::size_t>(pre.n_r);
  const T* w = weights_xi.data();
  T* out_ptr = out.data();

  if constexpr (StorageMode == TensorStorageMode::SparseIndexed) {
    for (std::size_t i = 0; i < n; ++i) {
      const T wi = w[i];
      for (std::size_t j = 0; j < n; ++j) {
        const T wij = wi * w[j];
        const uint32_t a = pre.phi_index[internal::matrix_offset(i, j, n)];
        if (a == kInvalidBin || a >= n_r) {
          continue;
        }
        for (std::size_t k = 0; k < n; ++k) {
          const uint32_t b = pre.omega_next[internal::omega_next_offset(a, i, k, n)];
          if (b == kInvalidBin || b >= n_r) {
            continue;
          }
          const uint32_t c = pre.omega_next[internal::omega_next_offset(b, j, k, n)];
          if (c == kInvalidBin || c >= n_r) {
            continue;
          }
          out_ptr[c] += wij * w[k];
        }
      }
    }
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      const T wi = w[i];
      for (std::size_t j = 0; j < n; ++j) {
        const T wij = wi * w[j];
        const uint32_t a = pre.phi_index_lookup[internal::matrix_offset(i, j, n)];
        if (a == kInvalidBin || a >= n_r) {
          continue;
        }
        for (std::size_t k = 0; k < n; ++k) {
          const uint32_t b = pre.omega_next_lookup[internal::omega_next_offset(a, i, k, n)];
          if (b == kInvalidBin || b >= n_r) {
            continue;
          }
          const uint32_t c = pre.omega_next_lookup[internal::omega_next_offset(b, j, k, n)];
          if (c == kInvalidBin || c >= n_r) {
            continue;
          }
          out_ptr[c] += wij * w[k];
        }
      }
    }
  }

  return out;
}

template <typename T, TensorStorageMode StorageMode>
VectorT<T> compute_eec2_from_pairwise(
    const Eigen::Ref<const VectorT<T>>& weights_xi,
    const Eigen::Ref<const MatrixT<T>>& pairwise_R,
    const BinningConfigT<T>& cfg) {
  const auto pre = build_precompute<T, StorageMode>(weights_xi, pairwise_R, cfg);
  return compute_eec2<T, StorageMode>(weights_xi, pre);
}

template <typename T, TensorStorageMode StorageMode>
VectorT<T> compute_eec3_from_pairwise(
    const Eigen::Ref<const VectorT<T>>& weights_xi,
    const Eigen::Ref<const MatrixT<T>>& pairwise_R,
    const BinningConfigT<T>& cfg) {
  const auto pre = build_precompute<T, StorageMode>(weights_xi, pairwise_R, cfg);
  return compute_eec3<T, StorageMode>(weights_xi, pre);
}

using RuntimeVector = std::variant<Eigen::VectorXf, Eigen::VectorXd>;

RuntimeVector compute_eec2_runtime(
    const Eigen::Ref<const Eigen::VectorXd>& weights_xi,
    const Eigen::Ref<const Eigen::MatrixXd>& pairwise_R,
    const RuntimeConfig& cfg,
    const Eigen::Ref<const Eigen::VectorXd>& bin_edges);

RuntimeVector compute_eec3_runtime(
    const Eigen::Ref<const Eigen::VectorXd>& weights_xi,
    const Eigen::Ref<const Eigen::MatrixXd>& pairwise_R,
    const RuntimeConfig& cfg,
    const Eigen::Ref<const Eigen::VectorXd>& bin_edges);

}  // namespace tensoreecs
