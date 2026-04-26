#pragma once

#include "tensoreecs/internal/indexing.hpp"
#include "tensoreecs/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace tensoreecs::detail {

template <typename T>
inline void validate_inputs(
    const Eigen::Ref<const VectorT<T>>& weights_xi,
    const Eigen::Ref<const MatrixT<T>>& pairwise_R,
    const BinningConfigT<T>& cfg) {
  const auto n = weights_xi.size();
  if (n < 0) {
    throw std::invalid_argument("weights_xi has invalid size");
  }
  if (pairwise_R.rows() != n || pairwise_R.cols() != n) {
    throw std::invalid_argument("pairwise_R must be square and match weights_xi size");
  }

  for (Eigen::Index i = 0; i < n; ++i) {
    if (std::abs(pairwise_R(i, i)) > constants::Tolerances<T>::kDiagonalAbsTol) {
      throw std::invalid_argument("pairwise_R diagonal must be approximately zero");
    }
    for (Eigen::Index j = i + 1; j < n; ++j) {
      if (std::abs(pairwise_R(i, j) - pairwise_R(j, i)) > constants::Tolerances<T>::kSymmetryAbsTol) {
        throw std::invalid_argument("pairwise_R must be symmetric within tolerance");
      }
    }
  }

  if (cfg.mode == PhiMode::BinnedEdges) {
    if (cfg.bin_edges.size() < 2) {
      throw std::invalid_argument("bin_edges must have at least 2 entries in binned mode");
    }
    for (Eigen::Index i = 0; i + 1 < cfg.bin_edges.size(); ++i) {
      if (!(cfg.bin_edges(i) < cfg.bin_edges(i + 1))) {
        throw std::invalid_argument("bin_edges must be strictly increasing");
      }
    }
  }
}

template <typename T>
inline std::pair<VectorT<T>, std::vector<uint32_t>> build_rho_and_pair_to_bin(
    const Eigen::Ref<const MatrixT<T>>& pairwise_R,
    const BinningConfigT<T>& cfg) {
  const std::size_t n = static_cast<std::size_t>(pairwise_R.rows());
  std::vector<uint32_t> pair_to_bin(n * n, kInvalidBin);

  if (n == 0) {
    return {VectorT<T>(), pair_to_bin};
  }

  if (cfg.mode == PhiMode::BinnedEdges) {
    const auto n_r = static_cast<std::size_t>(cfg.bin_edges.size() - 1);
    VectorT<T> rho(static_cast<Eigen::Index>(n_r));
    for (std::size_t a = 0; a < n_r; ++a) {
      rho(static_cast<Eigen::Index>(a)) =
          static_cast<T>(0.5) * (cfg.bin_edges(static_cast<Eigen::Index>(a)) +
                                 cfg.bin_edges(static_cast<Eigen::Index>(a + 1)));
    }

    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        if (i == j) {
          continue;
        }
        const T d = pairwise_R(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
        const T* edges_begin = cfg.bin_edges.data();
        const T* edges_end = edges_begin + cfg.bin_edges.size();
        auto it = std::upper_bound(edges_begin, edges_end, d);
        if (it != edges_begin && it != edges_end) {
          const std::size_t idx = static_cast<std::size_t>(it - edges_begin - 1);
          if (idx < n_r) {
            pair_to_bin[internal::matrix_offset(i, j, n)] = static_cast<uint32_t>(idx);
          }
        } else if (it == edges_end && d == cfg.bin_edges(cfg.bin_edges.size() - 1)) {
          pair_to_bin[internal::matrix_offset(i, j, n)] = static_cast<uint32_t>(n_r - 1);
        }
      }
    }

    return {rho, pair_to_bin};
  }

  std::vector<std::tuple<T, std::size_t, std::size_t>> entries;
  entries.reserve(n * (n - 1) / 2);

  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      entries.emplace_back(
          pairwise_R(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)), i, j);
    }
  }

  std::stable_sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
    return std::get<0>(lhs) < std::get<0>(rhs);
  });

  std::vector<T> rho_values;
  rho_values.reserve(entries.size());

  for (const auto& item : entries) {
    const T d = std::get<0>(item);
    const std::size_t i = std::get<1>(item);
    const std::size_t j = std::get<2>(item);

    uint32_t bin = 0;
    if (rho_values.empty()) {
      rho_values.push_back(d);
      bin = 0;
    } else {
      const T representative = rho_values.back();
      if (std::abs(d - representative) <= cfg.exact_epsilon) {
        bin = static_cast<uint32_t>(rho_values.size() - 1);
      } else {
        rho_values.push_back(d);
        bin = static_cast<uint32_t>(rho_values.size() - 1);
      }
    }

    pair_to_bin[internal::matrix_offset(i, j, n)] = bin;
    pair_to_bin[internal::matrix_offset(j, i, n)] = bin;
  }

  VectorT<T> rho(static_cast<Eigen::Index>(rho_values.size()));
  for (std::size_t a = 0; a < rho_values.size(); ++a) {
    rho(static_cast<Eigen::Index>(a)) = rho_values[a];
  }

  return {rho, pair_to_bin};
}

template <typename T>
inline std::vector<uint32_t> build_sparse_omega_next(
    const Eigen::Ref<const MatrixT<T>>& pairwise_R,
    const VectorT<T>& rho,
    const std::vector<uint32_t>& pair_to_bin) {
  const std::size_t n = static_cast<std::size_t>(pairwise_R.rows());
  const std::size_t n_r = static_cast<std::size_t>(rho.size());
  std::vector<uint32_t> omega_next(n_r * n * n, kInvalidBin);

  for (std::size_t a_prev = 0; a_prev < n_r; ++a_prev) {
    const T rep = rho(static_cast<Eigen::Index>(a_prev));
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        if (i == j) {
          omega_next[internal::omega_next_offset(a_prev, i, j, n)] = static_cast<uint32_t>(a_prev);
          continue;
        }
        const T d = pairwise_R(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
        const uint32_t phi_bin = pair_to_bin[internal::matrix_offset(i, j, n)];
        if (d <= rep) {
          omega_next[internal::omega_next_offset(a_prev, i, j, n)] = static_cast<uint32_t>(a_prev);
        } else {
          omega_next[internal::omega_next_offset(a_prev, i, j, n)] = phi_bin;
        }
      }
    }
  }
  return omega_next;
}

template <typename T>
inline std::vector<uint8_t> build_dense_phi(
    std::size_t n,
    std::size_t n_r,
    const std::vector<uint32_t>& pair_to_bin) {
  std::vector<uint8_t> phi_dense(n_r * n * n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const uint32_t bin = pair_to_bin[internal::matrix_offset(i, j, n)];
      if (bin != kInvalidBin) {
        phi_dense[internal::phi_offset(bin, i, j, n)] = 1;
      }
    }
  }
  return phi_dense;
}

template <typename T>
inline std::vector<uint8_t> build_dense_omega(
    std::size_t n,
    std::size_t n_r,
    const std::vector<uint32_t>& omega_next) {
  std::vector<uint8_t> omega_dense(n_r * n_r * n * n, 0);
  for (std::size_t a_prev = 0; a_prev < n_r; ++a_prev) {
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        const uint32_t a_next = omega_next[internal::omega_next_offset(a_prev, i, j, n)];
        if (a_next != kInvalidBin) {
          omega_dense[internal::omega_dense_offset(a_next, a_prev, i, j, n, n_r)] = 1;
        }
      }
    }
  }
  return omega_dense;
}

template <typename T>
inline T dense_indicator_at_phi(
    const std::vector<uint8_t>& phi_dense,
    std::size_t n,
    std::size_t a,
    std::size_t i,
    std::size_t j) {
  return static_cast<T>(phi_dense[internal::phi_offset(a, i, j, n)]);
}

template <typename T>
inline T dense_indicator_at_omega(
    const std::vector<uint8_t>& omega_dense,
    std::size_t n,
    std::size_t n_r,
    std::size_t a_next,
    std::size_t a_prev,
    std::size_t i,
    std::size_t j) {
  return static_cast<T>(omega_dense[internal::omega_dense_offset(a_next, a_prev, i, j, n, n_r)]);
}

template <typename T>
inline BinningConfigT<T> to_binning_config(const RuntimeConfig& cfg, const VectorT<T>& bin_edges) {
  BinningConfigT<T> out;
  out.mode = cfg.phi_mode;
  out.exact_epsilon = static_cast<T>(cfg.exact_epsilon);
  out.bin_edges = bin_edges;
  return out;
}

}  // namespace tensoreecs::detail

namespace tensoreecs {

template <typename T, TensorStorageMode StorageMode>
PrecomputeResultT<T, StorageMode> build_precompute(
    const Eigen::Ref<const VectorT<T>>& weights_xi,
    const Eigen::Ref<const MatrixT<T>>& pairwise_R,
    const BinningConfigT<T>& cfg) {
  static_assert(is_supported_scalar_v<T>, "Unsupported scalar type");
  detail::validate_inputs(weights_xi, pairwise_R, cfg);

  auto [rho, pair_to_bin] = detail::build_rho_and_pair_to_bin(pairwise_R, cfg);

  PrecomputeResultT<T, StorageMode> result;
  result.rho = rho;
  result.n_particles = static_cast<uint32_t>(weights_xi.size());
  result.n_r = static_cast<uint32_t>(rho.size());

  if constexpr (StorageMode == TensorStorageMode::SparseIndexed) {
    result.phi_index = std::move(pair_to_bin);
    result.omega_next = detail::build_sparse_omega_next(pairwise_R, rho, result.phi_index);
  } else {
    const std::size_t n = static_cast<std::size_t>(weights_xi.size());
    const std::size_t n_r = static_cast<std::size_t>(rho.size());
    const auto omega_next = detail::build_sparse_omega_next(pairwise_R, rho, pair_to_bin);
    result.phi_dense = detail::build_dense_phi<T>(n, n_r, pair_to_bin);
    result.omega_dense = detail::build_dense_omega<T>(n, n_r, omega_next);
    result.phi_index_lookup = pair_to_bin;
    result.omega_next_lookup = omega_next;
  }

  return result;
}

}  // namespace tensoreecs
