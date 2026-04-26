#include "tensoreecs/eec.hpp"

namespace tensoreecs {

namespace {

template <typename T>
BinningConfigT<T> make_binning_cfg(const RuntimeConfig& cfg, const Eigen::Ref<const Eigen::VectorXd>& edges) {
  BinningConfigT<T> out;
  out.mode = cfg.phi_mode;
  out.exact_epsilon = static_cast<T>(cfg.exact_epsilon);
  if (cfg.phi_mode == PhiMode::BinnedEdges) {
    out.bin_edges = edges.template cast<T>();
  }
  return out;
}

template <typename T, TensorStorageMode StorageMode>
VectorT<T> dispatch_eec2(
    const Eigen::Ref<const Eigen::VectorXd>& weights_xi,
    const Eigen::Ref<const Eigen::MatrixXd>& pairwise_R,
    const RuntimeConfig& cfg,
    const Eigen::Ref<const Eigen::VectorXd>& bin_edges) {
  const auto xi = weights_xi.template cast<T>();
  const auto r = pairwise_R.template cast<T>();
  const auto bcfg = make_binning_cfg<T>(cfg, bin_edges);
  return compute_eec2_from_pairwise<T, StorageMode>(xi, r, bcfg);
}

template <typename T, TensorStorageMode StorageMode>
VectorT<T> dispatch_eec3(
    const Eigen::Ref<const Eigen::VectorXd>& weights_xi,
    const Eigen::Ref<const Eigen::MatrixXd>& pairwise_R,
    const RuntimeConfig& cfg,
    const Eigen::Ref<const Eigen::VectorXd>& bin_edges) {
  const auto xi = weights_xi.template cast<T>();
  const auto r = pairwise_R.template cast<T>();
  const auto bcfg = make_binning_cfg<T>(cfg, bin_edges);
  return compute_eec3_from_pairwise<T, StorageMode>(xi, r, bcfg);
}

}  // namespace

RuntimeVector compute_eec2_runtime(
    const Eigen::Ref<const Eigen::VectorXd>& weights_xi,
    const Eigen::Ref<const Eigen::MatrixXd>& pairwise_R,
    const RuntimeConfig& cfg,
    const Eigen::Ref<const Eigen::VectorXd>& bin_edges) {
  if (cfg.numeric_type == NumericType::Float32) {
    if (cfg.storage_mode == TensorStorageMode::SparseIndexed) {
      return dispatch_eec2<float, TensorStorageMode::SparseIndexed>(weights_xi, pairwise_R, cfg, bin_edges);
    }
    return dispatch_eec2<float, TensorStorageMode::MaterializedDense>(weights_xi, pairwise_R, cfg, bin_edges);
  }

  if (cfg.storage_mode == TensorStorageMode::SparseIndexed) {
    return dispatch_eec2<double, TensorStorageMode::SparseIndexed>(weights_xi, pairwise_R, cfg, bin_edges);
  }
  return dispatch_eec2<double, TensorStorageMode::MaterializedDense>(weights_xi, pairwise_R, cfg, bin_edges);
}

RuntimeVector compute_eec3_runtime(
    const Eigen::Ref<const Eigen::VectorXd>& weights_xi,
    const Eigen::Ref<const Eigen::MatrixXd>& pairwise_R,
    const RuntimeConfig& cfg,
    const Eigen::Ref<const Eigen::VectorXd>& bin_edges) {
  if (cfg.numeric_type == NumericType::Float32) {
    if (cfg.storage_mode == TensorStorageMode::SparseIndexed) {
      return dispatch_eec3<float, TensorStorageMode::SparseIndexed>(weights_xi, pairwise_R, cfg, bin_edges);
    }
    return dispatch_eec3<float, TensorStorageMode::MaterializedDense>(weights_xi, pairwise_R, cfg, bin_edges);
  }

  if (cfg.storage_mode == TensorStorageMode::SparseIndexed) {
    return dispatch_eec3<double, TensorStorageMode::SparseIndexed>(weights_xi, pairwise_R, cfg, bin_edges);
  }
  return dispatch_eec3<double, TensorStorageMode::MaterializedDense>(weights_xi, pairwise_R, cfg, bin_edges);
}

}  // namespace tensoreecs
