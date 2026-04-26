#include "tensoreecs/eec.hpp"

#include <Eigen/Dense>

#include <chrono>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::high_resolution_clock;

struct EventData {
  Eigen::VectorXd weights;
  Eigen::MatrixXd distances;
};

struct BenchResult {
  bool ok = false;
  double precompute_ms = std::numeric_limits<double>::quiet_NaN();
  double eec2_compute_ms = std::numeric_limits<double>::quiet_NaN();
  double eec3_compute_ms = std::numeric_limits<double>::quiet_NaN();
  double eec2_total_ms = std::numeric_limits<double>::quiet_NaN();
  double eec3_total_ms = std::numeric_limits<double>::quiet_NaN();
  std::string status = "error";
};

constexpr std::size_t kMaxDenseBytes = static_cast<std::size_t>(700) * 1024 * 1024;

EventData make_event(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u01(0.0, 1.0);

  EventData out;
  out.weights = Eigen::VectorXd(static_cast<Eigen::Index>(n));
  double sum_w = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double w = u01(rng) + 1.0e-6;
    out.weights(static_cast<Eigen::Index>(i)) = w;
    sum_w += w;
  }
  out.weights /= sum_w;

  out.distances = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const double d = u01(rng);
      out.distances(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = d;
      out.distances(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) = d;
    }
  }

  return out;
}

Eigen::VectorXd make_uniform_edges(std::size_t bins) {
  Eigen::VectorXd edges(static_cast<Eigen::Index>(bins + 1));
  for (std::size_t i = 0; i <= bins; ++i) {
    edges(static_cast<Eigen::Index>(i)) = static_cast<double>(i) / static_cast<double>(bins);
  }
  return edges;
}

template <typename Func>
double time_ms(Func&& fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) {
    (void)fn();
  }

  const auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) {
    (void)fn();
  }
  const auto t1 = Clock::now();
  const std::chrono::duration<double, std::milli> dt = t1 - t0;
  return dt.count() / static_cast<double>(iters);
}

template <typename T, tensoreecs::TensorStorageMode StorageMode>
BenchResult benchmark_pair(
    const EventData& event,
    tensoreecs::PhiMode phi_mode,
    std::size_t n_r_hint,
    const Eigen::VectorXd& edges,
    int warmup,
    int iters) {
  const std::size_t n = static_cast<std::size_t>(event.weights.size());
  if constexpr (StorageMode == tensoreecs::TensorStorageMode::MaterializedDense) {
    const long double n_ld = static_cast<long double>(n);
    const long double n_r_ld = static_cast<long double>(n_r_hint);
    const long double phi_bytes = n_r_ld * n_ld * n_ld;
    const long double omega_bytes = n_r_ld * n_r_ld * n_ld * n_ld;
    const long double lookup_bytes = 4.0L * n_ld * n_ld + 4.0L * n_r_ld * n_ld * n_ld;
    const long double total_bytes = phi_bytes + omega_bytes + lookup_bytes;
    if (total_bytes > static_cast<long double>(kMaxDenseBytes)) {
      return BenchResult{
          false,
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          "skipped_dense_memory_guard"};
    }
  }

  tensoreecs::BinningConfigT<T> cfg;
  cfg.mode = phi_mode;
  if (phi_mode == tensoreecs::PhiMode::BinnedEdges) {
    cfg.bin_edges = edges.template cast<T>();
  }

  const auto xi = event.weights.template cast<T>();
  const auto r = event.distances.template cast<T>();

  try {
    const double precompute_ms = time_ms(
        [&]() {
          const auto pre = tensoreecs::build_precompute<T, StorageMode>(xi, r, cfg);
          return pre.n_r;
        },
        warmup,
        iters);

    const auto pre = tensoreecs::build_precompute<T, StorageMode>(xi, r, cfg);

    const double eec2_compute_ms = time_ms(
        [&]() {
          const auto out = tensoreecs::compute_eec2<T, StorageMode>(xi, pre);
          return out.size();
        },
        warmup,
        iters);

    const double eec3_compute_ms = time_ms(
        [&]() {
          const auto out = tensoreecs::compute_eec3<T, StorageMode>(xi, pre);
          return out.size();
        },
        warmup,
        iters);

    BenchResult result;
    result.ok = true;
    result.precompute_ms = precompute_ms;
    result.eec2_compute_ms = eec2_compute_ms;
    result.eec3_compute_ms = eec3_compute_ms;
    result.eec2_total_ms = precompute_ms + eec2_compute_ms;
    result.eec3_total_ms = precompute_ms + eec3_compute_ms;
    result.status = "ok";
    return result;
  } catch (const std::bad_alloc&) {
    return BenchResult{
        false,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        "bad_alloc"};
  } catch (...) {
    return BenchResult{
        false,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        "runtime_error"};
  }
}

void write_result_csv_row(
    std::ofstream& csv,
    const std::string& benchmark_name,
    const std::size_t particles,
    const std::size_t bins,
    const std::string& phi_mode,
    const std::string& dtype,
    const std::string& storage,
    const BenchResult& result) {
  csv << benchmark_name << ","
      << particles << ","
      << bins << ","
      << phi_mode << ","
      << dtype << ","
      << storage << ",";

  if (result.ok) {
    csv << std::fixed << std::setprecision(6)
        << result.precompute_ms << ","
        << result.eec2_compute_ms << ","
        << result.eec3_compute_ms << ","
        << result.eec2_total_ms << ","
        << result.eec3_total_ms << ",";
  } else {
    csv << "NA,NA,NA,NA,NA,";
  }
  csv << result.status << "\n";
}

void print_console_row(
    const std::size_t x_value,
    const std::string& dtype,
    const std::string& storage,
    const BenchResult& result) {
  std::cout << x_value << "," << dtype << "," << storage << ",";
  if (result.ok) {
    std::cout << std::fixed << std::setprecision(6)
              << result.precompute_ms << ","
              << result.eec2_compute_ms << ","
              << result.eec3_compute_ms << ","
              << result.eec2_total_ms << ","
              << result.eec3_total_ms << ","
              << result.status << "\n";
  } else {
    std::cout << "NA,NA,NA,NA,NA," << result.status << "\n";
  }
}

void run_particle_scaling_benchmark(std::ofstream& csv) {
  const std::vector<std::size_t> particle_counts = {16, 24, 32, 40, 48, 64};
  const int warmup = 1;
  const int iters = 3;

  std::cout << "# exact_mode_particle_scaling\n";
  std::cout << "particles,dtype,storage,precompute_ms,eec2_compute_ms,eec3_compute_ms,eec2_total_ms,eec3_total_ms,status\n";

  for (const auto n : particle_counts) {
    const auto event = make_event(n, 1337u + n);
    const Eigen::VectorXd no_edges;

    const std::size_t n_r_hint = n * (n - 1) / 2;

    const auto f32_sparse = benchmark_pair<float, tensoreecs::TensorStorageMode::SparseIndexed>(
        event,
        tensoreecs::PhiMode::ExactUniqueDistances,
        n_r_hint,
        no_edges,
        warmup,
      iters);
    print_console_row(n, "float32", "sparse", f32_sparse);
    write_result_csv_row(csv, "particle_scaling", n, 0, "exact", "float32", "sparse", f32_sparse);

    const auto f32_dense = benchmark_pair<float, tensoreecs::TensorStorageMode::MaterializedDense>(
        event,
        tensoreecs::PhiMode::ExactUniqueDistances,
        n_r_hint,
        no_edges,
        warmup,
      iters);
    print_console_row(n, "float32", "dense", f32_dense);
    write_result_csv_row(csv, "particle_scaling", n, 0, "exact", "float32", "dense", f32_dense);

    const auto f64_sparse = benchmark_pair<double, tensoreecs::TensorStorageMode::SparseIndexed>(
        event,
        tensoreecs::PhiMode::ExactUniqueDistances,
        n_r_hint,
        no_edges,
        warmup,
      iters);
    print_console_row(n, "float64", "sparse", f64_sparse);
    write_result_csv_row(csv, "particle_scaling", n, 0, "exact", "float64", "sparse", f64_sparse);

    const auto f64_dense = benchmark_pair<double, tensoreecs::TensorStorageMode::MaterializedDense>(
        event,
        tensoreecs::PhiMode::ExactUniqueDistances,
        n_r_hint,
        no_edges,
        warmup,
      iters);
    print_console_row(n, "float64", "dense", f64_dense);
    write_result_csv_row(csv, "particle_scaling", n, 0, "exact", "float64", "dense", f64_dense);
  }
}

  void run_bin_scaling_benchmark(std::ofstream& csv) {
  const std::size_t n_particles = 64;
  const std::vector<std::size_t> bin_counts = {8, 16, 32, 64, 128, 256};
  const int warmup = 1;
  const int iters = 3;

  const auto event = make_event(n_particles, 4242u);

  std::cout << "\n# binned_mode_bin_scaling\n";
  std::cout << "bins,dtype,storage,precompute_ms,eec2_compute_ms,eec3_compute_ms,eec2_total_ms,eec3_total_ms,status\n";

  for (const auto bins : bin_counts) {
    const auto edges = make_uniform_edges(bins);

    const std::size_t n_r_hint = bins;

    const auto f32_sparse = benchmark_pair<float, tensoreecs::TensorStorageMode::SparseIndexed>(
        event,
        tensoreecs::PhiMode::BinnedEdges,
        n_r_hint,
        edges,
        warmup,
      iters);
    print_console_row(bins, "float32", "sparse", f32_sparse);
    write_result_csv_row(csv, "bin_scaling", n_particles, bins, "binned", "float32", "sparse", f32_sparse);

    const auto f32_dense = benchmark_pair<float, tensoreecs::TensorStorageMode::MaterializedDense>(
        event,
        tensoreecs::PhiMode::BinnedEdges,
        n_r_hint,
        edges,
        warmup,
      iters);
    print_console_row(bins, "float32", "dense", f32_dense);
    write_result_csv_row(csv, "bin_scaling", n_particles, bins, "binned", "float32", "dense", f32_dense);

    const auto f64_sparse = benchmark_pair<double, tensoreecs::TensorStorageMode::SparseIndexed>(
        event,
        tensoreecs::PhiMode::BinnedEdges,
        n_r_hint,
        edges,
        warmup,
      iters);
    print_console_row(bins, "float64", "sparse", f64_sparse);
    write_result_csv_row(csv, "bin_scaling", n_particles, bins, "binned", "float64", "sparse", f64_sparse);

    const auto f64_dense = benchmark_pair<double, tensoreecs::TensorStorageMode::MaterializedDense>(
        event,
        tensoreecs::PhiMode::BinnedEdges,
        n_r_hint,
        edges,
        warmup,
      iters);
    print_console_row(bins, "float64", "dense", f64_dense);
    write_result_csv_row(csv, "bin_scaling", n_particles, bins, "binned", "float64", "dense", f64_dense);
  }
}

}  // namespace

int main() {
  const std::string output_csv_path = "benchmarks/benchmark_results.csv";
  std::ofstream csv(output_csv_path, std::ios::out | std::ios::trunc);
  if (!csv.is_open()) {
    std::cerr << "Failed to open benchmark CSV output at: " << output_csv_path << "\n";
    return 1;
  }

  csv << "benchmark,particles,bins,phi_mode,dtype,storage,precompute_ms,eec2_compute_ms,eec3_compute_ms,eec2_total_ms,eec3_total_ms,status\n";

  std::cout << "TensorEECs benchmark suite\n";
  std::cout << "Writing CSV to: " << output_csv_path << "\n";
  run_particle_scaling_benchmark(csv);
  run_bin_scaling_benchmark(csv);
  csv.flush();
  return 0;
}
