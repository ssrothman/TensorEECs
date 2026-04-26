#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Always run in the WSL conda dev environment.
source /home/simon/miniforge3/etc/profile.d/conda.sh
# Some conda activation hooks read unset vars; relax nounset only for activation.
set +u
conda activate dev
set -u

echo "[bench] CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
echo "[bench] python=$(which python3)"
export MPLBACKEND=Agg

cmake -S . -B build -DBUILD_SHARED_LIB=ON -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON -DBUILD_PYTHON_BINDINGS=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure

BIN_DIR="${REPO_ROOT}/build/bin"

echo "[bench] Running test executables from ${BIN_DIR}"
"${BIN_DIR}/tensoreecs_test_eec2"
"${BIN_DIR}/tensoreecs_test_eec3"
"${BIN_DIR}/tensoreecs_test_invariance"
"${BIN_DIR}/tensoreecs_test_edge_cases"

echo "[bench] Running benchmark executable"
"${BIN_DIR}/tensoreecs_bench"
python3 benchmarks/plot_benchmarks.py --input benchmarks/benchmark_results.csv --output-dir benchmarks/plots

echo "[bench] CSV: ${REPO_ROOT}/benchmarks/benchmark_results.csv"
echo "[bench] Plots:"
ls -1 benchmarks/plots
