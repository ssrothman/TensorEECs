# TensorEECs

TensorEECs provides an Eigen-based implementation of projected 2-point and 3-point Energy-Energy Correlators (EECs) as described in [math.tex](math.tex).

## Features

- C++20 library target (`tensoreecs`)
- Optional pybind11 Python module (`_tensoreecs`)
- Optional C++ tests linked against the library
- Runtime-selectable storage mode:
	- sparse indexed representation
	- explicitly materialized dense tensors
- Runtime-selectable numeric type:
	- float32
	- float64

## Project Layout

- `include/tensoreecs`: public C++ API
- `src`: runtime dispatch implementation
- `python`: pybind11 module
- `tests`: C++ tests and Python smoke test
- `docs/API_SPEC.md`: implementation contract

## Dependencies

Linux packages expected in WSL:

- CMake (>= 3.20)
- C++ compiler with C++20 support
- Eigen3
- pybind11 (only when building Python bindings)
- Python 3 + NumPy (only when using Python bindings)

## Build in WSL

From WSL shell:

```bash
cd /mnt/c/Users/simon/Documents/TensorEECs/TensorEECs
cmake -S . -B build -DBUILD_SHARED_LIB=ON -DBUILD_TESTS=ON -DBUILD_PYTHON_BINDINGS=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Enable Python bindings:

```bash
cmake -S . -B build -DBUILD_SHARED_LIB=ON -DBUILD_TESTS=ON -DBUILD_PYTHON_BINDINGS=ON
cmake --build build -j
```

Run benchmarks (includes float32/float64 and sparse/dense matrix):

```bash
cmake -S . -B build -DBUILD_SHARED_LIB=ON -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON
cmake --build build -j
./build/benchmarks/tensoreecs_bench
python benchmarks/plot_benchmarks.py --input benchmarks/benchmark_results.csv --output-dir benchmarks/plots
```

Single-command benchmark runner in WSL (forces conda env `dev`):

```bash
bash benchmarks/run_benchmarks.sh
```

Benchmark output is written to a CSV file at `benchmarks/benchmark_results.csv` and includes:

- exact mode scaling versus number of particles
- binned mode scaling versus number of bins
- all combinations of dtype (float32, float64) and storage mode (sparse, dense)
- separate timing columns for precompute, compute-only, and total times
- a status column; dense cases that would exceed a memory threshold are reported as skipped_dense_memory_guard

The plotting script writes PNG files into `benchmarks/plots` for:

- exact mode scaling (particle count) for precompute and compute-only timings
- binned mode scaling (bin count) for precompute and compute-only timings

## Avoid In-Source Builds

This project enforces out-of-source CMake builds. Running CMake with the source and build directories set to the same path is intentionally blocked.

Correct usage:

```bash
cmake -S . -B build
cmake --build build -j
```

If an in-source configure/build happened previously, clean generated artifacts from the repository root:

```bash
git clean -ndX
git clean -fdX
```

Then reconfigure using the out-of-source commands above.

## Configuration Defaults

- `TensorStorageMode`: sparse
- `NumericType`: float64
- `PhiMode`: exact unique distances

See [docs/API_SPEC.md](docs/API_SPEC.md) for full function contracts and tolerance constants.