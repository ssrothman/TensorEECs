#!/usr/bin/env python3
"""Plot TensorEECs benchmark CSV output.

Usage:
  python benchmarks/plot_benchmarks.py \
      --input benchmarks/benchmark_results.csv \
      --output-dir benchmarks/plots
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            parsed = dict(row)
            parsed["particles"] = int(row["particles"])
            parsed["bins"] = int(row["bins"])
            parsed["precompute_ms"] = float(row["precompute_ms"])
            parsed["eec2_compute_ms"] = float(row["eec2_compute_ms"])
            parsed["eec3_compute_ms"] = float(row["eec3_compute_ms"])
            parsed["eec2_total_ms"] = float(row["eec2_total_ms"])
            parsed["eec3_total_ms"] = float(row["eec3_total_ms"])
            rows.append(parsed)
    return rows


def series_key(row):
    return f"{row['dtype']}|{row['storage']}"


def plot_metric(rows, benchmark_name, x_field, y_field, title, output_path):
    subset = [r for r in rows if r["benchmark"] == benchmark_name]
    grouped = defaultdict(list)
    for row in subset:
        grouped[series_key(row)].append(row)

    plt.figure(figsize=(9, 5.5))
    for key, values in sorted(grouped.items()):
        values = sorted(values, key=lambda r: r[x_field])
        x = [v[x_field] for v in values]
        y = [v[y_field] for v in values]
        plt.plot(x, y, marker="o", linewidth=2, label=key)

    plt.title(title)
    plt.xlabel(x_field)
    plt.ylabel("milliseconds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot TensorEECs benchmark results")
    parser.add_argument(
        "--input",
        default="benchmarks/benchmark_results.csv",
        help="Input benchmark CSV path",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/plots",
        help="Output directory for PNG plots",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_rows(args.input)

    if not rows:
        raise SystemExit("No successful benchmark rows found in CSV")

    # Particle scaling plots (exact mode).
    plot_metric(
        rows,
        benchmark_name="particle_scaling",
        x_field="particles",
        y_field="precompute_ms",
        title="Exact Mode: Precompute vs Number of Particles",
        output_path=os.path.join(args.output_dir, "particle_scaling_precompute.png"),
    )
    plot_metric(
        rows,
        benchmark_name="particle_scaling",
        x_field="particles",
        y_field="eec2_compute_ms",
        title="Exact Mode: EEC2 Compute-Only vs Number of Particles",
        output_path=os.path.join(args.output_dir, "particle_scaling_eec2_compute.png"),
    )
    plot_metric(
        rows,
        benchmark_name="particle_scaling",
        x_field="particles",
        y_field="eec3_compute_ms",
        title="Exact Mode: EEC3 Compute-Only vs Number of Particles",
        output_path=os.path.join(args.output_dir, "particle_scaling_eec3_compute.png"),
    )
    plot_metric(
        rows,
        benchmark_name="particle_scaling",
        x_field="particles",
        y_field="eec3_total_ms",
        title="Exact Mode: EEC3 Total (Precompute + Compute) vs Number of Particles",
        output_path=os.path.join(args.output_dir, "particle_scaling_eec3_total.png"),
    )

    # Bin scaling plots (binned mode).
    plot_metric(
        rows,
        benchmark_name="bin_scaling",
        x_field="bins",
        y_field="precompute_ms",
        title="Binned Mode: Precompute vs Number of Bins",
        output_path=os.path.join(args.output_dir, "bin_scaling_precompute.png"),
    )
    plot_metric(
        rows,
        benchmark_name="bin_scaling",
        x_field="bins",
        y_field="eec2_compute_ms",
        title="Binned Mode: EEC2 Compute-Only vs Number of Bins",
        output_path=os.path.join(args.output_dir, "bin_scaling_eec2_compute.png"),
    )
    plot_metric(
        rows,
        benchmark_name="bin_scaling",
        x_field="bins",
        y_field="eec3_compute_ms",
        title="Binned Mode: EEC3 Compute-Only vs Number of Bins",
        output_path=os.path.join(args.output_dir, "bin_scaling_eec3_compute.png"),
    )
    plot_metric(
        rows,
        benchmark_name="bin_scaling",
        x_field="bins",
        y_field="eec3_total_ms",
        title="Binned Mode: EEC3 Total (Precompute + Compute) vs Number of Bins",
        output_path=os.path.join(args.output_dir, "bin_scaling_eec3_total.png"),
    )

    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
