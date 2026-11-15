"""Train the LSTM and 1D-CNN, export them to ONNX, quantise to INT8, and benchmark all variants."""

import argparse
import platform
from pathlib import Path

import onnxruntime

from sensor_edge.edge_experiment import EdgeConfig, run_edge_benchmark
from sensor_edge.experiment import ExperimentConfig

TABLE_HEADER = (
    "| Model | Variant | Size (KB) | p50 (ms) | p95 (ms) | Memory added (MB) | "
    "Test RMSE | RMSE change | NASA score | Max diff vs PyTorch fp32 (cycles) |\n"
    "|---|---|---|---|---|---|---|---|---|---|\n"
)


def format_row(row: dict) -> str:
    memory = "n/a" if row["rss_delta_mb"] is None else f"{row['rss_delta_mb']:.1f}"
    diff = "n/a" if row["max_diff_cycles"] is None else f"{row['max_diff_cycles']:.2f}"
    return (
        f"| {row['model']} | {row['variant']} | {row['size_kb']:.1f} | {row['p50_ms']:.3f} | "
        f"{row['p95_ms']:.3f} | {memory} | {row['test_rmse']:.2f} | {row['rmse_change']:+.2f} | "
        f"{row['test_nasa_score']:.0f} | {diff} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cmapss"))
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--workdir", type=Path, default=Path("artifacts/edge"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/edge.md"))
    parser.add_argument("--runs", type=int, default=1000)
    args = parser.parse_args()

    edge = EdgeConfig(runs=args.runs)
    rows = run_edge_benchmark(args.data_dir, args.subset, ExperimentConfig(), edge, args.workdir)

    report = (
        f"Device: {platform.machine()}, onnxruntime {onnxruntime.__version__}, "
        f"{edge.threads} thread, batch size 1, {edge.runs} timed predictions after "
        f"{edge.warmup} warm-up runs, {args.subset} test windows. Accuracy is also scored "
        f"one window per call. PyTorch latency is the median of 3 repeated timings.\n\n"
        + TABLE_HEADER
        + "\n".join(format_row(row) for row in rows)
        + "\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
