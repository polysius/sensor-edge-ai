"""Train the LSTM and 1D-CNN on C-MAPSS and write a table next to the M1 baselines."""

import argparse
from pathlib import Path

from sensor_edge.deep_experiment import run_deep_models
from sensor_edge.experiment import ExperimentConfig, run_baselines

TABLE_HEADER = (
    "| Subset | Model | Val RMSE | Test RMSE | Test NASA score | Train seconds |\n"
    "|---|---|---|---|---|---|\n"
)


def format_baseline_row(row: dict) -> str:
    return (
        f"| {row['subset']} | {row['model']} | {row['val_rmse']:.2f} | "
        f"{row['test_rmse']:.2f} | {row['test_nasa_score']:.0f} | |"
    )


def format_deep_row(row: dict) -> str:
    return (
        f"| {row['subset']} | {row['model']} | "
        f"{row['val_rmse']:.2f} ± {row['val_rmse_std']:.2f} | "
        f"{row['test_rmse']:.2f} ± {row['test_rmse_std']:.2f} | "
        f"{row['test_nasa_score']:.0f} ± {row['test_nasa_score_std']:.0f} | "
        f"{row['seconds']:.0f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cmapss"))
    parser.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("artifacts/deep_models.md"))
    args = parser.parse_args()

    config = ExperimentConfig()
    lines = []
    for subset in args.subsets:
        lines += [format_baseline_row(row) for row in run_baselines(args.data_dir, subset, config)]
        lines += [
            format_deep_row(row) for row in run_deep_models(args.data_dir, subset, config, args.seeds)
        ]
        print(f"finished {subset}", flush=True)

    report = (
        f"Deep model rows show mean ± standard deviation over seeds {args.seeds}.\n\n"
        + TABLE_HEADER
        + "\n".join(lines)
        + "\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
