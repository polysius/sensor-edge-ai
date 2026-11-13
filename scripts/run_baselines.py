"""Run the M1 baselines on C-MAPSS subsets and write a markdown table."""

import argparse
from pathlib import Path

from sensor_edge.experiment import ExperimentConfig, run_baselines

TABLE_HEADER = (
    "| Subset | Model | Val RMSE | Test RMSE | Test NASA score |\n"
    "|---|---|---|---|---|\n"
)


def format_table(rows: list[dict]) -> str:
    lines = [
        f"| {row['subset']} | {row['model']} | {row['val_rmse']:.2f} | "
        f"{row['test_rmse']:.2f} | {row['test_nasa_score']:.0f} |"
        for row in rows
    ]
    return TABLE_HEADER + "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cmapss"))
    parser.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--output", type=Path, default=Path("artifacts/baselines.md"))
    args = parser.parse_args()

    config = ExperimentConfig()
    rows = [row for subset in args.subsets for row in run_baselines(args.data_dir, subset, config)]

    # FD002 mixes six operating conditions, so it shows what per-condition
    # normalisation is worth. Re-run it with one global condition for comparison.
    ablation_rows = []
    if "FD002" in args.subsets:
        global_config = ExperimentConfig(n_conditions=1)
        ablation_rows = run_baselines(args.data_dir, "FD002", global_config)

    report = "## Baselines\n\n" + format_table(rows)
    if ablation_rows:
        report += "\n## FD002 with one global normalisation\n\n" + format_table(ablation_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
