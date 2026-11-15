"""Evaluate the isolation forest and the autoencoder on injected faults and end-of-life windows."""

import argparse
from pathlib import Path

from sensor_edge.anomaly_experiment import THRESHOLD_QUANTILE, run_anomaly_detection
from sensor_edge.autoencoder import AutoencoderDetector
from sensor_edge.data import load_subset
from sensor_edge.detectors import IsolationForestDetector
from sensor_edge.experiment import ExperimentConfig

TABLE_HEADER = (
    "| Subset | Detector | Check | Precision | Recall | F1 | AUC | False alarm rate | Healthy | Anomalous |\n"
    "|---|---|---|---|---|---|---|---|---|---|\n"
)


def format_row(row: dict) -> str:
    return (
        f"| {row['subset']} | {row['detector']} | {row['check']} | {row['precision']:.2f} | "
        f"{row['recall']:.2f} | {row['f1']:.2f} | {row['auc']:.3f} | "
        f"{row['false_alarm_rate']:.3f} | {row['n_healthy']} | {row['n_anomalous']} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cmapss"))
    parser.add_argument("--subsets", nargs="+", default=["FD001", "FD002"])
    parser.add_argument("--output", type=Path, default=Path("artifacts/anomaly.md"))
    args = parser.parse_args()

    config = ExperimentConfig()
    detectors = {
        "Isolation forest": lambda: IsolationForestDetector(seed=config.seed),
        "Autoencoder": lambda: AutoencoderDetector(seed=config.seed),
    }
    rows = []
    for name in args.subsets:
        rows += run_anomaly_detection(load_subset(args.data_dir, name), config, detectors)
        print(f"finished {name}", flush=True)

    report = (
        f"Alarm threshold: the {THRESHOLD_QUANTILE:.0%} quantile of scores on healthy validation windows.\n\n"
        + TABLE_HEADER
        + "\n".join(format_row(row) for row in rows)
        + "\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
