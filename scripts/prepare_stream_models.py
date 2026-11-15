"""Train the streaming models and save them to a folder for the consumer."""

import argparse
from pathlib import Path

from sensor_edge.experiment import ExperimentConfig
from sensor_edge.stream_prepare import prepare_stream_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cmapss"))
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/stream"))
    args = parser.parse_args()

    bundle = prepare_stream_bundle(args.data_dir, args.subset, ExperimentConfig(), args.out_dir)
    print(f"Saved the streaming bundle to {args.out_dir}")
    print(f"Sensors used: {len(bundle.sensors)}, alarm threshold: {bundle.alarm_threshold:.4f}")


if __name__ == "__main__":
    main()
