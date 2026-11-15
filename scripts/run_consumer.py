"""Score the live sensor stream and expose the scores to Prometheus."""

import argparse
import json
from pathlib import Path

from sensor_edge.stream_consumer import run_consumer
from sensor_edge.stream_producer import DEFAULT_TOPIC


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=Path("artifacts/stream"))
    parser.add_argument("--brokers", default="localhost:19092")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--metrics-port", type=int, default=8000)
    parser.add_argument("--max-messages", type=int, default=None)
    parser.add_argument("--idle-seconds", type=float, default=15.0, help="stop after this much quiet")
    parser.add_argument("--summary", type=Path, default=Path("artifacts/stream_summary.json"))
    args = parser.parse_args()

    summary = run_consumer(
        args.bundle_dir, args.brokers, args.topic, args.metrics_port,
        args.max_messages, args.idle_seconds, args.summary,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
