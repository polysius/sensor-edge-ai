"""Replay the C-MAPSS test engines into Kafka as a live stream."""

import argparse
from pathlib import Path

from sensor_edge.data import read_cmapss_table
from sensor_edge.stream_producer import DEFAULT_TOPIC, SpikeFault, run_producer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cmapss"))
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--brokers", default="localhost:19092")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--rate", type=float, default=50.0, help="messages per second, 0 for no limit")
    parser.add_argument("--max-messages", type=int, default=None)
    parser.add_argument(
        "--fault",
        nargs=4,
        action="append",
        metavar=("UNIT", "FIRST_CYCLE", "N_CYCLES", "SENSOR"),
        default=[],
        help="plant a spike, for example: --fault 5 60 6 sensor_11",
    )
    args = parser.parse_args()

    frame = read_cmapss_table(args.data_dir / f"test_{args.subset}.txt")
    faults = [SpikeFault(int(u), int(c), int(n), sensor) for u, c, n, sensor in args.fault]
    sent = run_producer(frame, args.brokers, args.topic, args.rate, faults, args.max_messages)
    print(f"Sent {sent} readings to {args.topic}")


if __name__ == "__main__":
    main()
