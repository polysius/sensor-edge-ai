"""Benchmark one ONNX model in a fresh process and print the result as JSON.

Each model runs in its own process so that its memory use is not mixed up with
other models or with PyTorch. Usage:
    python -m sensor_edge.edge_worker --model m.onnx --windows windows.npy
"""

import argparse
import json
from pathlib import Path

import numpy as np
import psutil

from sensor_edge.edge_bench import create_session, latency_percentiles
from sensor_edge.onnx_export import INPUT_NAME

BYTES_PER_MB = 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True, help=".npy file of float windows")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    windows = np.load(args.windows).astype(np.float32)
    process = psutil.Process()
    rss_before = process.memory_info().rss

    session = create_session(args.model, args.threads)

    def run_once(i: int) -> None:
        start = i % len(windows)
        session.run(None, {INPUT_NAME: windows[start : start + 1]})

    latency = latency_percentiles(run_once, args.warmup, args.runs)
    rss_after = process.memory_info().rss
    print(json.dumps({**latency, "rss_delta_mb": (rss_after - rss_before) / BYTES_PER_MB}))


if __name__ == "__main__":
    main()
