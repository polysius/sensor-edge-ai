"""Replay C-MAPSS engines into a Kafka topic as if they were running live."""

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from sensor_edge.data import COLUMN_NAMES

DEFAULT_TOPIC = "sensor-readings"
FLUSH_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class SpikeFault:
    """A spike planted in one sensor of one engine, to show that alerts fire.

    The size is a multiple of that sensor's standard deviation across the replayed data.
    """

    unit: int
    first_cycle: int
    n_cycles: int
    sensor: str
    size_in_std: float = 8.0


def readings_from_frame(frame: pd.DataFrame, faults: list[SpikeFault]) -> Iterator[dict]:
    """Yield one reading per row, one cycle at a time across all engines.

    Going cycle by cycle interleaves the engines, so the stream looks like a fleet
    reporting together and not one engine after another.
    """
    spike_sizes = {
        fault: fault.size_in_std * float(frame[fault.sensor].std()) for fault in faults
    }
    for row in frame.sort_values(["cycle", "unit"]).to_dict("records"):
        reading = {column: row[column] for column in COLUMN_NAMES}
        reading["unit"], reading["cycle"] = int(row["unit"]), int(row["cycle"])
        for fault, size in spike_sizes.items():
            in_fault = fault.first_cycle <= reading["cycle"] < fault.first_cycle + fault.n_cycles
            if reading["unit"] == fault.unit and in_fault:
                reading[fault.sensor] += size
        yield reading


def due_time(start: float, index: int, rate: float) -> float:
    """When message number `index` should be sent for a steady `rate` per second."""
    return start + index / rate


def run_producer(
    frame: pd.DataFrame,
    brokers: str,
    topic: str,
    rate: float,
    faults: list[SpikeFault],
    max_messages: int | None = None,
) -> int:
    """Send readings to `topic` at `rate` per second (0 means as fast as possible).

    Each message carries `sent_ts`, the wall clock time just before it was sent, so
    the consumer can measure the end to end delay. Returns the number of readings the
    broker confirmed, and raises if any were not delivered.
    """
    from confluent_kafka import Producer

    producer = Producer({"bootstrap.servers": brokers, "linger.ms": 0})
    delivered, failures = 0, []

    def on_delivery(error, message):
        nonlocal delivered
        if error is None:
            delivered += 1
        else:
            failures.append(str(error))

    start = time.time()
    for queued, reading in enumerate(readings_from_frame(frame, faults)):
        if max_messages is not None and queued >= max_messages:
            break
        if rate > 0:
            time.sleep(max(0.0, due_time(start, queued, rate) - time.time()))
        reading["sent_ts"] = time.time()
        producer.produce(
            topic, key=str(reading["unit"]), value=json.dumps(reading), on_delivery=on_delivery
        )
        producer.poll(0)
    still_queued = producer.flush(FLUSH_TIMEOUT_S)
    if still_queued or failures:
        raise RuntimeError(
            f"{still_queued} readings were never delivered and {len(failures)} failed "
            f"(first error: {failures[0] if failures else 'none'})"
        )
    return delivered
