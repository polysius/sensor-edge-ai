"""Consume sensor readings, score them, and expose the scores to Prometheus."""

import json
import time
from pathlib import Path

import numpy as np

from sensor_edge.stream_models import load_bundle
from sensor_edge.stream_scoring import EngineScorer

ALERTS_TOPIC = "sensor-alerts"
# Fine buckets at the low end, because scoring a reading takes a few milliseconds.
LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


def latency_summary(latencies_s: list[float], duration_s: float) -> dict:
    """Percentiles in milliseconds and throughput for a list of per-message delays."""
    milliseconds = np.array(latencies_s) * 1000
    return {
        "messages": len(latencies_s),
        "duration_s": duration_s,
        "throughput_per_s": len(latencies_s) / duration_s if duration_s > 0 else 0.0,
        "p50_ms": float(np.percentile(milliseconds, 50)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "p99_ms": float(np.percentile(milliseconds, 99)),
        "max_ms": float(milliseconds.max()),
    }


def _build_metrics():
    from prometheus_client import Counter, Gauge, Histogram

    return {
        "scored": Counter("readings_scored_total", "Readings scored"),
        "alerts": Counter("anomaly_alerts_total", "Readings whose anomaly score passed the threshold", ["unit"]),
        "end_to_end": Histogram(
            "end_to_end_latency_seconds",
            "Time from the producer sending a reading to the consumer finishing its score",
            buckets=LATENCY_BUCKETS,
        ),
        "scoring": Histogram(
            "scoring_seconds", "Time the consumer spent scoring one reading", buckets=LATENCY_BUCKETS
        ),
        "anomaly_score": Gauge("engine_anomaly_score", "Latest anomaly score per engine", ["unit"]),
        "predicted_rul": Gauge("engine_predicted_rul", "Latest predicted RUL in cycles per engine", ["unit"]),
        "threshold": Gauge("anomaly_alarm_threshold", "Anomaly score above which an alert is raised"),
    }


def run_consumer(
    bundle_dir: Path,
    brokers: str,
    topic: str,
    metrics_port: int,
    max_messages: int | None,
    idle_seconds: float,
    summary_path: Path | None,
) -> dict:
    """Score readings until `max_messages` have arrived or the stream goes quiet.

    Starts from the newest offset, so start this before the producer. Prints READY
    once Kafka has assigned the topic, which is when it is safe to start the producer.
    """
    from confluent_kafka import Consumer, Producer
    from prometheus_client import start_http_server

    bundle = load_bundle(bundle_dir)
    scorer = EngineScorer(bundle)
    metrics = _build_metrics()
    metrics["threshold"].set(bundle.alarm_threshold)
    start_http_server(metrics_port)

    consumer = Consumer(
        {
            "bootstrap.servers": brokers,
            "group.id": f"scorer-{time.time_ns()}",  # a new group each run, so no old offsets
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
        }
    )
    alert_producer = Producer({"bootstrap.servers": brokers})
    consumer.subscribe([topic])

    end_to_end, scoring = [], []
    ready_announced = False
    first_message_at = last_message_at = None
    while max_messages is None or len(end_to_end) < max_messages:
        message = consumer.poll(0.5)
        if not ready_announced and consumer.assignment():
            print("READY", flush=True)
            ready_announced = True
        if message is None or message.error():
            if last_message_at and time.time() - last_message_at > idle_seconds:
                break
            continue

        received_at = time.time()
        reading = json.loads(message.value())
        result = scorer.score(reading)
        finished_at = time.time()

        end_to_end.append(finished_at - reading["sent_ts"])
        scoring.append(finished_at - received_at)
        first_message_at = first_message_at or received_at
        last_message_at = finished_at

        unit = str(result.unit)
        metrics["scored"].inc()
        metrics["end_to_end"].observe(end_to_end[-1])
        metrics["scoring"].observe(scoring[-1])
        metrics["anomaly_score"].labels(unit).set(result.anomaly_score)
        metrics["predicted_rul"].labels(unit).set(result.predicted_rul)
        if result.is_alert:
            metrics["alerts"].labels(unit).inc()
            alert = {"unit": result.unit, "cycle": result.cycle, "score": result.anomaly_score}
            alert_producer.produce(ALERTS_TOPIC, key=unit, value=json.dumps(alert))
            alert_producer.poll(0)
    consumer.close()
    alert_producer.flush()

    if not end_to_end:
        raise RuntimeError("No messages were received. Start the producer after READY is printed.")
    duration = last_message_at - first_message_at
    summary = {
        "end_to_end": latency_summary(end_to_end, duration),
        "scoring": latency_summary(scoring, duration),
    }
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
