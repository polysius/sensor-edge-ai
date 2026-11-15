#!/usr/bin/env bash
# Replay the FD001 test engines through Redpanda and score them live.
#
# Needs the Docker stack running (docker compose -f docker/compose.yml up -d), the topics
# created, and a trained bundle (scripts/prepare_stream_models.py). See the README.
set -euo pipefail

PYTHON=${PYTHON:-python}
DATA_DIR=${DATA_DIR:-data/cmapss}
BUNDLE_DIR=${BUNDLE_DIR:-artifacts/stream}
RATE=${RATE:-100}                  # messages per second, 0 for no limit
SUMMARY=${SUMMARY:-artifacts/stream_summary.json}
# Optionally plant a spike, for example FAULT="7 100 6 sensor_11" (engine, first cycle, number
# of cycles, sensor). The isolation forest does not flag single-sensor spikes, so the stream
# alerts come from engines wearing out. See RESULTS.md.
FAULT=${FAULT:-}

export PYTHONPATH=src
LOG=$(mktemp)

# The consumer starts from the newest offset, so it must be listening before the producer starts.
$PYTHON scripts/run_consumer.py --bundle-dir "$BUNDLE_DIR" --summary "$SUMMARY" \
    --idle-seconds 20 >"$LOG" 2>&1 &
CONSUMER_PID=$!
# Never leave the consumer behind, because it would keep its metrics port busy.
trap 'kill "$CONSUMER_PID" 2>/dev/null || true' EXIT

until grep -q READY "$LOG"; do
    if ! kill -0 "$CONSUMER_PID" 2>/dev/null; then cat "$LOG"; exit 1; fi
    sleep 0.5
done

FAULT_ARGS=()
if [ -n "$FAULT" ]; then
    read -r -a FAULT_WORDS <<<"$FAULT"
    FAULT_ARGS=(--fault "${FAULT_WORDS[@]}")
fi
$PYTHON scripts/run_producer.py --data-dir "$DATA_DIR" --rate "$RATE" "${FAULT_ARGS[@]}"
consumer_status=0
wait "$CONSUMER_PID" || consumer_status=$?
cat "$LOG"  # shows the latency summary, or the error if the consumer failed
exit "$consumer_status"
