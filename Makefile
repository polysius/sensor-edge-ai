PYTHON ?= python
DATA_DIR ?= data/cmapss

.PHONY: data test lint baselines deep anomaly uncertainty edge stream-up stream-models stream-demo stream-down

data:
	$(PYTHON) scripts/download_data.py --data-dir $(DATA_DIR)

test:
	$(PYTHON) -m pytest

lint:
	ruff check .

baselines:
	PYTHONPATH=src $(PYTHON) scripts/run_baselines.py --data-dir $(DATA_DIR) --output artifacts/baselines.md

anomaly:
	PYTHONPATH=src $(PYTHON) scripts/run_anomaly.py --data-dir $(DATA_DIR) --output artifacts/anomaly.md

uncertainty:
	PYTHONPATH=src $(PYTHON) scripts/run_uncertainty.py --data-dir $(DATA_DIR) --output artifacts/uncertainty.md --plot docs/uncertainty.png

stream-up:
	docker compose -f docker/compose.yml up -d
	docker compose -f docker/compose.yml exec redpanda rpk topic create sensor-readings sensor-alerts -p 1 -r 1

stream-models:
	PYTHONPATH=src $(PYTHON) scripts/prepare_stream_models.py --data-dir $(DATA_DIR) --out-dir artifacts/stream

stream-demo:
	PYTHON=$(PYTHON) DATA_DIR=$(DATA_DIR) bash scripts/stream_demo.sh

stream-down:
	docker compose -f docker/compose.yml down

edge:
	PYTHONPATH=src $(PYTHON) scripts/run_edge.py --data-dir $(DATA_DIR) --output artifacts/edge.md

deep:
	PYTHONPATH=src $(PYTHON) scripts/run_deep.py --data-dir $(DATA_DIR) --subsets FD001 FD002 --output artifacts/deep_models.md