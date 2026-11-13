PYTHON ?= python
DATA_DIR ?= data/cmapss

.PHONY: data test lint baselines deep

data:
	$(PYTHON) scripts/download_data.py --data-dir $(DATA_DIR)

test:
	$(PYTHON) -m pytest

lint:
	ruff check .

baselines:
	PYTHONPATH=src $(PYTHON) scripts/run_baselines.py --data-dir $(DATA_DIR) --output artifacts/baselines.md

deep:
	PYTHONPATH=src $(PYTHON) scripts/run_deep.py --data-dir $(DATA_DIR) --output artifacts/deep_models.md