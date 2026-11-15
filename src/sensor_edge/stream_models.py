"""Everything the streaming consumer needs to score an engine, saved to and loaded from a folder."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib

from sensor_edge.detectors import IsolationForestDetector
from sensor_edge.preprocess import OperatingConditionNormaliser

NORMALISER_FILE = "normaliser.joblib"
FOREST_FILE = "forest.joblib"
SETTINGS_FILE = "settings.json"
ONNX_FILE = "rul_model.onnx"


@dataclass(frozen=True)
class StreamBundle:
    """Fitted preprocessing, an RUL model, and an anomaly detector for streaming scoring.

    Nothing here is fitted on test data. The alarm threshold is the 95th percentile
    of anomaly scores on healthy windows from the validation engines.
    """

    normaliser: OperatingConditionNormaliser
    forest: IsolationForestDetector
    sensors: list[str]
    window_length: int
    rul_cap: int
    alarm_threshold: float
    onnx_path: Path


def save_bundle(bundle: StreamBundle, folder: Path) -> None:
    """Write the bundle to `folder`. The ONNX file must already be at `folder / ONNX_FILE`."""
    folder.mkdir(parents=True, exist_ok=True)
    if bundle.onnx_path != folder / ONNX_FILE:
        raise ValueError(f"Put the ONNX model at {folder / ONNX_FILE} before saving the bundle")
    joblib.dump(bundle.normaliser, folder / NORMALISER_FILE)
    joblib.dump(bundle.forest, folder / FOREST_FILE)
    settings = {
        "sensors": bundle.sensors,
        "window_length": bundle.window_length,
        "rul_cap": bundle.rul_cap,
        "alarm_threshold": bundle.alarm_threshold,
    }
    (folder / SETTINGS_FILE).write_text(json.dumps(settings, indent=2), encoding="utf-8")


def load_bundle(folder: Path) -> StreamBundle:
    folder = Path(folder)
    settings = json.loads((folder / SETTINGS_FILE).read_text(encoding="utf-8"))
    onnx_path = folder / ONNX_FILE
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing {onnx_path}. Run scripts/prepare_stream_models.py first.")
    return StreamBundle(
        normaliser=joblib.load(folder / NORMALISER_FILE),
        forest=joblib.load(folder / FOREST_FILE),
        sensors=settings["sensors"],
        window_length=settings["window_length"],
        rul_cap=settings["rul_cap"],
        alarm_threshold=settings["alarm_threshold"],
        onnx_path=onnx_path,
    )
