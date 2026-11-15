"""INT8 quantisation of ONNX models with ONNX Runtime."""

from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

from sensor_edge.onnx_export import INPUT_NAME


class WindowCalibrationReader(CalibrationDataReader):
    """Feeds example windows to static quantisation so it can measure activation ranges."""

    def __init__(self, windows: np.ndarray, batch_size: int = 32):
        self._batches = iter(
            [
                {INPUT_NAME: windows[start : start + batch_size].astype(np.float32)}
                for start in range(0, len(windows), batch_size)
            ]
        )

    def get_next(self) -> dict | None:
        return next(self._batches, None)


def quantize_dynamic_int8(fp32_path: Path, output_path: Path) -> None:
    """Quantise weights to INT8 now and activations on the fly at run time.

    Needs no example data. Only MatMul and LSTM nodes are quantised, because the
    CPU kernel for dynamically quantised convolutions does not accept INT8 weights.
    """
    quantize_dynamic(
        str(fp32_path),
        str(output_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "LSTM"],
    )


def quantize_static_int8(fp32_path: Path, output_path: Path, calibration_windows: np.ndarray) -> None:
    """Quantise weights and activations to INT8, with activation ranges from example windows."""
    quantize_static(
        str(fp32_path),
        str(output_path),
        WindowCalibrationReader(calibration_windows),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
