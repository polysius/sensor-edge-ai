"""Export a trained PyTorch RUL model to ONNX."""

from pathlib import Path

import torch
from torch import nn

from sensor_edge.onnx_io import INPUT_NAME, OUTPUT_NAME

OPSET_VERSION = 17


def export_onnx(model: nn.Module, window_length: int, n_sensors: int, path: Path) -> None:
    """Write `model` to `path` with a batch dimension that can vary at run time."""
    model.eval()
    example = torch.zeros(1, window_length, n_sensors)
    torch.onnx.export(
        model,
        example,
        str(path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=OPSET_VERSION,
        dynamo=False,
    )
