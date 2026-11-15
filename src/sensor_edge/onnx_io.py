"""Input and output names of the exported ONNX models.

Kept in a module with no heavy imports so the benchmark worker can run on a
device that has only ONNX Runtime installed, without PyTorch.
"""

INPUT_NAME = "window"
OUTPUT_NAME = "rul_scaled"  # RUL divided by the training cap, as the networks output it
