"""ONNX Runtime inference wrapper."""

from pathlib import Path

import numpy as np
import onnxruntime as ort


class ONNXClassifier:
    """Loads an ONNX model and provides batch inference."""

    def __init__(self, model_path: Path):
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Run batch inference.

        Args:
            images: (N, 3, H, W) float32 array, normalized.

        Returns:
            (N, num_classes) logits array.
        """
        outputs = self.session.run(None, {self.input_name: images})
        return outputs[0]
