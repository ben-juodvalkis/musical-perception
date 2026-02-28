"""
OpenWakeWord wake word detection wrapper.

DISPOSABLE — thin wrapper around OpenWakeWord for keyword spotting.
Processes 80ms audio frames (16-bit 16kHz PCM) and returns confidence scores.
"""

from __future__ import annotations

import numpy as np


def load_model(model_path: str | None = None) -> object:
    """
    Load an OpenWakeWord model.

    Args:
        model_path: Path to a custom .tflite or .onnx model file.
            If None, uses the bundled 'hey_jarvis' model as a placeholder.

    Returns:
        An OpenWakeWord Model instance.
    """
    try:
        from openwakeword.model import Model
    except ImportError:
        raise ImportError(
            "openwakeword is required for wake word detection. "
            "Install with: pip install -e '.[trigger]'"
        )

    if model_path is not None:
        return Model(wakeword_models=[model_path])
    # Default: use bundled pre-trained model as placeholder
    return Model()


def detect(model: object, audio_chunk: np.ndarray) -> dict[str, float]:
    """
    Process an audio chunk and return wake word confidence scores.

    Args:
        model: An OpenWakeWord Model instance from load_model().
        audio_chunk: Audio data as int16 numpy array, 1280 samples
            (80ms at 16kHz). OpenWakeWord expects 16-bit 16kHz mono PCM.

    Returns:
        Dict mapping model name to confidence (0.0-1.0).
    """
    prediction = model.predict(audio_chunk)
    return prediction
