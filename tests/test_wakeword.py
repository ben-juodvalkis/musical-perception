"""Tests for the wakeword perception wrapper.

These tests require openwakeword to be installed.
Skipped automatically in CI if the dependency is missing.
"""

import pytest
import numpy as np

try:
    import openwakeword
    HAS_OPENWAKEWORD = True
except ImportError:
    HAS_OPENWAKEWORD = False

# openwakeword's default models are .tflite: without a tflite interpreter
# (tflite-runtime has no macOS arm64 wheel) load_model() cannot run at all.
try:
    from tflite_runtime import interpreter as _tflite  # noqa: F401
    HAS_TFLITE = True
except ImportError:
    try:
        from tensorflow import lite as _tflite  # noqa: F401
        HAS_TFLITE = True
    except ImportError:
        HAS_TFLITE = False

pytestmark = pytest.mark.skipif(
    not (HAS_OPENWAKEWORD and HAS_TFLITE),
    reason="openwakeword or its tflite runtime not installed",
)


def test_load_default_model():
    """Default model loads without error."""
    from musical_perception.perception.wakeword import load_model
    model = load_model()
    assert model is not None


def test_detect_returns_dict():
    """Detection returns a dict of model_name → confidence."""
    from musical_perception.perception.wakeword import load_model, detect
    model = load_model()
    # 80ms of silence at 16kHz
    chunk = np.zeros(1280, dtype=np.int16)
    scores = detect(model, chunk)
    assert isinstance(scores, dict)
    assert len(scores) > 0
    for name, score in scores.items():
        assert isinstance(name, str)
        assert isinstance(score, float)


def test_silence_low_confidence():
    """Silence should produce low wake word confidence."""
    from musical_perception.perception.wakeword import load_model, detect
    model = load_model()
    chunk = np.zeros(1280, dtype=np.int16)
    scores = detect(model, chunk)
    for score in scores.values():
        assert score < 0.5, f"Silence triggered wake word with score {score}"
