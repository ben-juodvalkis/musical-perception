"""
WhiStress stress detection wrapper.

DISPOSABLE — thin wrapper around the WhiStress model.

WhiStress extends Whisper with a stress detection head, outputting
per-token binary stress labels (1 = stressed, 0 = unstressed).
This complements Praat's numeric measurements with learned emphasis labels.

Requires manual installation:
    git clone https://github.com/slp-rl/WhiStress.git
    cd WhiStress
    pip install -r requirements.txt
    python download_weights.py
"""

from musical_perception.types import TimestampedWord


def load_model(device: str | None = None):
    """
    Load the WhiStress inference client.

    Args:
        device: "cuda" or "cpu". Auto-detected if None.
    """
    from whistress import WhiStressInferenceClient

    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    return WhiStressInferenceClient(device=device)


def predict_stress(
    client,
    audio_path: str,
    words: list[TimestampedWord] | None = None,
) -> list[tuple[str, int]]:
    """
    Predict per-word stress labels from audio.

    Args:
        client: Loaded WhiStressInferenceClient
        audio_path: Path to audio file
        words: Optional pre-existing transcription (used as hint text).
            If provided, passes the joined text as the transcription argument
            so WhiStress aligns stress labels to known words.

    Returns:
        List of (word, stress_label) tuples where stress_label is
        1 (stressed) or 0 (unstressed).
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(audio_path, sr=16000)
    audio = {"array": np.array(y, dtype=np.float32), "sampling_rate": 16000}

    transcription = None
    if words:
        transcription = " ".join(w.word for w in words)

    return client.predict(audio=audio, transcription=transcription, return_pairs=True)
