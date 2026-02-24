"""
Prosodic feature extraction using Praat.

KEEP (gray zone) — signal processing, but future audio models might
expose internal pitch/energy representations.
"""

import tempfile

import numpy as np

from musical_perception.types import TimestampedWord, WordFeatures, MarkerType


def extract_prosody_contours(
    audio_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract pitch and intensity contours using Praat via parselmouth.

    Args:
        audio_path: Path to audio file

    Returns:
        Tuple of (pitch_times, pitch_values, intensity_times, intensity_values)
        - pitch_values: F0 in Hz, 0 = unvoiced
        - intensity_values: dB
    """
    import librosa
    import parselmouth
    import soundfile as sf

    # Load with librosa (handles compressed AIFF and various formats)
    y, sr = librosa.load(audio_path, sr=None)

    # Save to temp WAV for Praat (parselmouth can't read compressed AIFF)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, y, sr)
        sound = parselmouth.Sound(f.name)

    # Extract pitch (F0)
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=50, pitch_ceiling=300)
    pitch_values = pitch.selected_array['frequency']  # 0 = unvoiced
    pitch_times = pitch.xs()

    # Extract intensity (dB)
    intensity = sound.to_intensity(time_step=0.01)
    intensity_values = intensity.values[0]
    intensity_times = intensity.xs()

    return pitch_times, pitch_values, intensity_times, intensity_values


def extract_word_features(
    words: list[TimestampedWord],
    marker_types: list[MarkerType | None],
    pitch_times: np.ndarray,
    pitch_values: np.ndarray,
    intensity_times: np.ndarray,
    intensity_values: np.ndarray,
) -> list[WordFeatures]:
    """
    Extract prosodic features for each recognized rhythmic marker.

    Args:
        words: Timestamped words from transcription
        marker_types: Pre-classified marker type for each word (same length as words)
        pitch_times, pitch_values: Pitch contour from Praat
        intensity_times, intensity_values: Intensity contour from Praat

    Returns:
        List of WordFeatures (only for words that are rhythmic markers)
    """
    features = []

    for word, marker_type in zip(words, marker_types):
        if marker_type is None:
            continue

        # Find pitch frames within this word's time range
        p_mask = (pitch_times >= word.start) & (pitch_times <= word.end)
        word_pitches = pitch_values[p_mask]
        word_pitches = word_pitches[word_pitches > 0]  # voiced only

        # Find intensity frames within this word's time range
        i_mask = (intensity_times >= word.start) & (intensity_times <= word.end)
        word_intensities = intensity_values[i_mask]

        mean_pitch = float(np.mean(word_pitches)) if len(word_pitches) > 0 else np.nan
        mean_intensity = float(np.mean(word_intensities)) if len(word_intensities) > 0 else np.nan
        duration = word.end - word.start

        features.append(WordFeatures(
            word=word.word.strip().lower().strip(".,!?;:"),
            start=word.start,
            end=word.end,
            marker_type=marker_type,
            pitch_hz=mean_pitch,
            intensity_db=mean_intensity,
            duration=duration,
        ))

    return features
