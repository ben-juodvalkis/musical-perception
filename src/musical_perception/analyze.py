"""
Main analysis pipeline.

Orchestrates perception, scaffolding, and precision layers
to produce MusicalParameters from audio input.
"""

from musical_perception.types import MusicalParameters, MarkerType


def analyze(
    audio_path: str,
    model=None,
    model_name: str = "base.en",
    extract_signature: bool = False,
    detect_exercise_type: bool = True,
) -> MusicalParameters:
    """
    Analyze an audio file and return structured musical parameters.

    This is the primary entry point for the package.

    Args:
        audio_path: Path to audio file (wav, mp3, aif, etc.)
        model: Pre-loaded Whisper model (optional, avoids reloading)
        model_name: Whisper model to load if model not provided
        extract_signature: Whether to extract counting signature (requires prosody deps)
        detect_exercise_type: Whether to run exercise detection

    Returns:
        MusicalParameters with extracted musical information
    """
    from musical_perception.perception.whisper import load_model, transcribe
    from musical_perception.scaffolding.markers import classify_marker, extract_markers
    from musical_perception.precision.tempo import calculate_tempo
    from musical_perception.precision.subdivision import analyze_subdivisions

    if model is None:
        model = load_model(model_name)

    # Transcribe audio
    words = transcribe(model, audio_path)

    # Classify words into markers
    markers = extract_markers(words)

    # Calculate tempo from beat timestamps
    beat_timestamps = [m.timestamp for m in markers if m.marker_type == MarkerType.BEAT]
    tempo = calculate_tempo(beat_timestamps)

    # Analyze subdivisions
    subdivision = analyze_subdivisions(markers)

    # Detect exercise type
    exercise = None
    if detect_exercise_type:
        from musical_perception.scaffolding.exercise import detect_exercise
        exercise = detect_exercise(words)

    # Extract counting signature (optional — requires prosody deps)
    signature = None
    if extract_signature:
        from musical_perception.perception.prosody import extract_prosody_contours, extract_word_features
        from musical_perception.precision.signature import compute_signature

        pitch_times, pitch_values, intensity_times, intensity_values = extract_prosody_contours(audio_path)

        # Classify each word for prosody extraction
        marker_types = [classify_marker(w.word.strip().lower().strip(".,!?;:")) for w in words]

        features = extract_word_features(
            words, marker_types,
            pitch_times, pitch_values,
            intensity_times, intensity_values,
        )
        signature = compute_signature(features)

    return MusicalParameters(
        tempo=tempo,
        subdivision=subdivision,
        exercise=exercise,
        counting_signature=signature,
        words=words,
        markers=markers,
    )
