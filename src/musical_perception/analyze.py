"""
Main analysis pipeline.

Orchestrates perception, scaffolding, and precision layers
to produce MusicalParameters from audio input.
"""

from pathlib import Path

from musical_perception.types import (
    GeminiAnalysisResult,
    MarkerType,
    MusicalParameters,
    TimedMarker,
    TimestampedWord,
)

_VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".webm"}


def _normalize_word(word: str) -> str:
    """Normalize a word for comparison."""
    return word.strip().lower().strip(".,!?;:")


def _merge_gemini_with_timestamps(
    gemini_result: GeminiAnalysisResult,
    whisper_words: list[TimestampedWord],
) -> list[TimedMarker]:
    """
    Merge Gemini's word classifications with Whisper's timestamps.

    Uses sequential word alignment: walk both lists in order, matching
    by normalized word text. When Gemini classifies a word as a marker
    and Whisper has a timestamp for that word, produce a TimedMarker.

    Args:
        gemini_result: Classifications from Gemini (no timestamps).
        whisper_words: Timestamped words from Whisper.

    Returns:
        List of TimedMarker with both classification and timestamp.
    """
    markers = []
    whisper_idx = 0

    for gw in gemini_result.words:
        if gw.marker_type is None:
            continue

        gw_norm = _normalize_word(gw.word)

        # Look ahead for the next matching word in whisper output
        # without consuming non-matching words (they may match later Gemini words)
        for i in range(whisper_idx, len(whisper_words)):
            ww_norm = _normalize_word(whisper_words[i].word)
            if ww_norm == gw_norm:
                markers.append(TimedMarker(
                    marker_type=gw.marker_type,
                    beat_number=gw.beat_number,
                    timestamp=whisper_words[i].start,
                    raw_word=whisper_words[i].word,
                ))
                whisper_idx = i + 1
                break
        # If no match found, skip this Gemini word (no timestamp available)

    return markers


def analyze(
    audio_path: str,
    model=None,
    model_name: str = "base.en",
    extract_signature: bool = False,
    detect_exercise_type: bool = True,
    detect_stress: bool = False,
    use_gemini: bool = False,
    gemini_client=None,
    gemini_model: str = "gemini-2.5-flash",
    use_pose: bool = False,
) -> MusicalParameters:
    """
    Analyze an audio or video file and return structured musical parameters.

    This is the primary entry point for the package.

    Args:
        audio_path: Path to audio file (wav, mp3, aif, etc.) or video file (mov, mp4)
        model: Pre-loaded Whisper/WhisperX model (optional, avoids reloading)
        model_name: Model to load if model not provided
        extract_signature: Whether to extract counting signature (requires prosody deps)
        detect_exercise_type: Whether to run exercise detection
        detect_stress: Whether to run WhiStress stress detection (requires WhiStress)
        use_gemini: Whether to use Gemini for word classification and exercise detection
        gemini_client: Pre-loaded Gemini client (optional, avoids re-init)
        gemini_model: Gemini model to use if gemini_client not provided
        use_pose: Whether to run pose estimation for movement quality (requires pose deps, video only)

    Returns:
        MusicalParameters with extracted musical information
    """
    from musical_perception.precision.tempo import calculate_tempo
    from musical_perception.precision.subdivision import analyze_subdivisions

    from musical_perception.perception.whisper import load_model, transcribe
    from musical_perception.scaffolding.markers import extract_markers

    if model is None:
        model = load_model(model_name)

    # Whisper + scaffolding markers always own timestamps and tempo
    words = transcribe(model, audio_path)
    markers = extract_markers(words)

    # Exercise detection + qualitative analysis:
    # Gemini (multimodal) or scaffolding (pattern matching)
    exercise = None
    meter = None
    quality = None
    structure = None
    if use_gemini:
        from musical_perception.perception.gemini import load_client, analyze_media

        if gemini_client is None:
            gemini_client = load_client(model=gemini_model)
        gemini_result = analyze_media(gemini_client, audio_path)
        exercise = gemini_result.exercise
        meter = gemini_result.meter
        quality = gemini_result.quality
        structure = gemini_result.structure
    elif detect_exercise_type:
        from musical_perception.scaffolding.exercise import detect_exercise
        exercise = detect_exercise(words)

    # Pose estimation + dynamics (optional — requires pose deps, video only)
    is_video = Path(audio_path).suffix.lower() in _VIDEO_EXTENSIONS
    if use_pose and not is_video:
        import warnings
        warnings.warn("--pose requires video input; skipping pose estimation for audio file")
    if use_pose and is_video:
        from musical_perception.perception.pose import load_model as load_pose, extract_landmarks
        from musical_perception.precision.dynamics import compute_quality, synthesize

        landmarker = load_pose()
        landmark_series = extract_landmarks(landmarker, audio_path)
        pose_quality = compute_quality(landmark_series)
        quality = synthesize(gemini=quality, pose=pose_quality)

    # Calculate tempo from beat timestamps
    beat_timestamps = [m.timestamp for m in markers if m.marker_type == MarkerType.BEAT]
    tempo = calculate_tempo(beat_timestamps)

    # Analyze subdivisions
    subdivision = analyze_subdivisions(markers)

    # Extract counting signature (optional — requires prosody deps)
    signature = None
    if extract_signature:
        from musical_perception.scaffolding.markers import classify_marker
        from musical_perception.perception.prosody import extract_prosody_contours, extract_word_features
        from musical_perception.precision.signature import compute_signature

        pitch_times, pitch_values, intensity_times, intensity_values = extract_prosody_contours(audio_path)

        marker_types = [classify_marker(w.word.strip().lower().strip(".,!?;:")) for w in words]

        features = extract_word_features(
            words, marker_types,
            pitch_times, pitch_values,
            intensity_times, intensity_values,
        )
        signature = compute_signature(features)

    # Detect stress labels (optional — requires WhiStress)
    stress_labels = None
    if detect_stress:
        from musical_perception.perception.whistress import load_model as load_whistress, predict_stress
        whistress_client = load_whistress()
        stress_labels = predict_stress(whistress_client, audio_path, words)

    return MusicalParameters(
        tempo=tempo,
        subdivision=subdivision,
        meter=meter,
        exercise=exercise,
        quality=quality,
        counting_signature=signature,
        structure=structure,
        words=words,
        markers=markers,
        stress_labels=stress_labels,
    )
