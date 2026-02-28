"""
Main analysis pipeline.

Orchestrates perception and precision layers
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
    detect_stress: bool = False,
    gemini_client=None,
    gemini_model: str = "gemini-2.5-flash",
    use_pose: bool = False,
) -> MusicalParameters:
    """
    Analyze an audio or video file and return structured musical parameters.

    This is the primary entry point for the package. Requires a Gemini API key
    (set GEMINI_API_KEY env var or pass a pre-loaded gemini_client).

    Args:
        audio_path: Path to audio file (wav, mp3, aif, etc.) or video file (mov, mp4)
        model: Pre-loaded Whisper/WhisperX model (optional, avoids reloading)
        model_name: Model to load if model not provided
        extract_signature: Whether to extract counting signature (requires prosody deps)
        detect_stress: Whether to run WhiStress stress detection (requires WhiStress)
        gemini_client: Pre-loaded Gemini client (optional, avoids re-init)
        gemini_model: Gemini model to use if gemini_client not provided
        use_pose: Whether to run pose estimation for movement quality (requires pose deps, video only)

    Returns:
        MusicalParameters with extracted musical information
    """
    from musical_perception.precision.tempo import calculate_tempo, interpret_meter
    from musical_perception.precision.subdivision import analyze_subdivisions
    from musical_perception.precision.rhythm import detect_onset_tempo
    from musical_perception.perception.whisper import load_model, transcribe
    from musical_perception.perception.gemini import load_client, analyze_media

    if model is None:
        model = load_model(model_name)

    # Whisper owns timestamps, Gemini owns word classification
    words = transcribe(model, audio_path)

    # Onset-based tempo (Gemini-independent, runs on Whisper timestamps alone)
    onset_tempo = detect_onset_tempo(words)

    # Pass onset BPM to Gemini as a calibration hint
    onset_bpm = onset_tempo.bpm if onset_tempo is not None else None

    if gemini_client is None:
        gemini_client = load_client(model=gemini_model)
    gemini_result = analyze_media(gemini_client, audio_path, onset_bpm=onset_bpm)

    # Merge Gemini classifications with Whisper timestamps
    markers = _merge_gemini_with_timestamps(gemini_result, words)

    exercise = gemini_result.exercise
    meter = gemini_result.meter
    quality = gemini_result.quality
    structure = gemini_result.structure

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
        from musical_perception.perception.prosody import extract_prosody_contours, extract_word_features
        from musical_perception.precision.signature import compute_signature

        pitch_times, pitch_values, intensity_times, intensity_values = extract_prosody_contours(audio_path)

        # Derive marker types from Gemini merge (keyed by word start time)
        marker_lookup = {m.timestamp: m.marker_type for m in markers}
        marker_types = [marker_lookup.get(w.start) for w in words]

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

    # Coherent metric interpretation: BPM + meter + subdivision as one answer
    gemini_subdivision = (
        gemini_result.counting_structure.subdivision_type
        if gemini_result.counting_structure else None
    )
    normalized_tempo = interpret_meter(
        onset_tempo=onset_tempo,
        gemini_tempo=tempo,
        gemini_meter=gemini_result.meter,
        gemini_subdivision=gemini_subdivision,
    )

    # Backward compat: populate old fields from normalized_tempo
    normalized_bpm = normalized_tempo.bpm if normalized_tempo else None
    tempo_multiplier = normalized_tempo.tempo_multiplier if normalized_tempo else None
    if normalized_tempo:
        meter = normalized_tempo.meter

    return MusicalParameters(
        tempo=tempo,
        onset_tempo=onset_tempo,
        normalized_tempo=normalized_tempo,
        normalized_bpm=normalized_bpm,
        tempo_multiplier=tempo_multiplier,
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
