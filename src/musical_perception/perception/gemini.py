"""
Gemini multimodal analysis wrapper.

DISPOSABLE — thin wrapper around the Gemini API for exercise detection
and word classification. Sends video + extracted audio, receives structured
JSON. Does NOT replace the precision layer — Gemini cannot provide word
timestamps or numeric prosodic measurements.

Requires:
    pip install -e ".[gemini]"
    Set GEMINI_API_KEY environment variable (or .env file)
"""

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from musical_perception.types import (
    ExerciseDetectionResult,
    ExerciseMatch,
    GeminiAnalysisResult,
    GeminiCountingStructure,
    GeminiWord,
    MarkerType,
)

_VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".webm"}
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".aif", ".aiff", ".aac", ".flac", ".ogg", ".m4a"}

_MARKER_TYPE_MAP = {
    "beat": MarkerType.BEAT,
    "and": MarkerType.AND,
    "ah": MarkerType.AH,
    "none": None,
}

_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "words": {
            "type": "ARRAY",
            "description": "Every word heard in the audio, in order",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "word": {"type": "STRING", "description": "The word as spoken"},
                    "marker_type": {
                        "type": "STRING",
                        "description": (
                            "Rhythmic role: 'beat' for counted numbers (1,2,3...), "
                            "'and' for 'and' subdivisions, 'ah' for 'ah' subdivisions, "
                            "or 'none' for non-rhythmic speech"
                        ),
                        "enum": ["beat", "and", "ah", "none"],
                    },
                    "beat_number": {
                        "type": "INTEGER",
                        "description": (
                            "Which beat number this word belongs to (1-16+). "
                            "For 'and'/'ah', the preceding beat number. "
                            "Null for non-rhythmic words."
                        ),
                        "nullable": True,
                    },
                },
                "required": ["word", "marker_type", "beat_number"],
            },
        },
        "exercise": {
            "type": "OBJECT",
            "description": "The dance exercise being demonstrated",
            "properties": {
                "exercise_type": {
                    "type": "STRING",
                    "description": (
                        "Canonical exercise name in snake_case "
                        "(e.g. plie, tendu, chaine_turn, pirouette). "
                        "Use 'unknown' if unclear."
                    ),
                },
                "display_name": {
                    "type": "STRING",
                    "description": "Pretty display name (e.g. Plié, Chaîné Turn)",
                },
                "confidence": {
                    "type": "NUMBER",
                    "description": "Confidence 0.0-1.0",
                },
                "reasoning": {
                    "type": "STRING",
                    "description": "Brief explanation of why this exercise was identified",
                },
            },
            "required": ["exercise_type", "display_name", "confidence", "reasoning"],
        },
        "counting_structure": {
            "type": "OBJECT",
            "description": "The rhythmic structure of the counting",
            "properties": {
                "total_counts": {
                    "type": "INTEGER",
                    "description": "Total number of beats counted",
                },
                "prep_counts": {
                    "type": "STRING",
                    "description": "Any preparatory counts before the main phrase (e.g. '5, 6, 7, 8')",
                    "nullable": True,
                },
                "subdivision_type": {
                    "type": "STRING",
                    "description": (
                        "Whether the counting uses subdivisions: "
                        "'none' (just beats), 'duple' (1-and-2-and), "
                        "'triplet' (1-and-ah-2-and-ah)"
                    ),
                    "enum": ["none", "duple", "triplet"],
                },
                "estimated_bpm": {
                    "type": "NUMBER",
                    "description": "Estimated tempo in beats per minute",
                    "nullable": True,
                },
            },
            "required": ["total_counts", "prep_counts", "subdivision_type", "estimated_bpm"],
        },
        "meter": {
            "type": "OBJECT",
            "description": "The meter/time signature of the exercise",
            "properties": {
                "beats_per_measure": {
                    "type": "INTEGER",
                    "description": (
                        "Beats per measure: 2, 3, 4, or 6. "
                        "3 for waltz/balancé, 4 for most exercises, "
                        "6 for 6/8 compound time."
                    ),
                },
                "beat_unit": {
                    "type": "INTEGER",
                    "description": "Note value that gets one beat: 4 for quarter note, 8 for eighth note",
                },
            },
            "required": ["beats_per_measure", "beat_unit"],
        },
        "quality": {
            "type": "OBJECT",
            "description": "The movement and musical quality/style",
            "properties": {
                "descriptors": {
                    "type": "ARRAY",
                    "description": (
                        "2-5 musical/movement quality words that describe how the exercise "
                        "should be performed. Examples: legato, staccato, sharp, flowing, "
                        "sustained, marcato, bouncy, smooth, lyrical, crisp, grounded, airy"
                    ),
                    "items": {"type": "STRING"},
                },
            },
            "required": ["descriptors"],
        },
        "structure": {
            "type": "OBJECT",
            "description": "The phrase structure of the exercise",
            "properties": {
                "counts": {
                    "type": "INTEGER",
                    "description": "Total counts in one full phrase (e.g. 16, 32)",
                },
                "sides": {
                    "type": "INTEGER",
                    "description": (
                        "Number of sides/repetitions (1 if one-sided, "
                        "2 if the exercise repeats left and right)"
                    ),
                },
            },
            "required": ["counts", "sides"],
        },
    },
    "required": ["words", "exercise", "counting_structure", "meter", "quality", "structure"],
}

_PROMPT = """\
Analyze this dance class audio/video. A teacher is counting aloud to \
demonstrate rhythm for a ballet exercise.

For each spoken word, classify it:
- "beat": counted numbers (1, 2, 3... 8) — assign beat_number
- "and": subdivision words ("and", "&") — assign the beat_number of the preceding beat
- "ah": subdivision words ("ah", "a", "uh") — assign the beat_number of the preceding beat
- "none": non-rhythmic speech (instructions, exercise names, etc.) — beat_number is null

Identify the ballet exercise type from speech and/or movement.

For counting_structure, report what you observe about the counting pattern.

For meter, determine the time signature from the counting pattern and movement quality \
(e.g. waltz = 3/4, most barre work = 4/4).

For quality, describe the movement/musical style with 2-5 descriptors \
(e.g. legato, sharp, flowing, sustained, marcato, bouncy, lyrical, crisp).

For structure, report the phrase length in counts and whether the exercise \
is performed on one side or both sides."""


@dataclass
class _GeminiClient:
    """Wrapper around google.genai.Client with model name."""
    client: object  # google.genai.Client
    model: str


def load_client(
    model: str = "gemini-2.5-flash",
    api_key: str | None = None,
) -> _GeminiClient:
    """
    Initialize a Gemini API client.

    Args:
        model: Gemini model name. Defaults to 2.5 Flash (reliable audio from video).
        api_key: API key. If None, reads from GEMINI_API_KEY env var or .env file.

    Returns:
        _GeminiClient wrapper ready for analyze_media().
    """
    try:
        from google import genai
    except ImportError as e:
        raise ImportError(
            "google-genai is not installed. Install with:\n"
            "  pip install -e '.[gemini]'"
        ) from e

    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")

    if api_key is None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
        except ImportError:
            pass

    if not api_key:
        raise ValueError(
            "Gemini API key not found. Either:\n"
            "  - Set GEMINI_API_KEY environment variable\n"
            "  - Add GEMINI_API_KEY=... to .env file\n"
            "  - Pass api_key= to load_client()"
        )

    client = genai.Client(api_key=api_key)
    return _GeminiClient(client=client, model=model)


def _extract_audio(video_path: str) -> str | None:
    """
    Extract audio track from video file using ffmpeg.

    Returns path to temporary AAC file, or None if extraction fails.
    Caller is responsible for cleanup.
    """
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".m4a", delete=False)
        tmp.close()

        result = subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-acodec", "aac", tmp.name, "-y"],
            capture_output=True,
            timeout=30,
        )

        if result.returncode != 0:
            os.unlink(tmp.name)
            return None

        return tmp.name
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _upload_and_wait(client, file_path: str, uploaded_files: list, timeout: float = 120.0):
    """
    Upload a file to Gemini and wait for it to become ACTIVE.

    The file is appended to uploaded_files immediately after upload
    (before waiting), so the caller's finally block can clean it up
    even if waiting raises.

    Raises:
        TimeoutError: If file doesn't become ACTIVE within timeout.
        RuntimeError: If file enters FAILED state.
    """
    uploaded = client.files.upload(file=file_path)
    uploaded_files.append(uploaded)

    start = time.time()
    while uploaded.state.name == "PROCESSING":
        if time.time() - start > timeout:
            raise TimeoutError(f"File {file_path} still processing after {timeout}s")
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)

    if uploaded.state.name == "FAILED":
        raise RuntimeError(f"File upload failed: {file_path}")

    return uploaded


def _parse_response(raw: dict, model: str) -> GeminiAnalysisResult:
    """Convert raw JSON response to GeminiAnalysisResult."""
    words = []
    for w in raw.get("words", []):
        marker_type = _MARKER_TYPE_MAP.get(w.get("marker_type", "none"))
        words.append(GeminiWord(
            word=w["word"],
            marker_type=marker_type,
            beat_number=w.get("beat_number"),
        ))

    ex_raw = raw.get("exercise", {})
    exercise = ExerciseDetectionResult(
        primary_exercise=ex_raw.get("exercise_type"),
        display_name=ex_raw.get("display_name"),
        confidence=ex_raw.get("confidence", 0.0),
        all_matches=[ExerciseMatch(
            exercise_type=ex_raw.get("exercise_type", "unknown"),
            display_name=ex_raw.get("display_name", "Unknown"),
            matched_text=ex_raw.get("reasoning", ""),
            timestamp=0.0,
            confidence=ex_raw.get("confidence", 0.0),
        )],
    )

    cs_raw = raw.get("counting_structure", {})
    counting_structure = GeminiCountingStructure(
        total_counts=cs_raw.get("total_counts"),
        prep_counts=cs_raw.get("prep_counts"),
        subdivision_type=cs_raw.get("subdivision_type"),
        estimated_bpm=cs_raw.get("estimated_bpm"),
    )

    meter_raw = raw.get("meter", {})
    meter = None
    if meter_raw:
        meter = {
            "beats_per_measure": meter_raw.get("beats_per_measure"),
            "beat_unit": meter_raw.get("beat_unit"),
        }

    quality_raw = raw.get("quality", {})
    quality = None
    if quality_raw.get("descriptors"):
        quality = {"descriptors": quality_raw["descriptors"]}

    structure_raw = raw.get("structure", {})
    structure = None
    if structure_raw:
        structure = {
            "counts": structure_raw.get("counts"),
            "sides": structure_raw.get("sides"),
        }

    return GeminiAnalysisResult(
        words=words,
        exercise=exercise,
        counting_structure=counting_structure,
        meter=meter,
        quality=quality,
        structure=structure,
        model=model,
    )


def analyze_media(
    client: _GeminiClient,
    media_path: str,
) -> GeminiAnalysisResult:
    """
    Analyze a media file (video or audio) using Gemini.

    For video files, automatically extracts and sends audio separately
    to ensure reliable audio processing across Gemini model versions.

    Args:
        client: Initialized _GeminiClient from load_client().
        media_path: Path to video (.mov, .mp4) or audio (.wav, .mp3, .aif) file.

    Returns:
        GeminiAnalysisResult with word classifications, exercise detection,
        and counting structure observations.
    """
    from google.genai import types

    ext = Path(media_path).suffix.lower()
    is_video = ext in _VIDEO_EXTENSIONS

    audio_tmp_path = None
    uploaded_files = []

    try:
        # Upload main media file
        main_file = _upload_and_wait(client.client, media_path, uploaded_files)

        # For video, extract and upload audio separately
        if is_video:
            audio_tmp_path = _extract_audio(media_path)
            if audio_tmp_path:
                audio_file = _upload_and_wait(client.client, audio_tmp_path, uploaded_files)

        # Build content parts
        parts = []
        for f in uploaded_files:
            parts.append(types.Part.from_uri(
                file_uri=f.uri,
                mime_type=f.mime_type,
            ))
        parts.append(types.Part.from_text(text=_PROMPT))

        # Call Gemini with structured output
        response = client.client.models.generate_content(
            model=client.model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_RESPONSE_SCHEMA,
            ),
        )

        raw = json.loads(response.text)
        return _parse_response(raw, client.model)

    finally:
        # Clean up temp audio file
        if audio_tmp_path:
            try:
                os.unlink(audio_tmp_path)
            except OSError:
                pass

        # Best-effort cleanup of uploaded files from Gemini
        for f in uploaded_files:
            try:
                client.client.files.delete(name=f.name)
            except Exception:
                pass
