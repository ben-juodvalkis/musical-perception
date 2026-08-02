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
    Meter,
    PhraseQuality,
    PhraseStructure,
    QualityProfile,
)

_VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm"}
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".aif", ".aiff", ".aac", ".flac", ".ogg", ".m4a"}

_MARKER_TYPE_MAP = {
    "beat": MarkerType.BEAT,
    "and": MarkerType.AND,
    "ah": MarkerType.AH,
    "none": None,
}

def _words_schema(with_index: bool) -> dict:
    """Schema for the words array; index-keyed when a transcript is provided."""
    properties = {
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
    }
    required = ["word", "marker_type", "beat_number"]
    description = "Every word heard in the audio, in order"
    if with_index:
        properties["index"] = {
            "type": "INTEGER",
            "description": "The [index] of this word in the provided transcript",
        }
        required = ["index"] + required
        description = "One entry per word of the provided indexed transcript, in order"
    return {
        "type": "ARRAY",
        "description": description,
        "items": {
            "type": "OBJECT",
            "properties": properties,
            "required": required,
        },
    }


_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "words": _words_schema(with_index=False),
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
            "type": "ARRAY",
            "description": (
                "Per-phrase quality ratings. Break the combination into distinct phrases "
                "(typically 8-count sections or sections with consistent movement character). "
                "Return at least 2 phrases if the video is long enough."
            ),
            "items": {
                "type": "OBJECT",
                "properties": {
                    "phrase_number": {
                        "type": "INTEGER",
                        "description": "Sequential phrase number starting at 1",
                    },
                    "description": {
                        "type": "STRING",
                        "description": "Brief description of what happens in this phrase (1 sentence)",
                    },
                    "articulation": {
                        "type": "NUMBER",
                        "description": (
                            "0.0 = staccato/sharp/detached, 1.0 = legato/smooth/flowing. "
                            "Calibration: frappé/batterie ~0.1, grand battement/dégagé ~0.25, "
                            "tendu/waltz ~0.5, rond de jambe/développé ~0.7, "
                            "adagio/port de bras/contemporary floorwork ~0.9"
                        ),
                    },
                    "weight": {
                        "type": "NUMBER",
                        "description": (
                            "0.0 = light/buoyant/airy, 1.0 = heavy/grounded/pressing. "
                            "Calibration: petit allegro/sautés ~0.1, "
                            "assemblé/glissade ~0.3, tendu/dégagé ~0.45, "
                            "fondu/contemporary release ~0.65, "
                            "grand plié/modern floorwork/character stamps ~0.9"
                        ),
                    },
                    "energy": {
                        "type": "NUMBER",
                        "description": (
                            "0.0 = calm/controlled/gentle, 1.0 = energetic/active/explosive. "
                            "Calibration: port de bras/balance ~0.1, "
                            "adagio/slow développé ~0.25, tendu/plié ~0.4, "
                            "grand battement/turns ~0.65, "
                            "grand allegro/manège/jumps ~0.9"
                        ),
                    },
                    "primary": {
                        "type": "BOOLEAN",
                        "description": (
                            "True if this phrase represents the core/defining movement of the exercise. "
                            "False for transitions, preparations, port de bras breaks, or connecting phrases. "
                            "Example: in a grand battement combination, the battement phrases are primary; "
                            "the port de bras between sets is not."
                        ),
                    },
                },
                "required": ["phrase_number", "description", "articulation", "weight", "energy", "primary"],
            },
        },
        "structure": {
            "type": "OBJECT",
            "description": "The phrase structure of the exercise",
            "properties": {
                "counts": {
                    "type": "INTEGER",
                    "description": (
                        "Length of ONE complete phrase in counts, as "
                        "demonstrated: the core repeating pattern only. "
                        "Derive it from the counting — the total span of "
                        "counts before the pattern repeats or the exercise "
                        "moves on (counting 1..8 four times through "
                        "different steps = 32 counts, not 8). EXCLUDE "
                        "preparation counts before one AND any closing "
                        "port de bras, balance, or finish after the "
                        "pattern completes. Do NOT multiply by sides — a "
                        "32-count phrase done both sides is counts=32, "
                        "sides=2, never 64."
                    ),
                },
                "sides": {
                    "type": "INTEGER",
                    "description": (
                        "How many times the phrase pattern repeats with a "
                        "different side, facing, or direction (2 if it "
                        "repeats left/right or front/back, 1 if performed "
                        "once). Repetitions belong here, never in counts."
                    ),
                },
            },
            "required": ["counts", "sides"],
        },
    },
    "required": ["words", "exercise", "counting_structure", "meter", "quality", "structure"],
}

_PROMPT_TEMPLATE = """\
Analyze this dance class audio/video. A teacher is marking rhythm for a \
ballet exercise — they may count with numbers OR with step names spoken \
in rhythm (e.g. "tendu front brush through" at a steady pulse).

For each spoken word, classify it:
- "beat": any word spoken ON a rhythmic beat — this includes counted numbers \
(1, 2, 3…8) AND step names / directions spoken in steady rhythm \
(e.g. "tendu", "front", "side", "close", "brush", "through", "plié"). \
Assign beat_number sequentially (1, 2, 3…).
- "and": subdivision words ("and", "&") — assign the beat_number of the preceding beat
- "ah": subdivision words ("ah", "a", "uh") — assign the beat_number of the preceding beat
- "none": non-rhythmic speech (explanations, corrections, setup talk that is \
NOT part of the rhythmic counting) — beat_number is null

IMPORTANT: Many ballet teachers never say numbers — they mark the rhythm \
entirely with step names. If words are spoken at a regular rhythmic pulse, \
they ARE beats, even if they are not numbers.
{transcript_block}{onset_context}
Identify the ballet exercise type from speech and/or movement.

For counting_structure, report what you observe about the counting pattern.

For meter, determine the time signature from the counting pattern and movement quality \
(e.g. waltz/balancé = 3/4, most barre work = 4/4). \
Listen for groups of 3 vs groups of 4 in the rhythmic feel.

For quality, break the combination into distinct phrases and rate each one on \
three numeric dimensions (0.0–1.0). Rate what you actually observe, not what \
the exercise should ideally look like. Look for CHANGES in quality between \
phrases — a combination that goes from slow port de bras into sharp frappés \
should have very different ratings per phrase.

Calibration:
- articulation: frappé/batterie ~0.1, grand battement/dégagé ~0.25, \
tendu/waltz ~0.5, rond de jambe/développé ~0.7, adagio/port de bras ~0.9
- weight: petit allegro/sautés ~0.1, assemblé/glissade ~0.3, tendu ~0.45, \
fondu/release ~0.65, grand plié/floorwork/stamps ~0.9
- energy: port de bras/balance ~0.1, adagio ~0.25, tendu/plié ~0.4, \
grand battement/turns ~0.65, grand allegro/manège/jumps ~0.9

Mark each phrase as "primary" (true/false). Primary phrases are the core/defining \
movements of the exercise. Transitions, preparations, and port de bras breaks \
between sets are NOT primary.

For structure: counts is the length of ONE complete phrase as demonstrated — \
the core repeating pattern only. Follow the counting and take the full span \
before the pattern repeats to another side/direction or the exercise ends \
(a combination counted 1..8 four times over different steps is 32 counts, \
not 8). EXCLUDE preparation counts before one, and EXCLUDE any closing port \
de bras, balance, or finishing position after the pattern completes — those \
are not part of the phrase. sides is how many times the phrase repeats to \
another side or direction (front/back or left/right). Never multiply counts \
by sides, and never report just one 8-count as the phrase if the combination \
continues."""


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
        index = w.get("index")
        words.append(GeminiWord(
            word=w["word"],
            marker_type=marker_type,
            beat_number=w.get("beat_number"),
            index=index if isinstance(index, int) else None,
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
        meter = Meter(
            beats_per_measure=meter_raw.get("beats_per_measure", 4),
            beat_unit=meter_raw.get("beat_unit", 4),
        )

    quality_raw = raw.get("quality", [])
    quality = None
    if quality_raw:
        # Parse per-phrase quality
        phrases = []
        for p in quality_raw:
            phrases.append(PhraseQuality(
                phrase_number=p.get("phrase_number", 0),
                description=p.get("description", ""),
                articulation=p.get("articulation", 0.5),
                weight=p.get("weight", 0.5),
                energy=p.get("energy", 0.5),
                primary=p.get("primary", True),
            ))

        # Compute aggregate from primary phrases only
        primary = [p for p in phrases if p.primary] or phrases
        quality = QualityProfile(
            articulation=round(sum(p.articulation for p in primary) / len(primary), 2),
            weight=round(sum(p.weight for p in primary) / len(primary), 2),
            energy=round(sum(p.energy for p in primary) / len(primary), 2),
            phrases=phrases,
        )

    structure_raw = raw.get("structure", {})
    structure = None
    if structure_raw:
        structure = PhraseStructure(
            counts=structure_raw.get("counts", 16),
            sides=structure_raw.get("sides", 1),
        )

    return GeminiAnalysisResult(
        words=words,
        exercise=exercise,
        counting_structure=counting_structure,
        meter=meter,
        quality=quality,
        structure=structure,
        model=model,
        raw_response=raw,
    )


# Public seam for trace replay (ADR-009): frozen raw JSON goes back through
# the current parser, keeping parsing under test.
parse_raw_response = _parse_response


def analyze_media(
    client: _GeminiClient,
    media_path: str,
    onset_bpm: float | None = None,
    transcript_words: list[str] | None = None,
) -> GeminiAnalysisResult:
    """
    Analyze a media file (video or audio) using Gemini.

    For video files, automatically extracts and sends audio separately
    to ensure reliable audio processing across Gemini model versions.

    Args:
        client: Initialized _GeminiClient from load_client().
        media_path: Path to video (.mov, .mp4) or audio (.wav, .mp3, .aif) file.
        onset_bpm: Optional BPM hint from onset-based tempo detection.
            Included in the prompt to help Gemini calibrate its analysis.
        transcript_words: Optional ASR transcript (word list, in order).
            When provided, Gemini classifies these exact tokens and returns
            each classification keyed by transcript index, so the merge with
            timestamps is a lookup instead of fragile text matching.

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

        # Build prompt with optional transcript and onset context
        if transcript_words is not None:
            indexed = " ".join(f"[{i}] {w}" for i, w in enumerate(transcript_words))
            transcript_block = (
                f"\nA speech recognizer produced this indexed transcript of the "
                f"same audio:\n\n{indexed}\n\n"
                f"Classify EVERY indexed word above, returning each word's index "
                f"with its classification. Use the audio to judge rhythm and "
                f"timing; use this list as the definitive tokenization — do not "
                f"add, remove, or reorder words. If the recognizer misheard a "
                f"word, classify what was actually spoken at that position.\n"
            )
        else:
            transcript_block = ""
        if onset_bpm is not None:
            onset_context = (
                f"\nContext: An independent rhythm detector estimated the speech pulse "
                f"at approximately {onset_bpm:.0f} BPM. Use this as a rough guide — "
                f"the true beat rate should be in the 70–140 BPM range typical for "
                f"ballet class. If your estimate is outside that range, consider whether "
                f"you are counting at a subdivision or measure level instead of the beat level.\n"
            )
        else:
            onset_context = ""
        prompt = _PROMPT_TEMPLATE.format(
            transcript_block=transcript_block, onset_context=onset_context,
        )

        # Index-keyed words schema when a transcript was provided
        schema = _RESPONSE_SCHEMA
        if transcript_words is not None:
            schema = {
                **_RESPONSE_SCHEMA,
                "properties": {
                    **_RESPONSE_SCHEMA["properties"],
                    "words": _words_schema(with_index=True),
                },
            }

        # Build content parts
        parts = []
        for f in uploaded_files:
            parts.append(types.Part.from_uri(
                file_uri=f.uri,
                mime_type=f.mime_type,
            ))
        parts.append(types.Part.from_text(text=prompt))

        # Call Gemini with structured output. temperature=0: classification
        # should be reproducible run-to-run, not sampled.
        response = client.client.models.generate_content(
            model=client.model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.0,
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
