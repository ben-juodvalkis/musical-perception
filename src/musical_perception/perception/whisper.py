"""
Whisper/WhisperX transcription wrapper.

DISPOSABLE — this will be replaced by multimodal model API calls.
Thin wrapper: load model, transcribe, return TimestampedWord list.

Uses WhisperX when available (tighter word boundaries via forced alignment),
falls back to plain Whisper otherwise.
"""

from dataclasses import dataclass

from musical_perception.types import TimestampedWord

# Decoding bias for ballet-class speech. Whisper conditions on this the way it
# conditions on preceding audio context, so French step names survive instead
# of degrading into English-ish syllables ("grand babma", "polybra").
BALLET_VOCABULARY_PROMPT = (
    "Ballet barre exercise, teacher counting and naming steps: "
    "one and two and three and four, five six seven eight. "
    "Grand battement, battement tendu, battement fondu, battement frappé, "
    "plié, demi-plié, grand plié, relevé, port de bras, passé, retiré, "
    "développé, rond de jambe, dégagé, piqué, coupé, cou-de-pied, "
    "arabesque, attitude, sauté, jeté, glissade, chassé, balancé, "
    "pas de bourrée, pirouette, soutenu, échappé, sissonne, "
    "en croix, en dehors, en dedans, devant, derrière, à la seconde, "
    "croisé, effacé, épaulement, adagio, petit allegro, grand allegro, "
    "first position, fifth position, demi-pointe, en pointe, barre, centre."
)


@dataclass
class _LoadedModel:
    """Wrapper to track which backend loaded the model."""
    model: object
    backend: str  # "whisperx" or "whisper"
    device: str | None = None
    initial_prompt: str | None = None  # used at transcribe time by plain-whisper path


def _detect_device() -> tuple[str, str]:
    """Detect best available device and compute type."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
        if torch.backends.mps.is_available():
            return "cpu", "int8"  # WhisperX doesn't support MPS directly
    except ImportError:
        pass
    return "cpu", "int8"


def load_model(
    model_name: str = "large-v3-turbo",
    initial_prompt: str | None = BALLET_VOCABULARY_PROMPT,
) -> _LoadedModel:
    """
    Load a transcription model.

    Tries WhisperX first (better word boundaries), falls back to plain Whisper.

    Recommended models:
    - "tiny.en": Fastest, least accurate
    - "base.en": Fast, fine for quick experiments
    - "small.en": More accurate, slower
    - "large-v3-turbo": Best accuracy, still fast (default; ~1.6 GB download)

    Args:
        model_name: Whisper model to load.
        initial_prompt: Vocabulary-bias prompt (defaults to ballet terms);
            pass None to disable.
    """
    try:
        import whisperx
        device, compute_type = _detect_device()
        model = whisperx.load_model(
            model_name, device, compute_type=compute_type,
            language="en",  # skip per-chunk detection on multilingual models
            asr_options={"initial_prompt": initial_prompt},
        )
        return _LoadedModel(
            model=model, backend="whisperx", device=device,
            initial_prompt=initial_prompt,
        )
    except ImportError:
        import whisper
        model = whisper.load_model(model_name)
        return _LoadedModel(
            model=model, backend="whisper", initial_prompt=initial_prompt,
        )


def transcribe(loaded: _LoadedModel, audio_path: str) -> list[TimestampedWord]:
    """
    Transcribe audio and extract word-level timestamps.

    If the model was loaded with WhisperX, runs forced alignment for
    tighter word boundaries. Otherwise falls back to Whisper's native
    word timestamps.

    Args:
        loaded: Loaded model wrapper from load_model()
        audio_path: Path to audio file (wav, mp3, aif, etc.)

    Returns:
        List of words with timestamps
    """
    if loaded.backend == "whisperx":
        return _transcribe_whisperx(loaded, audio_path)
    return _transcribe_whisper(loaded, audio_path)


def _transcribe_whisperx(loaded: _LoadedModel, audio_path: str) -> list[TimestampedWord]:
    """Transcribe with WhisperX (forced alignment for tighter word boundaries)."""
    import whisperx

    device = loaded.device
    audio = whisperx.load_audio(audio_path)
    batch_size = 16 if device == "cuda" else 4
    result = loaded.model.transcribe(audio, batch_size=batch_size)

    # Forced alignment via wav2vec2.
    # NOTE: load_align_model reloads on each call. For batch use, consider
    # loading once and passing the align model through.
    align_model, metadata = whisperx.load_align_model(
        language_code=result.get("language", "en"),
        device=device,
    )
    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    words = []
    for word_info in aligned["word_segments"]:
        # Some words may lack timing if alignment failed
        if "start" not in word_info or "end" not in word_info:
            continue
        words.append(TimestampedWord(
            word=word_info["word"].strip().lower(),
            start=word_info["start"],
            end=word_info["end"],
        ))

    return words


def _transcribe_whisper(loaded: _LoadedModel, audio_path: str) -> list[TimestampedWord]:
    """Transcribe with plain Whisper (fallback)."""
    result = loaded.model.transcribe(
        audio_path,
        word_timestamps=True,
        language="en",
        initial_prompt=loaded.initial_prompt,
    )

    words = []
    for segment in result["segments"]:
        if "words" in segment:
            for word_info in segment["words"]:
                words.append(TimestampedWord(
                    word=word_info["word"].strip().lower(),
                    start=word_info["start"],
                    end=word_info["end"],
                ))

    return words
