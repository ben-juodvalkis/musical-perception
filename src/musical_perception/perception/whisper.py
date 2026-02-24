"""
Whisper transcription wrapper.

DISPOSABLE — this will be replaced by multimodal model API calls.
Thin wrapper: load model, transcribe, return TimestampedWord list.
"""

from musical_perception.types import TimestampedWord


def load_model(model_name: str = "base.en"):
    """
    Load a Whisper model.

    Recommended models:
    - "tiny.en": Fastest, least accurate
    - "base.en": Good balance (recommended for experiments)
    - "small.en": More accurate, slower
    """
    import whisper
    return whisper.load_model(model_name)


def transcribe(model, audio_path: str) -> list[TimestampedWord]:
    """
    Transcribe audio and extract word-level timestamps.

    Args:
        model: Loaded Whisper model
        audio_path: Path to audio file (wav, mp3, aif, etc.)

    Returns:
        List of words with timestamps
    """
    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        language="en",
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
