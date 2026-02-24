"""
Word classification for rhythmic markers.

SCAFFOLDING — AI models will replace this with direct marker classification.
These word lists need constant tuning per accent/voice. A multimodal model
just understands language.
"""

from musical_perception.types import MarkerType, TimestampedWord, TimedMarker


# Words that indicate counted beats (downbeats)
BEAT_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "1", "2", "3", "4", "5", "6", "7", "8",
}

# Map beat words to their numeric value
BEAT_WORD_TO_NUMBER = {
    "one": 1, "1": 1,
    "two": 2, "2": 2,
    "three": 3, "3": 3,
    "four": 4, "4": 4,
    "five": 5, "5": 5,
    "six": 6, "6": 6,
    "seven": 7, "7": 7,
    "eight": 8, "8": 8,
}

# Words that indicate "and" subdivisions (half-beat in duple, 2nd of 3 in triplet)
AND_WORDS = {"and", "&", "an", "n", "in"}

# Words that indicate "ah" subdivisions (3rd subdivision in triplet)
AH_WORDS = {"ah", "a", "uh", "la", "the", "da", "ta"}

# Words that indicate "e" subdivisions (for future 16th note support: 1-e-and-a)
E_WORDS = {"e", "ee"}


def classify_marker(word: str) -> MarkerType | None:
    """
    Classify a word as a rhythmic marker type.

    Args:
        word: A transcribed word (will be cleaned/lowercased)

    Returns:
        MarkerType if the word is a recognized marker, None otherwise
    """
    clean = word.strip().lower().strip(".,!?;:")

    if clean in BEAT_WORDS:
        return MarkerType.BEAT
    if clean in AND_WORDS:
        return MarkerType.AND
    if clean in AH_WORDS:
        return MarkerType.AH
    if clean in E_WORDS:
        return MarkerType.E

    return None


def extract_markers(words: list[TimestampedWord]) -> list[TimedMarker]:
    """
    Convert transcribed words to rhythmic markers with beat associations.

    Each subdivision marker is associated with the most recent beat marker.

    Args:
        words: List of timestamped words from transcription

    Returns:
        List of TimedMarker objects with beat associations
    """
    markers = []
    current_beat_number: int | None = None

    for word in words:
        clean_word = word.word.strip().lower().strip(".,!?;:")
        marker_type = classify_marker(clean_word)

        if marker_type is None:
            continue

        if marker_type == MarkerType.BEAT:
            current_beat_number = BEAT_WORD_TO_NUMBER.get(clean_word)

        markers.append(TimedMarker(
            marker_type=marker_type,
            beat_number=current_beat_number,
            timestamp=word.start,
            raw_word=word.word,
        ))

    return markers
