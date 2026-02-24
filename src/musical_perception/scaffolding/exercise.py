"""
Exercise type detection from transcribed words.

SCAFFOLDING — A multimodal model understands "tendus to the front" natively.
This pattern matcher will be replaced by a single API call.
"""

from musical_perception.types import TimestampedWord, ExerciseMatch, ExerciseDetectionResult


# Confidence calculation parameters
BASE_CONFIDENCE = 0.5
CONFIDENCE_PER_CHAR = 1 / 20
CONFIDENCE_BOOST_PER_MENTION = 0.1
DUPLICATE_TIMESTAMP_THRESHOLD = 0.1  # seconds


# Each exercise maps to a set of words/phrases Whisper might transcribe.
EXERCISE_WORDS: dict[str, set[str]] = {
    # Barre exercises (typical order)
    "plie": {"plie", "plié", "pliés", "plies", "plea", "please", "play"},
    "tendu": {"tendu", "tendus", "tendue", "tend to", "tondo", "tan do"},
    "degage": {
        "degage", "dégagé", "degages", "dégagés",
        "day ga jay", "dig a j", "the ga j", "de gosh", "de ga",
    },
    "rond_de_jambe": {
        "rond de jambe", "rondejambe", "ron de jon", "round de jambe",
        "rond", "ronds",
    },
    "fondu": {"fondu", "fondus", "fondue", "fond do", "fun do", "fondo"},
    "frappe": {"frappe", "frappé", "frappes", "frappés", "frap", "for pay", "frapeze"},
    "adagio": {"adagio", "adage", "a dog", "a dojo"},
    "grand_battement": {
        "grand battement", "grand batma", "gran batma",
        "grand batman", "grandma", "grand bat",
        "grand bop ma", "grand bopma",
        "battement", "batma", "bop ma",
    },
    # Additional barre exercises
    "releve": {"releve", "relevé", "relevés", "relay", "relay vay"},
    "balance": {"balance", "balancé", "balances"},
    "stretch": {"stretch", "stretches", "stretching"},
    # Center exercises
    "pirouette": {"pirouette", "pirouettes", "pirou", "peer wet"},
    "waltz": {"waltz", "waltzes", "walts", "walls"},
    "allegro": {"allegro", "a leg row", "allegra"},
    "petit_allegro": {"petit allegro", "petty allegro", "small allegro"},
    "grand_allegro": {"grand allegro", "gran allegro", "big allegro"},
    "saute": {"saute", "sauté", "sautés", "so tay", "sautee"},
    "changement": {"changement", "changements", "sha ma", "change ma"},
    "jete": {"jete", "jeté", "jetés", "jet", "jets", "je tay"},
    "assemble": {"assemble", "assemblé", "assembly", "a som blay"},
    "glissade": {"glissade", "glissades", "gliss odd", "glee sod"},
    "pas_de_bourree": {
        "pas de bourree", "pas de bourrée", "pa de boo ray",
        "bourree", "bourrée", "boo ray",
    },
    "tour": {"tour", "tours", "turn", "turns"},
    "reverence": {"reverence", "révérence", "reveronce"},
}

EXERCISE_DISPLAY_NAMES: dict[str, str] = {
    "plie": "Plié",
    "tendu": "Tendu",
    "degage": "Dégagé",
    "rond_de_jambe": "Rond de Jambe",
    "fondu": "Fondu",
    "frappe": "Frappé",
    "adagio": "Adagio",
    "grand_battement": "Grand Battement",
    "releve": "Relevé",
    "balance": "Balancé",
    "stretch": "Stretch",
    "pirouette": "Pirouette",
    "waltz": "Waltz",
    "allegro": "Allegro",
    "petit_allegro": "Petit Allegro",
    "grand_allegro": "Grand Allegro",
    "saute": "Sauté",
    "changement": "Changement",
    "jete": "Jeté",
    "assemble": "Assemblé",
    "glissade": "Glissade",
    "pas_de_bourree": "Pas de Bourrée",
    "tour": "Tour",
    "reverence": "Révérence",
}


def _clean_text(text: str) -> str:
    """Normalize text for matching."""
    return text.strip().lower().strip(".,!?;:'\"")


def find_exercise_matches(
    words: list[TimestampedWord],
    search_window_seconds: float | None = None,
) -> list[ExerciseMatch]:
    """
    Find all exercise type mentions in the transcription.

    Args:
        words: List of timestamped words from transcription
        search_window_seconds: If set, only search within first N seconds.

    Returns:
        List of ExerciseMatch objects, sorted by timestamp
    """
    if not words:
        return []

    if search_window_seconds is not None:
        words = [w for w in words if w.start < search_window_seconds]

    matches: list[ExerciseMatch] = []
    full_text = " ".join(_clean_text(w.word) for w in words)

    for exercise_key, variants in EXERCISE_WORDS.items():
        for variant in variants:
            variant_clean = _clean_text(variant)

            if variant_clean in full_text:
                variant_words = variant_clean.split()
                first_variant_word = variant_words[0]

                for i, word in enumerate(words):
                    if _clean_text(word.word) == first_variant_word:
                        # For multi-word matches, verify subsequent words
                        if len(variant_words) > 1:
                            match_found = True
                            for j, vw in enumerate(variant_words[1:], 1):
                                if i + j >= len(words):
                                    match_found = False
                                    break
                                if _clean_text(words[i + j].word) != vw:
                                    match_found = False
                                    break
                            if not match_found:
                                continue

                        confidence = min(1.0, BASE_CONFIDENCE + len(variant_clean) * CONFIDENCE_PER_CHAR)

                        matches.append(ExerciseMatch(
                            exercise_type=exercise_key,
                            display_name=EXERCISE_DISPLAY_NAMES.get(exercise_key, exercise_key),
                            matched_text=variant,
                            timestamp=word.start,
                            confidence=confidence,
                        ))
                        break

    matches.sort(key=lambda m: m.timestamp)

    # Remove duplicate exercises at same timestamp
    seen: set[tuple[str, float]] = set()
    unique_matches = []
    for match in matches:
        key = (match.exercise_type, round(match.timestamp / DUPLICATE_TIMESTAMP_THRESHOLD) * DUPLICATE_TIMESTAMP_THRESHOLD)
        if key not in seen:
            seen.add(key)
            unique_matches.append(match)

    return unique_matches


def detect_exercise(
    words: list[TimestampedWord],
    search_window_seconds: float | None = 5.0,
) -> ExerciseDetectionResult:
    """
    Detect the primary exercise type from a transcription.

    Args:
        words: List of timestamped words from transcription
        search_window_seconds: Focus on first N seconds (None for full search)

    Returns:
        ExerciseDetectionResult with primary exercise and all matches
    """
    all_matches = find_exercise_matches(words, search_window_seconds)

    if not all_matches:
        # Try full transcription if windowed search failed
        if search_window_seconds is not None:
            all_matches = find_exercise_matches(words, None)

    if not all_matches:
        return ExerciseDetectionResult(
            primary_exercise=None,
            display_name=None,
            confidence=0.0,
            all_matches=[],
        )

    primary = all_matches[0]

    # Boost confidence if exercise is mentioned multiple times
    exercise_counts: dict[str, int] = {}
    for match in all_matches:
        exercise_counts[match.exercise_type] = exercise_counts.get(match.exercise_type, 0) + 1

    mention_count = exercise_counts.get(primary.exercise_type, 1)
    confidence = min(1.0, primary.confidence + (mention_count - 1) * CONFIDENCE_BOOST_PER_MENTION)

    return ExerciseDetectionResult(
        primary_exercise=primary.exercise_type,
        display_name=primary.display_name,
        confidence=round(confidence, 2),
        all_matches=all_matches,
    )
