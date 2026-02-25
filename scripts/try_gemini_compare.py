"""Compare Gemini models: 2.5 Flash vs 3.1 Pro vs 3.1 Pro with separate audio."""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("GEMINI_API_KEY not set in .env")
    sys.exit(1)

client = genai.Client(api_key=api_key)

# --- Upload files ---
video_path = Path(__file__).resolve().parents[1] / "video" / "IMG_7843.MOV"
audio_path = Path(__file__).resolve().parents[1] / "video" / "IMG_7843_audio.m4a"

print(f"Uploading video ({video_path.stat().st_size / 1024 / 1024:.1f} MB)...")
video_file = client.files.upload(file=video_path)

print(f"Uploading audio ({audio_path.stat().st_size / 1024:.0f} KB)...")
audio_file = client.files.upload(file=audio_path)

# Wait for both to process
for f, name in [(video_file, "video"), (audio_file, "audio")]:
    while f.state.name == "PROCESSING":
        print(f"  waiting for {name}...")
        time.sleep(2)
        f = client.files.get(name=f.name)
    if name == "video":
        video_file = f
    else:
        audio_file = f
print("Both files ready.\n")

# --- Schema (same for all) ---
schema = {
    "type": "OBJECT",
    "properties": {
        "words": {
            "type": "ARRAY",
            "description": "Every word heard in the audio, in order",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "word": {"type": "STRING"},
                    "marker_type": {
                        "type": "STRING",
                        "enum": ["beat", "and", "ah", "none"],
                    },
                    "beat_number": {
                        "type": "INTEGER",
                        "nullable": True,
                    },
                },
                "required": ["word", "marker_type", "beat_number"],
            },
        },
        "exercise": {
            "type": "OBJECT",
            "properties": {
                "exercise_type": {"type": "STRING"},
                "display_name": {"type": "STRING"},
                "confidence": {"type": "NUMBER"},
                "reasoning": {"type": "STRING"},
            },
            "required": ["exercise_type", "display_name", "confidence", "reasoning"],
        },
        "counting_structure": {
            "type": "OBJECT",
            "properties": {
                "total_counts": {"type": "INTEGER"},
                "prep_counts": {"type": "STRING", "nullable": True},
                "subdivision_type": {
                    "type": "STRING",
                    "enum": ["none", "duple", "triplet"],
                },
                "estimated_bpm": {"type": "NUMBER", "nullable": True},
            },
            "required": ["total_counts", "prep_counts", "subdivision_type", "estimated_bpm"],
        },
    },
    "required": ["words", "exercise", "counting_structure"],
}

prompt = """You are analyzing a video of a dance teacher counting and demonstrating an exercise.

From the audio and video, extract:
1. Every word spoken, classified by its rhythmic role (beat number, "and" subdivision, "ah" subdivision, or non-rhythmic speech)
2. The dance exercise being demonstrated (from both what you hear AND what you see)
3. The counting structure (total counts, prep counts, subdivisions, tempo)

Listen carefully to the audio for all spoken words and counting. Watch the video for the type of movement."""

config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=schema,
)


def run_test(name, model, parts):
    print(f"\n{'=' * 60}")
    print(f"  {name} ({model})")
    print(f"{'=' * 60}")

    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],
        config=config,
    )

    result = json.loads(response.text)

    # Summary
    ex = result["exercise"]
    cs = result["counting_structure"]
    beat_words = [w for w in result["words"] if w["marker_type"] == "beat"]
    and_words = [w for w in result["words"] if w["marker_type"] == "and"]
    ah_words = [w for w in result["words"] if w["marker_type"] == "ah"]
    other_words = [w for w in result["words"] if w["marker_type"] == "none"]

    print(f"Exercise: {ex['display_name']} ({ex['confidence']})")
    print(f"  Why: {ex['reasoning']}")
    print(f"Counting: {cs['total_counts']} counts, prep={cs.get('prep_counts')}, {cs['subdivision_type']}, ~{cs.get('estimated_bpm')} BPM")
    print(f"Words: {len(result['words'])} total — {len(beat_words)} beats, {len(and_words)} ands, {len(ah_words)} ahs, {len(other_words)} other")
    print(f"  All: {', '.join(w['word'] for w in result['words'])}")

    # Token breakdown
    usage = response.usage_metadata
    print(f"Tokens: {usage.prompt_token_count} in, {usage.candidates_token_count} out, {usage.thoughts_token_count} thinking")
    if usage.prompt_tokens_details:
        for detail in usage.prompt_tokens_details:
            print(f"  {detail.modality.value}: {detail.token_count}")

    return result


# --- Run all three ---
video_part = types.Part.from_uri(file_uri=video_file.uri, mime_type=video_file.mime_type)
audio_part = types.Part.from_uri(file_uri=audio_file.uri, mime_type=audio_file.mime_type)
text_part = types.Part.from_text(text=prompt)

# Test 1: 2.5 Flash — video only (has audio track embedded)
r1 = run_test("2.5 Flash — video file", "gemini-2.5-flash", [video_part, text_part])

# Test 2: 3.1 Pro — video only
r2 = run_test("3.1 Pro — video file", "gemini-3.1-pro-preview", [video_part, text_part])

# Test 3: 3.1 Pro — video + separate audio
r3 = run_test("3.1 Pro — video + separate audio", "gemini-3.1-pro-preview", [video_part, audio_part, text_part])

print(f"\n{'=' * 60}")
print("  DONE")
print(f"{'=' * 60}")
