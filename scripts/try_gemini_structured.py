"""Experiment 2: Gemini with schema-enforced JSON output.

Ask Gemini to return structured data matching what the scaffolding
layer currently produces: word classification + exercise detection.
"""

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

# --- Upload video ---
video_path = Path(__file__).resolve().parents[1] / "video" / "IMG_7843.MOV"
print(f"Uploading {video_path.name} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)...")

video_file = client.files.upload(file=video_path)
while video_file.state.name == "PROCESSING":
    print("  waiting for processing...")
    time.sleep(2)
    video_file = client.files.get(name=video_file.name)

if video_file.state.name == "FAILED":
    print(f"Upload failed: {video_file.state}")
    sys.exit(1)
print(f"Ready.\n")

# --- Define the output schema ---
# This mirrors what scaffolding/markers.py and scaffolding/exercise.py produce.

schema = {
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
                        "description": "Rhythmic role: 'beat' for counted numbers (1,2,3...), 'and' for 'and' subdivisions, 'ah' for 'ah' subdivisions, or 'none' for non-rhythmic speech",
                        "enum": ["beat", "and", "ah", "none"],
                    },
                    "beat_number": {
                        "type": "INTEGER",
                        "description": "Which beat number this word belongs to (1-16+). For 'and'/'ah', the preceding beat number. Null for non-rhythmic words.",
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
                    "description": "Canonical exercise name in snake_case (e.g. plie, tendu, chaine_turn, pirouette, balance). Use 'unknown' if unclear.",
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
                    "description": "Whether the counting uses subdivisions: 'none' (just beats), 'duple' (1-and-2-and), 'triplet' (1-and-ah-2-and-ah)",
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
    },
    "required": ["words", "exercise", "counting_structure"],
}

prompt = """You are analyzing a video of a dance teacher counting and demonstrating an exercise.

From the audio and video, extract:
1. Every word spoken, classified by its rhythmic role (beat number, "and" subdivision, "ah" subdivision, or non-rhythmic speech)
2. The dance exercise being demonstrated (from both what you hear AND what you see)
3. The counting structure (total counts, prep counts, subdivisions, tempo)

Listen carefully to the audio for all spoken words and counting. Watch the video for the type of movement."""

print("Sending to Gemini 2.5 Flash (structured output)...")
print("=" * 60)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=video_file.uri,
                    mime_type=video_file.mime_type,
                ),
                types.Part.from_text(text=prompt),
            ],
        )
    ],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
    ),
)

result = json.loads(response.text)
print(json.dumps(result, indent=2))
print("=" * 60)

# --- Summary ---
print("\n--- Summary ---")
ex = result["exercise"]
print(f"Exercise: {ex['display_name']} (confidence: {ex['confidence']})")
print(f"  Reasoning: {ex['reasoning']}")

cs = result["counting_structure"]
print(f"Counting: {cs['total_counts']} counts, {cs['subdivision_type']} subdivision")
if cs.get("prep_counts"):
    print(f"  Prep: {cs['prep_counts']}")
if cs.get("estimated_bpm"):
    print(f"  Tempo: ~{cs['estimated_bpm']} BPM")

beat_words = [w for w in result["words"] if w["marker_type"] == "beat"]
and_words = [w for w in result["words"] if w["marker_type"] == "and"]
ah_words = [w for w in result["words"] if w["marker_type"] == "ah"]
other_words = [w for w in result["words"] if w["marker_type"] == "none"]

print(f"Words: {len(result['words'])} total — {len(beat_words)} beats, {len(and_words)} ands, {len(ah_words)} ahs, {len(other_words)} other")
print(f"  Beats: {', '.join(w['word'] for w in beat_words)}")
if and_words:
    print(f"  Ands: {', '.join(w['word'] for w in and_words)}")
if other_words:
    print(f"  Other: {', '.join(w['word'] for w in other_words)}")

print(f"\nUsage: {response.usage_metadata}")
