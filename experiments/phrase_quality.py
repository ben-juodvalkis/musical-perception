"""
Experiment: per-phrase quality analysis → maestro-modern style mapping.

Asks Gemini to rate quality (articulation, weight, energy) for each phrase
of a dance combination, then derives:
  - melodic_rhythmic: mean(1 - articulation)  → staccato=rhythmic, legato=melodic
  - sparse_dense:     mean(energy)            → calm=sparse, energetic=dense
  - safe_weird:       variability across phrases → consistent=safe, varied=weird

Usage:
    python experiments/phrase_quality.py video/IMG_7843.MOV
    python experiments/phrase_quality.py --all   # run on all test files
"""

import json
import os
import sys
import tempfile
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Schema: per-phrase quality
# ---------------------------------------------------------------------------

_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "phrases": {
            "type": "ARRAY",
            "description": (
                "Quality ratings for each distinct phrase/section of the combination. "
                "A phrase is typically an 8-count or a movement section with consistent character. "
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
        "overall_quality": {
            "type": "OBJECT",
            "description": "Overall quality for the entire video (for comparison with per-phrase)",
            "properties": {
                "articulation": {"type": "NUMBER"},
                "weight": {"type": "NUMBER"},
                "energy": {"type": "NUMBER"},
            },
            "required": ["articulation", "weight", "energy"],
        },
        "exercise_type": {
            "type": "STRING",
            "description": "What exercise/combination is being demonstrated",
        },
    },
    "required": ["phrases", "overall_quality", "exercise_type"],
}

_PROMPT = """\
Analyze this dance class video/audio. A teacher is demonstrating or marking an exercise.

Break the combination into distinct phrases (typically 8-count sections or sections \
with different movement character). For EACH phrase, rate the movement quality on \
three dimensions (0.0-1.0):

- articulation: How sharp vs smooth is the movement?
  0.0 = staccato/sharp/percussive/detached
  0.1 = frappé, batterie, quick petit allegro
  0.25 = grand battement, dégagé, sharp jumps
  0.5 = tendu, waltz, moderate turns
  0.7 = rond de jambe, développé, sustained balances
  0.9 = adagio, port de bras, contemporary floorwork, lyrical phrases

- weight: How light vs heavy is the movement?
  0.0 = light/buoyant/airy/suspended
  0.1 = petit allegro, sautés, quick jumps
  0.3 = assemblé, glissade, brisé
  0.45 = tendu, dégagé, moderate barre work
  0.65 = fondu, contemporary release, lunges
  0.9 = grand plié, modern floorwork, character stamps, grounded movement

- energy: How calm vs explosive is the movement?
  0.0 = calm/controlled/gentle/still
  0.1 = port de bras, balance, stretches
  0.25 = adagio, slow développé, sustained movements
  0.4 = tendu, plié, moderate barre work
  0.65 = grand battement, turns, quick footwork
  0.9 = grand allegro, manège, big jumps, explosive combinations

Rate what you actually observe, not what the exercise should ideally look like.
Look for CHANGES in quality between phrases — a combination that goes from \
slow port de bras into sharp frappés should have very different ratings per phrase.

Mark each phrase as "primary" (true/false). Primary phrases are the core/defining \
movements of the exercise. Transitions, preparations, and port de bras breaks \
between sets are NOT primary. For example, in a grand battement combination, the \
battement phrases are primary but the port de bras recovery is not.

Also provide an overall quality rating for the entire video for comparison."""

_VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".webm", ".m4v"}
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".aif", ".aiff", ".aac", ".flac", ".ogg", ".m4a"}


# ---------------------------------------------------------------------------
# Gemini helpers (standalone, no dependency on musical_perception.perception)
# ---------------------------------------------------------------------------

def _extract_audio(video_path: str) -> str | None:
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".m4a", delete=False)
        tmp.close()
        result = subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-acodec", "aac", tmp.name, "-y"],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            os.unlink(tmp.name)
            return None
        return tmp.name
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _upload_and_wait(client, file_path: str, uploaded: list, timeout=120.0):
    f = client.files.upload(file=file_path)
    uploaded.append(f)
    start = time.time()
    while f.state.name == "PROCESSING":
        if time.time() - start > timeout:
            raise TimeoutError(f"Still processing after {timeout}s")
        time.sleep(2)
        f = client.files.get(name=f.name)
    if f.state.name == "FAILED":
        raise RuntimeError(f"Upload failed: {file_path}")
    return f


def call_gemini(media_path: str) -> dict:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    ext = Path(media_path).suffix.lower()
    is_video = ext in _VIDEO_EXTENSIONS

    audio_tmp = None
    uploaded = []

    try:
        main_file = _upload_and_wait(client, media_path, uploaded)

        if is_video:
            audio_tmp = _extract_audio(media_path)
            if audio_tmp:
                _upload_and_wait(client, audio_tmp, uploaded)

        parts = []
        for f in uploaded:
            parts.append(types.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type))
        parts.append(types.Part.from_text(text=_PROMPT))

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_RESPONSE_SCHEMA,
            ),
        )
        return json.loads(response.text)

    finally:
        if audio_tmp:
            try:
                os.unlink(audio_tmp)
            except OSError:
                pass
        for f in uploaded:
            try:
                client.files.delete(name=f.name)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Mapping: per-phrase quality → maestro-modern style axes
# ---------------------------------------------------------------------------

def compute_style_axes(raw: dict) -> dict:
    phrases = raw["phrases"]
    overall = raw["overall_quality"]

    if len(phrases) == 0:
        return {"error": "no phrases detected"}

    # Use primary phrases for style axes; fall back to all if none marked primary
    primary = [p for p in phrases if p.get("primary", True)]
    if not primary:
        primary = phrases

    arts = [p["articulation"] for p in primary]
    weights = [p["weight"] for p in primary]
    energies = [p["energy"] for p in primary]

    # --- Melodic ↔ Rhythmic (from articulation) ---
    # legato (high articulation) → melodic (low value)
    # staccato (low articulation) → rhythmic (high value)
    melodic_rhythmic = 1.0 - np.mean(arts)

    # --- Sparse ↔ Dense (from energy) ---
    # calm → sparse, energetic → dense
    sparse_dense = np.mean(energies)

    # --- Safe ↔ Weird (from variability) ---
    # Low variance across phrases → safe, high variance → weird
    stds = [np.std(arts), np.std(weights), np.std(energies)]
    variability = np.mean(stds)
    # Normalize: std of ~0.25+ is pretty high variance for 0-1 scale
    safe_weird = float(np.clip(variability / 0.25, 0.0, 1.0))

    # Convert to maestro-modern 1-10 scale
    def to_slider(v):
        return round(v * 9) + 1

    return {
        "melodic_rhythmic": {
            "raw": float(melodic_rhythmic),
            "slider": to_slider(melodic_rhythmic),
            "label": f"{'Rhythmic' if melodic_rhythmic > 0.5 else 'Melodic'} ({to_slider(melodic_rhythmic)}/10)",
            "source": f"mean articulation={np.mean(arts):.2f} → inverted",
        },
        "sparse_dense": {
            "raw": float(sparse_dense),
            "slider": to_slider(sparse_dense),
            "label": f"{'Dense' if sparse_dense > 0.5 else 'Sparse'} ({to_slider(sparse_dense)}/10)",
            "source": f"mean energy={np.mean(energies):.2f}",
        },
        "safe_weird": {
            "raw": float(safe_weird),
            "slider": to_slider(safe_weird),
            "label": f"{'Weird' if safe_weird > 0.5 else 'Safe'} ({to_slider(safe_weird)}/10)",
            "source": f"variability={variability:.3f} (stds: art={stds[0]:.3f} wt={stds[1]:.3f} en={stds[2]:.3f})",
        },
        "per_phrase": [
            {
                "phrase": p["phrase_number"],
                "desc": p["description"],
                "art": p["articulation"],
                "wt": p["weight"],
                "en": p["energy"],
                "primary": p.get("primary", True),
            }
            for p in phrases
        ],
        "overall_comparison": {
            "articulation": overall["articulation"],
            "weight": overall["weight"],
            "energy": overall["energy"],
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_TEST_FILES = [
    "/Users/Music/Desktop/Swanson Clean.mov",
    "/Users/Music/Desktop/Swanson Clean 2.mov",
    "video/IMG_7843.MOV",
    "video/youtube/Exercise 1 Demo.m4v",
    "video/youtube/Exercise 3 Demo.m4v",
    "video/youtube/ballet class.mov",
    "video/youtube/plies demo.m4v",
]


def run_one(path: str):
    name = Path(path).name
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    try:
        raw = call_gemini(path)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    result = compute_style_axes(raw)

    print(f"\n  Exercise: {raw.get('exercise_type', '?')}")
    print(f"  Phrases: {len(result['per_phrase'])}")
    for p in result["per_phrase"]:
        tag = " *" if p["primary"] else "  "
        print(f"   {tag}{p['phrase']:2d}. {p['desc']}")
        print(f"        art={p['art']:.2f}  wt={p['wt']:.2f}  en={p['en']:.2f}")

    print(f"\n  Overall (Gemini single-shot): "
          f"art={result['overall_comparison']['articulation']:.2f} "
          f"wt={result['overall_comparison']['weight']:.2f} "
          f"en={result['overall_comparison']['energy']:.2f}")

    print(f"\n  --- Maestro-Modern Style Mapping ---")
    for axis in ["melodic_rhythmic", "sparse_dense", "safe_weird"]:
        info = result[axis]
        print(f"  {axis:20s}: {info['label']:20s}  ({info['source']})")

    return result


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        files = ALL_TEST_FILES
    elif len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        print("Usage: python experiments/phrase_quality.py <video_path>")
        print("       python experiments/phrase_quality.py --all")
        sys.exit(1)

    results = {}
    for f in files:
        r = run_one(f)
        if r:
            results[Path(f).name] = r

    if len(results) > 1:
        print(f"\n\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        print(f"  {'File':<30s} {'Mel/Rhy':>8s} {'Spr/Den':>8s} {'Saf/Wrd':>8s}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
        for name, r in results.items():
            mr = r["melodic_rhythmic"]["slider"]
            sd = r["sparse_dense"]["slider"]
            sw = r["safe_weird"]["slider"]
            print(f"  {name:<30s} {mr:>8d} {sd:>8d} {sw:>8d}")


if __name__ == "__main__":
    main()
