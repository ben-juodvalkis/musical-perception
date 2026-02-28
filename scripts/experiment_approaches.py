"""
Experiment: Two approaches to tempo detection from step-name counting.

Hypothesis: When a ballet teacher says step names in rhythm (instead of numbers),
the tempo is still recoverable. We test two approaches:

A) Updated Gemini prompt that recognizes step names as beat markers
B) Onset-based tempo detection directly from Whisper word timestamps

Both are tested independently against the same audio to compare.
"""

import json
import os
import time
import subprocess
import tempfile
from pathlib import Path

import numpy as np

VIDEO_PATH = "video/youtube/Exercise 1 Demo.m4v"


# ─────────────────────────────────────────────────────────────
# Step 0: Get Whisper timestamps (shared by both experiments)
# ─────────────────────────────────────────────────────────────

def get_whisper_words(model_name="base.en"):
    """Transcribe and return timestamped words."""
    from musical_perception.perception.whisper import load_model, transcribe
    model = load_model(model_name)
    words = transcribe(model, VIDEO_PATH)
    return words


def print_words_with_timing(words):
    """Print words with their onset times for visual inspection."""
    print(f"\n  {'Time':>6}  {'Dur':>5}  Word")
    print(f"  {'─'*6}  {'─'*5}  {'─'*20}")
    for w in words:
        dur = w.end - w.start
        print(f"  {w.start:6.2f}  {dur:5.3f}  {w.word}")


# ─────────────────────────────────────────────────────────────
# Experiment A: Gemini with updated prompt
# ─────────────────────────────────────────────────────────────

_PROMPT_STEP_NAMES = """\
Analyze this dance class audio/video. A teacher is demonstrating rhythm \
for a ballet exercise. They may count with NUMBERS (1, 2, 3...) or with \
STEP NAMES (tendu, plié, brush, close, etc.) — both are rhythmic markers.

For each spoken word, classify it:
- "beat": words that land ON a beat pulse — counted numbers (1, 2, 3...) \
  OR step names spoken rhythmically (tendu, plié, brush, close, front, back, \
  side, point, etc.). The key indicator is regular temporal spacing.
- "and": subdivision words ("and", "&") between beats
- "ah": subdivision words ("ah", "a", "uh") between beats
- "none": non-rhythmic speech (instructions, explanations, transitions \
  like "we're going to...", "same to the...", "you start the...")

IMPORTANT: A word is "beat" if it falls on a regular rhythmic pulse, \
regardless of whether it's a number or a step name. Listen for the \
temporal pattern — evenly-spaced step names ARE beats. Assign beat_number \
sequentially (1, 2, 3...) within each phrase.

Identify the ballet exercise type from speech and/or movement.

For counting_structure, report what you observe about the counting pattern.

For meter, determine the time signature from the counting pattern and movement quality \
(e.g. waltz = 3/4, most barre work = 4/4).

For quality, rate the movement on three numeric dimensions (0.0–1.0). \
Rate what you actually observe, not what the exercise should ideally look like. \
Use these calibration points: articulation (frappé=0.1, tendu=0.5, port de bras=0.9), \
weight (petit allegro=0.2, tendu=0.5, grand plié=0.9), \
energy (adagio=0.2, tendu=0.5, grand allegro=0.9).

For structure, report the phrase length in counts and whether the exercise \
is performed on one side or both sides."""


def experiment_a_gemini_step_names():
    """Run Gemini with updated prompt that recognizes step names as beats."""
    from google import genai
    from google.genai import types
    from musical_perception.perception.gemini import (
        _RESPONSE_SCHEMA,
        _VIDEO_EXTENSIONS,
        _extract_audio,
        _upload_and_wait,
        _parse_response,
    )

    print("\n" + "=" * 70)
    print("  EXPERIMENT A: Gemini with step-name-aware prompt")
    print("=" * 70)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash"

    uploaded_files = []
    audio_tmp_path = None

    try:
        t0 = time.time()

        # Upload video
        main_file = _upload_and_wait(client, VIDEO_PATH, uploaded_files)

        # Extract and upload audio
        audio_tmp_path = _extract_audio(VIDEO_PATH)
        if audio_tmp_path:
            audio_file = _upload_and_wait(client, audio_tmp_path, uploaded_files)

        # Build content with NEW prompt
        parts = []
        for f in uploaded_files:
            parts.append(types.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type))
        parts.append(types.Part.from_text(text=_PROMPT_STEP_NAMES))

        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_RESPONSE_SCHEMA,
            ),
        )

        raw = json.loads(response.text)
        result = _parse_response(raw, model)
        elapsed = time.time() - t0

        print(f"  Completed in {elapsed:.1f}s")

        # Show what Gemini classified
        beat_words = [w for w in result.words if w.marker_type is not None]
        none_words = [w for w in result.words if w.marker_type is None]
        print(f"\n  Total words from Gemini: {len(result.words)}")
        print(f"  Classified as markers: {len(beat_words)}")
        print(f"  Classified as none: {len(none_words)}")

        print(f"\n  Marker words:")
        for w in beat_words:
            mt = w.marker_type.value if w.marker_type else "none"
            print(f"    [{mt:4s}] beat={w.beat_number}  \"{w.word}\"")

        return result, raw

    finally:
        if audio_tmp_path:
            try:
                os.unlink(audio_tmp_path)
            except OSError:
                pass
        for f in uploaded_files:
            try:
                client.files.delete(name=f.name)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────
# Experiment B: Onset-based tempo from Whisper timestamps
# ─────────────────────────────────────────────────────────────

def experiment_b_onset_tempo(words):
    """
    Detect tempo from word onset times using periodicity detection.

    Approach: Look at inter-onset intervals (IOIs) between consecutive
    words. If there's a regular pulse, the IOI histogram will have a
    peak at the beat period.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT B: Onset-based tempo from word timestamps")
    print("=" * 70)

    if len(words) < 3:
        print("  Not enough words for onset analysis")
        return

    onsets = np.array([w.start for w in words])
    iois = np.diff(onsets)

    print(f"\n  Total words: {len(words)}")
    print(f"  Time span: {onsets[0]:.2f}s – {onsets[-1]:.2f}s ({onsets[-1]-onsets[0]:.1f}s)")
    print(f"\n  IOI statistics:")
    print(f"    Mean:   {np.mean(iois):.3f}s")
    print(f"    Median: {np.median(iois):.3f}s")
    print(f"    Std:    {np.std(iois):.3f}s")
    print(f"    Min:    {np.min(iois):.3f}s")
    print(f"    Max:    {np.max(iois):.3f}s")

    # --- Method B1: IOI Histogram Peak ---
    print(f"\n  --- B1: IOI Histogram ---")

    # Bin IOIs and look for the most common interval
    # Focus on musically-plausible range: 0.2s (300 BPM) to 2.0s (30 BPM)
    musical_iois = iois[(iois >= 0.2) & (iois <= 2.0)]

    if len(musical_iois) > 0:
        hist, bin_edges = np.histogram(musical_iois, bins=36, range=(0.2, 2.0))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        peak_idx = np.argmax(hist)
        peak_period = bin_centers[peak_idx]
        peak_bpm = 60.0 / peak_period

        print(f"    Musical IOIs (0.2-2.0s): {len(musical_iois)} of {len(iois)}")
        print(f"    Peak IOI: {peak_period:.3f}s → {peak_bpm:.1f} BPM")
        print(f"    Peak count: {hist[peak_idx]} occurrences")

        # Show top 3 bins
        top_bins = np.argsort(hist)[::-1][:3]
        print(f"    Top 3 IOI bins:")
        for i, idx in enumerate(top_bins):
            if hist[idx] > 0:
                print(f"      {i+1}. {bin_centers[idx]:.3f}s ({60/bin_centers[idx]:.1f} BPM) — {hist[idx]} hits")

    # --- Method B2: Autocorrelation of onset train ---
    print(f"\n  --- B2: Autocorrelation ---")

    # Create a binary onset signal (1 at word onset, 0 elsewhere)
    # Quantize to 10ms resolution
    resolution = 0.01  # 10ms
    duration = onsets[-1] + 1.0
    n_samples = int(duration / resolution)
    onset_signal = np.zeros(n_samples)
    for onset in onsets:
        idx = int(onset / resolution)
        if 0 <= idx < n_samples:
            onset_signal[idx] = 1.0

    # Smooth slightly (50ms gaussian)
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(onset_signal, sigma=5)

    # Autocorrelation
    autocorr = np.correlate(smoothed, smoothed, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # Keep positive lags only

    # Look for peaks in musically-plausible range
    min_lag = int(0.3 / resolution)   # 200 BPM
    max_lag = int(2.0 / resolution)   # 30 BPM

    if max_lag <= len(autocorr):
        search_region = autocorr[min_lag:max_lag]
        if len(search_region) > 0:
            peak_lag = np.argmax(search_region) + min_lag
            peak_period = peak_lag * resolution
            peak_bpm = 60.0 / peak_period

            print(f"    Peak lag: {peak_lag} samples ({peak_period:.3f}s)")
            print(f"    → {peak_bpm:.1f} BPM")

            # Find secondary peaks (for subdivision detection)
            # Look for peaks at half and third of the main period
            half_lag = peak_lag // 2
            third_lag = peak_lag // 3
            if half_lag > min_lag:
                half_strength = autocorr[half_lag] / autocorr[peak_lag]
                print(f"    Half-period strength: {half_strength:.2f} (>0.5 suggests subdivisions)")
            if third_lag > min_lag:
                third_strength = autocorr[third_lag] / autocorr[peak_lag]
                print(f"    Third-period strength: {third_strength:.2f} (>0.5 suggests triplets)")

    # --- Method B3: Cluster-based (group similar IOIs) ---
    print(f"\n  --- B3: IOI Clustering ---")

    if len(musical_iois) >= 3:
        # Simple approach: find the most common IOI cluster
        sorted_iois = np.sort(musical_iois)

        # Use a tolerance-based clustering (10% window)
        clusters = []
        current_cluster = [sorted_iois[0]]
        for ioi in sorted_iois[1:]:
            if ioi <= current_cluster[-1] * 1.15:  # 15% tolerance
                current_cluster.append(ioi)
            else:
                clusters.append(current_cluster)
                current_cluster = [ioi]
        clusters.append(current_cluster)

        # Sort by cluster size
        clusters.sort(key=len, reverse=True)

        print(f"    Found {len(clusters)} IOI clusters:")
        for i, cluster in enumerate(clusters[:5]):
            mean_ioi = np.mean(cluster)
            bpm = 60.0 / mean_ioi
            print(f"      {i+1}. {mean_ioi:.3f}s ({bpm:.1f} BPM) — {len(cluster)} intervals, "
                  f"range [{min(cluster):.3f}, {max(cluster):.3f}]")

        # Check for hierarchical relationships (beat vs subdivision)
        if len(clusters) >= 2:
            c1_mean = np.mean(clusters[0])
            c2_mean = np.mean(clusters[1])
            ratio = max(c1_mean, c2_mean) / min(c1_mean, c2_mean)
            print(f"\n    Top-2 cluster ratio: {ratio:.2f}")
            if 1.8 <= ratio <= 2.2:
                print(f"    → Suggests duple relationship (beat + subdivision)")
            elif 2.7 <= ratio <= 3.3:
                print(f"    → Suggests triplet relationship")

    # --- Method B4: Segmented analysis ---
    print(f"\n  --- B4: Segmented analysis (looking for rhythmic vs non-rhythmic sections) ---")

    # Look at word density in sliding windows
    window_size = 3.0  # seconds
    step = 1.0
    t = onsets[0]
    segments = []
    while t + window_size <= onsets[-1]:
        mask = (onsets >= t) & (onsets < t + window_size)
        count = np.sum(mask)
        segment_onsets = onsets[mask]
        if len(segment_onsets) >= 3:
            segment_iois = np.diff(segment_onsets)
            cv = np.std(segment_iois) / np.mean(segment_iois) if np.mean(segment_iois) > 0 else 999
            segments.append({
                'start': t,
                'end': t + window_size,
                'words': count,
                'mean_ioi': np.mean(segment_iois),
                'cv': cv,
                'is_rhythmic': cv < 0.4,  # Low CV = regular spacing = rhythmic
            })
        t += step

    rhythmic_segments = [s for s in segments if s['is_rhythmic']]
    print(f"    Total 3s windows: {len(segments)}")
    print(f"    Rhythmic windows (CV < 0.4): {len(rhythmic_segments)}")

    if rhythmic_segments:
        print(f"\n    Rhythmic sections detected:")
        for s in rhythmic_segments:
            bpm = 60.0 / s['mean_ioi']
            # Get the actual words in this window
            mask = (onsets >= s['start']) & (onsets < s['end'])
            section_words = [words[i].word for i in range(len(words)) if mask[i]]
            print(f"      {s['start']:.1f}-{s['end']:.1f}s: "
                  f"{bpm:.0f} BPM (CV={s['cv']:.2f}) "
                  f"words: {' '.join(section_words)}")

        # Weighted average of rhythmic sections
        weighted_bpm = np.mean([60.0 / s['mean_ioi'] for s in rhythmic_segments])
        print(f"\n    Weighted avg BPM from rhythmic sections: {weighted_bpm:.1f}")


# ─────────────────────────────────────────────────────────────
# Experiment C: Combined — Gemini classifications + onset timing
# ─────────────────────────────────────────────────────────────

def experiment_c_combined(gemini_result, words):
    """
    Use Gemini's step-name-aware classifications merged with Whisper
    timestamps, then run through the existing precision layer.
    """
    from musical_perception.analyze import _merge_gemini_with_timestamps
    from musical_perception.precision.tempo import calculate_tempo
    from musical_perception.precision.subdivision import analyze_subdivisions
    from musical_perception.types import MarkerType

    print("\n" + "=" * 70)
    print("  EXPERIMENT C: Gemini step-name prompt + Whisper merge → Precision layer")
    print("=" * 70)

    markers = _merge_gemini_with_timestamps(gemini_result, words)

    print(f"\n  Merged markers: {len(markers)}")
    for m in markers:
        print(f"    {m.timestamp:6.2f}s  [{m.marker_type.name:4s}]  beat={m.beat_number}  \"{m.raw_word}\"")

    beat_timestamps = [m.timestamp for m in markers if m.marker_type == MarkerType.BEAT]
    tempo = calculate_tempo(beat_timestamps)
    subdivision = analyze_subdivisions(markers)

    print(f"\n  Tempo: ", end="")
    if tempo:
        print(f"{tempo.bpm} BPM ({tempo.confidence:.0%} confidence, {tempo.beat_count} beats)")
        print(f"  Intervals: {[f'{i:.3f}' for i in tempo.intervals]}")
    else:
        print("not detected")

    print(f"  Subdivision: {subdivision.subdivision_type} ({subdivision.confidence:.0%})")


def main():
    print("Getting Whisper transcription...")
    words = get_whisper_words("small.en")  # best for ballet terminology

    print(f"\n{'#'*70}")
    print(f"  RAW WHISPER DATA")
    print(f"{'#'*70}")
    print_words_with_timing(words)

    # Experiment A: Gemini with step-name prompt
    gemini_result, raw_json = experiment_a_gemini_step_names()

    # Experiment B: Onset-based tempo
    experiment_b_onset_tempo(words)

    # Experiment C: Combined (Gemini step-name + merge)
    if gemini_result:
        experiment_c_combined(gemini_result, words)

    print(f"\n\n{'#'*70}")
    print(f"  DONE — Compare approaches above")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
