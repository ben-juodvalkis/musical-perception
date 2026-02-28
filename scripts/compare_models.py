"""
Compare multiple model combinations on the same video file.

Runs different Whisper models + Gemini models + optional features
and prints a side-by-side comparison.
"""

import json
import sys
import time
import traceback
from dataclasses import asdict

VIDEO_PATH = "video/youtube/Exercise 1 Demo.m4v"


def run_analysis(
    whisper_model_name: str,
    gemini_model_name: str,
    extract_signature: bool = False,
    use_pose: bool = False,
    label: str = "",
):
    """Run one analysis configuration and return results + timing."""
    from musical_perception.analyze import analyze
    from musical_perception.perception.whisper import load_model as load_whisper

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Whisper: {whisper_model_name} | Gemini: {gemini_model_name}")
    if extract_signature:
        print(f"  + Prosody/Signature extraction")
    if use_pose:
        print(f"  + Pose estimation")
    print(f"{'='*70}")

    try:
        t0 = time.time()

        # Load whisper model
        print(f"  Loading Whisper {whisper_model_name}...")
        t_whisper_load = time.time()
        whisper_model = load_whisper(whisper_model_name)
        print(f"  Whisper loaded in {time.time() - t_whisper_load:.1f}s")

        # Run full analysis
        print(f"  Running analysis...")
        result = analyze(
            VIDEO_PATH,
            model=whisper_model,
            model_name=whisper_model_name,
            gemini_model=gemini_model_name,
            extract_signature=extract_signature,
            use_pose=use_pose,
        )

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        # Print results
        print(f"\n  --- Words transcribed (Whisper) ---")
        word_text = " ".join(w.word for w in result.words)
        print(f"  {word_text}")
        print(f"  ({len(result.words)} words)")

        print(f"\n  --- Markers (Gemini + Whisper merged) ---")
        for m in result.markers:
            print(f"  {m.timestamp:.2f}s  [{m.marker_type.name:4s}]  beat={m.beat_number}  \"{m.raw_word}\"")
        print(f"  ({len(result.markers)} markers)")

        print(f"\n  --- Tempo ---")
        if result.tempo:
            print(f"  BPM: {result.tempo.bpm}")
            print(f"  Confidence: {result.tempo.confidence:.0%}")
            print(f"  Beats detected: {result.tempo.beat_count}")
            if result.tempo.intervals:
                print(f"  Intervals: {[f'{i:.3f}' for i in result.tempo.intervals]}")
        else:
            print(f"  Could not detect tempo")

        print(f"\n  --- Subdivision ---")
        if result.subdivision:
            print(f"  Type: {result.subdivision.subdivision_type}")
            print(f"  Confidence: {result.subdivision.confidence:.0%}")
            if result.subdivision.subdivisions_per_beat is not None:
                print(f"  Subdivisions/beat: {result.subdivision.subdivisions_per_beat}")

        if result.exercise and result.exercise.primary_exercise:
            print(f"\n  --- Exercise ---")
            print(f"  Type: {result.exercise.display_name}")
            print(f"  Confidence: {result.exercise.confidence:.0%}")

        if result.meter:
            print(f"\n  --- Meter ---")
            print(f"  {result.meter.beats_per_measure}/{result.meter.beat_unit}")

        if result.quality:
            print(f"\n  --- Quality ---")
            q = result.quality
            print(f"  articulation: {q.articulation:.2f}  (staccato → legato)")
            print(f"  weight:       {q.weight:.2f}  (light → heavy)")
            print(f"  energy:       {q.energy:.2f}  (calm → energetic)")

        if result.structure:
            print(f"\n  --- Structure ---")
            print(f"  {result.structure.counts} counts, {result.structure.sides} side(s)")

        if result.counting_signature:
            print(f"\n  --- Counting Signature ---")
            sig = result.counting_signature
            if sig.weight_placement:
                placement_desc = {
                    "on_beat": "Weight ON the beat (grounded, marcato)",
                    "after_beat": "Weight AFTER the beat (flowing, lilting)",
                    "even": "Weight evenly distributed",
                }
                print(f"  Weight: {placement_desc.get(sig.weight_placement, sig.weight_placement)}")
            if sig.beat_vs_and_intensity_db is not None:
                sign = "+" if sig.beat_vs_and_intensity_db >= 0 else ""
                print(f"  Beat vs And intensity: {sign}{sig.beat_vs_and_intensity_db:.1f} dB")
            if sig.beat_stats:
                print(f"  Beat stats: pitch={sig.beat_stats.mean_pitch_hz:.1f}Hz, "
                      f"intensity={sig.beat_stats.mean_intensity_db:.1f}dB, "
                      f"duration={sig.beat_stats.mean_duration:.3f}s")
            if sig.and_stats:
                print(f"  And stats:  pitch={sig.and_stats.mean_pitch_hz:.1f}Hz, "
                      f"intensity={sig.and_stats.mean_intensity_db:.1f}dB, "
                      f"duration={sig.and_stats.mean_duration:.3f}s")

        return {
            "label": label,
            "whisper_model": whisper_model_name,
            "gemini_model": gemini_model_name,
            "elapsed_s": elapsed,
            "n_words": len(result.words),
            "n_markers": len(result.markers),
            "bpm": result.tempo.bpm if result.tempo else None,
            "tempo_confidence": result.tempo.confidence if result.tempo else None,
            "beat_count": result.tempo.beat_count if result.tempo else None,
            "subdivision": result.subdivision.subdivision_type if result.subdivision else None,
            "sub_confidence": result.subdivision.confidence if result.subdivision else None,
            "exercise": result.exercise.display_name if result.exercise else None,
            "meter": f"{result.meter.beats_per_measure}/{result.meter.beat_unit}" if result.meter else None,
            "quality": {
                "articulation": result.quality.articulation,
                "weight": result.quality.weight,
                "energy": result.quality.energy,
            } if result.quality else None,
            "structure": f"{result.structure.counts} counts, {result.structure.sides} sides" if result.structure else None,
            "words_text": word_text,
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return {
            "label": label,
            "whisper_model": whisper_model_name,
            "gemini_model": gemini_model_name,
            "error": str(e),
        }


def print_comparison(results):
    """Print a side-by-side comparison table."""
    print(f"\n\n{'#'*70}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'#'*70}")

    # Filter successful results
    ok = [r for r in results if "error" not in r]
    if not ok:
        print("No successful runs!")
        return

    header = f"{'Config':<35} {'BPM':>5} {'Conf':>5} {'Beats':>5} {'Sub':>8} {'Exercise':>15} {'Meter':>5} {'Words':>5} {'Time':>6}"
    print(f"\n{header}")
    print("-" * len(header))

    for r in ok:
        bpm = f"{r['bpm']}" if r.get('bpm') else "—"
        conf = f"{r['tempo_confidence']:.0%}" if r.get('tempo_confidence') else "—"
        beats = f"{r['beat_count']}" if r.get('beat_count') else "—"
        sub = r.get('subdivision', '—') or '—'
        ex = (r.get('exercise') or '—')[:15]
        meter = r.get('meter', '—') or '—'
        words = f"{r['n_words']}"
        elapsed = f"{r['elapsed_s']:.1f}s"
        label = r['label'][:35]
        print(f"{label:<35} {bpm:>5} {conf:>5} {beats:>5} {sub:>8} {ex:>15} {meter:>5} {words:>5} {elapsed:>6}")

    # Quality comparison
    quality_results = [r for r in ok if r.get('quality')]
    if quality_results:
        print(f"\nQuality Profile Comparison:")
        print(f"{'Config':<35} {'Artic':>6} {'Weight':>6} {'Energy':>6}")
        print("-" * 60)
        for r in quality_results:
            q = r['quality']
            label = r['label'][:35]
            print(f"{label:<35} {q['articulation']:>6.2f} {q['weight']:>6.2f} {q['energy']:>6.2f}")


def main():
    results = []

    # === WHISPER MODEL COMPARISON (all with Gemini 2.5 Flash) ===
    for whisper_model in ["tiny.en", "base.en", "small.en"]:
        r = run_analysis(
            whisper_model_name=whisper_model,
            gemini_model_name="gemini-2.5-flash",
            label=f"Whisper {whisper_model} + Flash 2.5",
        )
        results.append(r)

    # === GEMINI MODEL COMPARISON (all with Whisper base.en) ===
    for gemini_model in ["gemini-2.0-flash-001", "gemini-2.5-pro"]:
        r = run_analysis(
            whisper_model_name="base.en",
            gemini_model_name=gemini_model,
            label=f"Whisper base.en + {gemini_model.replace('gemini-', 'Gemini ')}",
        )
        results.append(r)

    # === WITH PROSODY/SIGNATURE ===
    r = run_analysis(
        whisper_model_name="base.en",
        gemini_model_name="gemini-2.5-flash",
        extract_signature=True,
        label="base.en + Flash 2.5 + Signature",
    )
    results.append(r)

    # === SUMMARY ===
    print_comparison(results)


if __name__ == "__main__":
    main()
