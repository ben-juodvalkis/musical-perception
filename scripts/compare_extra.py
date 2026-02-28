"""
Additional comparisons: signature extraction + extra Gemini models.
"""

import sys
import time
import traceback

VIDEO_PATH = "video/youtube/Exercise 1 Demo.m4v"


def run_one(whisper_model_name, gemini_model_name, extract_signature=False, label=""):
    from musical_perception.analyze import analyze
    from musical_perception.perception.whisper import load_model as load_whisper

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Whisper: {whisper_model_name} | Gemini: {gemini_model_name}")
    if extract_signature:
        print(f"  + Prosody/Signature")
    print(f"{'='*70}")

    try:
        t0 = time.time()
        whisper_model = load_whisper(whisper_model_name)
        result = analyze(
            VIDEO_PATH,
            model=whisper_model,
            model_name=whisper_model_name,
            gemini_model=gemini_model_name,
            extract_signature=extract_signature,
        )
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        # Words
        word_text = " ".join(w.word for w in result.words)
        print(f"\n  Words ({len(result.words)}): {word_text[:120]}...")

        # Markers
        print(f"\n  Markers ({len(result.markers)}):")
        for m in result.markers:
            print(f"    {m.timestamp:.2f}s  [{m.marker_type.name:4s}]  beat={m.beat_number}  \"{m.raw_word}\"")

        # Tempo
        if result.tempo:
            print(f"\n  Tempo: {result.tempo.bpm} BPM ({result.tempo.confidence:.0%} conf, {result.tempo.beat_count} beats)")
            print(f"  Intervals: {[f'{i:.3f}' for i in result.tempo.intervals]}")
        else:
            print(f"\n  Tempo: not detected")

        # Subdivision
        if result.subdivision:
            print(f"  Subdivision: {result.subdivision.subdivision_type} ({result.subdivision.confidence:.0%} conf)")

        # Exercise
        if result.exercise and result.exercise.primary_exercise:
            print(f"  Exercise: {result.exercise.display_name} ({result.exercise.confidence:.0%})")

        # Meter
        if result.meter:
            print(f"  Meter: {result.meter.beats_per_measure}/{result.meter.beat_unit}")

        # Quality
        if result.quality:
            q = result.quality
            print(f"  Quality: art={q.articulation:.2f} wt={q.weight:.2f} en={q.energy:.2f}")

        # Structure
        if result.structure:
            print(f"  Structure: {result.structure.counts} counts, {result.structure.sides} side(s)")

        # Signature
        if result.counting_signature:
            sig = result.counting_signature
            print(f"\n  --- Counting Signature ---")
            if sig.weight_placement:
                print(f"  Weight placement: {sig.weight_placement}")
            if sig.beat_vs_and_intensity_db is not None:
                sign = "+" if sig.beat_vs_and_intensity_db >= 0 else ""
                print(f"  Beat vs And intensity: {sign}{sig.beat_vs_and_intensity_db:.1f} dB")
            if sig.beat_stats:
                print(f"  Beat stats: pitch={sig.beat_stats.mean_pitch_hz:.1f}Hz, "
                      f"int={sig.beat_stats.mean_intensity_db:.1f}dB, "
                      f"dur={sig.beat_stats.mean_duration:.3f}s")
            if sig.and_stats:
                print(f"  And stats:  pitch={sig.and_stats.mean_pitch_hz:.1f}Hz, "
                      f"int={sig.and_stats.mean_intensity_db:.1f}dB, "
                      f"dur={sig.and_stats.mean_duration:.3f}s")
            if sig.ah_stats:
                print(f"  Ah stats:   pitch={sig.ah_stats.mean_pitch_hz:.1f}Hz, "
                      f"int={sig.ah_stats.mean_intensity_db:.1f}dB, "
                      f"dur={sig.ah_stats.mean_duration:.3f}s")

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()


def main():
    # Run with signature extraction (base.en + Flash 2.5)
    run_one("base.en", "gemini-2.5-flash", extract_signature=True,
            label="base.en + Flash 2.5 + Signature")

    # Try Gemini Flash-Lite (cheaper, faster)
    run_one("base.en", "gemini-2.0-flash-lite",
            label="base.en + Flash Lite 2.0")

    # Try running the same config twice to check reproducibility
    run_one("base.en", "gemini-2.5-flash",
            label="base.en + Flash 2.5 (run 2 - reproducibility)")

    # Try small.en with signature too
    run_one("small.en", "gemini-2.5-flash", extract_signature=True,
            label="small.en + Flash 2.5 + Signature")


if __name__ == "__main__":
    main()
