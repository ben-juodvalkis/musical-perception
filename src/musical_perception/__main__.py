"""CLI entry point: python -m musical_perception <audio_file>"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m musical_perception <audio_file> [--signature] [--stress] [--gemini]")
        sys.exit(1)

    audio_file = sys.argv[1]
    extract_sig = "--signature" in sys.argv
    detect_stress = "--stress" in sys.argv
    use_gemini = "--gemini" in sys.argv

    from musical_perception.analyze import analyze

    if use_gemini:
        print("Analyzing with Gemini...")
    else:
        print("Loading model...")
    result = analyze(
        audio_file,
        extract_signature=extract_sig,
        detect_stress=detect_stress,
        use_gemini=use_gemini,
    )

    print("\n--- Tempo ---")
    if result.tempo:
        print(f"BPM: {result.tempo.bpm} ({result.tempo.confidence:.0%} confidence)")
        print(f"Beats detected: {result.tempo.beat_count}")
    else:
        print("Could not detect tempo")

    print("\n--- Subdivision ---")
    if result.subdivision:
        print(f"Type: {result.subdivision.subdivision_type}")
        print(f"Confidence: {result.subdivision.confidence:.0%}")

    if result.exercise and result.exercise.primary_exercise:
        print(f"\n--- Exercise ---")
        print(f"Type: {result.exercise.display_name}")
        print(f"Confidence: {result.exercise.confidence:.0%}")

    if result.meter:
        print(f"\n--- Meter ---")
        print(f"{result.meter.beats_per_measure}/{result.meter.beat_unit}")

    if result.quality:
        print(f"\n--- Quality ---")
        q = result.quality
        print(f"  smoothness:   {q.smoothness:.2f}")
        print(f"  energy:       {q.energy:.2f}")
        print(f"  groundedness: {q.groundedness:.2f}")
        print(f"  attack:       {q.attack:.2f}")
        print(f"  weight:       {q.weight:.2f}")
        print(f"  sustain:      {q.sustain:.2f}")

    if result.structure:
        print(f"\n--- Structure ---")
        print(f"{result.structure.counts} counts, {result.structure.sides} side(s)")

    if result.counting_signature:
        print(f"\n--- Counting Signature ---")
        sig = result.counting_signature
        if sig.weight_placement:
            placement_desc = {
                "on_beat": "Weight ON the beat (grounded, marcato)",
                "after_beat": "Weight AFTER the beat (flowing, lilting)",
                "even": "Weight evenly distributed",
            }
            print(f"Weight: {placement_desc.get(sig.weight_placement, sig.weight_placement)}")
        if sig.beat_vs_and_intensity_db is not None:
            sign = "+" if sig.beat_vs_and_intensity_db >= 0 else ""
            print(f"Beat vs And intensity: {sign}{sig.beat_vs_and_intensity_db:.1f} dB")

    if result.stress_labels:
        print(f"\n--- Stress (WhiStress) ---")
        stressed = [w for w, s in result.stress_labels if s == 1]
        print(f"Stressed words: {', '.join(stressed) if stressed else '(none)'}")


if __name__ == "__main__":
    main()
