"""CLI entry point: python -m musical_perception <audio_file>"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m musical_perception <audio_file> [--signature] [--stress]")
        sys.exit(1)

    audio_file = sys.argv[1]
    extract_sig = "--signature" in sys.argv
    detect_stress = "--stress" in sys.argv

    from musical_perception.analyze import analyze

    print("Loading model...")
    result = analyze(audio_file, extract_signature=extract_sig, detect_stress=detect_stress)

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
