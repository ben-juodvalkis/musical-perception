"""CLI entry point: python -m musical_perception <audio_file>"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m musical_perception <audio_file> [--signature] [--stress] [--pose]")
        sys.exit(1)

    audio_file = sys.argv[1]
    extract_sig = "--signature" in sys.argv
    detect_stress = "--stress" in sys.argv
    use_pose = "--pose" in sys.argv

    from musical_perception.analyze import analyze

    print("Analyzing with Gemini...")
    result = analyze(
        audio_file,
        extract_signature=extract_sig,
        detect_stress=detect_stress,
        use_pose=use_pose,
    )

    if result.normalized_bpm is not None:
        mult = result.tempo_multiplier
        if mult == 1:
            note = ""
        elif mult == 2:
            note = " (raw doubled from measure level)"
        elif mult == 3:
            note = " (raw tripled from triple-meter measure level)"
        elif mult == -2:
            note = " (raw halved from subdivision level)"
        elif mult == -3:
            note = " (raw divided by 3 from triplet subdivision)"
        else:
            note = f" (multiplier={mult})"
        print(f"\n--- Tempo (normalized) ---")
        print(f"BPM: {result.normalized_bpm}{note}")

    print("\n--- Tempo (raw signals) ---")
    if result.tempo:
        print(f"Gemini-based: {result.tempo.bpm} BPM ({result.tempo.confidence:.0%} confidence, {result.tempo.beat_count} beats)")
    else:
        print("Gemini-based: no beats detected")

    if result.onset_tempo:
        ot = result.onset_tempo
        print(f"Onset-based:  {ot.bpm} BPM ({ot.confidence:.0%} confidence, {ot.rhythmic_coverage:.0%} coverage)")
        if ot.ioi_histogram_peak_bpm:
            print(f"  Histogram cross-check: {ot.ioi_histogram_peak_bpm} BPM")
        for section in ot.rhythmic_sections:
            print(f"  {section.start:.1f}-{section.end:.1f}s: {section.bpm:.0f} BPM "
                  f"(CV={section.cv:.2f}) [{' '.join(section.words)}]")
    else:
        print("Onset-based:  no rhythmic sections detected")

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
        print(f"  articulation: {q.articulation:.2f}  (staccato → legato)")
        print(f"  weight:       {q.weight:.2f}  (light → heavy)")
        print(f"  energy:       {q.energy:.2f}  (calm → energetic)")

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
