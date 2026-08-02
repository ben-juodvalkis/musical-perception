"""CLI entry point: python -m musical_perception <audio_file>"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m musical_perception <audio_file> [--signature] [--stress] [--pose] [--model NAME] [--record-traces [DIR]]")
        sys.exit(1)

    audio_file = sys.argv[1]
    extract_sig = "--signature" in sys.argv
    detect_stress = "--stress" in sys.argv
    use_pose = "--pose" in sys.argv
    model_name = "large-v3-turbo"
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 >= len(sys.argv):
            print("--model requires a name (e.g. base.en, small.en, large-v3-turbo)")
            sys.exit(1)
        model_name = sys.argv[idx + 1]

    gemini_model = "gemini-2.5-flash"
    trace_dir = None
    if "--record-traces" in sys.argv:
        from pathlib import Path
        from musical_perception.evals.traces import slugify

        idx = sys.argv.index("--record-traces")
        nxt = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        if nxt and not nxt.startswith("--"):
            trace_dir = Path(nxt)
        else:
            trace_dir = Path("evals/traces") / slugify(Path(audio_file).stem)

    from musical_perception.analyze import analyze

    bundle = None
    if trace_dir is not None:
        from musical_perception.bundle import build_default_bundle
        from musical_perception.evals.traces import make_recording_bundle

        bundle = make_recording_bundle(
            build_default_bundle(model_name=model_name, gemini_model=gemini_model),
            trace_dir,
        )
        print(f"Recording trace to {trace_dir}")

    print("Analyzing with Gemini...")
    result = analyze(
        audio_file,
        model_name=model_name,
        extract_signature=extract_sig,
        detect_stress=detect_stress,
        use_pose=use_pose,
        bundle=bundle,
    )

    if trace_dir is not None:
        from musical_perception.evals.traces import write_meta

        write_meta(
            trace_dir, audio_file,
            use_pose=use_pose,
            whisper_model_name=model_name,
            gemini_model=gemini_model,
        )

    if result.normalized_tempo is not None:
        nt = result.normalized_tempo
        m = nt.meter
        sub = f", {nt.subdivision} subdivision" if nt.subdivision != "none" else ""
        print(f"\n--- Tempo ---")
        print(f"BPM: {nt.bpm} ({m.beats_per_measure}/{m.beat_unit}{sub})")
        if nt.tempo_multiplier != 1:
            print(f"  (raw {nt.raw_bpm} BPM, multiplier={nt.tempo_multiplier})")

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

    if result.subdivision:
        print(f"\n--- Subdivision (raw) ---")
        print(f"Type: {result.subdivision.subdivision_type}")
        print(f"Confidence: {result.subdivision.confidence:.0%}")

    if result.exercise and result.exercise.primary_exercise:
        print(f"\n--- Exercise ---")
        print(f"Type: {result.exercise.display_name}")
        print(f"Confidence: {result.exercise.confidence:.0%}")

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
