"""W5 diagnostic: per-clip evidence inventory for the joint posterior.

Read-only. For every tier-1 case, replays the frozen trace exactly as the
harness does and dumps what the rhythm core actually has to work with:
the classified marker stream (classes, beat-number cycle, beat IOIs), the
raw word-onset stream, Gemini's meter/subdivision/count claims, and what
the current stack answers. This is the observation-stream inventory the
W5 pre-registration's per-clip predictions are derived from. Not part of
the eval harness.
"""
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")

from musical_perception.analyze import _markers_from_gemini
from musical_perception.evals.cases import load_cases
from musical_perception.evals.traces import replay_bundle
from musical_perception.precision.rhythm import detect_onset_tempo
from musical_perception.precision.tempo import calculate_tempo, interpret_meter
from musical_perception.types import MarkerType

EVALS = Path("evals")


def beat_number_cycle(markers) -> dict:
    """How the counted numbers behave: max, cycle length, resets."""
    nums = [m.beat_number for m in markers
            if m.marker_type == MarkerType.BEAT and m.beat_number is not None]
    if not nums:
        return {"n_numbered": 0}
    resets = [i for i in range(1, len(nums)) if nums[i] < nums[i - 1]]
    cycle_lengths = Counter()
    prev = 0
    for i in resets:
        cycle_lengths[nums[i - 1]] += 1
        prev = i
    return {
        "n_numbered": len(nums),
        "max": max(nums),
        "n_resets": len(resets),
        "cycle_tops": dict(cycle_lengths.most_common(4)),
        "head": nums[:12],
    }


def ioi_stats(times: list[float]) -> dict:
    if len(times) < 3:
        return {"n": len(times)}
    iois = np.diff(sorted(times))
    med = float(np.median(iois))
    return {
        "n": len(times),
        "ioi_median_s": round(med, 3),
        "bpm_at_level": round(60.0 / med, 1) if med > 0 else None,
        "ioi_cv": round(float(np.std(iois) / np.mean(iois)), 3),
    }


rows = []
for case in load_cases(EVALS / "cases"):
    try:
        bundle, meta = replay_bundle(EVALS / case.trace)
    except Exception as e:
        rows.append({"id": case.id, "error": str(e)})
        continue
    words = bundle.transcribe("x")
    ot = detect_onset_tempo(words)
    gr = bundle.analyze_media("x")
    markers = _markers_from_gemini(gr, words)

    by_class = Counter(m.marker_type.value for m in markers)
    beats = [m.timestamp for m in markers if m.marker_type == MarkerType.BEAT]
    gt = calculate_tempo(beats)
    counting = gr.counting_structure
    gsub = counting.subdivision_type if counting else None
    current = interpret_meter(ot, gt, gr.meter, gsub)

    truth_meter = case.expect.get("meter")
    rows.append({
        "id": case.id,
        "prov": case.provisional,
        "truth": {
            "bpm": case.expected_bpm,
            "meter": (f"{truth_meter.beats_per_measure}/{truth_meter.beat_unit}"
                      if truth_meter else None),
            "subdivision": case.expect.get("subdivision"),
            "counts": case.expect.get("counts"),
        },
        "words": ioi_stats([w.start for w in words]),
        "markers": {
            "classes": dict(by_class),
            "beat_iois": ioi_stats(beats),
            "numbers": beat_number_cycle(markers),
        },
        "onset": None if ot is None else {
            "bpm": ot.bpm, "conf": ot.confidence,
            "cov": ot.rhythmic_coverage, "hist": ot.ioi_histogram_peak_bpm,
        },
        "gemini": {
            "meter": (f"{gr.meter.beats_per_measure}/{gr.meter.beat_unit}"
                      if gr.meter else None),
            "subdivision": gsub,
            "est_bpm": counting.estimated_bpm if counting else None,
            "total_counts": counting.total_counts if counting else None,
            "structure_counts": gr.structure.counts if gr.structure else None,
        },
        "current": None if current is None else {
            "bpm": current.bpm,
            "meter": (f"{current.meter.beats_per_measure}"
                      f"/{current.meter.beat_unit}"),
            "subdivision": current.subdivision,
            "mult": current.tempo_multiplier,
            "conf": current.confidence,
        },
    })

print(json.dumps(rows))
