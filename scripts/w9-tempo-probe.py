"""W9 diagnostic: dump the tempo path's internal state per tier-1 case.

Read-only. Replays each frozen trace exactly as the harness does, but
records what each tempo arm said BEFORE arbitration, which arm won, and
how normalize_tempo folded it. Not part of the eval harness.
"""
import sys, warnings, json
from pathlib import Path

warnings.simplefilter("ignore")

from musical_perception.evals.cases import load_cases
from musical_perception.evals.traces import replay_bundle
from musical_perception.precision.rhythm import detect_onset_tempo
from musical_perception.precision.tempo import calculate_tempo, normalize_tempo
from musical_perception.types import MarkerType
from musical_perception.analyze import _markers_from_gemini

EVALS = Path("evals")

rows = []
for case in load_cases(EVALS / "cases"):
    try:
        bundle, meta = replay_bundle(EVALS / case.trace)
    except Exception as e:
        rows.append({"id": case.id, "error": str(e)}); continue
    words = bundle.transcribe("x")
    ot = detect_onset_tempo(words)
    gr = bundle.analyze_media("x")
    markers = _markers_from_gemini(gr, words)
    beats = [m.timestamp for m in markers if m.marker_type == MarkerType.BEAT]
    gt = calculate_tempo(beats)

    onset_at_beat = ot is not None and ot.confidence >= 0.3 and 70.0 <= ot.bpm <= 140.0
    marker_at_beat = (gt is not None and gt.confidence >= 0.6
                      and gt.beat_count >= 8 and 70.0 <= gt.bpm <= 140.0)
    marker_strong = marker_at_beat and not onset_at_beat
    if marker_strong: arm, raw = "marker", gt.bpm
    elif ot is not None and ot.confidence >= 0.3: arm, raw = "onset", ot.bpm
    elif gt is not None: arm, raw = "marker-fallback", gt.bpm
    elif ot is not None: arm, raw = "onset-lowconf", ot.bpm
    else: arm, raw = "none", None
    norm, mult = normalize_tempo(raw) if raw else (None, 0)

    rows.append({
        "id": case.id,
        "truth": case.expected_bpm,
        "prov": case.provisional,
        "onset_bpm": None if ot is None else ot.bpm,
        "onset_conf": None if ot is None else ot.confidence,
        "onset_cov": None if ot is None else ot.rhythmic_coverage,
        "onset_hist": None if ot is None else ot.ioi_histogram_peak_bpm,
        "gem_bpm": None if gt is None else gt.bpm,
        "gem_conf": None if gt is None else gt.confidence,
        "gem_n": None if gt is None else gt.beat_count,
        "arm": arm, "raw": raw, "norm": norm, "mult": mult,
    })

print(json.dumps(rows, indent=None))
