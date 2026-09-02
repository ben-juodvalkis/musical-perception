#!/usr/bin/env python
"""EB-1 Arm B: off-the-shelf beat trackers on the eight barre-6 demos.

W3 (2026-08-21) benchmarked these tools, but the barre-6 demos did not
exist then and W3 scored pulse F, not step-one tempo. This scores the
SAME frozen step-one criterion the rest of EB-1 uses: committed pulse
within +-8% of the in-band truth, after the same x/{2,3} projection.

Trackers are reused verbatim from scripts/baseline_benchmark.py (W3) so
this is not a reimplementation. A tool that fails is reported BLOCKED by
name. REPORTED-ONLY; writes nothing under evals/ or src/.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util as _iu  # noqa: E402

_spec = _iu.spec_from_file_location("w3", str(ROOT / "scripts" / "baseline_benchmark.py"))
w3 = _iu.module_from_spec(_spec)
sys.modules["w3"] = w3   # dataclass needs the module registered
_spec.loader.exec_module(w3)

from musical_perception.evals.aggregate import acc2, octave_errors  # noqa: E402
from musical_perception.evals.cases import load_cases  # noqa: E402

BAND_LO, BAND_HI = 70.0, 140.0
PASS_TOL = 0.08
FACTORS = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0)
TOOLS = {
    "librosa_plp": w3.track_librosa_plp,
    "essentia_re2013": w3.track_essentia,
    "beat_this": w3.track_beat_this,
}


def project(bpm):
    for f in FACTORS:
        c = bpm * f
        if BAND_LO <= c <= BAND_HI:
            return c, f
    return None, None


def main() -> int:
    cases = {c.id: c for c in load_cases(ROOT / "evals" / "cases")}
    demos = sorted(cid for cid, c in cases.items()
                   if c.tags.get("clip_role") == "demo" and not c.reference
                   and c.maturity == "verified")
    print(f"Arm B — {len(TOOLS)} trackers x {len(demos)} demos (step-one tempo, +-8%)\n")
    out, blocked = {}, {}
    with tempfile.TemporaryDirectory() as td:
        wavs = {}
        for cid in demos:
            src = ROOT / cases[cid].media
            w = Path(td) / f"{cid}.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                            "-vn", "-ac", "1", "-ar", "44100", str(w)], check=True)
            wavs[cid] = w
        for tool, fn in TOOLS.items():
            rows = {}
            for cid in demos:
                truth = float(cases[cid].expected_bpm)
                try:
                    _beats, bpm = fn(wavs[cid])
                except Exception as e:  # noqa: BLE001
                    blocked.setdefault(tool, []).append(f"{cid}: {type(e).__name__}: {e}")
                    rows[cid] = {"error": f"{type(e).__name__}"}
                    continue
                if bpm is None:
                    rows[cid] = {"raw": None, "bpm": None, "pass": False, "truth": truth}
                    continue
                b, fac = project(float(bpm))
                r = {"raw": round(float(bpm), 2), "truth": truth,
                     "bpm": None if b is None else round(b, 2), "factor": fac}
                if b is not None:
                    r["pass"] = abs(b - truth) / truth <= PASS_TOL
                    r["acc2"] = bool(acc2(b, truth, PASS_TOL))
                    _, oe2 = octave_errors(b, truth)
                    r["between_levels"] = bool(0.08 < abs(oe2) <= 0.585)
                else:
                    r.update({"pass": False, "acc2": False, "between_levels": False})
                rows[cid] = r
            out[tool] = rows
            ok = sum(1 for r in rows.values() if r.get("pass"))
            print(f"{tool:<18} {ok}/{len(demos)} pass")
            for cid in demos:
                r = rows[cid]
                nm = cid.replace("barre6-", "").replace("-demo", "")
                if "error" in r:
                    print(f"    {nm:<18} FAILED {r['error']}")
                else:
                    print(f"    {nm:<18} raw {str(r['raw']):>7} -> {str(r['bpm']):>7}"
                          f" (truth {r['truth']:.0f}){'  PASS' if r.get('pass') else ''}")
    if blocked:
        print("\nBLOCKED by name:")
        for t, errs in blocked.items():
            print(f"  {t}: {errs[0]}  (+{len(errs)-1} more)" if len(errs) > 1 else f"  {t}: {errs[0]}")
    json.dump({"tools": out, "blocked": blocked},
              open(ROOT / "docs/research/eb1-arm-b-trackers.json", "w"), indent=1, sort_keys=True)
    print("\nwrote docs/research/eb1-arm-b-trackers.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
