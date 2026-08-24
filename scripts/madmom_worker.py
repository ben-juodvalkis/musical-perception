"""madmom beat-tracking worker — runs in its own interpreter.

madmom cannot share the project venv: the PyPI release (0.16.1, Nov 2017)
imports `collections.MutableSequence`, removed in Python 3.10. Review 4 §d
predicted exactly this and prescribed a dedicated environment; git main
installs cleanly on 3.12. This worker is the boundary between the two.

Reads a JSON list of wav paths on stdin, writes {path: [beat_times]} on
stdout. Invoked by scripts/baseline_benchmark.py via .venv-madmom/bin/python.
"""

import json
import sys

from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor

# min_bpm=40, not madmom's default 55: Review 4 §(a) flags that the default
# silently octave-doubles 40-54 BPM clips, and this corpus has 60 and 63 BPM
# markings.
MIN_BPM = 40
MAX_BPM = 210


def main() -> None:
    paths = json.load(sys.stdin)
    act_proc = RNNBeatProcessor()
    dbn = DBNBeatTrackingProcessor(min_bpm=MIN_BPM, max_bpm=MAX_BPM, fps=100)
    out = {}
    for path in paths:
        try:
            beats = dbn(act_proc(path))
            out[path] = [float(b) for b in beats]
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            out[path] = {"error": f"{type(exc).__name__}: {exc}"}
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
