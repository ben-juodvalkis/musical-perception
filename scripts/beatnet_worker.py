"""BeatNet beat-tracking worker — runs in the madmom interpreter.

Review 4 §(a)'s optional sixth tool. BeatNet is a joint beat/downbeat CRNN
whose inference stage is madmom's DBN, so it can only live in the same
dedicated environment madmom does (`.venv-madmom`), never in the project
venv. Two further install facts, recorded because the rung asks for exact
failures: BeatNet needs `librosa` at import, and `BeatNet.BeatNet` imports
`pyaudio` unconditionally at module load even in `mode='offline'`, so the
offline path cannot be used without a portaudio toolchain.

Reads a JSON list of wav paths on stdin, writes {path: [beat_times]} on
stdout. Invoked by scripts/baseline_benchmark.py via .venv-madmom/bin/python.
"""

import json
import sys


def main() -> None:
    paths = json.load(sys.stdin)
    from BeatNet.BeatNet import BeatNet

    # mode='offline' + inference_model='DBN' is the whole-file condition, the
    # only one comparable with every other tool in the table; BeatNet's
    # 'realtime'/'PF' particle filter is a different (causal) task.
    est = BeatNet(1, mode="offline", inference_model="DBN", plot=[], thread=False,
                  device="cpu")
    out = {}
    for path in paths:
        try:
            res = est.process(path)
            # (n, 2) array of [time, beat_number]; column 0 is the beat grid.
            out[path] = [float(row[0]) for row in res] if len(res) else []
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            out[path] = {"error": f"{type(exc).__name__}: {exc}"}
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
