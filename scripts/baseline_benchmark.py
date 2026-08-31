"""Rung 6 / W3 — the Review-4 off-the-shelf beat-tracker benchmark.

Runs the Review-4 §(a) tool plan against the owner-verified beat grids and
writes `docs/research/baseline-benchmark.{json,md}`.

Two input conditions, per the rung's condition:

* **raw** — the clip's own media, decoded to 22.05 kHz mono wav.
* **markers** — a click track synthesised at the Whisper word-start times
  from the clip's frozen trace. Every tool in the plan takes audio, so the
  marker stream is realised *as* audio; that keeps the two conditions
  comparable tool-for-tool and needs no media file, which is why this
  condition covers all 30 DEV clips and `raw` covers only the 6 whose media
  is on this machine.

Nothing here writes to `evals/`. Scoring imports `mir_eval` for the beat
metrics and the *existing* tier-1 tempo helpers from
`musical_perception.evals.aggregate` (read-only import, so Acc1/Acc2/OE
mean exactly what they mean in tier-1 reporting) — no scorer code is
modified.

AMLt-with-triples is computed here rather than taken from mir_eval:
mir_eval's allowed metric variations are duple-only (original, off-beat,
double, half-odd, half-even), so on a corpus containing 3/4, 6/8 and
triplet subdivisions its AMLt understates agreement by construction. See
`amlt_with_triples`.

Usage:
    python scripts/baseline_benchmark.py [--tools a,b] [--conditions raw,markers]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from musical_perception.evals.aggregate import (  # noqa: E402  read-only import
    ACC_TOL_HOUSE,
    ACC_TOL_STANDARD,
    acc1,
    acc2,
    octave_errors,
)

GRID_DIR = REPO / "evals" / "grids"
TRACE_DIR = REPO / "evals" / "traces"
CASE_DIR = REPO / "evals" / "cases"
OUT_JSON = REPO / "docs" / "research" / "baseline-benchmark.json"
OUT_MD = REPO / "docs" / "research" / "baseline-benchmark.md"
MADMOM_PY = REPO / ".venv-madmom" / "bin" / "python"

SR = 22050
HOP = 512
F_TOL = 0.07          # Review 2 §4.3 / mir_eval default
CLICK_FREQ = 1000.0


# ---------------------------------------------------------------- references


@dataclass
class Clip:
    clip_id: str
    provisional: bool
    beats: np.ndarray
    media: str | None
    annotation_method: str | None
    count_style: str | None
    truth_meter: str | None
    markers: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def ref_bpm(self) -> float | None:
        """Tempo the grid itself implies (median inter-beat interval).

        Deliberately grid-derived rather than the case file's
        `marking_bpm`: these metrics score a tool against *these* beats, so
        the tempo they are graded on must be the tempo of the same
        annotation. The two agree closely on verified grids; where they do
        not, the ledger's `marking_bpm` rulings are about the metronome,
        not the annotation.
        """
        if self.beats.size < 3:
            return None
        return float(60.0 / np.median(np.diff(self.beats)))


def load_clips() -> list[Clip]:
    clips = []
    for path in sorted(GRID_DIR.glob("*.yaml")):
        g = yaml.safe_load(path.read_text())
        clip_id = g["clip"]
        beats = np.array(sorted(float(b) for b in g.get("beats", [])))
        case_path = CASE_DIR / f"{clip_id}.yaml"
        count_style = truth_meter = None
        if case_path.exists():
            case = yaml.safe_load(case_path.read_text())
            count_style = (case.get("tags") or {}).get("count_style")
            truth_meter = (case.get("expect") or {}).get("meter")
        clips.append(
            Clip(
                clip_id=clip_id,
                provisional=bool(g.get("provisional", True)),
                beats=beats,
                media=g.get("media"),
                annotation_method=g.get("annotation_method"),
                count_style=count_style,
                truth_meter=truth_meter,
                markers=_markers(clip_id, g.get("media")),
            )
        )
    return clips


_TRACE_BY_MEDIA: dict[str, Path] | None = None


def _trace_dir(clip_id: str, media: str | None) -> Path | None:
    """Locate a clip's trace directory.

    The five pre-rig grids carry an ADR-prefixed clip id
    (`adr006-8-counts-2x`) while their trace directories are named for the
    media (`8-counts-2x`), so id matching alone silently drops them. Media
    path is the join key that holds for both.
    """
    global _TRACE_BY_MEDIA
    direct = TRACE_DIR / clip_id
    if (direct / "whisper.json").exists():
        return direct
    if _TRACE_BY_MEDIA is None:
        _TRACE_BY_MEDIA = {}
        for meta in TRACE_DIR.glob("*/meta.json"):
            m = json.loads(meta.read_text()).get("media")
            if m:
                _TRACE_BY_MEDIA[m] = meta.parent
    return _TRACE_BY_MEDIA.get(media) if media else None


def _markers(clip_id: str, media: str | None) -> np.ndarray:
    """Whisper word starts from the frozen trace — the pipeline's marker stream."""
    tdir = _trace_dir(clip_id, media)
    if tdir is None:
        return np.array([])
    wpath = tdir / "whisper.json"
    if not wpath.exists():
        return np.array([])
    words = json.loads(wpath.read_text()).get("words", [])
    return np.array(sorted(float(w["start"]) for w in words if w.get("start") is not None))


# -------------------------------------------------------------------- audio


def raw_wav(clip: Clip, cache: Path) -> Path | None:
    """Decode the clip's media to mono wav, or None if it is not on this machine."""
    if not clip.media:
        return None
    src = REPO / clip.media
    if not src.exists():
        return None
    dst = cache / f"{clip.clip_id}.raw.wav"
    if not dst.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR), str(dst)],
            check=True,
            capture_output=True,
        )
    return dst


def marker_wav(clip: Clip, cache: Path) -> Path | None:
    """Click track at the marker times — the marker-stream condition as audio."""
    import librosa
    import soundfile as sf

    if clip.markers.size == 0:
        return None
    dst = cache / f"{clip.clip_id}.markers.wav"
    if not dst.exists():
        end = max(float(clip.markers[-1]), float(clip.beats[-1]) if clip.beats.size else 0.0)
        length = int((end + 1.0) * SR)
        y = librosa.clicks(times=clip.markers, sr=SR, click_freq=CLICK_FREQ, length=length)
        sf.write(dst, y, SR)
    return dst


# ----------------------------------------------------------------- trackers


def _bpm_from_beats(beats: np.ndarray) -> float | None:
    if beats.size < 3:
        return None
    return float(60.0 / np.median(np.diff(beats)))


def track_librosa_dp(path: Path) -> tuple[np.ndarray, float | None]:
    import librosa

    y, sr = librosa.load(path, sr=SR, mono=True)
    tempo, beats = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=HOP, start_bpm=100.0, tightness=100, units="time"
    )
    return np.asarray(beats, dtype=float), float(np.atleast_1d(tempo)[0])


def track_librosa_plp(path: Path) -> tuple[np.ndarray, float | None]:
    import librosa

    y, sr = librosa.load(path, sr=SR, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP, aggregate=np.median)
    pulse = librosa.beat.plp(
        onset_envelope=env, sr=sr, hop_length=HOP, tempo_min=40, tempo_max=208
    )
    peaks = np.flatnonzero(librosa.util.localmax(pulse) & (pulse > 0.5 * pulse.max()))
    beats = librosa.frames_to_time(peaks, sr=sr, hop_length=HOP)
    return np.asarray(beats, dtype=float), _bpm_from_beats(beats)


def track_beat_this(path: Path) -> tuple[np.ndarray, float | None]:
    from beat_this.inference import File2Beats

    global _F2B
    try:
        f2b = _F2B
    except NameError:
        f2b = _F2B = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    beats, _downbeats = f2b(str(path))
    beats = np.asarray(beats, dtype=float)
    return beats, _bpm_from_beats(beats)


def track_essentia(path: Path) -> tuple[np.ndarray, float | None]:
    import essentia.standard as es

    audio = es.MonoLoader(filename=str(path), sampleRate=44100)()
    bpm, ticks, _conf, _est, _ivals = es.RhythmExtractor2013(
        method="multifeature", minTempo=40, maxTempo=208
    )(audio)
    return np.asarray(ticks, dtype=float), float(bpm)


def track_nuclei_hybrid(path: Path) -> tuple[np.ndarray, float | None]:
    """Review 4 §(a) item 5: syllable nuclei -> sparse envelope -> librosa DP.

    Uses this project's own peakRate nuclei extractor (rung 2) as the front
    end, which is the domain-native baseline the review argues for.
    """
    import librosa

    from musical_perception.precision.pulse import acoustic_pulse_events

    y, sr = librosa.load(path, sr=SR, mono=True)
    events = acoustic_pulse_events(y, sr)
    n_frames = 1 + len(y) // HOP
    env = np.zeros(n_frames, dtype=float)
    for t in events:
        frame = int(round(float(t) * sr / HOP))
        if 0 <= frame < n_frames:
            env[frame] = 1.0
    if env.sum() < 3:
        return np.array([]), None
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=env, sr=sr, hop_length=HOP, start_bpm=100.0, tightness=100,
        units="time",
    )
    return np.asarray(beats, dtype=float), float(np.atleast_1d(tempo)[0])


TRACKERS = {
    "librosa_dp": track_librosa_dp,
    "librosa_plp": track_librosa_plp,
    "beat_this": track_beat_this,
    "essentia_re2013": track_essentia,
    "nuclei_hybrid": track_nuclei_hybrid,
    # madmom_dbn and beatnet are handled out-of-process; see run_worker.
}
OUT_OF_PROCESS = {"madmom_dbn": "madmom_worker.py", "beatnet": "beatnet_worker.py"}
ALL_TOOLS = list(TRACKERS) + list(OUT_OF_PROCESS)


def run_worker(
    worker: str, wavs: dict[str, Path]
) -> dict[str, tuple[np.ndarray, float | None]]:
    """Batch an out-of-process tracker in the dedicated venv.

    madmom and BeatNet both need `.venv-madmom` (BeatNet's inference stage
    *is* madmom's DBN), and both are batched one subprocess per tool so the
    model loads once rather than per clip.
    """
    if not MADMOM_PY.exists():
        raise RuntimeError(f"madmom venv absent at {MADMOM_PY}")
    paths = [str(p) for p in wavs.values()]
    proc = subprocess.run(
        [str(MADMOM_PY), str(REPO / "scripts" / worker)],
        input=json.dumps(paths),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{worker} exit {proc.returncode}: {proc.stderr[-400:]}")
    raw = json.loads(proc.stdout)
    out = {}
    for key, path in wavs.items():
        got = raw.get(str(path))
        if isinstance(got, dict):
            raise RuntimeError(got.get("error", "unknown madmom error"))
        beats = np.asarray(got or [], dtype=float)
        out[key] = (beats, _bpm_from_beats(beats))
    return out


# ------------------------------------------------------------------ scoring


def amlt_with_triples(reference: np.ndarray, estimated: np.ndarray) -> float:
    """AMLt extended with triple/third metric variations.

    mir_eval's AMLt maximises continuity over duple variations only. A
    corpus with 3/4, 6/8 and triplet subdivisions needs the triple family
    too, or a tool that locks onto the triplet level is scored as wrong for
    a reason that is purely an artefact of the metric's variation set. Each
    extra variation is scored with mir_eval's own continuity loop, so the
    only thing added here is the candidate set.
    """
    import mir_eval.beat as B

    if reference.size < 2 or estimated.size < 2:
        return 0.0
    best = B.continuity(reference, estimated)[3]      # mir_eval's own AMLt
    idx = np.arange(0, reference.shape[0] - 1e-9, 1.0 / 3.0)
    triple = np.interp(idx, np.arange(reference.shape[0]), reference)
    variations = [triple]
    for phase in range(3):
        third = reference[phase::3]
        if third.size >= 2:
            variations.append(third)
    for var in variations:
        if var.size < 2:
            continue
        try:
            best = max(best, B.continuity(var, estimated)[1])
        except ValueError:
            continue
    return float(best)


def score(clip: Clip, beats: np.ndarray, bpm: float | None) -> dict:
    import mir_eval.beat as B

    ref = clip.beats
    est = np.asarray(sorted(float(b) for b in beats), dtype=float)
    row: dict = {
        "n_ref": int(ref.size),
        "n_est": int(est.size),
        "est_beats": [round(float(b), 4) for b in est],
    }
    if ref.size < 2 or est.size < 2:
        row.update(f_measure=0.0, cmlt=0.0, amlt=0.0, amlt_triples=0.0)
    else:
        ref_t, est_t = B.trim_beats(ref), B.trim_beats(est)
        if ref_t.size < 2 or est_t.size < 2:
            row.update(f_measure=0.0, cmlt=0.0, amlt=0.0, amlt_triples=0.0)
        else:
            row["f_measure"] = float(B.f_measure(ref_t, est_t, f_measure_threshold=F_TOL))
            # Untrimmed twin: mir_eval's trim_beats drops everything before
            # 5 s (MIREX convention), but the stage-1 suite does not trim.
            # Without this column the comparison against the pipeline's own
            # pulse F would be off-by-a-convention rather than a measurement.
            row["f_measure_untrimmed"] = float(
                B.f_measure(ref, est, f_measure_threshold=F_TOL)
            )
            _cmlc, cmlt, _amlc, amlt = B.continuity(ref_t, est_t)
            row["cmlt"] = float(cmlt)
            row["amlt"] = float(amlt)
            row["amlt_triples"] = amlt_with_triples(ref_t, est_t)
    ref_bpm = clip.ref_bpm
    if bpm and ref_bpm:
        oe1, oe2 = octave_errors(bpm, ref_bpm)
        row.update(
            est_bpm=round(float(bpm), 2),
            ref_bpm=round(ref_bpm, 2),
            acc1_04=bool(acc1(bpm, ref_bpm, ACC_TOL_STANDARD)),
            acc1_08=bool(acc1(bpm, ref_bpm, ACC_TOL_HOUSE)),
            acc2_04=bool(acc2(bpm, ref_bpm, ACC_TOL_STANDARD)),
            acc2_08=bool(acc2(bpm, ref_bpm, ACC_TOL_HOUSE)),
            oe1=round(oe1, 4),
            oe2=round(oe2, 4),
        )
    return row


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    def rate(key):
        vals = [r[key] for r in rows if key in r]
        return round(float(np.mean(vals)), 3) if vals else None

    oe2s = [abs(r["oe2"]) for r in rows if "oe2" in r]
    return {
        "n": len(rows),
        "f_measure": mean("f_measure"),
        "f_measure_untrimmed": mean("f_measure_untrimmed"),
        "cmlt": mean("cmlt"),
        "amlt": mean("amlt"),
        "amlt_triples": mean("amlt_triples"),
        "acc1_04": rate("acc1_04"),
        "acc1_08": rate("acc1_08"),
        "acc2_04": rate("acc2_04"),
        "acc2_08": rate("acc2_08"),
        "oe2_abs_median": round(float(np.median(oe2s)), 3) if oe2s else None,
        "n_tempo": len(oe2s),
    }


# --------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", default=",".join(ALL_TOOLS))
    ap.add_argument("--conditions", default="raw,markers")
    args = ap.parse_args()
    tools = [t for t in args.tools.split(",") if t]
    conditions = [c for c in args.conditions.split(",") if c]

    clips = load_clips()
    verified = [c for c in clips if not c.provisional]
    print(f"{len(clips)} grids · {len(verified)} verified · tools={tools}")

    cache = Path(tempfile.gettempdir()) / "mp-baseline-cache"
    cache.mkdir(exist_ok=True)

    wavs: dict[str, dict[str, Path]] = {"raw": {}, "markers": {}}
    for clip in clips:
        if "raw" in conditions:
            p = raw_wav(clip, cache)
            if p:
                wavs["raw"][clip.clip_id] = p
        if "markers" in conditions:
            p = marker_wav(clip, cache)
            if p:
                wavs["markers"][clip.clip_id] = p
    for cond in conditions:
        print(f"  condition {cond}: {len(wavs[cond])} clips with input")

    by_id = {c.clip_id: c for c in clips}
    results: dict = {"per_clip": [], "failures": {}}

    for cond in conditions:
        for tool in tools:
            t0 = time.time()
            try:
                if tool in OUT_OF_PROCESS:
                    outputs = run_worker(OUT_OF_PROCESS[tool], wavs[cond])
                else:
                    fn = TRACKERS[tool]
                    outputs = {cid: fn(p) for cid, p in wavs[cond].items()}
            except Exception as exc:  # noqa: BLE001 - the rung asks for exact failures
                results["failures"][f"{tool}/{cond}"] = f"{type(exc).__name__}: {exc}"
                print(f"  FAIL {tool}/{cond}: {type(exc).__name__}: {exc}")
                continue
            for cid, (beats, bpm) in outputs.items():
                clip = by_id[cid]
                row = score(clip, beats, bpm)
                row.update(
                    clip=cid, tool=tool, condition=cond,
                    provisional=clip.provisional, count_style=clip.count_style,
                    truth_meter=clip.truth_meter,
                )
                results["per_clip"].append(row)
            print(f"  {tool}/{cond}: {len(outputs)} clips in {time.time() - t0:.1f}s")

    # aggregates: verified-only headline, provisional reported separately
    agg: dict = {}
    for cond in conditions:
        for tool in tools:
            sel = [r for r in results["per_clip"] if r["tool"] == tool and r["condition"] == cond]
            ver = [r for r in sel if not r["provisional"]]
            agg[f"{tool}/{cond}"] = {
                "verified": aggregate(ver),
                "provisional": aggregate([r for r in sel if r["provisional"]]),
                "by_count_style": {
                    style: aggregate([r for r in ver if r["count_style"] == style])
                    for style in sorted({r["count_style"] for r in ver if r["count_style"]})
                },
            }
    results["aggregates"] = agg
    results["meta"] = {
        "f_tolerance_s": F_TOL,
        "sr": SR,
        "n_grids": len(clips),
        "n_verified": len(verified),
        "conditions": {c: sorted(wavs[c]) for c in conditions},
        "tools": tools,
    }
    OUT_JSON.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {OUT_JSON.relative_to(REPO)}")
    print_summary(results, tools, conditions)
    render_md(results)
    print(f"wrote {OUT_MD.relative_to(REPO)}")
    return 0


def _fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def render_md(results: dict) -> None:
    """Render the committed results document from the JSON."""
    meta = results["meta"]
    tools, conditions = meta["tools"], list(meta["conditions"])
    lines = [
        "# Baseline benchmark — off-the-shelf beat trackers on the DEV grids",
        "",
        "Rung 6 / marathon workstream W3. Generated by "
        "`scripts/baseline_benchmark.py`; the raw per-clip rows are in",
        "`baseline-benchmark.json`. Reference is the owner-verified beat grid "
        f"({meta['n_verified']} of {meta['n_grids']} grids verified; the two "
        "provisional grids are aggregated separately and gate nothing).",
        "",
        "**Conditions.** `raw` = the clip's own media. `markers` = a click "
        "track at the frozen trace's Whisper word-start times, so every tool "
        "sees the marker stream on the same scale it sees audio.",
        "",
        f"`raw` covers **{len(meta['conditions'].get('raw', []))} of "
        f"{meta['n_grids']}** clips and `markers` covers "
        f"**{len(meta['conditions'].get('markers', []))} of {meta['n_grids']}**. "
        "The 2026-08-21 run reached only 6 raw clips because `audio/rig/*.mp3` "
        "was not on the runner; those 24 files were committed on 2026-08-28, "
        "so both conditions now cover the same rows and are comparable "
        "tool-for-tool. That comparison was invalid as printed in the "
        "2026-08-21 table and is the main thing this run adds.",
        "",
        "**`essentia_re2013` is non-deterministic and its numbers are single "
        "draws.** Three back-to-back calls on one *markers* wav returned 93.8, "
        "107.8 and 121.9 BPM. Repeated whole-suite passes average out (raw "
        "F 0.697-0.706 over 5 passes, sd 0.004; markers F 0.418-0.424), so the "
        "aggregates below are usable, but no single Essentia cell should be "
        "quoted as a measurement. Every other tool here is bit-identical "
        "across repeat runs.",
        "",
        "## Summary — verified grids only",
        "",
        "| tool | condition | n | F@70ms | CMLt | AMLt | AMLt+triples | Acc1@4% | Acc2@4% | \\|OE2\\| median |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cond in conditions:
        for tool in tools:
            a = results["aggregates"].get(f"{tool}/{cond}", {}).get("verified") or {}
            if not a:
                lines.append(f"| `{tool}` | {cond} | — | — | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| `{tool}` | {cond} | {a['n']} | {_fmt(a['f_measure'])} | "
                f"{_fmt(a['cmlt'])} | {_fmt(a['amlt'])} | {_fmt(a['amlt_triples'])} | "
                f"{_fmt(a['acc1_04'])} | {_fmt(a['acc2_04'])} | "
                f"{_fmt(a['oe2_abs_median'])} |"
            )
    lines += ["", "## By count style — verified grids, per condition", ""]
    for cond in conditions:
        styles = sorted(
            {
                s
                for tool in tools
                for s in (results["aggregates"].get(f"{tool}/{cond}", {}).get("by_count_style") or {})
            }
        )
        if not styles:
            continue
        lines += [
            f"### condition `{cond}` — mean F@70ms",
            "",
            "| tool | " + " | ".join(styles) + " |",
            "| --- | " + " | ".join("---:" for _ in styles) + " |",
        ]
        for tool in tools:
            by = results["aggregates"].get(f"{tool}/{cond}", {}).get("by_count_style") or {}
            cells = []
            for s in styles:
                a = by.get(s) or {}
                cells.append(f"{_fmt(a.get('f_measure'))} (n={a['n']})" if a else "—")
            lines.append(f"| `{tool}` | " + " | ".join(cells) + " |")
        lines.append("")
    if results["failures"]:
        lines += ["## Tools that did not run", ""]
        for k, v in sorted(results["failures"].items()):
            lines.append(f"- `{k}` — `{v}`")
        lines.append("")
    lines += [
        "## Install notes (exact failures, per the rung condition)",
        "",
        "- **librosa 1.0.0, Essentia 2.1b6.dev1389, Beat This! 1.1.0, "
        "mir_eval 0.8.2** — installed into the project venv without incident.",
        "- **madmom 0.16.1 (PyPI) FAILED twice, and is not what ran.** First: "
        "`ModuleNotFoundError: No module named 'Cython'` under build "
        "isolation. With Cython present it builds, then fails at import with "
        "`ModuleNotFoundError: No module named 'pkg_resources'` "
        "(setuptools 84 removed it), and with `setuptools<81` fails at "
        "`ImportError: cannot import name 'MutableSequence' from "
        "'collections'` — the Python 3.10 removal, in a package last "
        "released in Nov 2017. Review 4 §d predicted exactly this.",
        "- **madmom git main works on Python 3.12** and is what these numbers "
        "come from, installed in a dedicated `.venv-madmom` (gitignored) and "
        "driven out-of-process by `scripts/madmom_worker.py`. It reports "
        "`0.17.dev0` and — contra Review 4 §d, which is stale on this point "
        "— installs happily against **numpy 2.5.2**, so the `numpy<2` pin is "
        "no longer a reason to avoid it.",
        "- **BeatNet** was not attempted (Review 4 lists it optional, "
        "conditional on the madmom venv existing; it now does, so it is a "
        "cheap follow-up).",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


def print_summary(results: dict, tools: list[str], conditions: list[str]) -> None:
    print("\nverified-grid means (F@70ms / CMLt / AMLt / AMLt+triples / Acc1@4% / Acc2@4%)")
    header = f"{'tool':<18}{'cond':<9}{'n':>3}  {'F':>6}{'CMLt':>7}{'AMLt':>7}{'AMLt3':>7}{'Acc1':>7}{'Acc2':>7}"
    print(header)
    print("-" * len(header))
    for cond in conditions:
        for tool in tools:
            a = results["aggregates"].get(f"{tool}/{cond}", {}).get("verified") or {}
            if not a:
                print(f"{tool:<18}{cond:<9}{'--':>3}   (no result)")
                continue
            def f(k):
                v = a.get(k)
                return "  n/a" if v is None else f"{v:6.3f}"
            print(
                f"{tool:<18}{cond:<9}{a['n']:>3}  {f('f_measure')}{f('cmlt')}"
                f"{f('amlt')}{f('amlt_triples')}{f('acc1_04')}{f('acc2_04')}"
            )
    if results["failures"]:
        print("\nfailures:")
        for k, v in results["failures"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    raise SystemExit(main())
