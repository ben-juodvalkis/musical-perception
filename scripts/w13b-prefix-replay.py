#!/usr/bin/env python3
"""W13(b) — the prefix-replay convergence twin.

Read-only over committed traces and case files: no media, no models, no
API key. Reproduce with `python scripts/w13b-prefix-replay.py`.

Replays every frozen trace on prefixes of its own evidence and records
when each output field's answer stops moving — the machine-side twin of
W13(a)'s owner convergence curve. Pre-registered 2026-08-31 (P1-P6).

Two conditions:
  A "granted"  — timed evidence truncated, Gemini's clip-level semantic
                 fields left whole-clip (the trace holds exactly one
                 whole-clip Gemini answer). Convergence times are a LOWER
                 bound on the pipeline's true time-to-commitment.
  B "withheld" — same truncation plus the clip-level Gemini fields
                 suppressed: the timing-only ablation, and the direct
                 probe of memo hypothesis H1.

Nothing under evals/ is written; no scorer/harness code is touched.
"""
from __future__ import annotations

import json
import statistics
import warnings
from dataclasses import replace
from pathlib import Path

from musical_perception.analyze import analyze
from musical_perception.bundle import PerceptionBundle
from musical_perception.evals.cases import load_cases
from musical_perception.evals.traces import replay_bundle

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT / "evals" / "traces"
CASES = ROOT / "evals" / "cases"
OUT_JSON = ROOT / "docs" / "research" / "w13b-prefix-convergence.json"
OUT_MD = ROOT / "docs" / "research" / "w13b-prefix-convergence.md"

NUMERIC_TOL = 0.04          # Standing Lesson 7: sub-4% is noise by construction
CONDITIONS = ("granted", "withheld")
COMMITTED_FIELDS = ("exercise", "meter", "grouping", "division", "tempo_bpm", "counts")
CHANNEL_FIELDS = ("onset_bpm", "marker_bpm")
FIELDS = COMMITTED_FIELDS + CHANNEL_FIELDS
NUMERIC_FIELDS = {"tempo_bpm", "onset_bpm", "marker_bpm"}
# Teacher-demo material, the closest thing in the corpus to W13(a)'s clip
# (a demo video, 37.8s). Named explicitly rather than tag-derived: the
# Barre-1 takes carry no `expect` labels, and the four originals are the
# only verified demo videos.
DEMO_VERIFIED = ("exercise-1-demo", "frappe", "grande-battement", "plies-demo")


# --- the prefix bundle -------------------------------------------------

def prefix_bundle(inner: PerceptionBundle, t: float | None, *,
                  withhold_semantics: bool) -> PerceptionBundle:
    """Wrap a replay bundle so it only knows the clip up to time `t`.

    A word counts as heard once it has finished (`end <= t`). Gemini's
    per-word classifications are filtered to the surviving transcript
    indices, so markers past the prefix vanish; its clip-level fields are
    kept (condition A) or suppressed (condition B). `t is None` = the
    full clip, used for the identity check.
    """
    full_words = inner.transcribe("replay")

    def transcribe(audio_path: str):
        if t is None:
            return list(full_words)
        return [w for w in full_words if w.end <= t]

    def analyze_media(media_path: str, *, onset_bpm=None, transcript_words=None):
        result = inner.analyze_media(media_path, onset_bpm=onset_bpm,
                                     transcript_words=transcript_words)
        n_kept = len(transcribe(media_path))
        words = [gw for gw in result.words
                 if gw.index is None or gw.index < n_kept]
        if any(gw.index is None for gw in result.words) and t is not None:
            # Legacy free-transcription traces pair by text, not index;
            # truncating the whisper list already drops late markers.
            words = result.words
        out = replace(result, words=words)
        if withhold_semantics:
            out = replace(out, exercise=None, counting_structure=None,
                          meter=None, quality=None, structure=None)
        return out

    return PerceptionBundle(transcribe=transcribe, analyze_media=analyze_media,
                            extract_landmarks=None)


def fields_of(params) -> dict:
    """The tracked answer at one prefix."""
    nt = params.normalized_tempo
    return {
        "exercise": (params.exercise.primary_exercise if params.exercise else None),
        "meter": (f"{params.meter.beats_per_measure}/{params.meter.beat_unit}"
                  if params.meter else None),
        "grouping": (nt.meter.beats_per_measure if nt and nt.meter else None),
        "division": (nt.subdivision if nt else None),
        "tempo_bpm": (float(nt.bpm) if nt else None),
        "counts": (params.structure.counts if params.structure else None),
        "onset_bpm": (float(params.onset_tempo.bpm) if params.onset_tempo else None),
        "marker_bpm": (float(params.tempo.bpm) if params.tempo else None),
    }


def matches(field: str, value, final) -> bool:
    if value is None or final is None:
        return value is None and final is None
    if field in NUMERIC_FIELDS:
        return abs(value - final) <= NUMERIC_TOL * abs(final)
    return value == final


def change_log(times: list[float], series: list[dict]) -> list[dict]:
    """Every time an answer moved: the curve itself, compactly."""
    log = []
    for i, (t, vals) in enumerate(zip(times, series)):
        prev = series[i - 1] if i else {f: None for f in FIELDS}
        for f in FIELDS:
            if not matches(f, vals[f], prev[f]):
                log.append({"t": round(t, 3), "field": f, "to": vals[f]})
    return log


def convergence(times: list[float], series: list[dict], final: dict) -> dict:
    """Per field: the earliest grid time from which the value never leaves
    the final value again. None when the final value is None (excluded)."""
    out = {}
    for f in FIELDS:
        if final[f] is None:
            out[f] = None
            continue
        t_star = None
        for i in range(len(times) - 1, -1, -1):
            if matches(f, series[i][f], final[f]):
                t_star = times[i]
            else:
                break
        out[f] = t_star
    return out


# --- the run -----------------------------------------------------------

def run_clip(name: str, condition: str) -> dict:
    inner, meta = replay_bundle(TRACES / name)
    words = inner.transcribe("replay")
    span = max((w.end for w in words), default=0.0)
    # t=0.0 is the honest zero point: what the pipeline answers before
    # any evidence at all (in condition A, Gemini's granted semantics).
    grid = [0.0] + sorted({w.end for w in words})
    withhold = condition == "withheld"

    def at(t):
        b = prefix_bundle(inner, t, withhold_semantics=withhold)
        return fields_of(analyze("replay.wav", bundle=b))

    series = [at(t) for t in grid]
    final_full = at(None)
    row = {
        "clip": name, "span": span, "n_grid": len(grid),
        "final": final_full,
        "identity_ok": (bool(grid) and series[-1] == final_full),
        "convergence": convergence(grid, series, final_full) if grid else {f: None for f in FIELDS},
    }
    row["changes"] = change_log(grid, series)
    row["n_changes"] = {f: sum(c["field"] == f for c in row["changes"]) for f in FIELDS}
    row["norm"] = {
        f: (None if row["convergence"][f] is None or not span
            else round(row["convergence"][f] / span, 4))
        for f in FIELDS
    }
    return row


def truth_for(cases) -> dict:
    """clip name -> (maturity, expect dict), read-only."""
    out = {}
    for c in cases:
        clip = Path(c.trace.rstrip("/")).name
        out[clip] = {"case": c.id, "provisional": c.provisional, "expect": c.expect}
    return out


def final_correct(field: str, final: dict, expect: dict) -> bool | None:
    """Is the clip's final answer right? None = no truth label for it."""
    if field == "tempo_bpm":
        exp = expect.get("marking_bpm") or expect.get("performance_bpm")
        if exp is None or final["tempo_bpm"] is None:
            return None
        return abs(final["tempo_bpm"] - exp) <= 0.08 * exp
    if field in ("meter", "grouping"):
        exp = expect.get("meter")
        if exp is None or final["meter"] is None:
            return None
        return final["meter"] == f"{exp.beats_per_measure}/{exp.beat_unit}"
    if field == "division":
        exp = expect.get("subdivision")
        return None if exp is None or final["division"] is None else final["division"] == exp
    if field == "counts":
        exp = expect.get("counts")
        return None if exp is None or final["counts"] is None else final["counts"] == exp
    return None


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None


def main() -> None:
    warnings.simplefilter("ignore")
    truth = truth_for(load_cases(CASES))
    clips = sorted(p.name for p in TRACES.iterdir() if (p / "whisper.json").is_file())

    results = {c: {} for c in CONDITIONS}
    for cond in CONDITIONS:
        for name in clips:
            results[cond][name] = run_clip(name, cond)
            print(f"  {cond:9s} {name:32s} "
                  f"grid={results[cond][name]['n_grid']:4d} "
                  f"identity={'ok' if results[cond][name]['identity_ok'] else 'MISMATCH'}")

    payload = {
        "generated_by": "scripts/w13b-prefix-replay.py",
        "numeric_tol": NUMERIC_TOL,
        "conditions": {
            "granted": "timed evidence truncated; Gemini clip-level fields whole-clip (LOWER bound)",
            "withheld": "timed evidence truncated; Gemini clip-level fields suppressed (ablation)",
        },
        "truth": {k: {"case": v["case"], "provisional": v["provisional"]} for k, v in truth.items()},
        "clips": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=1, default=str) + "\n")
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)}")
    write_report(results, truth)
    print(f"wrote {OUT_MD.relative_to(ROOT)}")


def slice_rows(results, truth, cond, want_provisional=None, only_correct=False):
    rows = []
    for name, row in results[cond].items():
        t = truth.get(name)
        if want_provisional is not None:
            if t is None or t["provisional"] != want_provisional:
                continue
        rows.append((name, row, t))
    if only_correct:
        rows = [(n, r, t) for n, r, t in rows if t is not None]
    return rows


def field_stats(rows, field, only_correct=False):
    vals, n_excl, n_wrongfinal = [], 0, 0
    for name, row, t in rows:
        v = row["norm"][field]
        if v is None:
            n_excl += 1
            continue
        if only_correct:
            ok = final_correct(field, row["final"], t["expect"]) if t else None
            if ok is not True:
                n_wrongfinal += 1
                continue
        vals.append(v)
    return {"n": len(vals), "median_norm": med(vals), "excluded": n_excl,
            "dropped_wrong_final": n_wrongfinal,
            "frac_before_30pct": (round(sum(v < 0.30 for v in vals) / len(vals), 3)
                                  if vals else None)}


def write_report(results, truth):
    from datetime import datetime, timezone
    L = []
    A = L.append
    A("# W13(b) — the machine's time-to-commitment curve\n")
    A(f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by "
      "`scripts/w13b-prefix-replay.py` (read-only over frozen traces; no media, "
      "no models, no API key). Pre-registered in the ledger, 2026-08-31.\n")
    A("Convergence time t\\* = the earliest prefix from which a field's answer "
      "never leaves its final value again; normalized by the clip's voiced span "
      "(last word end). Numeric fields match within 4% (Standing Lesson 7).\n")
    A("**Condition A (granted) times are a LOWER bound**: the frozen trace holds "
      "one whole-clip Gemini answer, so the semantic fields are granted at t=0. "
      "Condition B suppresses them entirely.\n")

    for cond in CONDITIONS:
        A(f"\n## Condition {cond}\n")
        for label, want in (("verified", False), ("provisional", True)):
            rows = slice_rows(results, truth, cond, want_provisional=want)
            A(f"\n### {label} slice (n={len(rows)} clips)\n")
            A("| field | n scored | median t\\*/span | converged before 30% of span "
              "| median answer moves | excluded (final None) |")
            A("|---|---|---|---|---|---|")
            for f in FIELDS:
                s = field_stats(rows, f)
                nch = med([r["n_changes"][f] for _, r, _ in rows])
                A(f"| `{f}` | {s['n']} | {s['median_norm']} | {s['frac_before_30pct']} "
                  f"| {nch} | {s['excluded']} |")
        rows = slice_rows(results, truth, cond, want_provisional=False)
        A(f"\n### verified slice, clips whose FINAL answer is correct\n")
        A("| field | n right | median t\\*/span | dropped (final wrong/unlabelled) |")
        A("|---|---|---|---|")
        for f in COMMITTED_FIELDS:
            if f == "exercise":
                A("| `exercise` | — | — | no truth label in any case file |")
                continue
            s = field_stats(rows, f, only_correct=True)
            A(f"| `{f}` | {s['n']} | {s['median_norm']} | {s['dropped_wrong_final']} |")

    A("\n## Granted vs withheld: does Gemini's clip-level meter change the answer?\n")
    A("Per field, over all 52 clips: how many end at a DIFFERENT final value "
      "when Gemini's clip-level fields are suppressed, and how the median "
      "convergence time moves. This is the P5 probe.\n")
    A("| field | clips with both finals non-None | different final | median t\\*/span granted | withheld |")
    A("|---|---|---|---|---|")
    for f in FIELDS:
        both, diff, g_vals, w_vals = 0, 0, [], []
        for name in results["granted"]:
            gf, wf = results["granted"][name]["final"][f], results["withheld"][name]["final"][f]
            if gf is not None and wf is not None:
                both += 1
                if not matches(f, wf, gf):
                    diff += 1
            if results["granted"][name]["norm"][f] is not None:
                g_vals.append(results["granted"][name]["norm"][f])
            if results["withheld"][name]["norm"][f] is not None:
                w_vals.append(results["withheld"][name]["norm"][f])
        A(f"| `{f}` | {both} | {diff} | {med(g_vals)} | {med(w_vals)} |")

    A("\n## Absolute seconds, teacher-demo material (the W13(a) comparison)\n")
    A("W13(a)'s clip was a 37.8s demo video. The owner committed at: exercise "
      "~3s, meter ~3s, tempo ~9-12s, quality ~9-12s, structure ~30-33s.\n")
    A("| subset | field | median t\\* (s) | median span (s) | n |")
    A("|---|---|---|---|---|")
    demo_prov = tuple(sorted(n for n in results["granted"] if n.startswith("barre1-") and n.endswith("-d")))
    for label, names in (("verified demo videos", DEMO_VERIFIED),
                         ("Barre-1 demo takes (provisional)", demo_prov)):
        for f in COMMITTED_FIELDS:
            secs = [results["granted"][n]["convergence"][f] for n in names
                    if results["granted"][n]["convergence"][f] is not None]
            spans = [results["granted"][n]["span"] for n in names]
            A(f"| {label} | `{f}` | {med(secs)} | {med(spans)} | {len(secs)}/{len(names)} |")

    A("\n## Per-clip convergence (condition granted, seconds)\n")
    A("| clip | span | " + " | ".join(f"`{f}`" for f in FIELDS) + " | maturity |")
    A("|---" * (len(FIELDS) + 3) + "|")
    for name, row in results["granted"].items():
        t = truth.get(name)
        mat = "—" if t is None else ("provisional" if t["provisional"] else "verified")
        cells = " | ".join(
            "—" if row["convergence"][f] is None else f"{row['convergence'][f]:.1f}"
            for f in FIELDS)
        A(f"| {name} | {row['span']:.1f} | {cells} | {mat} |")

    A("\n## Identity check (P1)\n")
    bad = [(c, n) for c in CONDITIONS for n, r in results[c].items() if not r["identity_ok"]]
    A(f"Clips where the full prefix reproduced the untruncated replay exactly: "
      f"{sum(len(results[c]) for c in CONDITIONS) - len(bad)} / "
      f"{sum(len(results[c]) for c in CONDITIONS)}.")
    if bad:
        A("\nMismatches: " + ", ".join(f"{c}/{n}" for c, n in bad))
    OUT_MD.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
