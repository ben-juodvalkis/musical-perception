#!/usr/bin/env python3
"""W14 — the commitment stopping rule.

Scores two families of stopping rule over the frozen W13(b) prefix replay:

  F1  k-stable-prefixes — commit once the answer has not moved for k
      consecutive grid points.
  F2  confidence >= theta — commit once the confidence the pipeline
      ALREADY computes for that field first reaches theta.

Read-only over `docs/research/w13b-prefix-convergence.json`: no media, no
models, no API key, nothing under `evals/` read or written, no scorer or
harness code touched. REPORTED-ONLY — this pins no outcome and is wired
into no pipeline path (Standing Lesson 9). Pre-registered in the ledger,
2026-08-31 (P1-P7). Reproduce with `python scripts/w14-stopping-rule.py`.
"""
from __future__ import annotations

import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# W14-c: see the note in w13b-prefix-replay.py. Empty by default.
_SUF = os.environ.get("MP_ARTIFACT_SUFFIX", "")
IN_JSON = ROOT / "docs" / "research" / f"w13b-prefix-convergence{_SUF}.json"
OUT_JSON = ROOT / "docs" / "research" / f"w14-stopping-rule{_SUF}.json"
OUT_MD = ROOT / "docs" / "research" / f"w14-stopping-rule{_SUF}.md"

NUMERIC_TOL = 0.04
NUMERIC_FIELDS = {"tempo_bpm", "onset_bpm", "marker_bpm"}
FIELDS = ("exercise", "meter", "grouping", "division", "tempo_bpm", "counts",
          "onset_bpm", "marker_bpm")
CONDITIONS = ("granted", "withheld")
K_SWEEP = tuple(range(1, 9))
THETA_SWEEP = tuple(round(0.10 + 0.05 * i, 2) for i in range(17))  # 0.10 .. 0.90
PREMATURE_CEILING = 0.10    # pre-registered, not tuned

# W13(a), the owner's own curve on a 37.8s demo clip.
OWNER_SPAN = 37.8
OWNER_CURVE = {"exercise": 3.0, "meter": 3.0, "tempo_bpm": 10.5, "counts": 31.5}


def matches(field: str, a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if field in NUMERIC_FIELDS:
        return abs(a - b) <= NUMERIC_TOL * abs(b)
    return a == b


def series_of(row: dict, field: str) -> list:
    """The per-prefix answer for one field.

    Numeric fields come from the recorded `series_num` stream: the published
    `changes` log is a >4%-move log and is LOSSY for them (sub-threshold
    drift accumulates unrecorded). Non-numeric fields reconstruct exactly
    from the change log, and the reconstruction is checked in `main`.
    """
    if field in NUMERIC_FIELDS:
        return row["series_num"][field]
    grid = row["grid"]
    moves = {c["t"]: c["to"] for c in row["changes"] if c["field"] == field}
    out, cur = [], None
    for t in grid:
        if t in moves:
            cur = moves[t]
        out.append(cur)
    return out


def commit_k(values: list, k: int) -> int | None:
    """First index whose non-None answer has held for k consecutive points."""
    for i in range(k - 1, len(values)):
        if values[i] is None:
            continue
        if all(_eq(values[j], values[i]) for j in range(i - k + 1, i)):
            return i
    return None


def _eq(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) <= NUMERIC_TOL * abs(b) if b else a == b
    return a == b


def commit_theta(values: list, confs: list, theta: float) -> int | None:
    for i, (v, c) in enumerate(zip(values, confs)):
        if v is not None and c is not None and c >= theta:
            return i
    return None


def score(rows, field, commit_fn, conf_key=None):
    """rows: list of (clip, row). Returns the metric block for one setting."""
    n_elig = premature = no_commit = 0
    times = []
    for clip, row in rows:
        final = row["final"][field]
        if final is None:
            continue                      # no answer at all: same exclusion as W13(b)
        n_elig += 1
        values = series_of(row, field)
        i = commit_fn(values, row) if conf_key is None else commit_fn(values, row)
        if i is None:
            no_commit += 1
            continue
        if not matches(field, values[i], final):
            premature += 1
        span = row["span"]
        if span:
            times.append(row["grid"][i] / span)
    committed = n_elig - no_commit
    return {
        "n": n_elig,
        "committed": committed,
        "no_commit_rate": round(no_commit / n_elig, 4) if n_elig else None,
        "premature_rate": round(premature / committed, 4) if committed else None,
        "premature_n": premature,
        "median_commit_norm": round(statistics.median(times), 4) if times else None,
    }


def best_point(sweep_block: dict) -> tuple | None:
    """Pre-registered rule: smallest median commit time among settings whose
    premature rate is <= the ceiling. None = the family has no operating
    point, which is a result, not a reason to relax the ceiling."""
    ok = [(k, m) for k, m in sweep_block.items()
          if m["premature_rate"] is not None
          and m["premature_rate"] <= PREMATURE_CEILING
          and m["median_commit_norm"] is not None]
    return min(ok, key=lambda kv: kv[1]["median_commit_norm"]) if ok else None


def main() -> None:
    data = json.loads(IN_JSON.read_text())
    truth = data["truth"]
    conf_src = data["conf_streams"]

    slices = {}
    for cond in CONDITIONS:
        rows = list(data["clips"][cond].items())
        slices[(cond, "verified")] = [(c, r) for c, r in rows
                                      if c in truth and not truth[c]["provisional"]]
        slices[(cond, "provisional")] = [(c, r) for c, r in rows
                                         if c in truth and truth[c]["provisional"]]

    # self-check: the rebuilt series must end on the published final answer
    bad = []
    for cond in CONDITIONS:
        for clip, row in data["clips"][cond].items():
            for f in FIELDS:
                if not matches(f, series_of(row, f)[-1], row["final"][f]):
                    bad.append(f"{cond}/{clip}/{f}")
    print(f"series reconstruction check: {'OK' if not bad else 'MISMATCH ' + str(bad[:5])}")

    results = {"f1": {}, "f2": {}}
    for (cond, mat), rows in slices.items():
        for f in FIELDS:
            results["f1"].setdefault(cond, {}).setdefault(mat, {})[f] = {
                str(k): score(rows, f, lambda v, r, k=k: commit_k(v, k))
                for k in K_SWEEP
            }
            src = conf_src[f]
            if src is None:
                results["f2"].setdefault(cond, {}).setdefault(mat, {})[f] = None
                continue
            results["f2"].setdefault(cond, {}).setdefault(mat, {})[f] = {
                str(th): score(rows, f,
                               lambda v, r, th=th, src=src: commit_theta(v, r["conf"][src], th))
                for th in THETA_SWEEP
            }

    payload = {
        "generated_by": "scripts/w14-stopping-rule.py",
        "source": "docs/research/w13b-prefix-convergence.json",
        "premature_ceiling": PREMATURE_CEILING,
        "k_sweep": list(K_SWEEP), "theta_sweep": list(THETA_SWEEP),
        "conf_streams": conf_src,
        "reconstruction_mismatches": bad,
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    write_report(results, slices, conf_src, slices[("granted", "verified")])
    print(f"wrote {OUT_MD.relative_to(ROOT)}")


def fmt(x):
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def write_report(results, slices, conf_src, conf_rows):
    L, A = [], None
    A = L.append
    A("# W14 — the commitment stopping rule\n")
    A(f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by "
      "`scripts/w14-stopping-rule.py`, read-only over the W13(b) prefix replay "
      "(no media, no models, no API key). Pre-registered in the ledger, "
      "2026-08-31.\n")
    A("**REPORTED-ONLY.** Nothing in `src/` changes, no eval suite gains a "
      "metric, no outcome is pinned. Two families are scored:\n")
    A("- **F1 k-stable-prefixes** — commit once the answer has held for `k` "
      "consecutive grid points (k = 1..8).")
    A("- **F2 confidence ≥ θ** — commit once the confidence the pipeline "
      "already computes first reaches θ (θ = 0.10..0.90).\n")
    A(f"**Operating point** = smallest median commit time among settings whose "
      f"premature-commit rate is ≤ **{PREMATURE_CEILING:.2f}** on the slice. "
      "The ceiling was fixed before any number was seen; a family with no "
      "qualifying setting has **no operating point**, and that is the result.\n")
    A("Premature-commit rate is computed over the clips where the rule fired; "
      "`no-commit` is reported separately. Clips whose final answer for a "
      "field is `None` are excluded, the same exclusion W13(b) used. Commit is "
      "only permitted on a non-`None` answer: a rule that \"commits\" to "
      "*no answer yet* is not a stopping rule.\n")

    A("## The confidence map (F2's hard limit, read off the types)\n")
    A("| field | confidence stream the pipeline computes |")
    A("|---|---|")
    for f in FIELDS:
        A(f"| `{f}` | {conf_src[f] or '**none — F2 cannot score this field**'} |")
    A("\nFour committed fields share one number (`NormalizedTempo.confidence`, "
      "the posterior mass of the committed ±8% neighbourhood, ADR-017), so F2 "
      "is a *metric-block* rule, not a per-field one. `counts` has no "
      "confidence at all.\n")

    for cond in CONDITIONS:
        for mat in ("verified", "provisional"):
            n = len(slices[(cond, mat)])
            A(f"\n## Condition {cond} · {mat} slice (n={n} clips)\n")
            A("| field | eligible n | F1 best k | F1 premature | F1 median t/span | "
              "F1 no-commit | F2 best θ | F2 premature | F2 median t/span | "
              "F2 no-commit |")
            A("|---|---|---|---|---|---|---|---|---|---|")
            for f in FIELDS:
                n_elig = results["f1"][cond][mat][f]["1"]["n"]
                b1 = best_point(results["f1"][cond][mat][f])
                blk2 = results["f2"][cond][mat][f]
                b2 = best_point(blk2) if blk2 else None
                c1 = (f"k={b1[0]} | {fmt(b1[1]['premature_rate'])} | "
                      f"{fmt(b1[1]['median_commit_norm'])} | "
                      f"{fmt(b1[1]['no_commit_rate'])}"
                      if b1 else "**none** | — | — | —")
                if blk2 is None:
                    c2 = "**n/a — no confidence** | — | — | —"
                elif b2:
                    c2 = (f"θ={b2[0]} | {fmt(b2[1]['premature_rate'])} | "
                          f"{fmt(b2[1]['median_commit_norm'])} | "
                          f"{fmt(b2[1]['no_commit_rate'])}")
                else:
                    c2 = "**none** | — | — | —"
                if n_elig == 0:
                    c1 = c2 = "no eligible clip | — | — | —"
                A(f"| `{f}` | {n_elig} | {c1} | {c2} |")

    A("\n## Full F1 sweep — condition granted, verified slice\n")
    A("| field | " + " | ".join(f"k={k}" for k in K_SWEEP) + " |")
    A("|---" * (len(K_SWEEP) + 1) + "|")
    for f in FIELDS:
        cells = []
        for k in K_SWEEP:
            m = results["f1"]["granted"]["verified"][f][str(k)]
            cells.append(f"{fmt(m['premature_rate'])}<br>@{fmt(m['median_commit_norm'])}")
        A(f"| `{f}` | " + " | ".join(cells) + " |")
    A("\nEach cell: premature-commit rate over the top, median commit time as a "
      "fraction of span underneath.\n")

    A("\n## Full F2 sweep — condition granted, verified slice\n")
    shown = [t for t in THETA_SWEEP if abs(t * 100 % 10) < 1e-6]
    A("| field | " + " | ".join(f"θ={t}" for t in shown) + " |")
    A("|---" * (len(shown) + 1) + "|")
    for f in FIELDS:
        blk = results["f2"]["granted"]["verified"][f]
        if blk is None:
            A(f"| `{f}` | " + " | ".join("n/a" for _ in shown) + " |")
            continue
        cells = [f"{fmt(blk[str(t)]['premature_rate'])}<br>@{fmt(blk[str(t)]['median_commit_norm'])}"
                 for t in shown]
        A(f"| `{f}` | " + " | ".join(cells) + " |")
    A("\n(θ shown every 0.10; the scored sweep steps by 0.05 and is complete in "
      "`w14-stopping-rule.json`.)\n")

    # W14-c: this section used to assert a fixed conclusion. The
    # conclusion is a property of the data, so it is now read off the
    # data — the same script must tell the truth before and after the
    # calibration fix.
    _nt_first, _nt_full = [], []
    for _clip, _row in conf_rows:
        _cs = [c for c in _row["conf"]["normalized_tempo"] if c is not None]
        if _cs:
            _nt_first.append(_cs[0])
        if _row["final_conf"]["normalized_tempo"] is not None:
            _nt_full.append(_row["final_conf"]["normalized_tempo"])
    _backwards = (
        bool(_nt_first) and bool(_nt_full)
        and statistics.median(_nt_first) > statistics.median(_nt_full)
    )
    if _backwards:
        A("\n## Why F2 fails: the confidence runs backwards\n")
        A("F2 is nearly flat in θ, which a threshold sweep alone would not "
          "explain. The reason is in the confidence stream itself, read "
          "straight off the recorded prefixes (condition granted, verified "
          "slice):\n")
    else:
        A("\n## The confidence stream behind F2\n")
        A("F2 depends entirely on the confidence the pipeline already "
          "computes, so that stream is reported here, read straight off the "
          "recorded prefixes (condition granted, verified slice):\n")
    A("| stream | median at the FIRST prefix that has one | median on the FULL clip "
      "| clips already ≥0.90 at that first prefix | clips never reaching 0.50 |")
    A("|---|---|---|---|---|")
    for stream in ("normalized_tempo", "onset_tempo", "marker_tempo", "exercise"):
        first, full, mx = [], [], []
        for clip, row in conf_rows:
            cs = [c for c in row["conf"][stream] if c is not None]
            if not cs:
                continue
            first.append(cs[0])
            mx.append(max(cs))
            if row["final_conf"][stream] is not None:
                full.append(row["final_conf"][stream])
        if not first:
            continue
        A(f"| `{stream}` | {statistics.median(first):.3f} | "
          f"{statistics.median(full):.3f} | {sum(c >= 0.9 for c in first)}/{len(first)} "
          f"| {sum(m < 0.5 for m in mx)}/{len(mx)} |")
    if _backwards:
        A("\nThe metric block's confidence is **at its maximum when the "
          "pipeline knows the least** and *falls* as evidence arrives. That "
          "is not a miscalibrated threshold; it is a signal pointing the "
          "wrong way — consistency over two or three intervals is trivially "
          "perfect, and only starts paying for its mistakes once there are "
          "enough intervals to disagree. No θ can fix it, which is why the "
          "sweep is flat rather than merely badly placed.\n")
    else:
        A("\nThe metric block's confidence **rises with evidence**: it is "
          "lowest at the first prefix and higher on the full clip. A θ "
          "threshold is therefore meaningful here, and the F2 columns above "
          "are read as a real sweep rather than as an artifact of a signal "
          "pointing the wrong way.\n")

    A("\n## The owner's curve, laid against the best operating points\n")
    A("W13(a): on a 37.8s demo the owner committed exercise ~3s, meter ~3s, "
      "tempo ~9–12s, structure ~30–33s.\n")
    A("| field | owner t/span | F1 best (granted, verified) | F2 best | verdict |")
    A("|---|---|---|---|---|")
    for f, secs in OWNER_CURVE.items():
        owner = secs / OWNER_SPAN
        b1 = best_point(results["f1"]["granted"]["verified"][f])
        blk2 = results["f2"]["granted"]["verified"][f]
        b2 = best_point(blk2) if blk2 else None
        m1 = b1[1]["median_commit_norm"] if b1 else None
        m2 = b2[1]["median_commit_norm"] if b2 else None
        best = min([m for m in (m1, m2) if m is not None], default=None)
        verdict = ("no operating point" if best is None else
                   ("**earlier than the owner**" if best <= owner
                    else f"later by {best - owner:+.3f} of span"))
        A(f"| `{f}` | {owner:.3f} | {fmt(m1)} | "
          f"{'n/a' if blk2 is None else fmt(m2)} | {verdict} |")
    OUT_MD.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
