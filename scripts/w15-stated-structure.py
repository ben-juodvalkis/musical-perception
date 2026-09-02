#!/usr/bin/env python3
"""W15 — the stated-structure channel (REPORTED-ONLY).

Reads the frozen Whisper transcripts under ``evals/traces/`` and emits, per
clip, typed claims about *stated* musical structure: the moments where the
teacher says a quantity out loud.  The point of the workstream is the
**disambiguation** — deciding what a spoken number is a quantity *of* — not
the pattern matching, so the typing rules are enumerated and each claim
records which rule fired.

Gates nothing.  Wired into no pipeline path (Standing Lesson 9: build the
replay path before betting on the channel).  Touches no file under
``evals/`` and no file under ``src/musical_perception/``.

Usage:
    python scripts/w15-stated-structure.py [--json OUT] [--only PREFIX]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "evals"

# --------------------------------------------------------------------------
# tokenisation
# --------------------------------------------------------------------------

NUMERALS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "sixteen": 16, "twenty": 20, "twenty-four": 24,
    "thirty-two": 32,
}
COUNT_UNITS = {"count", "counts"}
BAR_UNITS = {"bar", "bars", "barre", "barres", "measure", "measures"}
EIGHT_UNITS = {"eight", "eights", "eighth", "eighths"}
TIME_UNITS = {"time", "times"}
UNITS = COUNT_UNITS | BAR_UNITS | EIGHT_UNITS | TIME_UNITS
# a spoken boundary after the numeral: discourse particles or clause end
BOUNDARY_WORDS = {"please", "ok", "okay", "yeah", "yes", "good", "here",
                  "thanks", "thank", "and"}
SPATIAL = {"to", "at", "on", "from", "of", "in", "behind", "front", "near"}
DETERMINERS = {"the", "a", "an", "your", "our", "this", "that"}

# A5-30 redaction: committed artifacts must not quote transcript lines that
# name steps.  Audit windows keep only the tokens the typing rules read; every
# other content word is masked.  The frozen transcripts remain the source of
# truth for anyone who needs the words.
REDACTION_KEEP = (
    set(NUMERALS) | UNITS | BOUNDARY_WORDS | SPATIAL | DETERMINERS
    | {"more", "last", "each", "same", "every", "beat", "beats", "we", "you",
       "go", "do", "it", "is", "then", "so", "now", "up", "down", "for",
       "with", "all", "way", "around", "one", "and"}
)


def redact(tokens: list[str]) -> str:
    """Mask step-naming content words, keep what the rules read."""
    return " ".join(t if (t in REDACTION_KEEP or t.isdigit()) else "\u00b7"
                    for t in tokens)


@dataclass
class Token:
    i: int
    raw: str
    bare: str
    t: float
    ends_clause: bool          # the raw token carried . , ; ! ?
    value: int | None          # numeral value, else None


def tokenize(words: list[dict]) -> list[Token]:
    """Whisper words -> tokens.  Hyphenated numeral+unit pairs are split."""
    out: list[Token] = []
    for w in words:
        raw = w.get("word", "")
        t = float(w.get("start", 0.0))
        cleaned = re.sub(r"[^\w'-]", "", raw).lower()
        pieces = [cleaned]
        if "-" in cleaned and cleaned not in NUMERALS:
            head, _, tail = cleaned.partition("-")
            if head in NUMERALS and tail in UNITS:
                pieces = [head, tail]
        for k, piece in enumerate(pieces):
            if not piece:
                continue
            out.append(Token(
                i=len(out), raw=raw, bare=piece, t=t,
                ends_clause=bool(re.search(r"[.,;!?]$", raw)) and k == len(pieces) - 1,
                value=NUMERALS.get(piece, int(piece) if piece.isdigit() else None),
            ))
    return out


def counting_run_members(toks: list[Token]) -> set[int]:
    """Indices inside an ascending numeral run of length >= 3.

    These are the teacher counting ("one, two, three, ..."), not stating a
    quantity, and they never generate a candidate.
    """
    members: set[int] = set()
    nums = [k for k, tk in enumerate(toks) if tk.value is not None]
    pos = {k: n for n, k in enumerate(nums)}
    used: set[int] = set()
    for k in nums:
        if k in used:
            continue
        seq = [k]
        j = k
        while True:
            nxt = None
            for cand in nums[pos[j] + 1:]:
                if cand - j > 4:
                    break
                if all(toks[m].bare in {"and", "a", "uh", "ah"} or toks[m].value is not None
                       for m in range(j + 1, cand)):
                    nxt = cand
                break
            if nxt is not None and toks[nxt].value == toks[j].value + 1:
                seq.append(nxt)
                j = nxt
            else:
                break
        if len(seq) >= 3:
            members.update(seq)
            used.update(seq)
    return members


# --------------------------------------------------------------------------
# typing rules  (the deliverable)
# --------------------------------------------------------------------------

# quantity vocabulary.  The condition pre-registered four types; the corpus
# forces three more (declared in the pre-registration).  FOLD maps back.
FOLD = {
    "beats_per_bar": "beats-per-bar",
    "bars": "bars",
    "phrases": "bars",          # a length in metric groups, coarser than a bar
    "step_duration": "unknown",  # not one of the four; the four cannot hold it
    "repetitions": "repetitions",
    "entry_point": "unknown",
    "unknown": "unknown",
}


@dataclass
class Claim:
    case_id: str
    t: float
    frame: str
    rule: str
    value: int | None
    quantity: str
    folded: str
    unit: str | None
    window: str          # +-3 bare tokens, for audit; no raw punctuation
    naive_bpb: int | None   # what the naive "spoken number = bar grouping" reads


def _bare_at(toks, k):
    return toks[k].bare if 0 <= k < len(toks) else ""


def classify(toks: list[Token], k: int, case_id: str) -> Claim:
    """Type the numeral at index k.  Rules are tried in the pre-registered
    precedence: explicit meter frame, unit-bearing frames, then bare frames."""
    tk = toks[k]
    nxt, nxt2, nxt3 = _bare_at(toks, k + 1), _bare_at(toks, k + 2), _bare_at(toks, k + 3)
    prev, prev2 = _bare_at(toks, k - 1), _bare_at(toks, k - 2)
    window = redact([t.bare for t in toks[max(0, k - 3):k + 4]])

    def claim(frame, rule, quantity, unit=None, naive=None):
        return Claim(case_id, tk.t, frame, rule, tk.value, quantity,
                     FOLD[quantity], unit, window, naive)

    # rule 7 — explicit beats-per-bar frame, the only route to that type
    if nxt in {"beat", "beats"} and nxt2 in {"to", "per"} and (
            nxt3 in BAR_UNITS or _bare_at(toks, k + 4) in BAR_UNITS):
        return claim("N_beats_per_bar", "R7", "beats_per_bar", "bar", tk.value)

    # unit-bearing frames.  One optional 'more' may intervene.
    off = 2 if nxt == "more" else 1
    unit = _bare_at(toks, k + off)

    if unit in COUNT_UNITS:
        # rule 3 — a count is always a beat (owner ruling), so 'N counts' is a
        # duration in beats, never a bar length.
        frame = "in_N_counts" if prev == "in" else "N_counts"
        return claim(frame, "R3", "step_duration", "count", tk.value)

    if unit in EIGHT_UNITS:
        # rule 4 — a length in 8-count phrases: a grouping rung, not a bar.
        return claim("N_eights", "R4", "phrases", "eight", tk.value)

    if unit in BAR_UNITS:
        # rule 2 — the homonym gate.  'bar' is furniture in a ballet class
        # unless a numeral quantifies it and no spatial frame owns it.
        spatial_owner = prev in DETERMINERS and prev2 in SPATIAL
        if spatial_owner:
            return claim("N_bars_rejected", "R2-reject", "unknown", "bar", None)
        return claim("N_bars", "R2", "bars", "bar", tk.value)

    if unit in TIME_UNITS:
        # rule 1 — 'time(s)' is a claim only as 'N more time' / 'N last time'.
        if nxt == "more":
            return claim("N_more_time", "R1", "repetitions", "time", None)
        return claim("N_time_rejected", "R1-reject", "unknown", "time", None)

    # rule 6 — 'N more' with no unit
    if nxt == "more":
        return claim("N_more", "R6", "repetitions", None, None)

    # rule 5 — bare 'in/at/on N' as the count to come in on
    if prev in {"in", "at", "on"}:
        follows_ascending = toks[k + 1].value == (tk.value or 0) + 1 if (
            k + 1 < len(toks) and toks[k + 1].value is not None) else False
        at_boundary = tk.ends_clause or nxt in BOUNDARY_WORDS or nxt == ""
        if follows_ascending:
            return claim("in_N_counting", "R5-reject", "unknown", None, None)
        if at_boundary:
            return claim("in_N", "R5", "entry_point", None, tk.value)
        return claim("in_N_quantified", "R5-reject", "unknown", None, None)

    # rule 8 — abstain
    return claim("bare_numeral", "R8", "unknown", None, None)


# --------------------------------------------------------------------------
# corpus pass
# --------------------------------------------------------------------------

def load_cases() -> dict:
    import yaml
    cases = {}
    for f in sorted((EVALS / "cases").glob("*.yaml")):
        d = yaml.safe_load(f.read_text())
        cases[d["id"]] = d
    return cases


def trace_dir(case: dict) -> Path:
    return EVALS / str(case["input"]["trace"]).rstrip("/")


def meter_numerator(meter: str | None) -> int | None:
    if not meter or "/" not in str(meter):
        return None
    return int(str(meter).split("/")[0])


def run(only: str | None = None) -> dict:
    cases = load_cases()
    per_clip = {}
    for cid, case in sorted(cases.items()):
        if only and not cid.startswith(only):
            continue
        wj = trace_dir(case) / "whisper.json"
        if not wj.exists():
            per_clip[cid] = {"skipped": "no whisper.json"}
            continue
        toks = tokenize(json.loads(wj.read_text()).get("words", []))
        runs = counting_run_members(toks)
        claims = [classify(toks, k, cid)
                  for k, tk in enumerate(toks)
                  if tk.value is not None and k not in runs]
        expect = case.get("expect") or {}
        per_clip[cid] = {
            "n_tokens": len(toks),
            "n_numerals": sum(1 for t in toks if t.value is not None),
            "n_in_counting_runs": len(runs),
            "n_candidates": len(claims),
            "truth_meter": expect.get("meter"),
            "truth_meter_numerator": meter_numerator(expect.get("meter")),
            "truth_subdivision": expect.get("subdivision"),
            "truth_counts": expect.get("counts"),
            "claims": [asdict(c) for c in claims],
        }
    return per_clip


# --------------------------------------------------------------------------
# hand gold — the audit set for type precision (P4)
# --------------------------------------------------------------------------

# Authored by reading each candidate's context in the frozen transcript.  Keys
# are (case_id, "%.2f" % t).  Every emitted claim and every rule-level
# rejection is listed; the parser is scored against this, not the other way
# round.  One disagreement is deliberate and is the honest miss (see the memo).
GOLD = {
    ("barre6-ballonne-demo", "0.85"): "entry_point",
    ("barre6-ballonne-take1", "33.96"): "phrases",
    ("barre6-ballonne-take1", "43.72"): "phrases",
    ("barre6-ballonne-take2", "39.45"): "phrases",
    ("barre6-degage-take1", "35.14"): "phrases",
    ("barre6-degage-take1", "44.10"): "repetitions",
    ("barre6-degage-take2", "37.54"): "phrases",
    ("barre6-frappe-demo", "46.70"): "step_duration",
    ("barre6-frappe-demo", "49.29"): "step_duration",
    ("barre6-frappe-take1", "16.59"): "step_duration",
    ("barre6-plie-take1", "97.36"): "step_duration",
    ("barre6-plie-take2", "94.60"): "step_duration",
    ("barre6-releve-finish-take1", "17.53"): "repetitions",
    ("barre6-rond-de-jambe-demo", "21.65"): "repetitions",
    ("barre6-rond-de-jambe-demo", "50.28"): "repetitions",
    # the tail of the preceding "N more on N" — the object of that claim, not a
    # second announcement.  The parser calls it entry_point; the gold says no.
    ("barre6-rond-de-jambe-demo", "51.02"): "unknown",
    ("barre6-rond-de-jambe-take1", "22.41"): "repetitions",
    ("barre6-rond-de-jambe-take1", "50.77"): "repetitions",
    ("barre6-rond-de-jambe-take2", "26.19"): "repetitions",
    ("barre6-rond-de-jambe-take2", "54.63"): "repetitions",
    ("barre6-tendu-warmup-demo", "7.10"): "repetitions",
    ("rig-mixed-4-4-104-quantities", "10.40"): "repetitions",
    ("rig-mixed-4-4-104-quantities", "13.35"): "step_duration",
    ("rig-mixed-4-4-104-quantities", "18.11"): "repetitions",
    # rule-level rejections: all correct, all "not a quantity claim"
    ("barre6-plie-demo", "6.76"): "unknown",
    ("barre6-plie-demo", "28.73"): "unknown",
    ("barre6-plie-demo", "66.76"): "unknown",
    ("barre6-plie-demo", "76.56"): "unknown",
    ("barre6-plie-take1", "61.87"): "unknown",
    ("rig-names-3-4-88-waltz", "7.58"): "unknown",
    ("rig-names-3-4-88-waltz", "15.84"): "unknown",
}

# claim types that, on their face, constrain the metric ladder
METER_BEARING = {"beats_per_bar", "bars"}


def score(per_clip: dict, meter_outcomes: dict | None = None) -> dict:
    """All the numbers the W15 report quotes, computed in one place."""
    claims = [(cid, c) for cid, v in per_clip.items() for c in v.get("claims", [])]
    typed = [(cid, c) for cid, c in claims if c["quantity"] != "unknown"]

    clips_with_candidate = {cid for cid, _ in claims}
    clips_with_typed = {cid for cid, _ in typed}

    # --- P1 -------------------------------------------------------------
    bpb = [(cid, c) for cid, c in typed if c["quantity"] == "beats_per_bar"]

    # --- P3: the naive "spoken number is the bar grouping" reading --------
    naive_rows = []
    for cid, v in sorted(per_clip.items()):
        num = v.get("truth_meter_numerator")
        fired = [c for c in v.get("claims", []) if c["naive_bpb"] is not None]
        if not fired:
            continue
        first = fired[0]
        naive_rows.append({
            "case_id": cid,
            "truth_numerator": num,
            "n_naive_claims": len(fired),
            "first_value": first["naive_bpb"],
            "first_frame": first["frame"],
            "first_agrees": first["naive_bpb"] == num,
            "any_agrees": any(c["naive_bpb"] == num for c in fired),
        })

    # --- P4: type precision against the hand gold -------------------------
    audited, right, disagreements = 0, 0, []
    seen_keys = set()
    for cid, c in claims:
        key = (cid, "%.2f" % c["t"])
        if key not in GOLD:
            continue
        seen_keys.add(key)
        audited += 1
        if c["quantity"] == GOLD[key]:
            right += 1
        else:
            disagreements.append({"case_id": cid, "t": c["t"], "frame": c["frame"],
                                  "parser": c["quantity"], "gold": GOLD[key],
                                  "window": c["window"]})
    missing_gold = sorted(set(GOLD) - seen_keys)

    # --- P5: the bar homonym gate ----------------------------------------
    rejected_by_bar_gate = [(cid, c) for cid, c in claims if c["rule"] == "R2-reject"]

    # --- P6: phrase claims vs the counts label ----------------------------
    phrase_clips = {}
    for cid, v in per_clip.items():
        ph = [c for c in v.get("claims", []) if c["quantity"] == "phrases"]
        if ph:
            counts = v.get("truth_counts")
            phrase_clips[cid] = {
                "values": [c["value"] for c in ph],
                "truth_counts": counts,
                "divisible_by_8": (counts % 8 == 0) if isinstance(counts, int) else None,
            }

    # --- the pre-registered verdict rule ----------------------------------
    strict, charitable = [], []
    if meter_outcomes:
        for cid, v in per_clip.items():
            if meter_outcomes.get(cid, {}).get("meter_triple") != "wrong":
                continue
            num = v.get("truth_meter_numerator")
            for c in v.get("claims", []):
                if c["quantity"] in METER_BEARING:
                    strict.append({"case_id": cid, "t": c["t"], "quantity": c["quantity"]})
                if c["quantity"] != "unknown" and c["value"] == num:
                    charitable.append({"case_id": cid, "t": c["t"],
                                       "quantity": c["quantity"], "value": c["value"],
                                       "truth_numerator": num})

    return {
        "n_clips": len(per_clip),
        "n_candidates": len(claims),
        "n_typed_claims": len(typed),
        "clips_with_candidate": len(clips_with_candidate),
        "clips_with_typed_claim": len(clips_with_typed),
        "by_quantity": {q: sum(1 for _, c in claims if c["quantity"] == q)
                        for q in sorted({c["quantity"] for _, c in claims})},
        "by_folded": {q: sum(1 for _, c in typed if c["folded"] == q)
                      for q in sorted({c["folded"] for _, c in typed})},
        "P1_beats_per_bar_claims": len(bpb),
        "P3_naive": {
            "clips_fired": len(naive_rows),
            "first_claim_agree": sum(1 for r in naive_rows if r["first_agrees"]),
            "any_claim_agree": sum(1 for r in naive_rows if r["any_agrees"]),
            "rows": naive_rows,
        },
        "P4_type_precision": {
            "audited": audited, "correct": right,
            "precision": round(right / audited, 4) if audited else None,
            "disagreements": disagreements,
            "gold_keys_not_matched": missing_gold,
        },
        "P5_bar_gate_rejections": len(rejected_by_bar_gate),
        "P6_phrase_clips": phrase_clips,
        "verdict_strict_rows": strict,
        "verdict_charitable_rows": charitable,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    ap.add_argument("--only")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-meter-outcomes", action="store_true",
                    help="skip the tier-1 replay used by the verdict rule")
    args = ap.parse_args(argv)

    per_clip = run(args.only)
    meter_outcomes = None
    if not args.no_meter_outcomes:
        import warnings
        from musical_perception.evals.runner import outcomes_map, run_tier1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            meter_outcomes = outcomes_map(run_tier1(EVALS))
    summary = score(per_clip, meter_outcomes)
    payload = {"summary": summary, "per_clip": per_clip}
    if args.json:
        args.json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.json}")
    if args.verbose:
        print(json.dumps(payload, indent=1)[:4000])
    print(json.dumps(
        {k: v for k, v in summary.items()
         if k not in {"P3_naive", "P4_type_precision", "P6_phrase_clips"}},
        indent=1))
    print("P3_naive:", {k: v for k, v in summary["P3_naive"].items() if k != "rows"})
    print("P4_type_precision:", {k: v for k, v in summary["P4_type_precision"].items()
                                 if k != "disagreements"})
    for d in summary["P4_type_precision"]["disagreements"]:
        print("   disagreement:", d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
