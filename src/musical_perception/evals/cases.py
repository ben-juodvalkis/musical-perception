"""
Eval cases: YAML files whose fields are a strict subset of the Vision 08
§8.2 annotation schema (so annotating a real class only ever *adds* cases).

§8.2 spellings are kept at the file level — `meter: "4/4"`, `sides: both`,
`marking_bpm` — and mapped to internal types here. `subdivision` is the one
additive field (the coherent-triple scorer needs it; flagged for a §8.2
amendment). Omitted expect fields are simply not scored.

`maturity` (charter W1.5) says who wrote the truth labels: `verified` is
owner-checked and gates; `provisional` is agent-proposed and gates
nothing. The key is optional and defaults to `verified`, so every case
written before W1.5 keeps exactly the meaning it had. Agent-authored
cases for new material MUST carry `maturity: provisional` — that is the
whole point of the charter's add-only ingestion carve-out.
"""

from dataclasses import dataclass, field
from pathlib import Path

from musical_perception.types import Meter

_EXPECT_KEYS = {
    "slot", "marking_bpm", "performance_bpm", "meter", "subdivision",
    "counts", "sides", "character",
}
_TOP_KEYS = {"id", "input", "tags", "expect", "notes", "maturity"}

VERIFIED = "verified"
PROVISIONAL = "provisional"
_MATURITIES = (VERIFIED, PROVISIONAL)

# Tag vocabulary (Vision 13 §13.6). Only the values this loader actually
# adjudicates live here; the rest of the vocabulary stays documentation.
# `accompanied` gained a third state at W1.5 (owner ruling B5,
# 2026-08-24): a recording that is accompaniment *and nothing else* — a
# pianist playing the exercise with no teacher voice present — is neither
# `false` nor the `true` of "a teacher counting over a pianist". Its truth,
# if ever labeled, comes from the piano's beat.
ACCOMPANIMENT_ONLY = "accompaniment_only"
_ACCOMPANIED_VALUES = (False, True, ACCOMPANIMENT_ONLY)


@dataclass
class Case:
    """One labeled clip: where its trace lives and what the truth is."""
    id: str
    trace: str                 # relative to the evals root, e.g. "traces/x/"
    media: str | None
    tags: dict = field(default_factory=dict)
    expect: dict = field(default_factory=dict)   # normalized values
    notes: str = ""
    maturity: str = VERIFIED     # verified (gates) | provisional (never gates)

    @property
    def provisional(self) -> bool:
        """True when this case's truth labels have not been owner-verified."""
        return self.maturity == PROVISIONAL

    @property
    def reference(self) -> bool:
        """True for rows the owner demoted out of the benchmark (reset,
        2026-09-01): the demo is the case, so piano takes are a
        reference-only slice — scored and reported with their own n,
        entering no headline aggregate, no gate, and never pinned by
        `bless`. Keyed on the existing `clip_role` tag so no case file
        changed. A second key, `step_one: deferred`, marks rows the owner
        deferred from step one by ruling (fast triple meters: no honest
        metric level sits inside 70-140, so the case waits for the meter
        step; its truth label is untouched). Orthogonal to `maturity`: a
        row both provisional and reference lands in the reference slice
        (demotion is the stronger exclusion)."""
        return (
            self.tags.get("clip_role") == "take"
            or self.tags.get("step_one") == "deferred"
        )

    @property
    def accompaniment_only(self) -> bool:
        """True for takes that are accompaniment and nothing else (B5)."""
        return self.tags.get("accompanied") == ACCOMPANIMENT_ONLY

    @property
    def expected_bpm(self) -> float | None:
        """performance_bpm wins when a case ever carries both (§8.2)."""
        return self.expect.get("performance_bpm") or self.expect.get("marking_bpm")


def parse_meter(text: str) -> Meter:
    beats, unit = str(text).split("/")
    return Meter(beats_per_measure=int(beats), beat_unit=int(unit))


def parse_sides(value) -> int:
    if isinstance(value, int):
        return value
    mapping = {"both": 2, "one": 1}
    if str(value) not in mapping:
        raise ValueError(f"sides must be 'both', 'one', or an int — got {value!r}")
    return mapping[str(value)]


def parse_maturity(value, case_id: str) -> str:
    """Default `verified`; anything outside the vocabulary is an error."""
    if value is None:
        return VERIFIED
    text = str(value)
    if text not in _MATURITIES:
        raise ValueError(
            f"case {case_id}: maturity must be one of {list(_MATURITIES)} "
            f"— got {value!r}"
        )
    return text


def _check_tags(tags: dict, case_id: str) -> dict:
    """Adjudicate the one tag whose vocabulary the harness depends on."""
    if "accompanied" in tags and tags["accompanied"] not in _ACCOMPANIED_VALUES:
        raise ValueError(
            f"case {case_id}: accompanied must be one of "
            f"{list(_ACCOMPANIED_VALUES)} — got {tags['accompanied']!r}"
        )
    return tags


def _normalize_expect(raw: dict, case_id: str) -> dict:
    unknown = set(raw) - _EXPECT_KEYS
    if unknown:
        raise ValueError(f"case {case_id}: unknown expect fields {sorted(unknown)}")
    expect = dict(raw)
    if "meter" in expect:
        expect["meter"] = parse_meter(expect["meter"])
    if "sides" in expect:
        expect["sides"] = parse_sides(expect["sides"])
    for key in ("marking_bpm", "performance_bpm"):
        if key in expect:
            expect[key] = float(expect[key])
    if "character" in expect and not isinstance(expect["character"], dict):
        raise ValueError(f"case {case_id}: character must be a mapping")
    return expect


def load_case_file(path: Path) -> Case:
    import yaml  # lazy: pyyaml lives in the [eval] extra

    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "id" not in raw:
        raise ValueError(f"{path}: not a case file (missing id)")
    unknown = set(raw) - _TOP_KEYS
    if unknown:
        raise ValueError(f"case {raw['id']}: unknown top-level keys {sorted(unknown)}")
    inp = raw.get("input", {})
    if "trace" not in inp:
        raise ValueError(f"case {raw['id']}: input.trace is required for tier 1")
    return Case(
        id=str(raw["id"]),
        trace=str(inp["trace"]),
        media=inp.get("media"),
        tags=_check_tags(dict(raw.get("tags") or {}), raw["id"]),
        expect=_normalize_expect(dict(raw.get("expect") or {}), raw["id"]),
        notes=str(raw.get("notes") or ""),
        maturity=parse_maturity(raw.get("maturity"), raw["id"]),
    )


def load_cases(cases_dir: Path) -> list[Case]:
    """Load every *.yaml case, sorted by id for stable run ordering."""
    cases = [load_case_file(p) for p in sorted(Path(cases_dir).glob("*.yaml"))]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate case ids in {cases_dir}")
    return sorted(cases, key=lambda c: c.id)
