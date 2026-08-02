"""
Eval cases: YAML files whose fields are a strict subset of the Vision 08
§8.2 annotation schema (so annotating a real class only ever *adds* cases).

§8.2 spellings are kept at the file level — `meter: "4/4"`, `sides: both`,
`marking_bpm` — and mapped to internal types here. `subdivision` is the one
additive field (the coherent-triple scorer needs it; flagged for a §8.2
amendment). Omitted expect fields are simply not scored.
"""

from dataclasses import dataclass, field
from pathlib import Path

from musical_perception.types import Meter

_EXPECT_KEYS = {
    "slot", "marking_bpm", "performance_bpm", "meter", "subdivision",
    "counts", "sides", "character",
}
_TOP_KEYS = {"id", "input", "tags", "expect", "notes"}


@dataclass
class Case:
    """One labeled clip: where its trace lives and what the truth is."""
    id: str
    trace: str                 # relative to the evals root, e.g. "traces/x/"
    media: str | None
    tags: dict = field(default_factory=dict)
    expect: dict = field(default_factory=dict)   # normalized values
    notes: str = ""

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
        tags=dict(raw.get("tags") or {}),
        expect=_normalize_expect(dict(raw.get("expect") or {}), raw["id"]),
        notes=str(raw.get("notes") or ""),
    )


def load_cases(cases_dir: Path) -> list[Case]:
    """Load every *.yaml case, sorted by id for stable run ordering."""
    cases = [load_case_file(p) for p in sorted(Path(cases_dir).glob("*.yaml"))]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate case ids in {cases_dir}")
    return sorted(cases, key=lambda c: c.id)
