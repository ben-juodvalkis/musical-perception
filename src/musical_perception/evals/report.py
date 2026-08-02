"""
Run artifacts: one JSON per run (metrics + per-case rows + reproducibility
hashes) plus a static HTML view and the markdown baseline table (ADR-009
rule 9 — ADR results tables stop being hand-pasted and start being
generated).
"""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np

from musical_perception.evals.aggregate import aggregate
from musical_perception.evals.runner import outcomes_map
from musical_perception.evals.scorers import CaseResult
from musical_perception.evals.traces import _git_sha

REPORT_SCHEMA = 1


def to_jsonable(obj):
    """JSON-safe deep conversion: dataclasses, Enums, numpy scalars, Paths."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(asdict(obj))
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def build_report(suite_results: dict[str, list[CaseResult]]) -> dict:
    version = None
    try:
        from importlib.metadata import version as _v
        version = _v("musical-perception")
    except Exception:
        pass
    return to_jsonable({
        "schema": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "package_version": version,
        "suites": {
            name: {
                "summary": aggregate(results),
                "outcomes": outcomes_map(results),
                "cases": results,
            }
            for name, results in suite_results.items()
        },
    })


def write_run(report: dict, runs_dir: Path) -> Path:
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = report["created_at"].replace(":", "").replace("-", "").replace("+0000", "Z")
    sha = (report.get("git_sha") or "nogit")[:7]
    path = runs_dir / f"run-{stamp}-{sha}.json"
    path.write_text(json.dumps(report, indent=1))
    path.with_suffix(".html").write_text(render_html(report))
    return path


_OUTCOME_COLOR = {"correct": "#2e7d32", "wrong": "#c62828", "abstained": "#8d6e63"}


def _field_rows(summary: dict) -> str:
    rows = []
    for name, s in summary["fields"].items():
        interval = s["accuracy_wilson95"]
        rows.append(
            f"<tr><td>{name}</td><td>{s['n']}</td><td>{s['correct']}</td>"
            f"<td>{s['wrong']}</td><td>{s['abstained']}</td>"
            f"<td>{s['accuracy'] if s['accuracy'] is not None else '—'}</td>"
            f"<td>{interval if interval else '—'}</td>"
            f"<td>{s['mean_credit']}</td>"
            f"<td>{', '.join(f'{k}×{v}' for k, v in s['failure_modes'].items()) or '—'}</td></tr>"
        )
    return "\n".join(rows)


def _case_rows(cases: list[dict]) -> str:
    rows = []
    for c in cases:
        if c.get("error"):
            rows.append(
                f"<tr><td>{c['case_id']}</td><td colspan=4 style='color:#c62828'>"
                f"ERROR: {c['error']}</td></tr>"
            )
            continue
        for s in c["scores"]:
            color = _OUTCOME_COLOR.get(s["outcome"], "#000")
            rows.append(
                f"<tr><td>{c['case_id']}</td><td>{s['field']}</td>"
                f"<td style='color:{color};font-weight:600'>{s['outcome']}</td>"
                f"<td>{s['predicted']} / {s['expected']}</td>"
                f"<td>{s['failure_mode'] or ''} {s['detail']}</td></tr>"
            )
    return "\n".join(rows)


def render_html(report: dict) -> str:
    sections = []
    for suite, data in report["suites"].items():
        summary = data["summary"]
        sections.append(f"""
<h2>{suite} — {summary['n_cases']} cases</h2>
<table>
<tr><th>field</th><th>n</th><th>correct</th><th>wrong</th><th>abstained</th>
<th>accuracy</th><th>wilson 95%</th><th>credit</th><th>failure modes</th></tr>
{_field_rows(summary)}
</table>
<p>ECE: {summary['ece'] if summary['ece'] is not None else 'n/a'}
&nbsp; errors: {', '.join(summary['errors']) or 'none'}</p>
<details><summary>per-case rows</summary>
<table>
<tr><th>case</th><th>field</th><th>outcome</th><th>pred / exp</th><th>notes</th></tr>
{_case_rows(data['cases'])}
</table></details>""")
    body = "\n".join(sections)
    return f"""<!doctype html><meta charset="utf-8">
<title>eval run {report['created_at']}</title>
<style>
body {{ font: 14px/1.5 -apple-system, sans-serif; margin: 2rem; max-width: 72rem; }}
table {{ border-collapse: collapse; margin: .5rem 0 1rem; }}
td, th {{ border: 1px solid #ccc; padding: .25rem .6rem; text-align: left; }}
th {{ background: #f3f3f3; }}
</style>
<h1>Eval run</h1>
<p>{report['created_at']} · git {report.get('git_sha') or '?'} ·
package {report.get('package_version') or '?'}</p>
{body}"""


def render_markdown_baseline(report: dict) -> str:
    """The published baseline table (Gate A2's deliverable)."""
    lines = [
        "# Eval Baseline",
        "",
        f"Generated {report['created_at']} at git `{(report.get('git_sha') or '?')[:7]}` "
        f"by `python -m musical_perception.evals bless`. Do not edit by hand.",
        "",
        "Outcomes are **correct / wrong / abstained** — abstention is never",
        "counted as wrong (ADR-009). n is small; intervals are the honest part.",
        "",
    ]
    for suite, data in report["suites"].items():
        summary = data["summary"]
        lines += [f"## {suite} ({summary['n_cases']} cases)", ""]
        lines += [
            "| field | n | correct | wrong | abstained | accuracy | wilson 95% | failure modes |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, s in summary["fields"].items():
            acc = s["accuracy"] if s["accuracy"] is not None else "—"
            wil = s["accuracy_wilson95"] or "—"
            modes = ", ".join(f"{k}×{v}" for k, v in s["failure_modes"].items()) or "—"
            lines.append(
                f"| {name} | {s['n']} | {s['correct']} | {s['wrong']} | "
                f"{s['abstained']} | {acc} | {wil} | {modes} |"
            )
        lines.append("")
        if data["summary"]["errors"]:
            lines.append(f"Case errors: {', '.join(summary['errors'])}")
            lines.append("")
    return "\n".join(lines)
