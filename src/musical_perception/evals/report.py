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
            name: _suite_block(results) for name, results in suite_results.items()
        },
    })


def _suite_block(results) -> dict:
    """CaseResult suites aggregate as before; dict suites (stage1) carry
    their own summary and pin no outcomes — provisional grids never gate."""
    if isinstance(results, dict):
        return {"summary": results, "outcomes": {}, "cases": []}
    return {
        "summary": aggregate(results),
        "outcomes": outcomes_map(results),
        "cases": results,
    }


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


def family_cell(summary_field: dict) -> str:
    """`hits/checked` for the ADR-014 truth-in-family measure, or — when
    the field never reported a metric-level family."""
    checked = summary_field.get("truth_in_family_n")
    if not checked:
        return "—"
    return f"{summary_field.get('truth_in_family') or 0}/{checked}"


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
            f"<td>{family_cell(s)}</td>"
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


def tempo_metrics_line(summary: dict) -> str | None:
    """One-line Acc1/Acc2 + OE1/OE2 readout (Review 2 §4.2), or None."""
    tm = summary.get("tempo_metrics")
    if not tm:
        return None
    return (
        f"tempo n={tm['n_committed']}: "
        f"Acc1 {tm['acc1']['tol_04']}@4% {tm['acc1']['tol_08']}@8% · "
        f"Acc2 {tm['acc2']['tol_04']}@4% {tm['acc2']['tol_08']}@8% · "
        f"OE1 median {tm['oe1']['median']} · |OE2| median {tm['oe2']['abs_median']} "
        f"(max {tm['oe2']['max_abs']}) · between-levels rows {tm['between_levels']}"
    )


def _stage1_rows(summary: dict) -> str:
    rows = []
    for c in summary["clips"]:
        rows.append(
            f"<tr><td>{c['case_id']}</td><td>{'yes' if c['provisional'] else 'no'}</td>"
            f"<td>{c['n_ref']}</td><td>{c['n_pred']}</td><td>{c['matched']}</td>"
            f"<td>{c['precision']}</td><td>{c['recall']}</td><td>{c['f_measure']}</td>"
            f"<td>{c['asynchrony_mean_ms']} ± {c['asynchrony_sd_ms']}</td></tr>"
        )
    return "\n".join(rows)


def _stage1_section(suite: str, summary: dict) -> str:
    agg = summary.get("aggregate_provisional")
    agg_line = (
        f"provisional pooled: P {agg['precision']} R {agg['recall']} "
        f"F {agg['f_measure']} (macro {agg['f_measure_macro']}) · asynchrony "
        f"mean {agg['asynchrony_mean_ms']} ms median {agg['asynchrony_median_ms']} ms"
        if agg else "no scored clips"
    )
    missing = ", ".join(summary["missing_grids"]) or "none"
    return f"""
<h2>{suite} — pulse vs beat grids ({summary['pulse_source']}, ±{summary['tolerance_s']}s)</h2>
<p><strong>PROVISIONAL slice — gates nothing until grids are owner-verified.</strong><br>
{agg_line}</p>
<table>
<tr><th>clip</th><th>provisional</th><th>n_ref</th><th>n_pred</th><th>matched</th>
<th>P</th><th>R</th><th>F</th><th>asynchrony ms</th></tr>
{_stage1_rows(summary)}
</table>
<p>missing grids: {missing}</p>"""


def _provisional_section(suite: str, summary: dict) -> str:
    """The W1.5 slice: agent-proposed truth, its own n, never pooled above."""
    prov = summary.get("provisional")
    if not prov:
        return ""
    tm_line = tempo_metrics_line(prov)
    return f"""
<h3>{suite} — provisional slice ({prov['n_cases']} cases)</h3>
<p><strong>Agent-proposed truth labels — gates nothing, pooled into nothing
above.</strong> Cases: {', '.join(prov['case_ids'])}</p>
<table>
<tr><th>field</th><th>n</th><th>correct</th><th>wrong</th><th>abstained</th>
<th>accuracy</th><th>wilson 95%</th><th>credit</th><th>truth in family</th>
<th>failure modes</th></tr>
{_field_rows(prov)}
</table>
<p>{tm_line + '<br>' if tm_line else ''}
ECE: {prov['ece'] if prov['ece'] is not None else 'n/a'}
&nbsp; errors: {', '.join(prov['errors']) or 'none'}</p>"""


def _reference_section(suite: str, summary: dict) -> str:
    """Reset 2026-09-01: owner-demoted piano takes — verified truth, out of
    the benchmark by ruling, reported with their own n."""
    ref = summary.get("reference")
    if not ref:
        return ""
    tm_line = tempo_metrics_line(ref)
    return f"""
<h3>{suite} — reference slice ({ref['n_cases']} cases)</h3>
<p><strong>Owner-demoted piano takes — verified truth, out of the benchmark,
gates nothing, pooled into nothing above.</strong> Cases: {', '.join(ref['case_ids'])}</p>
<table>
<tr><th>field</th><th>n</th><th>correct</th><th>wrong</th><th>abstained</th>
<th>accuracy</th><th>wilson 95%</th><th>credit</th><th>truth in family</th>
<th>failure modes</th></tr>
{_field_rows(ref)}
</table>
<p>{tm_line + '<br>' if tm_line else ''}
ECE: {ref['ece'] if ref['ece'] is not None else 'n/a'}
&nbsp; errors: {', '.join(ref['errors']) or 'none'}</p>"""


def render_html(report: dict) -> str:
    sections = []
    for suite, data in report["suites"].items():
        summary = data["summary"]
        if "clips" in summary:  # stage1-style suite
            sections.append(_stage1_section(suite, summary))
            continue
        tm_line = tempo_metrics_line(summary)
        sections.append(f"""
<h2>{suite} — {summary['n_cases']} cases</h2>
<table>
<tr><th>field</th><th>n</th><th>correct</th><th>wrong</th><th>abstained</th>
<th>accuracy</th><th>wilson 95%</th><th>credit</th><th>truth in family</th>
<th>failure modes</th></tr>
{_field_rows(summary)}
</table>
<p>{tm_line + '<br>' if tm_line else ''}
ECE: {summary['ece'] if summary['ece'] is not None else 'n/a'}
&nbsp; errors: {', '.join(summary['errors']) or 'none'}</p>
<details><summary>per-case rows</summary>
<table>
<tr><th>case</th><th>field</th><th>outcome</th><th>pred / exp</th><th>notes</th></tr>
{_case_rows(data['cases'])}
</table></details>{_provisional_section(suite, summary)}{_reference_section(suite, summary)}""")
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
        "**truth in family** (ADR-014) counts wrong answers whose reported",
        "metric-level family still contained the expected tempo — a selection",
        "failure rather than a measurement failure. It is informational and",
        "gates nothing; outcomes above are unaffected by it.",
        "",
    ]
    for suite, data in report["suites"].items():
        summary = data["summary"]
        if "clips" in summary:  # stage1-style suite — provisional, non-gating
            agg = summary.get("aggregate_provisional")
            lines += [f"## {suite} (pulse vs beat grids — PROVISIONAL, gates nothing)", ""]
            if agg:
                lines.append(
                    f"{summary['pulse_source']} ±{summary['tolerance_s']}s: "
                    f"P {agg['precision']} R {agg['recall']} F {agg['f_measure']} "
                    f"over {agg['n_clips']} clips; asynchrony mean "
                    f"{agg['asynchrony_mean_ms']} ms"
                )
            if summary["missing_grids"]:
                lines.append(f"Missing grids: {', '.join(summary['missing_grids'])}")
            lines.append("")
            continue
        lines += [f"## {suite} ({summary['n_cases']} cases)", ""]
        lines += [
            "| field | n | correct | wrong | abstained | accuracy | wilson 95% "
            "| truth in family | failure modes |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for name, s in summary["fields"].items():
            acc = s["accuracy"] if s["accuracy"] is not None else "—"
            wil = s["accuracy_wilson95"] or "—"
            modes = ", ".join(f"{k}×{v}" for k, v in s["failure_modes"].items()) or "—"
            lines.append(
                f"| {name} | {s['n']} | {s['correct']} | {s['wrong']} | "
                f"{s['abstained']} | {acc} | {wil} | {family_cell(s)} | {modes} |"
            )
        lines.append("")
        tm_line = tempo_metrics_line(summary)
        if tm_line:
            lines += [tm_line, ""]
        if data["summary"]["errors"]:
            lines.append(f"Case errors: {', '.join(summary['errors'])}")
            lines.append("")
        lines += _provisional_markdown(suite, summary)
        lines += _reference_markdown(suite, summary)
    return "\n".join(lines)


def _provisional_markdown(suite: str, summary: dict) -> list[str]:
    """W1.5 slice for the published baseline — separate table, separate n."""
    prov = summary.get("provisional")
    if not prov:
        return []
    lines = [
        f"### {suite} — provisional slice ({prov['n_cases']} cases)",
        "",
        "Agent-proposed truth labels (`maturity: provisional`). These rows",
        "gate nothing and are pooled into none of the numbers above; they",
        "become headline rows only when the owner verifies their labels.",
        "",
        f"Cases: {', '.join(prov['case_ids'])}",
        "",
        "| field | n | correct | wrong | abstained | accuracy | wilson 95% "
        "| truth in family | failure modes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in prov["fields"].items():
        acc = s["accuracy"] if s["accuracy"] is not None else "—"
        wil = s["accuracy_wilson95"] or "—"
        modes = ", ".join(f"{k}×{v}" for k, v in s["failure_modes"].items()) or "—"
        lines.append(
            f"| {name} | {s['n']} | {s['correct']} | {s['wrong']} | "
            f"{s['abstained']} | {acc} | {wil} | {family_cell(s)} | {modes} |"
        )
    lines.append("")
    return lines


def _reference_markdown(suite: str, summary: dict) -> list[str]:
    """Reset 2026-09-01 slice for the published baseline — the demoted
    piano takes, separate table, separate n."""
    ref = summary.get("reference")
    if not ref:
        return []
    lines = [
        f"### {suite} — reference slice ({ref['n_cases']} cases)",
        "",
        "Piano takes, demoted from the benchmark by owner ruling",
        "(reset 2026-09-01): the demo is the case; a take is one valid",
        "realization, kept as reference. Verified truth, gates nothing,",
        "pooled into none of the numbers above.",
        "",
        f"Cases: {', '.join(ref['case_ids'])}",
        "",
        "| field | n | correct | wrong | abstained | accuracy | wilson 95% "
        "| truth in family | failure modes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in ref["fields"].items():
        acc = s["accuracy"] if s["accuracy"] is not None else "—"
        wil = s["accuracy_wilson95"] or "—"
        modes = ", ".join(f"{k}×{v}" for k, v in s["failure_modes"].items()) or "—"
        lines.append(
            f"| {name} | {s['n']} | {s['correct']} | {s['wrong']} | "
            f"{s['abstained']} | {acc} | {wil} | {family_cell(s)} | {modes} |"
        )
    lines.append("")
    return lines
