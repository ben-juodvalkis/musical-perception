"""
Eval CLI:

    python -m musical_perception.evals run [--suite tier0,tier1,stage1]
    python -m musical_perception.evals bless [--run PATH]
    python -m musical_perception.evals live-check --case ID [--runs 3]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

DEFAULT_EVALS_ROOT = Path("evals")
BASELINE_NAME = "baseline.json"
BASELINE_MD = Path("docs/evals/baseline.md")


def _print_stage1(suite: str, summary: dict) -> None:
    print(f"  {suite}: {summary['pulse_source']} vs beat grids, "
          f"±{summary['tolerance_s']}s — PROVISIONAL slice, gates nothing")
    for c in summary["clips"]:
        prov = "prov" if c["provisional"] else "VERIFIED"
        print(f"    {c['case_id']:34s} {prov:8s} ref={c['n_ref']:3d} "
              f"pred={c['n_pred']:3d} P={c['precision']} R={c['recall']} "
              f"F={c['f_measure']} async={c['asynchrony_mean_ms']}"
              f"±{c['asynchrony_sd_ms']}ms")
    for label in ("aggregate_provisional", "aggregate_verified"):
        agg = summary.get(label)
        if agg:
            print(f"    {label}: clips={agg['n_clips']} P={agg['precision']} "
                  f"R={agg['recall']} F={agg['f_measure']} "
                  f"(macro {agg['f_measure_macro']}) asynchrony mean="
                  f"{agg['asynchrony_mean_ms']}ms median={agg['asynchrony_median_ms']}ms "
                  f"sd={agg['asynchrony_sd_ms']}ms")
    for style, agg in (summary.get("slices") or {}).items():
        print(f"    slice {style}: F={agg['f_measure']} "
              f"asynchrony mean={agg['asynchrony_mean_ms']}ms (n={agg['n_clips']})")
    if summary["missing_grids"]:
        print(f"    missing grids ({len(summary['missing_grids'])}): "
              f"{', '.join(summary['missing_grids'])}")
    for err in summary["errors"]:
        print(f"    ERROR {err}")


def _cmd_run(args) -> int:
    from musical_perception.evals.report import (
        build_report, family_cell, tempo_metrics_line, write_run,
    )
    from musical_perception.evals.runner import compare_outcomes, run_suites

    root = Path(args.evals_root)
    suites = [s.strip() for s in args.suite.split(",") if s.strip()]
    results = run_suites(suites, root)
    report = build_report(results)
    path = write_run(report, root / "runs")
    print(f"wrote {path}")
    for suite, data in report["suites"].items():
        summary = data["summary"]
        if "clips" in summary:  # stage1-style suite
            _print_stage1(suite, summary)
            continue
        for name, s in summary["fields"].items():
            print(f"  {suite:6s} {name:22s} n={s['n']:3d} correct={s['correct']:3d} "
                  f"wrong={s['wrong']:3d} abstained={s['abstained']:3d} "
                  f"accuracy={s['accuracy']} truth_in_family={family_cell(s)}")
        tm_line = tempo_metrics_line(summary)
        if tm_line:
            print(f"  {suite:6s} {tm_line}")

    baseline_path = root / BASELINE_NAME
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text())
        changes = []
        for suite in report["suites"]:
            base_suite = baseline.get("suites", {}).get(suite)
            if base_suite:
                changes += compare_outcomes(
                    report["suites"][suite]["outcomes"], base_suite["outcomes"]
                )
        if changes:
            print("\noutcome changes vs baseline:")
            for c in changes:
                print(f"  {c}")
        else:
            print("\nno outcome changes vs baseline")
    return 0


def _latest_run(runs_dir: Path) -> Path | None:
    runs = sorted(runs_dir.glob("run-*.json"))
    return runs[-1] if runs else None


def _cmd_bless(args) -> int:
    from musical_perception.evals.report import render_markdown_baseline

    root = Path(args.evals_root)
    run_path = Path(args.run) if args.run else _latest_run(root / "runs")
    if run_path is None or not run_path.is_file():
        print("no run to bless — do `evals run` first", file=sys.stderr)
        return 2
    shutil.copyfile(run_path, root / BASELINE_NAME)
    report = json.loads(run_path.read_text())
    BASELINE_MD.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_MD.write_text(render_markdown_baseline(report))
    print(f"blessed {run_path.name} -> {root / BASELINE_NAME}")
    print(f"regenerated {BASELINE_MD}")
    return 0


def _cmd_live_check(args) -> int:
    """Re-run the REAL pipeline on a case's media N times; every scored
    field must come back correct in every run. The acceptance gate for
    prompt/schema changes (needs local media + GEMINI_API_KEY)."""
    from musical_perception.analyze import analyze
    from musical_perception.evals.cases import load_cases
    from musical_perception.evals.runner import score_parameters

    root = Path(args.evals_root)
    cases = {c.id: c for c in load_cases(root / "cases")}
    if args.case not in cases:
        print(f"unknown case {args.case!r}; have {sorted(cases)}", file=sys.stderr)
        return 2
    case = cases[args.case]
    if not case.media or not Path(case.media).is_file():
        print(f"case media not available locally: {case.media}", file=sys.stderr)
        return 2

    wanted = [f.strip() for f in args.fields.split(",")] if args.fields else None
    failures = 0
    for i in range(args.runs):
        try:
            result = analyze(case.media)
        except Exception as e:  # transient API/network faults are a run error
            failures += 1
            print(f"run {i + 1}/{args.runs}: ERROR  {type(e).__name__}: {e}")
            continue
        scores = score_parameters(result, case)
        if wanted:
            scores = [s for s in scores if s.field in wanted]
        verdicts = ", ".join(f"{s.field}={s.outcome}({s.predicted})" for s in scores)
        ok = all(s.outcome == "correct" for s in scores)
        failures += 0 if ok else 1
        print(f"run {i + 1}/{args.runs}: {'PASS' if ok else 'FAIL'}  {verdicts}")
    print(f"\n{args.runs - failures}/{args.runs} runs fully correct")
    return 0 if failures == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m musical_perception.evals")
    parser.add_argument("--evals-root", default=str(DEFAULT_EVALS_ROOT))
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run suites, write a run artifact")
    p_run.add_argument("--suite", default="tier0,tier1")
    p_run.set_defaults(fn=_cmd_run)

    p_bless = sub.add_parser("bless", help="promote a run to the baseline")
    p_bless.add_argument("--run", default=None, help="run JSON (default: latest)")
    p_bless.set_defaults(fn=_cmd_bless)

    p_live = sub.add_parser("live-check", help="N live runs of one case must all pass")
    p_live.add_argument("--case", required=True)
    p_live.add_argument("--runs", type=int, default=3)
    p_live.add_argument("--fields", default=None, help="comma list, e.g. counts,sides")
    p_live.set_defaults(fn=_cmd_live_check)

    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
