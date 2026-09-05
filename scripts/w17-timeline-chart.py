"""W17 — render the tempo timeline, with the owner's windows overlaid.

    python scripts/w17-timeline-chart.py --clip barre6-frappe-demo

Two panels: `causal` (everything heard so far - when does a technique commit)
and `trailing` (a moving window - what is the tempo right now). The owner's
annotation is shaded if present; the study is designed so that annotation
happens BEFORE this chart is ever looked at.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "research" / "w17"

STYLE = {  # technique -> (colour, linestyle)
    "pulse_allpairs": ("#0b6fa4", "-"),
    "pulse_median":   ("#4da6d9", "--"),
    "markers_allpairs": ("#1a7f5a", "-"),
    "markers_median": ("#57b894", "--"),
    "words_allpairs": ("#8a4fa8", "-"),
    "words_median":   ("#b98bd0", "--"),
    "librosa_dp":     ("#c65a11", "-"),
    "librosa_plp":    ("#e8973f", "--"),
    "librosa_acf":    ("#8c6d3f", ":"),
    "grid_reference": ("#111111", "-"),
}
SHADE = {"fullout": ("#2e8b57", 0.16), "marking": ("#c0392b", 0.10),
         "talking": ("#7f8c8d", 0.08)}


def load(clip: str):
    d = json.loads((OUT / f"{clip}-timeline.json").read_text())
    ann_p = OUT / f"{clip}.owner-windows.json"
    ann = json.loads(ann_p.read_text()) if ann_p.is_file() else None
    import yaml
    case = yaml.safe_load((ROOT / "evals" / "cases" / f"{clip}.yaml").read_text())
    return d, ann, case["expect"].get("marking_bpm")


def series(rows, mode, tech):
    pts = [(r["t"], r["bpm"]) for r in rows
           if r["mode"] == mode and r["technique"] == tech and r["bpm"] is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="barre6-frappe-demo")
    ap.add_argument("--ymax", type=float, default=260.0)
    a = ap.parse_args()
    data, ann, truth = load(a.clip)
    rows, meta = data["rows"], data["meta"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)
    for ax, mode, title in zip(
            axes, ("causal", "trailing"),
            ("CAUSAL — everything heard from the start (when does it commit?)",
             f"TRAILING {meta['trailing_s']:.0f}s — what is the tempo right now?")):
        if ann:
            for r in ann["regions"]:
                key = r["text"].split("=")[0].split(":")[0].strip().lower()
                if key in SHADE:
                    c, al = SHADE[key]
                    ax.axvspan(r["start"], r["end"], color=c, alpha=al, lw=0)
            for p in ann["points"]:
                if p["text"].lower().startswith("commit"):
                    ax.axvline(p["start"], color="#111", lw=2.0, ls="-.")
                    ax.annotate("owner commits", (p["start"], 0.96),
                                xycoords=("data", "axes fraction"),
                                fontsize=9, rotation=90, va="top", ha="right")
        if truth:
            ax.axhline(truth, color="#111", lw=1.2, alpha=.55)
            ax.annotate(f"labelled truth {truth:g}", (0.995, truth),
                        xycoords=("axes fraction", "data"), fontsize=8,
                        va="bottom", ha="right", alpha=.75)
        for tech, (c, ls) in STYLE.items():
            xs, ys = series(rows, mode, tech)
            if not xs:
                continue
            ref = tech == "grid_reference"
            ax.plot(xs, ys, color=c, ls=ls, lw=2.4 if ref else 1.5,
                    alpha=1.0 if ref else .9,
                    label=("owner's taps (reference)" if ref else tech), zorder=3 if ref else 2)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("BPM")
        ax.set_ylim(0, a.ymax)
        ax.grid(alpha=.18)
    axes[1].set_xlabel("time in clip (s)")
    axes[0].legend(ncol=4, fontsize=8, loc="upper right", framealpha=.9)
    sub = "owner annotation overlaid" if ann else "NO owner annotation yet — shading appears once marked"
    fig.suptitle(f"{meta['clip']} — tempo over time by technique   ({sub})",
                 fontsize=12, y=.985)
    fig.tight_layout(rect=(0, 0, 1, .965))
    dest = OUT / f"{meta['clip']}-timeline.png"
    fig.savefig(dest, dpi=150)
    print(f"chart -> {dest}"
          f"{'' if ann else '   [no annotation yet]'}")


if __name__ == "__main__":
    main()
