"""§7 re-spec diagnostic (2026-09-05) — reproduces the two tables in §7.1/§7.2.

Neither is a score for any proposed mechanism: both are facts about the
INPUTS (PP-1's frozen per-clip pulse estimates, the owner's §6 bands, the
verified demo labels). Nothing here is tuned. Run: python scripts/spec7-multiple-diagnostic.py
"""

# truth = evals/cases/barre6-*-demo.yaml expect.marking_bpm (owner-verified)
# pulse = PP-1 RESULTS per-clip table, ledger 2026-09-03 (all-pairs period)
# band  = docs/research/pulse-next-step.md §6, dictated by the owner 2026-09-05
DEMOS = {
    "coupe-barre":   (108, 108.3, None),
    "degage":        (110, 113.9, (85, 105)),
    "fondu":         (86,   45.5, (100, 100)),   # point estimate -> +-8%
    "frappe":        (135, 144.0, (120, 140)),
    "plie":          (120,  42.2, (85, 125)),
    "rond-de-jambe": (96,   31.9, (85, 120)),
    "tendu":         (102,  None, (73, 120)),    # pulse refused (boundary artifact)
    "tendu-warmup":  (112,  42.2, None),
}
MULTIPLES = (1, 2, 3, 4)
TOL = 0.08


def integer_subdivision_table():
    print("§7.1 — is the pulse an integer subdivision of the truth?")
    hits = considered = 0
    for name, (truth, pulse, _) in DEMOS.items():
        if pulse is None:
            print(f"  {name:<15} pulse refused"); continue
        considered += 1
        ratio = truth / pulse
        k = min(MULTIPLES, key=lambda m: abs(ratio - m))
        err = abs(ratio - k) / k
        hits += err <= TOL
        print(f"  {name:<15} truth {truth:>5}  pulse {pulse:>6.1f}  "
              f"ratio {ratio:>4.2f}  x{k} off {err:>5.1%}"
              f"{'  <-- integer' if err <= TOL else ''}")
    print(f"  => {hits} of {considered} land within {TOL:.0%} of an integer subdivision\n")


def hard_filter_table():
    print("§7.2 — does a HARD band filter select the right multiple?")
    hit = miss = 0
    for name, (truth, pulse, band) in DEMOS.items():
        if pulse is None or band is None:
            print(f"  {name:<15} n/a ({'pulse refused' if pulse is None else 'band declined'})")
            continue
        lo, hi = band
        if lo == hi:
            lo, hi = lo * (1 - TOL), hi * (1 + TOL)
        inside = [(m, pulse * m) for m in MULTIPLES if lo <= pulse * m <= hi]
        if not inside:
            near = min((abs(pulse * m - lo), abs(pulse * m - hi)) for m in MULTIPLES)
            miss += 1
            print(f"  {name:<15} band {band[0]}-{band[1]}: none survive "
                  f"(closest multiple misses by {min(near):.1f} BPM)  MISS")
            continue
        _, pick = min(inside, key=lambda c: abs(c[1] - (lo + hi) / 2))
        ok = abs(pick - truth) / truth <= TOL
        hit += ok; miss += not ok
        print(f"  {name:<15} band {band[0]}-{band[1]}: picks {pick:.1f} vs truth "
              f"{truth}  {'HIT' if ok else 'MISS'}")
    print(f"  => {hit} hit / {miss} miss — a hard band is a fold; Standing Lesson 2 forbids it\n")


def band_contains_truth():
    print("§7.2 — does the owner's band contain the owner's own label?")
    inside = total = 0
    for name, (truth, _, band) in DEMOS.items():
        if band is None:
            continue
        lo, hi = band
        if lo == hi:
            lo, hi = lo * (1 - TOL), hi * (1 + TOL)
        total += 1; ok = lo <= truth <= hi; inside += ok
        print(f"  {name:<15} truth {truth:>5}  band {lo:g}-{hi:g}  "
              f"{'inside' if ok else 'OUTSIDE'}")
    print(f"  => {inside} of {total}; a hard band would veto the truth on {total - inside}")


if __name__ == "__main__":
    integer_subdivision_table()
    hard_filter_table()
    band_contains_truth()
