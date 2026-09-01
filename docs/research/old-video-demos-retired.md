# The four old video demos — retired 2026-09-01

Removed from the corpus by owner ruling, the same ruling that retired
Ballet Barre 1 and for the same reason: *"I've said several times that I
want to remove those from the corpus until we can re-analyze them... I
thought we were just using today's Barre 6 and the ones I recorded in my
own voice."*

| case | label it carried | provenance |
|---|---|---|
| `adr006-exercise-1-demo` | 117, 3/4, none | ADR-006 table |
| `adr007-plies-demo` | 118, **4/4, duple** | ADR-007 table |
| `adr010-grande-battement` | 104, 4/4, 32 counts, 2 sides | ADR-010 table |
| `frappe` | 160, 64 counts | early rung-2 material |

## Why, beyond the owner's preference

Every label came across from an ADR table and **had never been re-heard**.
`adr007-plies-demo`'s entire note was *"ADR-007's correct row (118.0, 4/4,
duple — 'Yes')"*.

And on the night they were retired, that clip's label was caught looking
wrong. The owner tapped nine beats into it and said they sat on **beats 1
and 3 of a three-beat bar**. That gives a bar of **1.497s**: a 3/4 bar at
118 is 1.525s (**1.8% off**), a 4/4 bar at 118 is 2.034s (**36% off**).
Independent autocorrelation over the tapped window is flat or negative at
*every* candidate — bar of 3, bar of 4, and the beat itself — so the
signal cannot arbitrate and only a human can.

That is one wrong meter found in the only one of the four anybody looked
at closely. The other three were never checked.

## What replaces them

Nothing, deliberately. The corpus is now **exactly two provenances**:

- **26 rig clips** — the owner's own voice against a confirmed metronome.
- **26 Ballet Barre 6 clips** — cut, labelled and read back by him on
  2026-09-01.

Both are material he can vouch for directly. That is the whole point.

## If they are ever re-analysed

Same recipe as [barre1-reanalysis-plan.md](barre1-reanalysis-plan.md).
Do not restore the deleted traces — re-cut and re-label from the media,
which is still on the owner's machine. And note that `adr007-plies-demo`
is the one with a live open question: **is it 3/4, or 4/4 with the owner
tapping the plié's down-and-up movement rather than the metric beat?**
He hedged ("a triplet or 3/4"), and that hedge is the thing to settle
first.

## Housekeeping done at the same time

Fifteen orphaned trace directories were deleted, of which **nine had
mangled names** (`arre6-plie-demo`, `rre6-rond-de-jambe-take2`, …). Those
were debris from a shell bug earlier in the session: a
`while read … done < file` loop whose Python child consumed the same
stdin, eating characters out of the filename list. They had been sitting
in `evals/traces/` looking like real traces.
