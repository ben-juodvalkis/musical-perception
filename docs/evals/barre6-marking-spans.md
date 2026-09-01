# Marking spans and time-to-commitment — Ballet Barre 6

*Owner-annotated 2026-09-01 in an attended session. Provisional: these
numbers gate nothing and no scorer reads this file yet.*

Two quantities, deliberately separated — they are **not** the same thing:

- **intended-tempo span** — where the teacher is at the tempo he wants,
  as opposed to explaining, or getting through it. This is Vision 08's
  *marking segmentation* metric, which had never been annotated.
- **time-to-know** — the frame by which the owner, a professional dance
  accompanist, had enough to start playing. The human ceiling to lay
  against W13(b)'s machine curve.

Frames are 25 fps, relative to the clip.

| demo | clip | intended-tempo span | % of clip | knew by | % |
|---|---|---|---|---|---|
| `tendu-warmup` | 55s | 4.0–40.0s | 65% | — | — |
| `plie` | 78s | 8.4–12.0s | 5% | — | — |
| `tendu` | 51s | 10.4–16.0s | 11% | — | — |
| `degage` | 55s | 5.2–22.4s | 31% | — | — |
| `ballonne` | 41s | 5.0–11.2s | 15% | — | — |
| `coupe-barre` | 38s | 3.8–34.0s | 79% | — | — |
| `rond-de-jambe` | 89s | 11.2–32.0s | 23% | 14.9s | 17% |
| `fondu` | 51s | 9.4–45.0s | 70% | 17.2s | 34% |
| `frappe` | 55s | 3.0–29.2s | 48% | 3.0s | 5% |
| `developpe` | 49s | 18.4–24.8s | 13% | 11.6s | 24% |

## What these say

**The tempo-bearing window is short and unpredictable.** The
intended-tempo fraction runs **5% to 79%**. On `plie` it is 3.6s of 78s
— about seven beats. A system estimating tempo across a whole demo reads
explanation 95% of the time on that clip.

**In tempo is not the same as at the intended tempo.** On `degage` 60%
of the clip is metrical but only 31% is at the tempo he wants. Both
sides of that boundary sound like a tempo, so no detector recovers it —
it is a judgement about intent. The grid format's region kinds
(`silent_beat`, `free_time`, `excluded_explanation`) have no value for
it; a fourth kind is **proposed, not ruled**.

**The expert commits far earlier than the pipeline.** Time-to-know spans
3.0–17.2s (5–34% of clip). W13(b) measured the machine settling at
**60–88%**. The owner's worst case sits inside the machine's best; the
distributions do not overlap.

**Fast is not the same as correct.** `frappe` was read fastest (3.0s)
and is also the clip where the metric level was most wrong — the marking
counts 135 where the piano plays 79, because the teacher counts at
double. Hearing a rate is quick; placing it on the ladder is not.

## Correction

`developpe` was first logged as known-by 24.8s. The owner flagged that
as the end of his *steady* stretch rather than time-to-know; re-checked
and corrected to **11.6s**. The original figure would have overstated
the human time-to-commitment by a factor of two.
