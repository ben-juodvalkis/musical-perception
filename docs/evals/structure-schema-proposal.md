# PROPOSED: a `structure:` block in the case schema

**Status: PROPOSED 2026-09-01, owner's to rule. EVAL-CHANGE — its own
increment, never bundled with ingestion or a pipeline change.**

Written out of the Ballet Barre 6 labelling session, which produced the
first corpus material rich enough to show what the current schema cannot
say. The worked example is [barre6-structure.yaml](barre6-structure.yaml),
which already holds all 32 clips in this shape.

## Why the current schema is short

`expect` carries one `counts`, one `bpm` and one `meter` per case. Three
things the barre-6 session established cannot be expressed in that:

### 1. ~~`count_unit`~~ — WITHDRAWN 2026-09-01

The proposal originally argued for a `count_unit` field, because `counts`
was being stored sometimes in bars and sometimes in beats. **The owner
removed the need rather than the symptom:** a count is always one beat,
and what varies between exercises is the phrase length (24 counts in a
3/4 exercise counted in 8 bars of 3; 8 counts in a 4/4 one). One meaning,
no tag.

Kept here as a record of a field that looked load-bearing and was not.
What survives is the weaker claim below: a section's length and tempo
still cannot be expressed in one `counts` and one `bpm`.

### 2. Sections need their own tempo

Five of the ten barre-6 exercises have a tail — a balance or a port de
bras — at a different tempo from the exercise it follows:

| exercise | exercise proper | tail |
|---|---|---|
| tendu | 112 | balance at **55** |
| rond de jambe | 95 | port de bras at **115** / **120** |
| dégagé | 121 | balance at 121 (same) |
| ballonné | 63 | balance at 63 (same) |
| développé | 80 → ~100 *within* the take | — |

A single `bpm` per case cannot say "64 counts at 112, then 16 at 55". The
session worked around it by cutting tails into separate clips and then
**deleting them** under the owner's single-tempo ruling — which kept the
corpus clean but threw the structure out of the cases. It survives only
in the structure record.

### 3. A demo describes the exercise, not the clip

A demo states a 64-count exercise inside a 51-second clip, because marking
is abbreviated. Any consumer assuming `counts` describes the recording is
wrong on every demo row. Deriving `count_unit` by fitting counts to *clip*
duration mis-set exactly one row for exactly this reason.

## The proposed shape

```yaml
structure:
  sides: 2
  sides_continuous: true          # true | false | unknown — the demo often cannot say
  describes: exercise             # exercise (demo rows) | clip (take rows)
  sections:
    - role: exercise              # exercise | balance | port_de_bras | preparation
      counts: 64
      count_unit: beat            # beat | bar
      bpm: 112
      meter: "4/4"
    - role: balance
      counts: 16
      count_unit: beat
      bpm: 55
      meter: "4/4"
      note: "ritardando at the very end"
```

Total length is derived, never stored. `music_seconds` per section is a
derived check against clip duration — the barre-6 record computes it for
all 32 clips and it caught two real errors.

## What it would take

- **Loader:** `_TOP_KEYS` currently makes an unknown top-level key a hard
  error, so `structure` cannot appear in a case file until the loader
  accepts it. Optional key, absent on all 78 existing cases → byte-identical
  output required as the gate, same proof style as W1.5 and W12.
- **§8.2:** an additive amendment, flagged the way `subdivision` was.
- **Scoring:** none initially. **REPORTED-ONLY** — it gates nothing until a
  separate owner ruling, per the W12 precedent.

## What it is not — and the deferred question

Not a phrase-shape representation. It says how long each section is and
at what tempo, not what happens inside it — no front/side/back, no
direction, no figure.

**DEFERRED by the owner, 2026-09-01.** A directional pass was offered
(ten exercises, ~15 minutes of his ear: "two eights front, two side, two
back, two side") and he chose to skip it for now. Recorded so a future
session does not re-litigate it, and so the reason is legible:

- It is **owner-only work.** Direction was mentioned in passing on two
  exercises and never as a full sequence; an agent filling it in would be
  inventing ballet, not transcribing it.
- It is **additive.** `sections` can gain a `phrases:` list later without
  touching anything above it, so nothing here forecloses it.
- The question it answers is **whether a generating model needs phrase
  shape or only phrase length** — whether a uniform 64 counts would sound
  wrong to a dancer, or whether a competent accompanist could play a
  generic 64 and have it work. That is an empirical question best settled
  by *listening to a model's output*, not by annotating in advance.

**Revisit when:** a model is generating class music and its output is the
right length but sounds wrong. That symptom is the trigger; until it
appears, the annotation has no consumer.

## Owner's position on record (2026-09-01)

> "If the schema needs to change, so be it. Today's work is much more
> granular and accurate. It's not the end of the world to throw out or
> repeat existing work that's lower value."

> "I think there needs to be the actual full dense information of the
> actual structure... we will eventually be giving this info to a model
> that will construct the proper music at the right structure and length."
