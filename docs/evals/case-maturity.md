# Case maturity — whose truth is this row?

**Status:** implemented 2026-08-25 (charter workstream W1.5, EVAL-CHANGE);
the pinning rule below added 2026-09-01 (W1.6, EVAL-CHANGE).
Companions: [beat grids](beat-grids.md) (the grid-level `provisional`
flag this generalizes), [the annotation convention](annotation-convention.md),
[the agent charter](../research/agent-charter.md) rule 2.

The charter lets an agent session create **new** case and trace files for
new material, and forbids it from inventing ground truth. Both hold at
once only if the harness can tell owner-verified truth from
agent-proposed truth and treat them differently. `maturity` is that
distinction.

## The key

```yaml
id: barre1-k-execution
maturity: provisional          # provisional | verified — default: verified
input:
  trace: traces/barre1-k-execution/
```

- **`verified`** — the owner checked these labels. The row gates, and it
  counts in every headline number. **This is the default**, so the 30
  cases written before W1.5 keep exactly the meaning they had; nothing
  about the blessed baseline moved when this landed.
- **`provisional`** — an agent proposed these labels. The row is scored
  and reported, in its own slice with its own n, and it **gates
  nothing**.

A typo (`maturity: probational`) is a load error, not a silent promotion
to `verified` — the failure mode this key exists to prevent is exactly a
guessed label quietly becoming a gate.

Flipping `provisional` → `verified` is an **owner act**, requested via a
BLOCKED ledger note. An agent never promotes its own labels.

## What "gates nothing" means, precisely

| Consumer | Provisional rows |
|---|---|
| `compare_outcomes` / the tier-1 pytest gate | **excluded** |
| tier-1 headline `fields`, `ece`, `slices`, `tempo_metrics`, `n_cases` | **excluded** |
| the `provisional` block of the same summary | included, own n |
| the **run** artifact's `outcomes` map | included (recorded, not gated) |
| the **blessed baseline**'s `outcomes` map | **excluded** (W1.6) |
| stage1 per-clip rows | included, flagged `provisional` |
| stage1 `aggregate_verified` | **excluded** |

The exclusion set the gate uses is the **union** of this run's
provisional ids and the baseline's own, read from
`summary.provisional.case_ids`. Two consequences worth stating:

1. The blessed baseline is **self-describing** — comparing runs never
   re-reads the case files.
2. A row that flips maturity in *either* direction still cannot gate on
   the run where it flipped, which is what makes owner verification a
   safe, reviewable event rather than a surprise red build.

A baseline blessed before W1.5 has no `provisional` key at all; that
degrades to "nothing excluded", which is the correct reading of a
corpus that was entirely verified.

## Which exclusion is the guarantee (W1.6)

There are two exclusions and they are **not** two implementations of one
idea. Read them as a pair:

- **Pinning time — `blessed_report`, in `runner.py`.** The baseline's
  `outcomes` map *is* the gating corpus, so `bless` writes only
  owner-verified rows into it and names what it withheld under
  `outcomes_withheld_provisional`. **This is the guarantee of record:**
  the gating set can only grow by the owner verifying a case *and*
  re-blessing.
- **Comparison time — `compare_outcomes(..., provisional=…)`.** A
  runtime filter for rows that are provisional *in the current run* —
  fresh ingestion the baseline has never seen — or whose maturity moved
  since the bless. Necessary, but it bounds one comparison, not what the
  baseline is allowed to claim.

Why the distinction had to be written down: until 2026-09-01 only the
second existed, and it hid the absence of the first. `bless` was
`shutil.copyfile(run, baseline.json)`, so the first bless after the
barre-1 ingestion pinned **52** rows, 22 of them agent-authored. Gating
*decisions* were unaffected — which is exactly why nothing caught it
except W1.5's tripwire,
`test_the_gating_corpus_is_exactly_the_blessed_thirty`.

**Consequence, by design:** the run after the owner verifies a case
reports it as `new case (not in baseline)` and the tier-1 gate fails
until a re-bless. That is not a regression. Growing the gating set is a
review event and should cost a deliberate act.

## Stage-1: a row is only as verified as its weakest label

Beat grids carried a `provisional` flag before cases did. W1.5 widens
that flag rather than adding a second one: a stage-1 row is provisional
when its **grid** is provisional **or** its **case** is. A verified grid
under a provisional case is still a provisional row.

## The `accompanied` tag gained a third state

Owner ruling B5 (2026-08-24). `accompanied` is no longer a boolean:

```yaml
accompanied: false                 # no accompaniment in the recording
accompanied: true                  # a teacher counting over a pianist
accompanied: accompaniment_only    # the pianist, and nothing else
```

`accompaniment_only` describes the six Ballet Barre 1 takes that are
class recordings of the pianist playing the exercise, with no teacher
voice and no dancer in frame. They are neither of the other two states,
and the difference is not cosmetic: their truth, if ever labeled, comes
from **the piano's beat** (owner-annotated), not from counted words.
They are candidate material for future accompaniment-following work —
not a pose testbed (that framing was corrected at B5) and not a counting
testbed.

Values outside the vocabulary are a load error, same as `maturity`.

## For the next ingestion session

1. Freeze traces (existing tooling).
2. Write case files with **`maturity: provisional`** on every one. Do not
   omit the key on new material and do not reason about what the owner
   would probably say — that is the whole failure this prevents.
3. Pre-annotate grids as `provisional` (unchanged).
4. Run the suite. The provisional slice prints its own numbers; the
   headline must not move. If the headline moved, something is mislabeled.
5. Write a BLOCKED ledger note asking the owner to verify.
