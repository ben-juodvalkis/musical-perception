# Rig Capture Checklist

Self-recorded fixtures with **truth manufactured at recording time**
(Vision 13 source A; ADR-009 bootstrap). Each clip: you, a metronome in an
earphone, and a label card filled in before you press record. No
annotation ever needed afterward.

## Ground rules (every clip)

- **Metronome in one earphone only** — it must never be audible on the mic.
- **Fill in the label card first** (below). Never say the tempo, meter, or
  count length on-mic — that would leak labels into the audio the system
  analyzes.
- 25–45 seconds per clip; **one exercise's marking per clip** (that is the
  unit `analyze()` consumes).
- Phone voice memos are fine — audio-only for speed. Re-do a handful on
  video later only if we want pose/quality signal from them.
- Quiet room for all of these (noisy variants are a later batch).
- Name files `rig-<style>-<meter>-<bpm>-<variant>` — e.g.
  `rig-numbers-3-4-90-clean.m4a` — and drop them in `audio/rig/`
  (gitignored; the trace is what gets committed).

## Label card (fill per clip, becomes the case YAML)

| field | example |
|---|---|
| file | rig-names-2-4-160-long.m4a |
| count_style | numbers / step_names / vocables / minimal |
| meter | 3/4 |
| metronome BPM | 90 |
| counts in ONE phrase (no prep, no coda) | 32 |
| sides | one / both |
| subdivision spoken | none / duple / triplet |
| variant | clean / explained / prep / half-tempo / coda / … |
| notes | anything odd you did |

## The clips

★ = record these ten first if time is short.

### Batch 1 — clean baseline (8)

Counting with numbers, then the same four as small marked combinations in
step names only (a tendu-ish pattern is fine), no numbers at all:

1. numbers · 2/4 · ~120 · 8 counts, twice through
2. ★ numbers · 3/4 · ~90 · count it the way you naturally count a waltz
   combination (note on the card exactly how you counted: in 3s? in 6s?)
3. numbers · 4/4 · ~104 · 8 counts, twice through
4. ★ numbers · 6/8 · ~100 · compound feel
5. step names · 2/4 · ~120
6. ★ step names · 3/4 · ~90
7. step names · 4/4 · ~104
8. step names · 6/8 · ~100

### Batch 2 — the known killers (8)

9. ★ numbers · 4/4 · ~104 · stop mid-phrase, explain something for ~5s,
   resume counting (the interleaved-explanation case)
10. ★ step names · 4/4 · ~104 · same interleaved explanation
11. numbers · 4/4 · ~104 · prep count first ("5, 6, 7, 8" → "1 …")
12. ★ half-tempo marking: exercise intended at ~120 but you speak at ~60
    (card carries BOTH numbers)
13. ★ step names · 2/4 · ~160 · a LONG combination, 32–64 counts,
    frappé-style fast (this is the shape we currently abstain on)
14. adagio: step names · 4/4 · ~60–66 · slow sustained marking
15. ★ triplet counting · 4/4 · ~80 · "1-and-a-2-and-a…" (the ÷2/÷3 defect's
    home turf)
16. duple counting · 4/4 · ~104 · "1-and-2-and…"

### Batch 3 — structure and style stressors (8)

17. vocables/minimal: "da da DA da… and… and…" · any meter, note it
18. ★ a 32-count phrase counted as **four rounds of 1–8** (numbers) — this
    is the cycle-vs-phrase ambiguity ADR-012 documents; label counts=32
19. same 32-count phrase, then "other side" and mark it again —
    sides=both
20. ★ waltz 3/4 with real lilt (balancé feel), step names
21. big-accent grand-allegro-ish marking · 4/4
22. ★ the grande-battement style: step names with **quantity numbers**
    mixed in ("we take two… one more…") — label card still carries the
    true phrase length
23. deliberately quiet, mumbled marking (energy floor)
24. a phrase with a **closing port de bras + balance coda** — counts on
    the card EXCLUDE the coda (ADR-011's boundary rule, tested for real)

## After recording (per clip, ~60 seconds)

```bash
python -m musical_perception "audio/rig/<file>" --record-traces
```

Then one case file in `evals/cases/rig-<name>.yaml`:

```yaml
id: rig-numbers-3-4-90-clean
input:
  trace: traces/rig-numbers-3-4-90-clean/
  media: "audio/rig/rig-numbers-3-4-90-clean.m4a"
tags: {source: rig, teacher: ben, slot: counting, count_style: numbers,
       explanation: none, lang: en, accompanied: false, snr_band: high}
expect:
  marking_bpm: 90
  meter: "3/4"
  subdivision: none
  counts: 8
  # sides only when the clip demonstrates both
notes: metronome-locked; label card 2026-08-__
```

Finish the session with:

```bash
python -m musical_perception.evals run --suite tier0,tier1
python -m musical_perception.evals bless
```

and commit the traces + cases + new baseline together — the diff IS the
finding.
