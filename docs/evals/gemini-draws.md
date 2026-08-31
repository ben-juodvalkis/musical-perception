# Gemini draw sidecars — `gemini-draws.json`

**Status:** format + loader landed 2026-08-30 (W6-a). **Nothing in the
shipping path reads these yet**, and no sidecar has been recorded —
recording needs live model calls and is W6-b.

Companion to [pulse sidecars](pulse-sidecars.md), whose rules this
format deliberately copies rather than reinventing.

## Why

ADR-011 measured 18, 18, 18, 32 BPM from four temperature-0 draws on
identical input. Standing Lesson 4 — *one draw is a coin flip* — has
been in the ledger since the loop began, and the pipeline has consumed
exactly one draw the whole time, because there was nowhere to put the
others. This is that place.

A draw is one model's classification of one transcript: for each
Whisper token, a class in `{beat, and, ah, e, none}` and, for beats, a
beat number. N draws frozen beside the trace are replayable offline
forever, which is Standing Lesson 9's rule — build the replay path
before betting on the channel.

## Format

```json
{
  "sidecar_format": 1,
  "media_sha256": "<the trace's media hash>",
  "transcript_sha256": "<fingerprint of the token sequence>",
  "recorded_at": "2026-09-01T00:00:00+00:00",
  "draws": [
    {
      "draw_id": "flash#0",
      "model": "gemini-2.5-flash",
      "params": {"temperature": 1.0, "top_p": 0.95},
      "words": [
        {"index": 0, "marker_type": "beat", "beat_number": 1},
        {"index": 1, "marker_type": "and"},
        {"index": 2, "marker_type": "none"}
      ]
    }
  ]
}
```

- **Add-only.** The file is created inside an existing trace directory
  under the owner-ratified sidecar carve-out (charter rule 2,
  2026-08-28). No existing file in the trace is touched.
- **`media_sha256`** must equal the trace's. `load_gemini_draws` raises
  `SidecarError` when it does not — a sidecar that drifted from its
  trace describes different audio, and which one is right is not a
  question the loader may guess at.
- **`transcript_sha256`** pins the token sequence the indices address
  (text and start time of every token). A draw's `index` into a
  different transcript is silently wrong rather than loudly wrong; this
  is the one failure mode the format can detect, so it does.
- **Per-draw `model` and `params`.** An ensemble whose members are not
  individually identified cannot be analysed after the fact — which
  draw came from which family is the whole question in "≥ 2 model
  families".
- **Omission means `none`.** A draw that returns only the markers it
  found is a complete draw; every unmentioned index is `none`.
- **A word with no `index` is an error**, not a text-matching fallback.

## How it is consumed

```python
side = load_gemini_draws(trace_dir, words=whisper_words)
beliefs = beliefs_from_draws(side.draws, whisper_words)
result = estimate_rhythm(words, markers, marker_beliefs=beliefs)
```

`beliefs_from_draws` mixes the draws into one `MarkerBelief` per token —
each draw votes 1/N for the class it assigned. `estimate_rhythm` takes
that as `marker_beliefs` and the per-frame Poisson emission charges
**expected support**: a token believed to be a beat with mass 0.4
contributes 0.4 of a beat event's log-rate credit.

Left `marker_beliefs=None`, the beliefs are built one-hot from the
markers and the answer is bit-for-bit the single-draw answer — proven
on the whole corpus at landing (byte-identical `suites` payload,
sha256 `4c27815c…`).

### What consumes the distribution and what does not

| consumer | reads | why |
|---|---|---|
| Poisson emission | the mixture (expected support) | a rate is defined under fractional mass |
| `_stream_support` (robust IOI CV) | the MAP decode | no agreed weighted form |
| `_division` (circular-concentration vet) | the MAP decode | no agreed weighted form |
| `_grouping_ladder` (counted cycle) | the MAP decode | beat numbers are labels, not mass |

Declared rather than discovered: the one-hot gate cannot see this
choice, because one draw makes MAP and expectation the same object.
W6-b revisits it with real draws in hand.

## The hazard, measured before any draws exist

Fractional belief is spent **per token and summed**, so a minority view
spread across many tokens is not a minority in likelihood terms. On a
synthetic clip where the on-beats are certain and the offbeats carry
`p(beat)`, the committed tempo flips from the beat to the half-beat at:

| offbeat tokens | flip point |
|---|---|
| 24 | p = 0.132 |
| 16 | p = 0.159 |
| 12 | p = 0.185 |
| 8 | p = 0.237 |

With N = 5 draws, **one dissenting draw is 0.2** — above the flip point
for any clip with a dozen or more contested tokens. An ensemble is
therefore not automatically more conservative than a single draw: on
level decisions it is *less* so, because it lets a 4–1 minority buy a
metric level the majority rejected. W6-b must either weight the mixture
or measure this directly; assuming a 3-of-5 mixture behaves like a
3-of-5 vote would be wrong.
