# Pulse sidecars — the acoustic pulse stream, frozen beside its trace

**Added at rung M workstream W11 (EVAL-CHANGE, 2026-08-28), branch
`agent/marathon`, under the owner-ratified sidecar carve-out
(agent-charter rule 2, 2026-08-28).**

A frozen trace records what the *models* said — Whisper's words, Gemini's
raw response, pose landmarks. The rung-2 acoustic pulse stream is not a
model output: it is a deterministic function of the media, and the media
is gitignored. Replaying it therefore used to mean re-deriving events
from files that exist only on the runner that recorded them, which is
Standing Lesson 9's failure mode — a channel nobody can replay is a
channel nobody iterates on.

`pulse.json` closes that gap.

## Format — `evals/traces/<case-id>/pulse.json`

```json
{
 "sidecar_format": 1,
 "extractor": "acoustic-pulse/1",
 "media": "audio/rig/rig-names-4-4-104-clean.mp3",
 "media_sha256": "fc9ddd09…",
 "recorded_at": "2026-08-29T05:50:00+00:00",
 "git_sha": "…",
 "params": {"peakrate": {…}, "events_per_nucleus": "all", …},
 "events": [2.3244, 3.0053, …]
}
```

`events` are `precision/pulse.acoustic_pulse_events` times in seconds,
rounded to 0.1 ms — voiced-gated peakRate events filtered to syllable
nuclei. `params` freezes the extractor constants that produced them, so
a sidecar recorded under different settings is self-identifying rather
than silently comparable.

## The checksum contract

The carve-out permits ADDING sidecars to existing trace directories only
when the source media is checksum-verified against the trace's stored
hash. That is enforced in code, not by hand:

- **Recording** hashes the media file and refuses to write unless it
  equals `meta.json`'s `media_sha256`. A clip whose media is missing or
  whose hash disagrees is skipped and named in the output — never
  recorded from whatever audio happens to sit at that path.
- **Loading** re-checks the sidecar's own `media_sha256` against the
  trace's. This is offline and needs no media, so every consumer gets
  the check for free. A mismatch raises `SidecarError`: which of the two
  files is the right one is not something this layer may guess.

## Recording

```bash
python -m musical_perception.evals record-pulse            # every case
python -m musical_perception.evals record-pulse --only rig-names-4-4-104-clean
python -m musical_perception.evals record-pulse --force    # re-record existing
```

Existing sidecars are skipped unless `--force`. Recording needs the
`[prosody]` extra (librosa + parselmouth) and, for video sources,
`ffmpeg`; it does not need any model or API key.

## Scoring with it

`stage1` keeps `whisper-word-starts` as its pulse source — the blessed
suite means exactly what it always meant. The acoustic stream is a
second, separately named suite:

```bash
python -m musical_perception.evals run --suite stage1,stage1-peakrate
```

Both are reported side by side and **both gate nothing**.

### Read these numbers with the anchoring caveat

Most verified grids were annotated `anchored` — seeded from this same
detector's onsets and then corrected by the owner. Scoring peakRate
against them is therefore partly circular: of the 895 matched pairs in
the first W11 run, **769 (86%) coincide with a frozen onset to within
1 ms**, and the two `provisional` grids (never owner-corrected) score a
meaningless P=R=F=1.000 because their beats *are* the detector's output.

The three `from_scratch` grids — `adr006-exercise-1-demo`,
`adr010-grande-battement`, `frappe` — carry **0 of 94** exact
coincidences. For any external magnitude claim, quote that cohort, per
the standing rung-2 anchoring caveat.

### Update 2026-09-03 (W11-c): the caveat now cuts both ways

The barre-6 traces gained sidecars on 2026-09-03, and the two cohorts
separated sharply.

**The circular cohort grew from 2 clips to 19 and now prints a headline.**
`aggregate_provisional: clips=19 P=1.0 R=0.999 F=1.0 async=0.0±0.0ms`.
Those grids — the 17 barre-6 takes, `barre6-releve-finish-take1` and
`barre6-ballonne-demo` — carry `annotator: peakrate-tap-assist/1` and
were never owner-corrected, so their beats *are* this detector's output.
**Never quote that aggregate.** A scorer-side fix (suppress or asterisk
rows whose grid annotator is the extractor being scored) is proposed in
the 2026-09-03 ledger entry and is not implemented here.

**The honest cohort grew too, and it is now where the rung is scored.**
The 8 owner-tapped barre-6 demo grids are `annotator: owner-live-tap/1`,
`annotation_method: from_scratch`, and **1 of their 419 beats** falls
within 1 ms of a frozen event. Together with the three original
`from_scratch` grids the quotable cohort is **11 clips / 513 beats**.
On the 8 demos the frozen stream scores macro **P 0.130 / R 0.347 /
F 0.187** against **F 0.139** for Whisper word starts on the same grids,
at a median **2.545 events per owner beat**.

**One consequence for reading the suite:** `slice step_names` moved
0.747 (n=11) → 0.374 (n=19) when the demos joined. That is coverage, not
regression — no pre-existing row changed.
