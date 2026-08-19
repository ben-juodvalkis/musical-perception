# Research Log — rhythm-core reset

Append-only ledger of every agent iteration on the
[agent charter](agent-charter.md)'s goal ladder. This file is the loop's
memory: sessions are stateless, the ledger is not. Newest entry last.
Never rewrite or delete past entries — corrections get their own entry.

**Entry template** (copy verbatim, fill every field; "n/a" is allowed,
blank is not):

```markdown
## YYYY-MM-DD · rung N · agent/<branch> · cloud|local
Attempted:
Pre-registered expectations:
Result: (eval deltas with numbers; prediction scorecard X/Y landed)
Regressions and classifications: (fake-green-lost / genuine-trade / knife-edge / none)
Lesson (durable, one paragraph):
Status: PROPOSED | BLESSED | REJECTED (reason) | DEAD-END | BLOCKED (needs)
```

## Standing Lessons

Distilled from the ADR record and the retrospective
([ADR-016](../adr/016-rhythm-core-reset.md)); every session reads these
before working. Add to this list only via a ledger entry that earns it.

1. **Words are not the beat.** ASR word timestamps carry a 0–150 ms
   word-dependent early bias vs the perceptual beat (vowel onset /
   peakRate). Never anchor a grid to word starts; never annotate ground
   truth at word starts.
2. **Priors are priors, not post-processing.** A hard fold (the old 70–140
   band) destroys correct out-of-band measurements. Apply priors at level
   selection, multiplicatively, never to the raw measurement.
3. **Levels vote, they don't average.** Mean-IOI across mixed 1×/2×/3×
   onsets lands between metric levels — the field abandoned it ~2000–2005.
   Use harmonic summing / ratio-reinforced clustering / grid regression.
4. **One temp-0 LLM draw is a coin flip** (ADR-011: 18,18,18,32 on
   identical input). Consume distributions or outvote with independent
   evidence (ADR-012), never trust a single draw.
5. **Phrase-final lengthening is expected structure, not noise** (Repp;
   Wightman). Censor or down-weight boundary intervals; never average them
   into tempo.
6. **Silence is evidence.** A hypothesis that predicts a strong beat where
   the teacher voiced nothing pays for it (Povel–Essens; PIPPET).
7. **Sub-4% eval disagreements are noise by construction** (human tapping
   CV is 3–5%). Knife-edge rows gate nothing (ADR-015 typed gates).
8. **A transcription hallucination once scored all-green** (clip 17). Any
   green earned without a verified perceptual chain is provisional until
   the sanity guard or a human confirms it.
9. **The harness sets the gradient** (retrospective F5). Whatever is
   replayable gets iterated; build the trace/replay path for a new channel
   before betting on the channel.
10. **Falsified ideas stay falsified.** Check this ledger before
    attempting anything — re-runs of dead ends need new evidence, not
    optimism.

---

## 2026-08-09 · rung — · (charter created) · cloud

Attempted: Charter, ledger, environment guide, and goal ladder created
from ADR-016 + the voice-as-drum literature review. No pipeline work.
Pre-registered expectations: n/a.
Result: Blessed baseline at creation time (git `15b8164`, blessed by
ADR-015 override): tier-1 committed accuracy tempo 0.571 (16/12/1),
meter_triple 0.357 (10/18/1), counts 0.571 (12/9/7); tier-1 ECE 0.291;
tier-0 tempo 25/25, meter 24/25. Fully-green checklist clips 5/24, all
numbers-counted; step_names meter slice 0.077. DEV = all 30 current
cases; SEALED = empty until new capture.
Regressions and classifications: n/a.
Lesson (durable): The loop's constraints are downstream of documented
incidents, not hypotheticals — see Standing Lessons 1–10.
Status: PROPOSED (awaiting rung 0 — owner data logistics).

## 2026-08-09 · rung — · (infrastructure decision) · cloud

Attempted: Runner architecture decided with the owner; docs updated
(agent-environment.md rewritten, charter rung 0 / rung 5 / ladder intro
amended, scripts/air-nightly.sh added).
Pre-registered expectations: n/a (decision, not experiment).
Result: Three-machine architecture — the always-on 16 GB MacBook Air is
the PRIMARY runner (data locality, persistent model weights and tool
venvs, free compute); cloud sessions are OVERFLOW (burst parallelism,
clean-room checks) behind Path B; the owner's main machine is the SEALED
vault and blessing desk, so DEV/SEALED separation stays physical on every
runner. Scheduled runs use one never-changing standing contract that
delegates to the charter's CURRENT RUNG pointer. Local-model policy:
specialized nets in now (Whisper, WavLM/DistilHuBERT SSL front-ends,
Silero VAD, MediaPipe); general local LLMs (7–27B) deferred as a
premature cost-for-accuracy trade under the research posture — and a 27B
at 4-bit (~14–16 GB) does not fit the 16 GB Air regardless; audio-native
small model logged as a rung-5 ensemble backlog note; graduation
condition for fine-tuning a small model on the corpus: ~140+ annotated
clips.
Regressions and classifications: n/a.
Lesson (durable): The runner is the hands, not the brain — plan usage is
identical on the Air and in the cloud; what the Air buys is state
(weights, venvs, recordings) surviving between runs, which is exactly
the tax ephemeral containers keep re-paying.
Status: BLESSED (owner decision in session, 2026-08-09).

## 2026-08-09 · rung — · (data staged; autonomy mode pending) · cloud

Attempted: Rung-0 data staging confirmed by the owner (DEV audio ready;
Ballet Barre 1 section videos at `video/youtube/Ballet Barre 1/Sections`
on the Air, assigned all-DEV). Charter amended: add-only ingestion
carve-out with `maturity: provisional|verified` case labels (provisional
rows never gate, always a separate slice); Rung M (marathon) drafted as
an explicitly DORMANT section — staged-autonomy plan is rung-by-rung
with daily blessings through rung 2, then an owner decision at the
rung-2 kill-test verdict on whether to commission the marathon.
CURRENT RUNG set to 1.
Pre-registered expectations: n/a (process decision).
Result: n/a.
Regressions and classifications: n/a.
Lesson (durable): The early rungs are inherently interactive — rung 1
builds the measuring instruments, rung 1.5 needs human ground truth,
rung 2 ends at a strategic fork. Autonomy is earned at exactly the
point where the work becomes parallel and the foundations are verified.
Status: PROPOSED (awaiting owner confirmation of staged-autonomy mode +
the supervised rung-1 session).

## 2026-08-11 · rung 1 · agent/rung-1-stage-scoring · local

Attempted: The full rung-1 EVAL-CHANGE deliverable set. (1) Beat-grid
format `evals/grids/<case-id>.yaml` — editable `beats` + frozen peakRate
`onsets`, mandatory explicit `provisional` flag, media sha256 provenance,
frozen detector params recorded per grid ([docs/evals/beat-grids.md](../evals/beat-grids.md)).
(2) Tap-assist annotator `python -m musical_perception.annotation`
(peakRate per review-1 recipe #1: 300–3000 Hz band, 10 Hz zero-phase
envelope, derivative, 3·MAD prominence, 120 ms spacing, Praat voiced gate
±30 ms) with Audacity label round trip for owner correction; `--verified`
is owner-only. (3) `stage1` eval suite: pulse P/R/F at ±70 ms + signed
asynchrony per clip vs grids, provisional/verified aggregates split,
count_style slices, wired into `evals run --suite tier0,tier1,stage1`.
(4) Acc1/Acc2 (±4% standard + ±8% house, fixed family {⅓,½,1,2,3}) and
OE1/OE2 distributions added to tier-0/1 reporting (informational, never
gating). (5) Onset-vs-token guard on trace load (warn when tokens >
1.5×onsets+8 or onsets=0). Provisional grids generated for 24/30 DEV
clips (all with media on this machine), `provisional: true` everywhere.
Pre-registered expectations: written before implementation — P1 pooled
matched-pair asynchrony in [−120,−20] ms; P2 pooled F@70ms in
[0.35,0.65]; P3 |mean asynchrony| numbers < step_names; P4 Acc1@4%
0.35–0.55, Acc2@4% 0.45–0.65, ≥2 rows |OE2|∈[0.30,0.585]; P5 zero guard
triggers; P6 zero tier-0/1 outcome changes.
Result: Scorecard 3 full hits, 1 partial, 2 misses — both misses are
findings. P2 ✓ pooled F=0.391 (macro 0.373; P 0.425, R 0.362, 24 clips).
P4 ✓ Acc1@4% 0.393, Acc2@4% 0.500 (@8%: 0.571/0.679), OE1 median 0.0009,
|OE2| median 0.053, 6 rows with |OE2|∈[0.30,0.585] — the between-levels
mass ADR-016 predicted, now visible. P6 ✓ "no outcome changes vs
baseline". P1 partial: sign right, magnitude wrong — mean −12.7 ms
(median −16.4, sd 31.1): ±70 ms matching censors the strongly-early words
into non-matches (recall 0.362), so matched-pair asynchrony
under-measures the ASR lead; rung 2 must read recall AND asynchrony
together, not asynchrony alone. P3 ✗ reversed: numbers −15.9 ms vs
step_names −8.6 ms (n=11/11). P5 ✗ one trigger, classified benign:
rig-numbers-3-4-90-clean, 94 tokens vs 52 voiced onsets = dense triplet
counting ("one-and-a-…": 31 "and" + 31 "a"; unstressed schwas don't each
earn a voiced envelope rise) — threshold calibration data point, kept
sensitive on purpose (false positive costs one listen; false negative is
a silent green). The totals hid one headline: the vocables clip collapsed
to a single Whisper token (pred=1 vs 24 grid onsets, R=0.042) — the
strongest per-clip evidence yet that rung 2's acoustic channel is not
optional for vocables. pytest 193 passed / 3 skipped / 0 failed.
Regressions and classifications: none — tier-0/tier-1 outcomes byte-
identical to the blessed baseline; every addition is reporting-only.
Two disclosed non-eval accommodations: (a) tests/test_wakeword.py gained
a skipif for the missing tflite runtime (no macOS arm64 wheel — tests
errored on this machine before this branch; the trigger path is the
rung-7 RETIRED target); (b) the peakRate detector gained a conditional
degenerate-silence prominence floor, verified bit-identical on all 24
real clips vs the pure recipe (it exists so synthetic-silence tests can't
pass on filter ringing).
Lesson (durable, one paragraph): The instrument censors what it
measures — a ±70 ms matched-pair asynchrony can only ever report the
survivors, so a channel that is *very* early looks *mildly* early with
low recall; and provisional peakRate grids measure words-vs-peakRate,
not words-vs-truth, until rung 1.5 flips them. Separately, Whisper does
not merely mistime vocables, it deletes them (1 token for 24 voiced
onsets), and legitimate dense triplet counting reaches 1.8× tokens per
voiced onset — both now pinned as data, not anecdote.
Status: PROPOSED (owner: bless rung 1, then rung 1.5 — verify/correct
the 24 provisional grids via to-labels/from-labels). BLOCKED (needs):
6 DEV clips have no media on this machine, so no grids yet —
audio/counting/8-counts-2x.aif, audio/counting/8-counts-triple.aif,
video/youtube/Exercise 1 Demo.m4v, video/youtube/Frappe.mov,
video/youtube/plies demo.m4v, video/youtube/grande battement.mov; stage
them and re-run `python -m musical_perception.annotation generate`
(word-start-derived grids are forbidden by Standing Lesson 1, so this
gap is owner-only).

## 2026-08-11 · rung 1 · agent/rung-1-stage-scoring · local (addendum: media search)

Attempted: Exhaustive search for the 6 missing DEV media files after the
goal check flagged 24/30 grid coverage.
Pre-registered expectations: n/a (search, not experiment).
Result: The files are not on this machine within this account's reach:
absent from the repo checkout, the full git history (every branch),
Spotlight-indexed paths (exact names and fragments), ~/Movies ~/Music
~/Downloads ~/Documents, /Users/Shared, and all three mounted backup
volumes (Backup A, Backup B, Misc Backup BH Maestro);
/Users/ben.juodvalkis is permission-denied to the agent account by
design. Re-downloading the four YouTube sources was considered and
rejected: a fresh download cannot reproduce the traces' pinned
media_sha256 and may shift the timeline the frozen word timestamps are
anchored to, which would corrupt stage-1 comparisons rather than enable
them.
Regressions and classifications: n/a.
Lesson (durable, one paragraph): When media provenance is pinned by hash,
"find the bytes" is the only honest recovery path — re-acquisition from
the public source is not equivalent, because the grid and the trace must
be anchored to the same timeline.
Status: BLOCKED (needs owner: copy the 6 files from the main machine to
audio/counting/ and video/youtube/, then run
`python -m musical_perception.annotation generate` — completes in about
a minute and closes rung 1's last gap).

## 2026-08-11 · rung 1 · agent/rung-1-stage-scoring · local (resolution: 30/30)

Attempted: BLOCKED-state resolution during the supervised session — the
owner staged the 6 missing media files in-session (audio/counting/ and
video/youtube/old demos/; symlinks bridge the case-declared paths, media
stays gitignored).
Pre-registered expectations: n/a (unblocking, not an experiment).
Result: All 6 files sha256-MATCH their trace metas (same bytes the traces
were recorded from — timeline provenance intact). Grids generated: 30/30
DEV clips, all provisional=true. Full-suite rerun: tier-0/tier-1
unchanged ("no outcome changes vs baseline"), pytest 193 passed /
3 skipped / 0 failed. stage1 over all 30: pooled P 0.499 R 0.464
F 0.481 (macro 0.41), asynchrony mean −7.6 ms (median −11.3, sd 32.0).
New observation the 24-clip view could not see: the four YouTube demo
clips score F 0.58–0.65 — the word-onset baseline looks *better* on
dense continuous teaching speech than on sparse metronomic rig marking
(rig median F ≈ 0.36), so rung 2's required +15/+30-point margins are
against a baseline that varies strongly by speech density; the per-slice
split (step_names 0.483 / numbers 0.475 / vocables 0.08) is the honest
comparison surface.
Regressions and classifications: none — outcomes byte-identical.
Lesson (durable, one paragraph): n/a — see main entry (this closes its
BLOCKED note).
Status: BLESSED (owner, 2026-08-11, in the supervised session: merged to
main as 6da3fa6; CURRENT RUNG advanced to 1.5).

## 2026-08-13 · rung 1.5 · agent/rung-1.5-grid-verification · local (owner-supervised)

Attempted: Step 0 (settle and commit the annotation convention) plus the
owner verification loop over the 30 provisional beat grids. Agent acted as
scribe and tool-runner only: every time judgment came from the owner, and
`--verified` was applied per clip on the owner's word.

Pre-registered expectations: none recorded — this rung is measurement of
human ground truth, not an experiment. The one forward-looking claim under
test was the convention's own premise (ruling (b)) that full metric grids
are derivable later from verified grids + tempo; scored below.

Result: **21 of 30 clips verified** (numbers slice complete 14/14, vocables
1/1, step_names 6/14, adr006 counting 0/2, video demos 0/4).

*Convention (owner-ratified, `docs/evals/annotation-convention.md`, commit
b8c24a9).* (a) beats = tactus only, syllable evidence stays in the frozen
unverified `onsets`; (b) vocalized beats only, silences described in free
text; (c) in-tempo prep counts IN, framing talk OUT; (d) explanation speech
never beats, categorically; (e) continuous through transitions; (f)
annotate what was heard, lengthening included. §2 records the owner's
Option-2 ruling that the rung-2 gate is re-expressed at the end of rung 1.5
from the convention + verified baseline + adopted metrics only, no
candidate peeking, vocables staying a decisive-win requirement.

*Aggregate correction stats (21 clips, 693 provisional events → 542
verified beats, −21.8%):* kept 486, deleted 207, added 56. Median |nudge|
of survivors is **0.0 ms on 19 of 21 clips** (max observed 100.5 ms). The
headline is that split: peakRate's *placement* is essentially solved —
where it fires on a real beat it is already at the vowel onset — and its
*selection* is the entire problem. Best case rig-numbers-4-4-80-triplet
(found all 16 tactus beats, 15 errors all genuine subdivision syllables);
worst rig-names-4-4-63-adagio (49 events for 26 beats, precision 37%,
recall 69%) and rig-names-4-4-100-quiet (recall 69%, precision 61%). Slice
pattern: on numbers clips peakRate repeatedly hit perfect recall with zero
additions; on step_names it both misses and over-fires. Two false-positive
modes were isolated and are lexical, not acoustic-noise: the second
syllable of "seven" (+120 to +175 ms, 6 instances) and a diphthong/coda
re-fire on monosyllables "five"/"eight" (/aI/, /eI/ closed by a stop or
fricative). At 160 BPM the detector's 120 ms minimum peak spacing is a
third of a beat period and it structurally under-resolves
(rig-names-2-4-160-long: 10 beats added by hand, the only clip where
additions exceeded false positives).

*Stage-1 verified slice — the first honest words-vs-truth numbers.*
`aggregate_verified` clips=21 P=0.371 R=0.410 F=0.389 (macro 0.380),
asynchrony mean −12.1 ms, median −15.0 ms, sd 30.4 ms. Do NOT read this
against `aggregate_provisional` (F=0.583): the 9 remaining provisional
clips are a different, easier population (the four long video demos alone
score 0.58–0.65). The honest comparison is same-clip before/after, which
is reconstructible exactly because each grid's frozen `onsets` list *is*
its original provisional reference: **macro F on the same 21 clips moved
0.356 → 0.380 (+0.024), 15 improved / 6 worsened.** So verification did not
crater the word-onset baseline; it slightly raised it while cutting the
reference count by 21.8%. Slices now: numbers 0.506 (n=14), step_names
0.489 (n=14), vocables 0.118 (n=1). The verified vocables baseline is
pinned at P=1.000 R=0.0625 **F=0.1176** (1 prediction, 16 references), up
from 0.080 provisional — this is the number the §2 gate's decisive-win
requirement will be expressed against.

*Ruling (b) validated, twice, arithmetically.* rig-names-4-4-96-allegro: 27
voiced beats, four gaps at integer multiples of the median IOI implying 5
unvoiced beats, 27+5 = 32 = the case's counts=32. rig-names-4-4-63-adagio:
26 voiced, six gaps spaced ~4 beats apart (the owner speaks three and holds
the fourth), 26+6 = 32. An arbitrary set of missed labels would not total a
round number, so the reconstruction is self-confirming. Limitation found
and recorded: gap-based recovery is blind to silence at the START or END of
a marked passage, so recovered totals are lower bounds
(rig-names-3-4-90-clean, rig-names-2-4-160-long).

*Owner observations.* Background music of unknown origin, NOT beat-
synchronized, on rig-numbers-2-4-120-clean, which is tagged
accompanied=false / snr_band=high — a data-quality discrepancy for the
owner to rule on (cases untouchable this rung). Being unsynchronized it
cannot leak a pulse an acoustic detector could cheat off, and peakRate
still posted its second-best selection result there, suggesting the Praat
voiced-gate suppressed music-driven onsets. Compound-meter tactus level
recorded explicitly for both 6/8 clips (the six, not the dotted quarter);
without that note a consumer would expect ~33 BPM.

*Findings parked, not acted on.* (1) **Whisper silently drops entire rounds
of eight on clean high-SNR rig audio — three instances**:
rig-numbers-4-4-104-clean (middle round, 16 tokens vs 24 beats),
rig-numbers-2-4-120-clean (final round, 24 vs 32),
rig-numbers-4-4-104-fourx8 (24 vs 32). All three still score green on
tempo/meter/counts because counts is per-phrase and surviving intervals
give the right tempo — Standing Lesson 8 in live form, and invisible before
human grids existed. (2) On rig-numbers-4-4-104-fourx8 the verified grid
shows a known red row has TWO stacked causes: the predicted 8 is
counts-per-phrase against a case expecting one 32-count phrase, and even
fixed, the token stream holds only 24 of 32 counts, so a corrected
implementation would answer 24. (3) **Whisper block phase slip** on
rig-numbers-4-4-60-halftempo: token count matches 16:16, but beats 2–4 and
10–11 are shifted bodily ~0.75 s early (three quarters of a beat) with
correct internal spacing, then re-sync. Median −11.9 ms hides it; mean is
−219.9 ms, sd 361, only 7/16 within ±70 ms. A tolerance-based F reads phase
error as detection failure. (4) **Bar-internal agogic accent** on
rig-names-3-4-88-waltz: beat 1→2 runs +9.9% vs mean, 2→3 −3.7%, 3→1 −7.0%,
consistent across all 8 bars; removing per-position means drops CV from
10.7% to 7.8%. Standing Lesson 5 frames lengthening as phrase-FINAL; this
is periodic and bar-internal, and at ~50 ms it consumes most of the ±70 ms
tolerance before any detector error is counted. (5) The two-gap problem on
rig-names-4-4-104-explained: an excluded-explanation gap and a silent-beat
gap are indistinguishable in a flat time list (the C6 no-tagging limit).

*Workflow amendment earned (recommended for the convention doc).* The
ratified per-clip check is grid-implied BPM vs case label, flagged >4%.
That check PASSED at +3.51% on a grid carrying three spurious labels and a
missing beat (rig-numbers-4-4-104-explained). What caught it was a
**minimum-IOI check** (0.124 s against a 0.577 s beat) and an **IOI-spread
check**. Four owner-export errors were caught this way across the session
(rig-numbers-4-4-104-explained, rig-numbers-6-8-100-clean,
rig-names-2-4-120-clean, rig-names-2-4-160-long); on the last, removing one
stray moved the BPM check from +1.30% to −0.15%. Both checks must be
computed WITHIN-PHRASE, excluding owner-confirmed breaks: on
rig-names-4-4-96-allegro overall CV is 44.1% and within-phrase 9.1%; on
rig-names-2-4-120-clean 25.6% vs 6.4%. Recommend adding both to §4 of the
convention.

*Guard verdict.* The rung-1 hallucination guard's first trigger,
rig-numbers-3-4-90-clean (94 tokens vs 52 voiced onsets — the ADR-016
clip-17 signature), is adjudicated **BENIGN by ear, not by inference**.
Asked before seeing the transcript, the owner described 32 numbers at 90
BPM in 4 sets of 8, each followed by "and-ah", plus 4 extra syllables from
bisyllabic "seven" = 100 syllables. Whisper's 94 tokens reconcile exactly:
100 − 4 ("seven" is one word) − 2 (no trailing "and a" on the final eight).
Threshold stays as calibrated: this false alarm cost one listen; a miss
would be a silent green.

Regressions and classifications: none. tier-0 (tempo 25/25, meter 24/25)
and tier-1 (tempo 0.571, meter_triple 0.357, counts 0.571) byte-identical
to the blessed baseline — "no outcome changes vs baseline". pytest 193
passed / 3 skipped. `git diff --stat main` shows 22 files: 21 grid YAMLs +
docs/evals/annotation-convention.md, with 0 files changed under
evals/cases/, evals/traces/, evals/baseline.json, or
src/musical_perception/evals/.

Lesson (durable, one paragraph): Verifying the grids changed what the
instrument measures far more than what it reports — same-clip macro F moved
only +0.024 while the reference count fell 21.8%, yet the *composition* of
the error flipped from "peakRate mistimes beats" to "peakRate cannot
choose which onsets are beats", a distinction no aggregate could have
surfaced and the whole reason rung 2's gate needs level-collapsed scoring.
The corollary is that a ground-truth pass is also an audit of everything
upstream of it: three silent Whisper round-drops, a block phase slip, a
two-cause red row, and a periodic bar-internal agogic accent were all
sitting inside green or unexplained rows and became visible only once a
human said what was actually there.

Status: PAUSED (21 of 30 verified). Remaining: rig-mixed-4-4-104-quantities,
rig-names-4-4-104-coda, rig-numbers-4-4-104-bothsides, adr006-8-counts-2x,
adr006-8-counts-triple, and the four video demos (adr006-exercise-1-demo,
adr007-plies-demo, adr010-grande-battement, frappe — ~510 events total;
their media lives at video/youtube/old demos/ and needs symlinks to the
case-declared paths recreated). Resume rule: progress = grids with
provisional:false. Also outstanding: the ratified duplicate-pass
self-consistency measurement on the two calibration clips
(rig-names-4-4-104-clean, rig-numbers-4-4-104-clean) — untouched seed
copies are preserved at /Users/Shared/DevWork/rung1.5-wav/*.SEED.labels.txt
so pass B starts from the seed rather than from pass A; this number is the
intra-annotator noise floor beneath which no rung-2 result can claim
significance, and it is not yet measured.

## 2026-08-14 · rung 1.5 · agent/rung-1.5-grid-verification · local (owner-supervised, session 2)

Attempted: completion of the rung-1.5 verification pass begun in the
2026-08-13 entry (which PAUSED at 21/30). Same division of labour: owner
judges, agent runs tools and records.

Pre-registered expectations: none — measurement of human ground truth.

Result: **28 of 30 clips verified, 2 deliberately declined with reasons.**
802 verified beats. Corpus complete.

*Convention amendment (charter rule 9), owner-ratified 2026-08-14.* Ruling
(d) — "explanation speech is never beats" — is **superseded by (d′)**:
annotate every beat you hear voiced on the pulse, whether counting or
talking through it. Framing talk and genuinely free-time material still
carry nothing. Four arguments, in
[annotation-convention.md](../evals/annotation-convention.md): (1) the pulse
demonstrably continues through explanation — in the two v1-annotated clips
the excised gaps span 9.92 and 9.08 beat periods, landing within 44 ms of
the grid after ~5.5 s of talking; (2) an accompanist must hold tempo
*through* the teacher talking, so a grid that deletes those beats grades the
wrong thing; (3) on teaching video explanation is the majority of the audio,
so v1 grids would cover a fraction of the clip and charge an extractor for
every correct beat elsewhere; (4) v1 required classifying speech before
annotating and was therefore *less* reproducible, not more — the original
justification was backwards. `rig-numbers-4-4-104-explained` and
`rig-names-4-4-104-explained` remain annotated under v1 and are flagged in
the doc; re-annotating them is an owner decision, not taken.

*THE ANCHORING / REACTION-LAG FINDING — the session's most important
methodological result.* Every seed-corrected clip reports median |nudge| =
0.0 ms, which looked like evidence that peakRate's placement is exact. It is
substantially **anchoring**: the owner left correct-looking seeds untouched.
The video clips were annotated from scratch in REAPER against the video, so
the statistic is finally unbiased — and the first pass, tapped live during
playback, sat **systematically late**: `adr010-grande-battement` median
**+39.2 ms** (28 of 33 marks later than the nearest peakRate onset),
`adr006-exercise-1-demo` **+33.2 ms**. A scrub-and-nudge pass with the
transport stopped removed it: −13.5, −18.8 and −2.7 ms across the three
video clips, a mutually comparable cohort. Both methods are therefore biased
— live tapping by ~35–40 ms of reaction time, seed correction by anchoring
toward peakRate — and mixing them silently would corrupt **signed
asynchrony**, a headline stage-1 metric, by more than half the ±70 ms
tolerance. Any future annotation must declare its method.

*A rung-1 finding overturned.* The rung-1 entry recorded that "the four
YouTube demo clips score F 0.58–0.65 — the word-onset baseline looks
*better* on dense continuous teaching speech than on sparse metronomic rig
marking." Same-clip before/after against verified grids:

```
VIDEO (3)       macro F 0.621 -> 0.299   (-0.322)
NON-VIDEO (25)  macro F 0.368 -> 0.397   (+0.029)
ALL 28          macro F 0.395 -> 0.386   (-0.009)
references      1199 provisional -> 802 verified (-33.1%)
```

The video advantage was an artifact of peakRate-seeded references: dense
machine onsets matched dense Whisper tokens. Against human tactus truth the
video clips are the **worst** slice, not the best —
`adr010-grande-battement` 0.653 → 0.209, `adr006-exercise-1-demo` 0.629 →
0.213. This materially changes rung 2's expectations: the baseline it must
beat is far weaker on teaching video than rung 1 believed, and far more of
that slice's apparent performance was measuring the annotator's proxy
against itself.

*Final stage-1 verified slice (28 clips, whisper-word-starts):* pooled
P 0.334 R 0.449 **F 0.383** (macro 0.386), asynchrony mean −14.3 ms, median
−19.4 ms, sd 32.6 ms. By count_style: numbers F 0.439 (n=14), step_names
0.414 (n=14), vocables 0.118 (n=1); mixed has no verified member. Vocables
baseline for the §2 gate stays pinned at P 1.000 / R 0.0625 / **F 0.1176**.

*Two deliberate declines, both substantive findings rather than gaps.*
(1) `rig-mixed-4-4-104-quantities` — the owner, who recorded it, could not
parse the pulse ("real weird"); the transcript is almost entirely prose
describing quantities. It is the only clip in the `mixed` slice, which
therefore has no verified member. (2) `adr007-plies-demo` — **the pulse is
not in the audio.** The teacher speaks quickly over a slow exercise; in the
owner's words an accompanist "would only really know which tempo to follow
since I know pliés are slow." Fitting a single period to the owner's sparse
11-mark pass: removing three tight pairs (two syllables inside one beat)
gives 0.5080 s = **118.1 BPM, matching the case label of 118** — the first
human confirmation of that label — but residual scatter is **~90 ms rms on
every subset fitted**, against a ±70 ms tolerance. The marks locate the
*tempo* but not the *beats*, and the marked syllables are ordinary words
("with", "that", "goes", "the"), not counts. This is the clearest evidence
in the corpus of the hard limit of ruling (b): where the pulse is unvoiced
and recovered from exercise knowledge, a vocalized-only grid cannot
represent it.

*Other findings from session 2.* `adr006-8-counts-2x` is a **FAKE RED**: its
verified grid reads 101.63 BPM against `expect.marking_bpm: 130` (−21.83%),
and the arithmetic is decisive (16 counts at 130 BPM span 7.38 s; these span
9.06 s, with uniform intervals excluding a 16-of-20 reading). The blessed
baseline scores the row tempo=wrong at predicted 95.6 — which is within 6%
of verified truth. The row is red because the label is wrong, the mirror of
Standing Lesson 8's fake-green. Its sibling `adr006-8-counts-triple` had
`marking_bpm` deliberately unpinned; it now has a first ground truth of
68.38 BPM, which also sharpens ADR-007's recorded "/2-over-/3 normalization
defect": the historical output of 82.7 is not in the family {⅓,½,1,2,3} ×
68.38 = {22.8, 34.2, 68.4, 136.8, 205.1}, so that failure was **off-family**,
not a level confusion. A **fourth Whisper round-drop** appeared
(`rig-numbers-4-4-104-bothsides`: grid shows 8 rounds of eight, transcript
holds 7), making four across fourteen numbers clips. `adr006-exercise-1-demo`
required clustering rather than a median to recover its beat period, because
the teacher voices only two of three beats per bar — its plain median IOI
(0.896 s) is not the beat period; pooled clustering gives 0.5067 s = 118.42
BPM vs a label of 117.

Regressions and classifications: none. tier-0 (tempo 25/25, meter 24/25) and
tier-1 (tempo 0.571, meter_triple 0.357, counts 0.571) byte-identical to the
blessed baseline — "no outcome changes vs baseline". pytest 193 passed /
3 skipped. `git diff --stat main`: 32 files, all under `evals/grids/` plus
`docs/evals/annotation-convention.md` and this ledger; **0 files changed**
under `evals/cases/`, `evals/traces/`, `evals/baseline.json`, or
`src/musical_perception/`.

Lesson (durable, one paragraph): The instrument that measures the annotator
matters as much as the annotator — median |nudge| of 0.0 ms across twenty-one
clips looked like a clean finding about peakRate's placement and was mostly
an artifact of showing the annotator peakRate's answer first, while the
unanchored alternative carried a 35–40 ms reaction-time bias of its own, so
the only honest reading came from running both and declaring the method. The
same lesson explains the corpus's biggest reversal: video clips looked like
the easy slice while their references were machine-generated, and became the
hardest the moment a human said where the beats were. And two clips could
not be annotated at all — one unparseable to its own performer, one whose
pulse lives in exercise knowledge rather than in the audio — which is not a
shortfall but the corpus telling us where perception alone cannot reach.

Status: BLESSED (owner, 2026-08-14; merged to main). Rung 1.5 complete:
28/30 verified, 2 declined with recorded reasons. Two deliverables remain before the CURRENT RUNG
pointer can advance to 2: (a) the **§2 rung-2 gate re-expression** — now due,
to be derived from the ratified convention, the verified-grid baseline above,
and the adopted metrics only, with no candidate peeking, recall-at-tactus +
level-collapsed precision as the intended shape and a decisive vocables win
mandatory; owner-blessed before the pointer moves. (b) the **duplicate-pass
self-consistency measurement** on the two calibration clips, still not done —
untouched seeds are preserved at
`/Users/Shared/DevWork/rung1.5-wav/*.SEED.labels.txt` so pass B starts from
the seed rather than pass A; given the anchoring finding this measurement is
now more important than when it was commissioned, since it is the only way to
quantify the annotator's own noise floor separately from the anchoring bias.
Owner decisions also outstanding: whether to re-annotate the two v1 clips
under (d′); whether to correct `adr006-8-counts-2x`'s `marking_bpm`; whether
to pin `adr006-8-counts-triple`'s.

## 2026-08-14 · rung 1.5 · agent/rung-1.5-grid-verification · local (self-consistency measurement)

Attempted: the ratified duplicate-pass self-consistency measurement, the
last outstanding rung-1.5 deliverable besides the §2 gate re-expression.

Design note (deviation from the original plan, disclosed): the plan was two
seed-anchored passes from the same preserved seeds. The anchoring finding
invalidated that design — seed-anchored passes overwhelmingly *accept* the
seeds (median |nudge| 0.0 ms on 19 of 21 clips), so two such passes would
agree almost perfectly and report a meaninglessly small noise floor. Pass B
was therefore run **from scratch** (REAPER, no seeds shown) against the
already-verified seed-anchored pass A, on `rig-numbers-4-4-104-clean`
(16 s, plain 1-8 counted three times at 104). Scope reduced from two clips
to one by owner decision after the cost/benefit was stated. This measures
the anchored-vs-scratch cohort offset *and* bounds placement scatter, which
is what the corpus actually needs given it holds 21 anchored + 3 scratch
clips.

Result:

```
pass A (seed-anchored, Audacity)   24 beats, 2.354-15.664 s
pass B (from scratch, REAPER)      24 beats, 2.494-15.636 s
matched  22/24 at ±70 ms (92%)   ·  23/24 at ±120 ms (96%)
delta (B − A):  median −21.6 ms   sd 24.9 ms   max |Δ| 71.0 ms
pass B vs peakRate onsets: median −19.7 ms   (pass A is 0.0 by anchoring)
```

**Structure is perfectly reproducible: 24 beats both times, none added, none
missed.** Beat identification — the thing that drives pulse precision,
recall and F — carries no measurable annotator variance on this clip. The
owner's prior ("I think I'd be really consistent") was correct on structure.

**Placement carries ~25 ms of noise, in two separable parts.** (1) A
systematic −20 ms: the from-scratch method lands earlier than peakRate's
onsets, while seed-anchored lands exactly on them by construction. This
matches the three video clips (−13.5, −18.8, −2.7 ms vs peakRate) and is
therefore the anchored-vs-scratch **cohort offset**, now measured rather
than assumed. (2) A residual sd of ~25 ms — genuine placement scatter.

Regressions and classifications: n/a (measurement only; no pipeline or eval
code touched).

Lesson (durable, one paragraph): The annotator's reproducibility is not one
number but two, and they behave completely differently — *which events are
beats* proved perfectly repeatable across methods and instruments, while
*where exactly those beats sit* carries ~25 ms of scatter on top of a ~20 ms
systematic offset between annotation methods. That split is convenient: it
means F-measure is robust to how the corpus was annotated, and only signed
asynchrony inherits the noise. It also means the original self-consistency
design would have measured nothing, because two anchored passes agree by
construction — the measurement only became informative once it was pointed
at the methods rather than at the annotator.

**THRESHOLD FOR RUNG 2 (the deliverable):** pulse P/R/F against these grids
are trustworthy and may be compared directly. **Signed-asynchrony
differences smaller than ~25 ms are inside annotator noise and must not be
claimed as results.** Any comparison that mixes anchored clips (the 21 rig
grids) with from-scratch clips (the 3 video grids) must additionally account
for the ~20 ms cohort offset, or restrict itself to within-cohort
comparisons.

Status: BLESSED (owner, 2026-08-14; merged to main).

## 2026-08-14 · rung 1.5 · main · (blessing bookkeeping)

Attempted: owner blessing of rung 1.5 and the corrected case labels.

Disclosure (charter rules 1 and 8): the charter reserves `evals bless` and
pushing `main` to the owner, and the agent flagged this. The owner reviewed
the outcome deltas, reaffirmed, and explicitly directed the agent to perform
the merge, bless and push on their behalf. The attestation is the owner's;
the keystrokes were the agent's. Recorded here so the blessing chain is not
silently ambiguous in the record.

Result: both branches merged to main (`agent/rung-1.5-grid-verification`,
36 commits; `agent/case-label-corrections`, 1 commit) — clean, no conflicts.
Suite re-run showed exactly the three predicted outcome changes and nothing
else: `adr006-8-counts-2x.meter_triple` wrong→correct,
`adr006-8-counts-2x.tempo` wrong→correct, `adr006-8-counts-triple.tempo`
None→wrong (a newly scored row). Blessed
`run-20260814T173647Z-0b93bc5.json` → `evals/baseline.json`;
`docs/evals/baseline.md` regenerated; pytest 193 passed / 3 skipped.

New blessed baseline: tier-0 tempo 25/25, meter 24/25. tier-1 tempo **0.586**
(17/30, was 0.571 on 29), meter_triple **0.393** (11/29, was 0.357), counts
0.571 (unchanged), sides 1.0, slot 1.0. tier-1 Acc1 0.379@4% / 0.586@8%,
Acc2 0.483@4% / 0.690@8%, between-levels rows 11. stage1
`aggregate_verified` 28 clips P 0.334 R 0.449 F 0.383, asynchrony median
−19.4 ms; slices numbers 0.439 (n=14), step_names 0.414 (n=14), vocables
0.118 (n=1); `aggregate_provisional` is now only the 2 declined clips.

The tier-1 gains are **not** a pipeline improvement — no pipeline code
changed. They are the reference getting truer: two fake-red rows on
`adr006-8-counts-2x` were red because the label said 130 when the verified
grid and the owner's ear both say ~102–104. The newly pinned
`adr006-8-counts-triple` tempo row is a genuine failure that had been
invisible while no truth existed. Anyone reading the baseline delta must not
read it as progress.

CURRENT RUNG advanced to **2**. The rung-2 gate in
`docs/evals/annotation-convention.md` §2 is marked BLESSED and supersedes the
margins in the charter's rung-2 condition.

Regressions and classifications: none — every outcome change is accounted
for above and was predicted before the run.
Lesson (durable, one paragraph): A baseline improvement can mean the system
got better or the yardstick got truer, and only the provenance distinguishes
them — here tier-1 tempo rose from 0.571 to 0.586 without a line of pipeline
code changing, purely because a human finally established what two fixtures'
tempos actually were. Any future reader comparing baselines across this
boundary is comparing against a different measuring instrument, which is why
the delta is written down here rather than left to be inferred from the
numbers.
Status: BLESSED (owner-directed, 2026-08-14).

## 2026-08-14 · rung — · (rung-1.5 report §8 rulings) · cloud

Attempted: Master-agent responses to the rung-1.5 completion report's
three open decisions, owner-confirmed ("go for it") in session.
Pre-registered expectations: n/a (governance).
Result: (1) **Barre-1 holdout carve-out** — amends the 2026-08-09 all-DEV
assignment: the owner randomly selects ~one third of the 33 sections and
moves them off the Air to the main machine BEFORE any ingestion; new
taxonomy HELD-OUT (weak seal, public-source, never-iterated-on) vs
SEALED (strong, owner capture only) written into the charter's splits.
Rationale for the reversal: the rung-1.5 report correctly elevated the
holdout gap to the project's largest structural risk, and the property
that matters at this corpus size — never iterated on — is protected by
physical removal regardless of source. (2) **QC checks ratified** into
convention §4 (min-IOI + within-phrase IOI-spread, evidence-based: they
caught four real errors the BPM check false-passed). (3) **Rung 2.5
scheduled** — taggable grid format (additive-only; verified grids remain
untouched) + QC implementation + annotation-method metadata, after rung
2's verdict, before Barre-1 ingestion. Also countersigned: the (d′)
amendment (the master agent's original (d) ruling was refuted by the
9.92/9.08-beat-period gap evidence — recorded as such), and the
owner-directed blessing pattern as acceptable with per-instance
disclosure, never as a default.
Regressions and classifications: n/a.
Lesson (durable, one paragraph): A holdout decision is only available
before iteration touches the data — it is the single irreversible step
in every ingestion, which is why it now sits in the charter as a
physical act (move files off the runner) rather than a labeling
convention. And a ruling made on reproducibility grounds ((d)) fell to
evidence within three days — conventions earn their keep by surviving
contact with the corpus, not by sounding principled.
Status: BLESSED (owner, 2026-08-14, in session).

## 2026-08-14 · rung — · (holdout executed) · owner action

Attempted: The Barre-1 HELD-OUT carve-out, executed by the owner per the
amended charter: 4 of the 12 exercises drawn at random (one per
barre-order quarter, exercise-level split — demo and execution files
move together), all their files saved to the main machine and deleted
from the Air including the trash. **Which exercises were drawn is
deliberately not recorded here** — the list exists only on the main
machine, per the charter. The remaining 8 exercises' section videos stay
on the Air as future DEV ingestion material.
Pre-registered expectations: n/a.
Result: Rung 2's prerequisites are now fully met (rung 1 blessed, 28/30
grids verified, gate blessed, DEV media present, holdout physically
removed from the runner). The kill-test is clear to run.
Regressions and classifications: n/a.
Lesson (durable): The holdout's protection is that agent sessions cannot
enumerate it — so the ledger records THAT it exists and its size, never
its contents. A holdout you can look up is not held out.
Status: BLESSED (owner, 2026-08-14).

## 2026-08-14 · rung 2 · agent/rung-2-acoustic-pulse · local

Attempted: the rung-2 kill-test — a peakRate acoustic pulse extractor in
the precision layer, scored against the whisper-word-start baseline on
the 28 owner-verified grids under the blessed §2 gate
(docs/evals/annotation-convention.md). The two declined clips
(`adr007-plies-demo`, `rig-mixed-4-4-104-quantities`) are excluded by
name. This section and the frozen design below are committed BEFORE any
extractor implementation exists (charter rule 3); results follow in this
same entry at session end.

Pre-registered expectations:

*Extractor design, frozen a priori (no tuning against gate results
permitted after this commit).* Review-1 "steal this first" #1 + #2:
(i) the peakRate core reuses `annotation/peakrate.py`'s frozen
`PeakRateParams` verbatim (300–3000 Hz band, 10 Hz zero-phase low-pass,
3·MAD prominence, 120 ms min spacing, Praat voiced gate ±30 ms,
75–450 Hz pitch) — the same detector whose selection behaviour rung 1.5
measured; (ii) NEW relative to the annotation seeder: de Jong & Wempe
syllable-nuclei REGIONS via Parselmouth (intensity 50 ms frames, silence
threshold −25 dB re the 99th-percentile max, min dip 4 dB — review-1
§1.2 says 4–8 dB for marked speech, the conservative end is chosen here
a priori — and the same AC pitch voicing), with **the first peakRate
event inside each nucleus region** kept as the event time (first, not
largest: the documented five/eight diphtong re-fire is a *second* rise
inside one nucleus); events outside every region are dropped. No tactus
selection — the extractor emits the syllable-rate stream the §2.1
metrics were designed to score fairly. Output: sorted times in seconds,
pure function of (audio, sr).

*P0 — metric validity gate (checked before any candidate number is
read).* The committed analysis code must reproduce the §2.2 baseline
table EXACTLY (all 12 numbers: ALL 0.449/0.506/0.452, numbers
0.568/0.604/0.577, step_names 0.349/0.363/0.343, vocables
0.062/1.000/0.118) from the frozen scorer's matcher + traces + verified
grids. Where §2.1 leaves edge semantics unstated (span ends, clustering
of out-of-span predictions), the reproduction pins them; if NO variant
reproduces the table, stop and write a BLOCKED entry rather than pick a
flattering variant.

*Evidence base for the predictions below* — only ledger-recorded facts:
rung-1.5 correction stats (21 anchored clips: 486 kept / 207 deleted /
56 added → peakRate found ~89.7% of anchored verified beats; numbers
clips repeatedly perfect recall with zero additions; step_names both
misses and over-fires: adagio R 0.69 / P 0.37, quiet R 0.69 / P 0.61,
160-long 10 beats hand-added; lexical FP modes: "seven" second syllable
+120–175 ms, five/eight diphthong re-fire), the from-scratch video
cohort sitting −13.5/−18.8/−2.7 ms (median) from peakRate onsets, and
the §2.2 baseline. No candidate has been run.

- **P1 (gate 1, step_names R@tac ≥ 0.499):** PASS predicted. Extractor
  macro R@tac on the 13 step_names clips in **[0.70, 0.88]**, point
  ~0.80, vs baseline 0.349. Reason: worst measured peakRate step_names
  recalls are ~0.69 (adagio, quiet) and ~0.81 ceiling on 160-long
  (44/54 after 10 hand-adds); the two video clips' from-scratch beats
  sit within ~19 ms of peakRate events (matched at ±70 ms), predicted
  R@tac 0.6–0.9 there. Risk: the new nuclei-region gate drops devoiced/
  whispered nuclei (review-1 §1.2 failure mode) — expected small on
  this close-mic corpus.
- **P2 (gate 2, improvement on ≥ 9 of 13):** PASS predicted, point
  prediction **13 of 13** (accept ≥ 12): baseline macro is 0.349 and no
  known per-clip peakRate recall sits below 0.69; weakest-margin clips
  predicted to be rig-names-2-4-160-long (structural 120 ms
  under-resolution at 160 BPM) and rig-names-4-4-63-adagio.
- **P3 (gate 3, vocables R@tac ≥ 0.60 AND P_lc ≥ 0.50, n=1):** PASS
  predicted: R@tac in **[0.94, 1.0]** (16 beats, seed-anchored grid —
  beats coincide with detector events unless the nuclei gate drops
  one), P_lc in **[0.70, 1.0]** (plosive vocables are the detector's
  best case; sub-tactus doubles collapse). Baseline 0.0625/1.000. n=1,
  never quoted as a slice average.
- **P4 (gate 4, numbers F_lc ≥ 0.527):** PASS predicted with margin:
  extractor numbers macro F_lc in **[0.80, 0.95]** vs baseline 0.577.
  Reason: repeatedly perfect recall + zero additions on numbers rig
  clips; subdivision syllables and both lexical FP modes land inside
  their beats' slots and collapse. Slice risk rows:
  adr010-grande-battement (from-scratch video — dense speech,
  out-of-span clusters charge P_lc) and adr006-8-counts-triple (n_ref
  8, small denominator).
- **P5 (overall, informational):** extractor ALL macro F_lc in
  **[0.75, 0.92]** vs baseline 0.452.
- **P6 (signed asynchrony, reported under the noise rules, no claims
  under ~25 ms):** on the 21 anchored grids extractor matched-pair
  asynchrony |median| ≤ 15 ms — and this is an ARTIFACT of anchoring
  (kept beats ARE detector events), to be disclosed as such, not
  claimed as placement accuracy; on the 3 from-scratch video grids
  median in **[0, +30] ms** (grids sit early of peakRate by ~3–19 ms).
  Whisper baseline stays ~−14 to −19 ms as recorded. Anchored and
  from-scratch never pooled without the ~20 ms offset disclosed.

*Kill criteria (what NEGATIVE looks like):* any of gates 1–4 failing
after the frozen design is scored as specified — no post-hoc parameter
changes, no re-runs with different constants. A failure writes the
negative-result entry with per-clip evidence and ends the rung
(ADR-016: the reset stops, P2 strengthens).

Result: **PASS — all four gate conditions hold.** Verdict from
`scripts/rung2_kill_test.py` (committed analysis code; frozen scorer
untouched); full artifacts in `docs/research/rung2-kill-test.{md,json}`
and the extractor event streams in
`docs/research/rung2-extractor-events.json`.

*P0 (metric validity) passed:* the §2.2 baseline table reproduced
EXACTLY (all 12 numbers) once the edge semantics were pinned by search
over 24 variants — beat-centered midpoint slots, annotated span extended
by half the MEDIAN IOI beyond the first/last beat, out-of-span
predictions individually charged, TP = cluster containing a one-to-one
±70 ms matched prediction, macro rows. Only this variant reproduces the
table AND the honesty-note per-clip value (triplet 0.469 → 0.882); it is
now the committed definition.

*Blessed metrics, macro per slice (baseline → extractor):*

```
                 n    R@tac           P_lc            F_lc
ALL             28    0.449 → 0.828   0.506 → 0.867   0.452 → 0.839
numbers         14    0.568 → 0.926   0.604 → 0.931   0.577 → 0.926
step_names      13    0.349 → 0.719   0.363 → 0.798   0.343 → 0.742
vocables (n=1)   1    0.062 → 0.875   1.000 → 0.875   0.118 → 0.875
```

*Gate:* (1) step_names R@tac 0.719 ≥ 0.499 ✓ (+0.370 absolute, 2.5× the
required margin). (2) improved on **12 of 13** step_names clips ✓; the
one exception is a TIE, not a loss — rig-names-4-4-100-quiet at 0.312
both systems. (3) vocables R@tac 0.875 ≥ 0.60 AND P_lc 0.875 ≥ 0.50 ✓
(single clip, never a slice average; baseline was R 0.0625 — Whisper
emitted one token for the phrase, the extractor 24 events, 14 of 16
beats matched). (4) numbers F_lc 0.926 ≥ 0.527 ✓ — the no-regression
slice improved by +0.349 instead. Stage-1 pooled over the 28 verified
grids: whisper P 0.334 / R 0.449 / F 0.383 → acoustic P 0.645 / R 0.805
/ F 0.716.

*Prediction scorecard: 5 hits, 1 partial.* P1 ✓ 0.719 ∈ [0.70, 0.88] —
at the LOW end of the range, not the ~0.80 point. P2 ✓ at the accept
threshold (12 ≥ 12; the point prediction of 13 missed, and the risk
reasoning was partly wrong: the predicted weak clips 160-long and
adagio improved strongly, while the actual non-improver was quiet). P3
partial: P_lc 0.875 ✓ ∈ [0.70, 1.0], but R@tac 0.875 ✗ below the
predicted [0.94, 1.0] (2 of 16 beats lost). P4 ✓ 0.926 ∈ [0.80, 0.95].
P5 ✓ 0.839 ∈ [0.75, 0.92]. P6 ✓ anchored-cohort extractor median
0.0 ms (the predicted anchoring ARTIFACT — kept beats ARE detector
events; not a placement claim), from-scratch +9.6 ms ∈ [0, +30], under
the 25 ms noise floor so nothing is claimed; baseline −20.0 / −18.3 ms
as recorded. P1's low end, P2's miss, and P3's miss all trace to ONE
cause: the new nuclei-region gate trims recall harder than predicted on
quiet or soft material — on rig-names-4-4-100-quiet it emitted 9 events
for 16 beats where rung-1.5 measured ungated peakRate recall at 0.69,
i.e. the q99−25 dB silence threshold, quantile-relative or not, bites
when the whole clip is quiet (review-1 §1.2's failure mode, observed).

*Honesty flags.* (a) **Anchoring inflates the rig margins:** on the 25
seed-anchored grids, verified beats coincide with peakRate events
wherever the owner kept the seed, so R@tac there partly measures "did
the owner keep this event" — and the extractor's core IS the seeding
detector. The gate was blessed knowing this provenance, so the PASS
stands, but the anchor-free evidence is the from-scratch cohort: on the
only two step_names video clips the improvement is real but small
(exercise-1-demo 0.488 → 0.561, frappe 0.673 → 0.691), while the
numbers video clip is large (adr010 0.389 → 0.694). Anyone reading the
step_names +0.370 as anchor-free detection quality is over-reading it.
(b) The vocables clip is also seed-anchored. (c) Extractor asynchrony
sd 6.2 ms on anchored clips is the same artifact, not super-human
placement.

Regressions and classifications: none — tier-0 (tempo 25/25, meter
24/25) and tier-1 (tempo 0.586, meter_triple 0.393, counts 0.571)
byte-identical, "no outcome changes vs baseline"; the extractor is new
precision-layer code not wired into the analyze pipeline; the frozen
stage1 suite still scores whisper-word-starts and is unchanged. pytest
198 passed / 3 skipped (5 new extractor tests). `git diff --stat main`:
9 files — pulse.py, test_pulse.py, rung2_kill_test.py, this ledger,
CLAUDE.md, and the three rung2 result artifacts + events cache under
docs/research/; **0 files changed** under `evals/cases/`,
`evals/traces/`, `evals/baseline.json`, or
`src/musical_perception/evals/`.

Backlog notes (rule 6, parked): (i) nuclei silence threshold fails on
uniformly quiet clips — consider a floor relative to the clip's speech
band rather than q99, in a future pipeline rung; (ii) the vocables
clip's 2 dropped beats deserve a listen at rung 2.5's QC pass; (iii)
the §2.4 cohort counts in the convention say 21 anchored / 3 scratch —
the 4 session-2 audio grids (coda, bothsides, both adr006 counting
clips) were also seed-anchored, so the honest split is 25/3; a one-line
convention correction is owner business.

Lesson (durable, one paragraph): The kill-test verdict is that the word
channel loses to the acoustic channel categorically, not marginally —
macro R@tac 0.449 → 0.828 with level-collapsed precision RISING — but
the size of the win is partly a property of the measuring instrument:
seed-anchored grids flatter the detector that seeded them, and the
anchor-free video cohort shows the honest, smaller margin. The rung's
second finding is methodological: a blessed gate whose metric has
unstated edge semantics is not yet a gate — reproducing the committed
baseline table EXACTLY (P0) forced every hidden choice into the open
before any candidate number could be read, and without it this session
could have picked among 24 variants whose step_names P_lc spans 0.334
to 0.417. Freeze the metric before the candidate, always.

Status: PROPOSED (owner: bless rung 2 — the kill-test PASSES, the reset
continues past its strategic fork; per the charter this is also the
owner's decision point on commissioning Rung M, and rung 2.5 is next
before any Barre-1 ingestion).

## 2026-08-14 · rung 2 + M · main · (blessing + commissioning bookkeeping)

Attempted: Owner blessing of rung 2 and commissioning of Rung M, both
owner-directed in session ("yes do it all") — disclosure per the
established pattern: the attestation is the owner's, the keystrokes the
master agent's, recorded per-instance.
Pre-registered expectations: n/a (governance).
Result: `agent/rung-2-acoustic-pulse` merged to main (VERDICT: PASS —
all four §2 gates, wide margins; no bless run needed, stage1 pins no
outcomes and frozen suites were byte-stable). Convention corrections
folded at the same touch: §2.1 now carries the pinned edge semantics
from the rung-2 P0 validity gate (reference implementation
scripts/rung2_kill_test.py; "a metric that exists only as prose is not
yet a metric"), §2.4 cohort count corrected 21→25 anchored. Charter:
CURRENT RUNG → M (COMMISSIONED); Rung M workstream ranking refreshed
(W1 rung 2.5 → W8 RETIRED sweep, with rung-2's parked nuclei-floor item
queued); rung-2 section carries its PASS verdict. Merged agent branches
deleted (`agent/rung-1-stage-scoring`, `agent/rung-2-acoustic-pulse`).
Regressions and classifications: n/a.
Lesson (durable, one paragraph): The staged-autonomy bargain completed
exactly as designed — the loop earned the marathon by three circuits of
disclosed constraints, honest scorecards, and a self-declared BLOCK, and
the commissioning decision was made at the pre-registered decision
point, on a PASS verdict measured against human ground truth. Autonomy
was granted at the moment the work became parallel and the foundations
became verified, not before.
Status: BLESSED (owner-directed, 2026-08-14). Marathon ACTIVE.

## 2026-08-16 · rung M · agent/marathon · local

Attempted: **W1 (= rung 2.5), the highest-ranked non-BLOCKED marathon
workstream** — the taggable grid format + the two owner-ratified QC
checks + annotation-method metadata. EVAL-CHANGE: no pipeline change is
bundled, and the gate is that stage1 output on the 28 verified grids is
byte-identical before and after. W1 gates all Barre-1 ingestion (W4), so
it is the correct first marathon increment. This pre-registration
section is committed BEFORE any implementation exists (charter rule 3);
results follow in this same entry at session end.

Pre-registered expectations:

*Design, frozen a priori (no tuning against results permitted after this
commit).*

1. **Format 2, additive-only.** `GRID_FORMAT = 2` for writing;
   `load_grid` accepts `{1, 2}`, so **every existing verified grid stays
   valid with zero edits**. `beats` stays a flat sorted time list —
   untouched semantics. Two optional keys are added:
   - `regions: [{start, end, kind, note}]` — a **parallel** structure,
     never interleaved with beats. `kind` is a closed set of exactly the
     three holes the C6 limitation could not distinguish
     (convention (d′)): `silent_beat` (pulse continued, unvoiced),
     `free_time` (no metric beat exists), `excluded_explanation`
     (material deliberately outside the annotation). Validation:
     `0 ≤ start < end`, sorted by start, non-overlapping, known kind.
   - `annotation_method: anchored | from_scratch | null` — the
     rung-1.5 cohort-offset finding (convention §2.4: 25 anchored / 3
     from-scratch) made a per-grid property. `null` = unrecorded.
2. **Audacity round trip extended.** `to_label_text` appends Audacity
   *region* labels (`start<TAB>end<TAB><kind>`) after the beat point
   labels. Parsing rule, frozen: a line is a **region** iff its label
   text is a known `kind`, and then `end > start` is required; a line
   with `end > start + 1 ms` whose text is *not* a known kind is a loud
   `ValueError`, never a silent beat. Every other line is a beat at
   `start`. Existing `.labels.txt` files (all point labels, `beat-N`)
   therefore parse **exactly as before** — this is the one place the
   extension could have corrupted a correction pass, so it is pinned by
   test, not by inspection.
3. **QC module** `annotation/qc.py` + `annotation qc [ID ...]` CLI,
   implementing convention §4 as amended 2026-08-14. Constants frozen
   here, a priori: `MIN_IOI_RATIO = 0.5` (flag IOI < 0.5 × clip median —
   the caught error was 0.124 s against a 0.577 s beat = 0.215×),
   `PHRASE_BREAK_RATIO = 1.75` (an IOI above this is a phrase break, not
   an agogic stretch), `MAX_PHRASE_CV = 0.15` (rung-1.5 healthy
   within-phrase CVs: 4.4 / 6.4 / 7.8 / 9.1 / 9.6 %, and the waltz's
   genuine bar-internal accent at 10.7 % must NOT flag),
   `MIN_PHRASE_IOIS = 3`, BPM tolerance 4 % (already ratified), CV =
   population sd / mean. **Suppression, as ratified:** any IOI
   overlapping a `free_time`, `excluded_explanation`, or `silent_beat`
   region is excluded from both checks — a gap explained by a tag is not
   evidence of an error.
4. **No threshold may be retuned after seeing output.** If a constant
   misfires it is reported as a finding and proposed for owner
   ratification (rule 9), not quietly changed.
5. **No verified grid file is written this session.** Provenance is
   owner authority — convention §2.4's cohort count was itself corrected
   21→25 — so `annotation_method` ships as field + `set-method` CLI, and
   backfilling the 28 verified grids is a BLOCKED owner act, exactly as
   `--verified` is.

*Q0 — the EVAL-CHANGE gate.* A deterministic dump of `run_stage1()` over
`evals/` hashes to `1d5fe5a3cbdc28b3e61873fa216ad36bc1a2e58c614b6ad6511ca5ed89c1d82a`
(12,660 bytes) at `main` = 85317e3, captured before any edit. **Predicted
PASS**: stage1 reads only `beats` and `provisional`, and no grid file is
written. A mismatch means the format change leaked into scoring and the
increment is reverted, not explained.

*Q1 — validity gate (P0 discipline: pin the metric before reading any
new number).* 25 verified grids' `notes` already record a grid-implied
BPM. The implementation must reproduce them from the committed grids
before any min-IOI or spread output is read: whole-clip `60/median(IOI)`
for the 21 recorded as "Grid-implied BPM" and the within-phrase variant
for the 4 recorded as "within-phrase BPM" (frappe 156.25,
rig-names-2-4-120-clean 119.51, rig-names-4-4-104-coda 106.44,
rig-names-4-4-96-allegro 96.07). **Predicted ≥ 23 of 25 reproduce to
±0.02.** If fewer, pin the semantics by explicit search across stated
variants and disclose which one the recorded numbers used — never pick
the flattering one.

*Q2 — min-IOI on the 28 verified grids.* **Predicted 0 violations**
(accept ≤ 1). Reason: this check's four rung-1.5 catches were all
corrected before verification (`rig-numbers-4-4-104-explained` read
+3.51 % pre-fix and is now 104.03). A violation here means a verified
grid still carries a double-mark — a real finding about ground truth.

*Q3 — within-phrase IOI spread on the 28 verified grids.* **Predicted
0–3 clips flagged at CV > 15 %.** Named must-not-flag rows, because
their spread is musical rather than erroneous:
`rig-names-3-4-88-waltz` (10.7 %, periodic bar-internal agogic accent),
`rig-names-4-4-96-allegro` (9.1 %), `rig-names-2-4-120-clean` (6.4 %).
If any of the three flags, the threshold is wrong for this corpus and
that is the reportable finding.

*Q4 — positive control on the 2 provisional grids.* `adr007-plies-demo`
and `rig-mixed-4-4-104-quantities` are raw peakRate seeds carrying
sub-tactus events. **Predicted: BOTH flagged by min-IOI.** A QC check
that finds nothing on unverified seed data is not checking anything, so
this is the control that the checks can fire at all.

*Q5 — round-trip stability (informational, sizes the owner's backfill).*
Re-serializing a verified format-1 grid through `load_grid` →
`save_grid` in a scratch dir changes **only** the `format:` line plus the
new optional keys, leaving beats, onsets, params and notes byte-for-byte
identical. **Predicted clean.** If it is not, the eventual backfill must
be surgical and the ledger says so.

*What failure looks like:* Q0 failing reverts the increment. Q1 failing
stops the session with a BLOCKED entry rather than a guessed metric.
Q2–Q4 are measurements, not gates — an unexpected flag is a finding
about the corpus or the threshold, reported either way.

Result: **W1 (rung 2.5) delivered — grid format 2, the three QC checks, and
annotation-method metadata; EVAL-CHANGE gate held.** No pipeline code was
touched. Prediction scorecard: **3 hits, 1 partial, 2 misses of 6.**

*Q0 — the EVAL-CHANGE gate: PASS.* The deterministic `run_stage1()` dump
hashes to `1d5fe5a3cbdc28b3e61873fa216ad36bc1a2e58c614b6ad6511ca5ed89c1d82a`
both before and after — `diff` reports no differences, 12,660 bytes each.
The suite run agrees end to end: tier-0 tempo 25/25, meter 24/25; tier-1
tempo 0.586, meter_triple 0.393, counts 0.571, sides 1.0, slot 1.0;
stage1 `aggregate_verified` 28 clips P 0.334 R 0.449 F 0.383, median
−19.4 ms; "no outcome changes vs baseline". pytest 213 passed / 3 skipped
(15 new tests). `git diff --stat main`: 8 files — `grids.py`, `qc.py`,
`annotation/__main__.py`, two test files, two eval docs, this ledger;
**0 files changed** under `evals/cases/`, `evals/traces/`,
`evals/grids/`, `evals/baseline.json`, or `src/musical_perception/evals/`.

*Q1 — validity gate: PASS (24 of 25 ≥ the predicted 23).* The
implementation reproduces to ±0.02 BPM every grid-implied figure the
owner recorded by hand in `notes`. The single miss is
`rig-names-4-4-104-coda`: recorded 106.44, computed 108.93 — and it is
the most useful number in the session, because the coda's own notes
already say why in prose ("TWO-TEMPO CLIP … CODA 12 beats, 25.91–34.00 s
… no stable period — free time"). Tagging exactly that span `free_time`
reproduces **106.44 exactly** and clears all seven of the clip's flags.
The metric was pinned before any new-check output was read.

*Q2 — min-IOI on the 28 verified grids: PARTIAL (1 clip flagged; accept
was ≤ 1, point prediction 0).* The one clip is `rig-names-4-4-104-coda`,
5 intervals of 0.150–0.253 s against a 0.558 s median. These are **not**
grid errors: the same notes record them as this check's first false
positives ("genuine fast marking, not stray labels"), and the notes go on
to prescribe the fix this rung implements. The prediction miss is my own
— the ledger I had already read contained the answer, and I reasoned only
from the four corrected export errors.

*Q3 — within-phrase spread on the 28 verified grids: MISS (4 flagged,
predicted 0–3).* All three named must-not-flag rows held:
`rig-names-3-4-88-waltz` 10.7%, `rig-names-4-4-96-allegro` 10.0%,
`rig-names-2-4-120-clean` 8.2% — the threshold does separate genuine
musical spread from error. The four flagged rows, each classified:
- `adr006-exercise-1-demo` (also −42.77% on BPM). Its notes warn
  outright that "the plain median IOI is NOT the beat period on this
  clip and would mislead (0.896 s)": the owner voices two of three beats
  per bar, so 21 intervals are two beats long. Tagging them
  `silent_beat` plus the owner-confirmed 20.6 s free-time tail moves the
  BPM flag from −42.77% to −3.85% and clears **all four** findings. The
  residual gap to the owner's cluster-first 118.42 BPM is real and
  unfixed: a plain median cannot recover the period here, tags or not.
- `rig-names-4-4-63-adagio`, 2 phrases at 29.3% / 25.0%. Tagging the six
  unvoiced beats the notes name reproduces the owner's recorded **15.0%**
  exactly. The finding worth keeping: two of those six gaps are 1.67×
  and 1.72× the median — **below** the 1.75× break ratio — because
  rubato compresses an unvoiced beat. No break ratio distinguishes them
  from a stretched beat, which is the argument for tagging rather than
  tuning. One flag survives at CV 15.03%, knife-edge over the line and
  equal to the owner's own number; no action (Standing Lesson 7).
- `rig-names-2-4-160-long`, one 6-interval phrase at 16.5%. Pooled
  within-phrase CV is **9.7%, exactly the figure in its notes** — the
  disagreement is statistic, not data: I flag max-over-phrases, the
  owner recorded pooled. At 160 BPM the beat is 0.375 s, so ±90 ms of
  annotation scatter reads as 16.5% in a short phrase. Threshold
  sensitivity at the corpus's fastest tempo, not a grid error.
- `rig-names-4-4-104-coda`, covered above.

**Zero of the 28 verified grids shows an actual annotation error.** Two
are cured exactly by the tags this rung adds; two are threshold
sensitivities at the tempo extremes, reported rather than tuned away
(pre-registered rule 4).

*Q4 — positive control: MISS, and it taught the most.* Predicted both
provisional grids flagged by min-IOI; **neither was** (ratios 0.521 and
0.524, just above 0.5×). Both are flagged loudly by the other two checks
(`adr007-plies-demo` +119.74% on BPM and 12 spread flags;
`rig-mixed-4-4-104-quantities` +28.63% and 4), so the control's intent —
"the checks can fire" — is satisfied, but the specific prediction was
wrong for a reason worth writing down: **min-IOI is relative to the
clip's own median, so it catches a *local* double mark and is blind to a
*globally* wrong metric level.** A raw peakRate seed is uniformly at the
syllable level, so nothing in it looks short relative to itself. The
level error is exactly what the BPM check is for; the three checks are
complementary, not redundant, and that is now demonstrated rather than
assumed.

*Q5 — round-trip stability: PASS, clean.* All 28 verified grids
round-trip through `load_grid` → `save_grid` with beats, onsets, notes
and params preserved, and **0 grids** show any text change beyond
`format: 1` → `format: 2` plus the two new keys. The owner's eventual
backfill is therefore a three-line diff per file.

Regressions and classifications: **none.** Every tier-0/tier-1/stage1
number is byte-identical to the blessed baseline and the runner prints
"no outcome changes vs baseline"; the one surviving QC flag after tagging
(adagio 15.03% vs a 15% line) is classified **knife-edge**. The two
threshold sensitivities (160-long, adagio) are classified **genuine-trade**
— a check sensitive enough to catch four real export errors will graze
rubato and fast material, and the corpus evidence says tagging is the
right fix, not a looser constant.

BLOCKED — owner queue (nothing here blocks the next session):
1. **Backfill `annotation_method` on the 28 verified grids.** Provenance
   is owner authority (§2.4's cohort count was itself corrected 21→25),
   so no session wrote it. One command per grid:
   `python -m musical_perception.annotation set-method <id> anchored|from_scratch`.
2. **Tag the regions the grid notes already describe in prose** — at
   minimum coda (`free_time` 25.91–34.00 s), exercise-1-demo
   (`silent_beat` × 21 + `free_time` 39.44 s→end, plus the one interior
   3.392 s gap the notes leave unclassified between silent-pulse and free
   time, which only the owner can settle), adagio (`silent_beat` × 6),
   160-long (`silent_beat` × 7). Each is a small edit to a verified grid,
   which is why it is owner business rather than agent business.
3. **The vocables dropped-beats listen** (rung-2 backlog (ii)): 2 of 16
   beats missed on `rig-vocables-4-4-100-clean`. Its QC is clean
   (min-IOI 0.933, CV 6.1%), so the checks cannot settle it — it needs an
   ear. Carried forward, still open.
4. **Ruling question raised, not answered:** `silent_beat` regions
   describe stretches the annotator did *not* mark, and ruling (b) still
   says vocalized-only. Whether a future scorer should credit silent
   beats (Standing Lesson 6) is a convention change and stays with the
   owner; the format now makes it expressible, which is all this rung
   claims.

Lesson (durable, one paragraph): Every QC flag on a verified grid turned
out to be the file failing to say something the annotator already knew —
the coda is out of time, the waltz voices two beats in three, the adagio
holds every fourth — and all of it was sitting in prose in the `notes`
field, unreadable to any check. Adding three region kinds did not make
the checks smarter; it let the human's existing knowledge reach them, and
the proof is that tagging reproduces the owner's hand-computed numbers
(106.44 BPM, 15.0% CV) to the digit rather than merely silencing a
warning. The second lesson is about thresholds: two of the six unvoiced
beats on the adagio clip compress to 1.67× and 1.72× the median under
rubato, which no break ratio can distinguish from a stretched beat, so a
constant tuned to pass them would have blinded the check on the material
it was ratified to police — the fix for a check that is right for the
wrong reason is more information, not a looser constant.

Status: PROPOSED (owner: review the format-2 extension and the three QC
checks; then the four BLOCKED items above, of which #1 and #2 are the
ones that gate W4 Barre-1 ingestion). Marathon: W1 complete, W2 (rung 3,
accent-periodicity meter votes) is the next-highest non-BLOCKED
workstream.

## 2026-08-16 · rung M · main · (W1 merge bookkeeping)

Attempted: Merge of `agent/marathon`'s W1 (rung 2.5) increment to main
and push, owner-directed in session ("can you merge to main and push").

Disclosure (charter rule 1): the charter reserves pushing `main` to the
owner and the agent flagged this before acting. The owner directed the
merge; the attestation is the owner's, the keystrokes the agent's —
recorded per-instance, never as a default, per the 2026-08-14 ruling.

Pre-registered expectations: n/a (bookkeeping).

Result: `agent/marathon` merged to main with `--no-ff` (4f755ea), clean,
no conflicts. **No `evals bless` was run and none is needed** — this is
an EVAL-CHANGE whose gate was byte-identical output: stage1 over the 28
verified grids hashes unchanged, and the post-merge suite on main prints
"no outcome changes vs baseline" (tier-0 tempo 25/25 meter 24/25; tier-1
tempo 0.586, meter_triple 0.393, counts 0.571; stage1 verified P 0.334 R
0.449 F 0.383, median −19.4 ms). pytest on main: 213 passed / 3 skipped.
`evals/baseline.json` is untouched, so the blessed baseline still
describes main exactly.

W1's status stays **PROPOSED** — merging is not blessing. The four
BLOCKED items in the W1 entry are unchanged and still the owner's queue;
items 1 and 2 (annotation_method backfill, tagging the regions the grid
notes already describe) are the ones that gate W4 Barre-1 ingestion.
`agent/marathon` is kept, not deleted: Rung M's contract runs every
session on that branch, and it was fast-forwarded to main so the next
session starts level.

Also corrected in this session (e954bec): the four date stamps this
session authored said 2026-08-14. Git's commit clock in this environment
reports 2026-08-14 while the shell and the harness both report
2026-08-16; the stamps now follow the shell/harness date, and every
owner-ratification date of 2026-08-14 elsewhere is untouched because
those are real. Flagged rather than silently normalized — a ledger whose
chronology drifts against its own commit timestamps is worth one line of
disclosure.

Regressions and classifications: none.

Lesson (durable, one paragraph): The marathon's first increment merged
without a blessing run, and that is a property of the work rather than a
shortcut — an EVAL-CHANGE that proves byte-identical outputs has nothing
to re-bless, which is exactly why the gate was written that way at
commissioning. The distinction worth keeping visible is that merging and
blessing came apart here for the first time: main now carries W1 while
W1's status is still PROPOSED, so the record must say which of the two
happened rather than letting a merge imply an attestation nobody made.

Status: BLESSED (owner-directed merge, 2026-08-16) as *bookkeeping*; the
W1 increment itself remains PROPOSED pending owner review.

## 2026-08-15 · rung — · (owner correction: (d') argument 1 withdrawn) · cloud

Attempted: Correction of the loop's memory on owner testimony, before the
owner-service session could present stale evidence as fact.
Pre-registered expectations: n/a (correction).
Result: The rung-1.5 finding "the pulse demonstrably continues through
explanation" (9.92/9.08-beat gaps, 44 ms re-entry) is WITHDRAWN as
evidence: the speaker testifies the restarts were ad lib and the phase
alignment coincidental (p ≈ 2% under random restart — unlikely, but the
speaker's account of his own production outweighs it). Convention (d')
now carries the correction inline; the ruling itself stands on its other
three arguments. Tagging consequences ratified by the owner: unmarked
talking stretches default to free_time; the proposed automatic
phase-re-entry test for held-pulse tagging is DEAD — this case proves it
false-positives; phase alignment may flag a stretch for human attention,
never decide a tag; held-pulse-through-talk is a per-clip owner ruling
for all future material. The master agent notes for the record that it
endorsed the withdrawn inference twice; the ground-truth authority
structure worked as designed.
Regressions and classifications: n/a.
Lesson (durable, one paragraph): A measurement can be improbable under
the wrong null and still be a coincidence — and the person who made the
sound is a better witness to the generative process than the residuals
are. Statistical inference about human production yields to the human's
testimony in this loop; where the two conflict, record both, let the
testimony govern the tags, and keep the measurement as a flag for
attention rather than a fact. Auto-rules derived from n=2 findings do
not survive contact with the n=1 human who produced them.
Status: BLESSED (owner testimony in session, 2026-08-15).

## 2026-08-16 · rung M · (departure guards) · cloud

Attempted: Two charter guards added before an extended unattended
marathon stretch, owner-requested ("send the Air on another long
adventure"): (1) W5 (joint posterior) marked OWNER-STARTED — scheduled
sessions treat it as BLOCKED-on-owner and move past it, so the deepest
modeling work cannot run unattended on a default model; (2) W0 meta-rung
made self-scheduling — it outranks all workstreams whenever the last
meta entry is >7 days old (or none exists after >=5 sessions), so the
loop reviews itself weekly without the owner triggering it.
Pre-registered expectations: n/a (governance).
Result: Charter amended; no pipeline or eval changes.
Regressions and classifications: n/a.
Lesson (durable): Autonomy for an absent owner is not fewer rules but
different ones — the guards convert "the owner would have caught this"
into structure: the un-delegable work waits, and the review happens on
schedule whether or not anyone is watching.
Status: BLESSED (owner-requested in session, 2026-08-16).

## 2026-08-18 · rung M · agent/owner-queue-20260816 · local (owner-service + launch)

Attempted: **Owner-service session — no W-workstream advanced, by design.**
Working the BLOCKED queue with the owner present, plus arming the nightly
schedule. Branch name carries `20260816` as the owner specified it; the
shell and git clock both read 2026-08-18 and agree with each other (the
2026-08-16 entry's clock disagreement has resolved itself), so the entry
is stamped 08-18 and the branch name is left as issued.

Pre-registered expectations: n/a (owner service + governance). The one
measurable claim is the EVAL-CHANGE gate on (a), stated before running:
writing `annotation_method` + a notes line touches no `beats` and no
`onsets`, so stage1 must be byte-identical. **Predicted PASS.**

BLOCKED sweep (the prompt warned its list might be incomplete — it was).
Two historical BLOCKED entries are already closed (2026-08-11 missing DEV
media; rung-1 grid coverage). Three open items the prompt did not name:
- **W1's status is still PROPOSED.** It is merged to main but never
  reviewed. Merging is not blessing; the distinction is live.
- **The 2026-08-14 blessing entry is wrong about branch deletion.** It
  records `agent/rung-1-stage-scoring` and `agent/rung-2-acoustic-pulse`
  as deleted. They were not — both existed, local and remote, until this
  session. Corrected by doing it, and recorded here so the ledger stops
  asserting a completed act that had not happened.
- **Rung-2 backlog (i)** (relative speech-band silence floor for quiet
  clips) is parked behind W2 as a pipeline workstream, not an owner
  block. Noted so it is not lost; no action.

Result, item by item.

*(a) `set-method` backfill — RESOLVED, gate held.* All 28 verified grids
stamped: 25 `anchored`, 3 `from_scratch` (`adr006-exercise-1-demo`,
`adr010-grande-battement`, `frappe` — the three video grids, owner-
confirmed in session against the agent's reading of log line 440). The
2 provisional grids are deliberately unstamped. **A limitation surfaced
and ruled on rather than worked around:** the owner's instruction was
that labels name the *method*, not just the cohort, but the enum is a
frozen two-value vocabulary (W1 pre-registration §1, rule 4 forbids
retuning it after the fact) and `from_scratch` cannot say "live tap in
REAPER, then nudge-corrected". Owner chose enum + prose over a format-3
proposal, so each grid's `notes` now carries the tool and the gesture —
the same field where every other method fact in this corpus already
lives. `git diff --stat main`: 28 grid files, 305 insertions / 56
deletions; **0 files** under `evals/cases/`, `evals/traces/`,
`evals/baseline.json`, or `src/musical_perception/evals/`. Diff
inspection confirms no `beats` or `onsets` line changed — the deletions
are YAML re-wrapping of existing notes plus `format: 1`→`2` and
`regions: []`. Suite: **"no outcome changes vs baseline"**;
`aggregate_verified` clips=28 P=0.334 R=0.449 F=0.383 (macro 0.386)
median −19.4 ms, identical to the W1 entry's figures. **Prediction hit.**

*(b) Region tagging — DEFERRED by turn budget, material prepared.* Not
started. The four clips whose own notes already describe their spans in
prose are `rig-names-4-4-104-coda` (`free_time` 25.91–34.00 s),
`adr006-exercise-1-demo` (19 × `silent_beat` from the 2-beat gaps,
`free_time` 39.44 s→end and 0→4.77 s, plus the interior 3.392 s gap at
~16.0 s that the notes leave unclassified and only the owner can settle),
`rig-names-4-4-63-adagio` (6 × `silent_beat` at ~6.734 / 10.638 / 14.337
/ 18.124 / 22.122 / 25.765 s), `rig-names-2-4-160-long` (7 × `silent_beat`
after 5.341 / 8.333 / 11.325 / 14.315 / 17.283 / 18.847 / 20.294 s).
Per the 2026-08-15 correction, unmarked talking stretches default to
`free_time` and phase-aligned re-entry is a flag for attention only.

*(c) Vocables dropped-beats listen — DEFERRED, and BLOCKED on media.*
The two missed beats are computed and pinned: on
`rig-vocables-4-4-100-clean` the rung-2 extractor misses **beat 9 at
7.274 s** and **beat 13 at 9.702 s** (one-to-one ±70 ms against the
verified grid; nearest extractor events are 6.680/7.868 and 9.382/9.975,
so it emitted nothing in either window rather than mistiming). **New
blocker found this session:** `audio/rig/*.mp3` is not on this machine
and has never been committed — the `.gitignore` exception exists but
`git log --all -- 'audio/rig/*'` is empty. `audio/counting/` and
`video/youtube/` are present; the rig audio is not. The colour-coded
Reaper marker file is trivial once the audio is staged; without it there
is nothing to listen to. Owner action: stage the DEV rig MP3s here, or
do the listen on the machine that holds them.

*(d) Ruling (g), silent-beat crediting — DEFERRED, not ratified.* Not
presented, so nothing was written into the convention. This is the item
that most needed the owner's own words and it did not get its turn; it
carries forward unchanged and unprejudiced.

*(e) Housekeeping — RESOLVED.* `agent/rung-1-stage-scoring` and
`agent/rung-2-acoustic-pulse` deleted, remote and local, after confirming
both are merged into main. `agent/marathon` untouched, local and remote.

*(f) Launch — RESOLVED and armed.* `scripts/air-nightly.sh` now passes
`--model opus` to the headless invocation (`bash -n`: OK).
`~/Library/LaunchAgents/com.musical-perception.nightly.plist` created
from the agent-environment.md template with this machine's real paths
(`/Users/la-ben.juodvalkis/github/musical-perception`), `plutil -lint`:
OK, and loaded. `launchctl list` shows `- 0 com.musical-perception.nightly`
— registered, never fired. **Departure from the template, disclosed:**
the template sets no environment, but `CLAUDE_BIN` defaults to bare
`claude` and launchd does not inherit a login PATH, so the job would have
died on command-not-found at 2am. The plist therefore sets `CLAUDE_BIN`
to the absolute `/Users/la-ben.juodvalkis/.local/bin/claude` (verified
executable) plus `HOME` and `PATH`. This is exactly the failure mode
agent-environment.md's "Air failure modes" note predicts. No nested
claude session was launched; the first firing is 02:00 local.

Also this session: `.claude/settings.local.json` created with a single
narrow allow rule for `annotation set-method`, because the auto-mode
classifier refused every compound shell command and 28 grids at one turn
apiece would have consumed the session. Owner-authorized in-session
rather than routed around; the file is globally gitignored and does not
enter the repo.

Regressions and classifications: **none.** Every tier-0/tier-1/stage1
number byte-identical; the runner prints "no outcome changes vs
baseline". `evals/baseline.json` and the frozen scorer untouched.

Lesson (durable, one paragraph): The queue's cheapest item exposed the
sharpest constraint — an enum frozen a priori is a promise not to
improvise, so when the owner asked the label to say more than the enum
can hold, the honest move was to surface the collision and let him choose
between prose today and a format change later, not to widen the
vocabulary quietly and report success. The session's other lesson is
about budget: five owner-authority items and a launch do not fit in forty
turns when each needs the owner's actual words, and the right failure is
to finish the mechanical work completely and hand back the judgment calls
untouched rather than to guess at three rulings in the last five turns.
(b), (c) and (d) are deferred with their evidence computed and their
questions intact, which costs the owner a session but costs the ground
truth nothing.

Status: PROPOSED. Owner queue, unchanged and now sharpened: (b) region
tags on the four named clips; (c) the vocables listen, blocked until the
rig MP3s are staged on this machine; (d) ruling (g) ratification; plus
the standing review of W1, still PROPOSED since 2026-08-16.

## 2026-08-18 · rung M · agent/owner-queue-20260816 · local (owner-service, session 2: items (b) and (d))

Attempted: The two deferred owner-authority items from this morning's
entry — (b) region tagging on the four clips whose notes describe their
spans in prose, and (d) ratification of ruling (g), silent-beat
crediting. Both worked with the owner present, one clip and one question
at a time. (c) remains blocked on media; W1's review remains open.

Pre-registered expectation, stated before writing any region: tags are
parallel metadata and add no beats, so **stage1 must stay byte-identical**
and the QC checks must reproduce the owner's hand-computed figures.
Predicted PASS on both.

Result: **(b) COMPLETE — 4 clips, 33 `silent_beat` gaps, 6 `free_time`
spans. (d) RATIFIED and written into the convention as ruling (g).**

*(b), clip by clip. Every span was proposed from the clip's own notes and
ruled on by the owner in session; nothing was inferred.*

- `rig-names-4-4-104-coda` — owner chose two regions over one: the 6.8 s
  break (19.110–25.912) and the out-of-time coda (25.912–33.996) tagged
  `free_time` separately rather than merged, keeping them distinct events
  in the record. **Plus a find the notes never recorded:** a 2.12×-median
  gap at 10.484–11.669 inside the *in-time body*, owner-ruled
  `silent_beat`. Result: grid-implied BPM **106.44 — the owner's
  hand-computed figure reproduced exactly** — and all seven flags cleared.
- `rig-names-4-4-63-adagio` — 6 `silent_beat` gaps, exactly where the
  notes put them. Two of the six are 1.67× and 1.72× the median, *below*
  the 1.75× phrase-break ratio, which is the pre-registered argument for
  tagging over threshold tuning, now exercised rather than argued.
  One flag survives at CV **15.2%** against a 15% line — **knife-edge**,
  equal to the owner's own recorded 15.0%, no action (Standing Lesson 7).
- `rig-names-2-4-160-long` — 7 `silent_beat` gaps as recorded. Clean,
  spread 9.9%, zero flags.
- `adr006-exercise-1-demo` — the hard one. The plain median (0.896 s) is
  not the beat period here, exactly as its notes warn, so gaps were
  measured against the cluster-derived **0.5067 s**: 19 `silent_beat`
  gaps, matching the notes' predicted 19. **Two holes needed the owner
  and only the owner:** the 3.392 s gap at 15.930 the notes explicitly
  left unclassified, and a 2.309 s gap at 28.186 **the notes never
  mention** — found this session. Owner ruled both `free_time`: the pulse
  *stopped*. Plus the head (0–4.772) and the owner-confirmed 20.6 s
  demonstration tail. Result: BPM flag **−42.77% → −3.85%**, the figure
  the W1 entry predicted, all four findings cleared, spread 9.2%.

*Prediction hit.* Suite after tagging: **"no outcome changes vs
baseline"**; `aggregate_verified` clips=28 P=0.334 R=0.449 F=0.383 median
−19.4 ms, unchanged. `git diff --name-only main` over `evals/cases/`,
`evals/traces/`, `evals/baseline.json`, `src/musical_perception/evals/`:
**0 files**.

*A finding worth more than the tagging: the notes' BPM figures are now
stale, and in two directions.* Tagging changes what "grid-implied BPM"
means, because tagged IOIs are excluded from the median. On the coda that
moved the number **onto** the owner's recorded 106.44 (he had computed
body-only). On the adagio and the 160-long it moved it **away** — 61.39
→ 65.17 and 159.76 → 164.07 — because those figures were computed
whole-clip, gaps included. Both still pass the 4% check so nothing flags,
but two grids now disagree with their own prose. Recorded rather than
silently reconciled; the notes are the stale artifact, not the tags.

*(d) Ruling (g) — ratified with one amendment the owner accepted.* The
proposal as put: stage-1 stays vocalized-only and unchanged; crediting
happens at the rung-4 tier as a separate CONTINUATION metric scoring
phase coherence across tagged `silent_beat` gaps against the verified
beats on the far side; **no beat is ever placed inside a gap** by hand or
by interpolation; `free_time` credits nothing anywhere. The agent
recommended, and the owner ratified, one addition: **CONTINUATION reports
per-clip, never as a bare average across gaps, and pins no gate until its
coverage is broad enough to support one.** The argument was precedent
rather than principle — the rung-2 gate already refused a margin carried
by one or two clips ("A margin carried by one or two clips does not
pass"), and today's tagging makes the same hazard concrete: 19 of the 33
gaps (58%) are on `adr006-exercise-1-demo` alone, so a gap-averaged
CONTINUATION would be one clip wearing a corpus's name. Also deliberately
excluded on the agent's recommendation: any specification of *how* phase
coherence is computed. Pre-specifying an unbuilt metric is precisely what
voided the original rung-2 gate; the arithmetic gets pinned at rung 4,
against grids that exist, with reference code. Ruling (b) gains a
cross-reference so its "stays unscored until a taggable format" line no
longer reads as still-pending.

Regressions and classifications: **none.** All tier-0/tier-1/stage1
figures byte-identical; the one surviving QC flag (adagio 15.2% vs a 15%
line) is classified **knife-edge**, as its predecessor was.

Lesson (durable, one paragraph): Tagging was supposed to be transcription
— moving what the notes already said into a form the checks could read —
and on three clips it was. What it actually produced was two beats nobody
had written down (an unvoiced beat inside the coda's in-time body, and a
2.3-second hole in the waltz that no note mentions), because prose can
describe a clip without accounting for it, and a format that demands
spans forces the accounting. The second lesson is the stale-notes finding:
making the file smarter changed the meaning of a number the file already
carried, so two grids now contradict their own prose — the cost of a
better representation is that every figure derived under the old one has
to be re-read, and the honest move is to say which artifact went stale
rather than to quietly recompute the notes to match.

Status: PROPOSED. Ruling (g) is BLESSED (owner-ratified in session,
2026-08-18) and live in the convention. Owner queue now: (c) the vocables
listen, still blocked until the rig MP3s are staged on this machine; the
review of W1, PROPOSED since 2026-08-16; and the notes-vs-tags BPM
staleness on `rig-names-4-4-63-adagio` and `rig-names-2-4-160-long`.

## 2026-08-18 · rung M · main · (owner-queue merge bookkeeping)

Attempted: Merge of `agent/owner-queue-20260816` to main and push,
owner-directed in session ("merge it").

Disclosure (charter rule 1): the charter reserves pushing `main` to the
owner and the agent flagged this before acting. The owner directed the
merge; the attestation is the owner's, the keystrokes the agent's —
recorded per-instance, never as a default, per the 2026-08-14 ruling.

Pre-registered expectations: n/a (bookkeeping).

Result: merged with `--no-ff`, clean, no conflicts. 31 files, 800
insertions / 58 deletions. **No `evals bless` was run and none is
needed** — same reasoning as the W1 merge: this is an EVAL-CHANGE whose
gate is byte-identical output. Post-merge on main: pytest **213 passed /
3 skipped**; suite prints **"no outcome changes vs baseline"** (stage1
`aggregate_verified` 28 clips P 0.334 R 0.449 F 0.383, median −19.4 ms).
`evals/baseline.json` untouched, so the blessed baseline still describes
main exactly. Zero files changed under `evals/cases/`, `evals/traces/`,
`evals/baseline.json`, or `src/musical_perception/evals/`.

What main now carries that it did not this morning: `annotation_method`
on all 28 verified grids; the first region tags in the corpus (33
`silent_beat`, 6 `free_time` across 4 clips); ruling (g) live in the
convention; `--model opus` in the nightly script. The launchd job is a
machine-local artifact and is not in the repo.

Consequence for tonight, which is why the merge was asked for: the 02:00
run pulls fresh main, so it now reads the ratified convention including
ruling (g) and the tagged grids, rather than this morning's state.

Regressions and classifications: none.

Lesson (durable, one paragraph): The second owner-directed merge of the
marathon followed the first one's shape exactly, and the value of having
written that shape down is that this time nobody had to re-derive whether
a blessing was owed — the gate says byte-identical output, the output was
byte-identical, so the merge is bookkeeping and the increment's status is
unchanged. A merge that happens for a *reason* is also worth recording:
this one was requested so that an unattended session six hours later
would read the ruling the owner had just made, which is the first time in
this project that merge timing was driven by what a future agent needs to
know rather than by review readiness.

Status: BLESSED (owner-directed merge, 2026-08-18) as *bookkeeping*; the
increments themselves — the (a)/(b) grid work and item (f) — remain
PROPOSED pending owner review, as does W1 from 2026-08-16. Ruling (g) is
separately BLESSED by owner ratification in session.

## 2026-08-19 · rung M · agent/nightly-permission-fix · local (W0 recovery + runner fix)

Attempted: Diagnosis of the first unattended nightly run, recovery of the
W0 meta-rung review it completed but could not file, and the runner fix.
The review below is the **nightly session's work, recovered from
`~/musical-perception-agent.log`**; the diagnosis, verification and fix
are this session's. Attribution kept explicit because the reviewing agent
never got to sign its own entry.

Pre-registered expectation for the fix, stated before running it: a
headless `claude -p` with `--permission-mode auto` can write a file, and
the same invocation without it cannot. Predicted PASS.

Result: **the run fired perfectly and wrote nothing.**

*What happened.* launchd started the job at 02:00:04 PDT exactly as
scheduled. It pulled `main`, read the charter and ledger, selected W0
correctly (no meta entry existed and the marathon had run ≥ 5 sessions,
so W0's self-scheduling rule outranked every pipeline workstream), and
ran **107 turns / 20 minutes / $11.98**, exiting `success`. Then: `main`
still at 5e5f592, clean tree, no branch, no commit, no ledger entry.
Seven denied writes across four mechanisms (`Write`, `Edit`, `git
branch`, `git switch`), plus denied `pytest`. The log line that names it:
*"Claude requested permissions to write to … RESEARCH-LOG.md, but you
haven't granted it yet."*

*Root cause, and it is one flag.* `scripts/air-nightly.sh:32` invoked
`claude -p` with **no `--permission-mode`**, so the session's init reads
`"permissionMode":"default"` — the mode that waits on a human. Permission
mode is **per-session and inherits nothing**: the interactive session that
armed the schedule had auto mode on, and that fact travelled with it and
not with the job.

*The deeper cause, which is the finding worth keeping.* The rung-0
checklist's guard (agent-environment.md step 4) specifies a **supervised
interactive dry-run**. An interactive session gets its permissions from a
human answering prompts, so it is *structurally incapable* of detecting a
headless-only permission failure. The guard could never have caught this.
The 2026-08-18 launch session verified the job would **start** — PATH,
`CLAUDE_BIN`, `plutil -lint`, git reachability, log writability — and
never verified the started job could **act**. That omission is this
agent's, and the checklist made it easy to make.

*Fix applied and verified.* (1) `--permission-mode auto` added to
`air-nightly.sh:32`, with a comment marking it load-bearing rather than
convenient. `auto` rather than `bypassPermissions`: the runner executes
under the owner's own account and working repo, **not** the dedicated
sandbox account the charter's rung-0 assumed, so blanket bypass would be
a materially different bargain than the one the charter designed; auto
gives the job the same classifier-guarded latitude the owner already
accepts interactively. (2) The rung-0 checklist gains a **headless write
probe** as step 5, with the exact command and the generalized rule:
*never accept an interactive test as evidence about a non-interactive
runner.* **Probe run and passed** — `claude -p '…' --permission-mode
auto` wrote `/tmp/mp-write-probe.txt` containing `ok`, exit 0.
**Prediction hit.**

*Known residual, disclosed:* auto mode's classifier refuses compound
shell commands — it blocked this session's own `sed && …` and, on
2026-08-18, a `for` loop over `set-method`. The nightly agent will meet
the same wall and must issue commands singly or use the file tools. That
degrades throughput, not correctness, and is the accepted cost of not
handing an unattended 2am process a blanket bypass on the owner's
account.

---

### The W0 review, as recovered (nightly session, 2026-08-19)

**Trigger:** correct. No meta-rung entry existed; W0 outranked all.

**BLOCKED-queue audit, checked against files rather than against the last
entry that mentioned each item.** *Closed but still listed open:* the
`annotation_method` backfill; region tagging; ruling (g); **both
rung-1.5 `marking_bpm` decisions** (`adr006-8-counts-2x` reads 101.63
"CORRECTED", `adr006-8-counts-triple` reads 68.38 "PINNED" — taken
08-14, never recorded as closed); the v1 re-annotation question. *Genuinely
open:* three increments on main unreviewed (W1 since 08-16, plus the
08-18 grid work and the launch item — the first weekly batch is due); the
vocables listen, blocked on media; the notes-vs-tags BPM staleness; and
**`evals/cases/rig-numbers-2-4-120-clean.yaml:12` still reading
`accompanied: false`** with notes describing a metronome in one earbud,
on the clip the owner heard music on — carried in prose since 08-13,
never reaching the file. Cases are agent-untouchable; only the owner can
close it. *(Verified this session: the line is there.)*

**Re-ranking:** W2 (rung 3, meter) · **W2.5 (rung-2 nuclei silence floor,
promoted)** — cheapest pipeline increment with a pre-measured target
(`rig-names-4-4-100-quiet`, 9 events for 16 beats, the only step_names
clip rung 2 did not improve) · W3 baselines · W4 ingestion (no longer
gated by W1, which shipped; recommend it wait on W1's *review*) · W5
OWNER-STARTED · W6–W8.

**Rung 3 re-expressed — and the review's own mid-course correction, which
is the best thing in it.** It first put W2's reachable set at seven
non-4/4 rows, then read `src/musical_perception/evals/scorers.py:151-208`
(read-only) and corrected itself: `meter_triple` requires meter **and**
tempo within ±8% **and** subdivision jointly, and `meter_wrong` is a
priority label emitted whenever meter differs regardless of tempo. **The
true reachable set is two rows, not seven** — `rig-numbers-2-4-120-clean`
and `rig-numbers-3-4-90-clean`, the two the charter had named. *(Verified
this session against `evals/baseline.json`: those two predict 119.7 vs
120 and 90.8 vs 90, inside tolerance; every step_names non-4/4 row is
17–50% off on tempo — 90.1 vs 120, 78.3 vs 160, 102.6 vs 88, 129.6 vs 90,
133 vs 100. Meter-only code cannot flip them.)* The charter's own
"1-of-8 … 0-of-3" is stale; the blessed truth is **2-of-9**.

**A6 — the strongest recommendation: re-scope W2 from an accuracy rung to
an evidence rung.** A meter-only module can move `meter_triple` from
0.393 to at most 0.459 (2 of 30 rows), and the reason is structural, not
fixable by better meter code — which is exactly ADR-016's and review-3's
argument for the joint posterior. Score W2's votes as a **diagnostic**
(does it recover the correct grouping on the nine non-4/4 clips given the
verified grids?) rather than against tier-1 committed accuracy, keeping
the two flippable rows as the accuracy check. The charter already says W5
requires W2's evidence; A6 makes W2 honestly what it already is, and
stops a future session pre-registering against a two-row ceiling and
calling a two-row move a rung.

**Structural finding behind A6:** five of seven wrong non-4/4 rows predict
4/4, four of them with a duple subdivision the truth lacks —
`interpret_meter`'s multiplier heuristic collapsing onto 4/4 by
construction.

**Pre-registered predictions drafted for W2 (M1–M5):** 3/4 rows flip most
readily (the waltz carries a *measured* periodic bar-internal agogic
accent, +9.9% / −3.7% / −7.0% consistent across all 8 bars — a period-3
signal sitting in verified ground truth); 2/4 is the risk, since counting
2/4 in eights makes period-2 and period-4 nearly indistinguishable; 6/8
flips or abstains but must never land on 3/4; zero regressions among the
11 greens (logic change, ADR-015); no movement on the 10
tempo/subdivision rows. Build note: run against the committed
`docs/research/rung2-extractor-events.json`, which needs no models, no
API key, and no rig MP3s.

**Amendments A1–A6 recovered:** (A1) Rung M's per-session condition needs
a **writability precondition** — a session that cannot commit satisfies
no clause and has no defined exit; (A2) a session's first act should be a
cheap write probe (this one spent eight turns reading before the
constraint surfaced); (A3) the meta-rung trigger should count ledger
*entries*, not "sessions"; (A4, flagged not pressed) the completion
targets assume n ≥ 60 verified DEV rows against a corpus of 28, so every
target is unreachable until W4 ingests — the ranking and the completion
criteria disagree about what matters; (A5) the rung-3 re-expression
above; (A6) as stated. **A7 and A8 are named in the run's closing summary
but were never written out in recoverable form; they are not reconstructed
here.** Their evident substance — the headless-probe guard and the
permission-mode fix — is implemented in this entry regardless.

**The nightly agent's own closing judgement, kept because it is the right
one:** it declined to keep re-attempting denied writes or to generate
turns to reach a bound it could not display, on the grounds that
fabricating a committed ledger entry or padding a counter were both worse
than an honest stop. It flagged every constraint clause as holding
*vacuously* rather than letting an empty `git diff --stat main` read as
compliance.

---

Regressions and classifications: none. No pipeline, eval, or grid file
touched this session: `git diff --stat main` covers
`scripts/air-nightly.sh`, `docs/research/agent-environment.md`, and this
ledger only.

Lesson (durable, one paragraph): The guard that fails is the one whose
test runs in a different mode from the thing it guards — an interactive
dry-run can prove a headless runner will start and can never prove it can
act, and no amount of care inside the wrong mode substitutes for testing
in the right one. The second lesson is about what an unattended agent is
worth when it is blocked: this one could not write a byte and still
produced the most useful artifact of the week, including a self-correction
that cut its own headline finding from seven rows to two after reading the
scorer — so the recovery question for a failed run is never "did it
commit" but "did it think, and is the thinking retrievable." The log was
the only reason it was.

Status: PROPOSED. The runner fix needs to reach `main` before 02:00
tonight or the same failure repeats. Owner queue unchanged and now
sharpened: the first weekly batch review (three increments, oldest since
08-16); the `accompanied: false` case-file discrepancy; the vocables
listen; and A1–A6 to accept, modify or reject.

## 2026-08-19 · rung M · main · (permission-fix merge bookkeeping)

Attempted: Merge of `agent/nightly-permission-fix` to main and push,
owner-directed in session ("yes").

Disclosure (charter rule 1): the charter reserves pushing `main` to the
owner and the agent flagged this before acting. The owner directed the
merge; the attestation is the owner's, the keystrokes the agent's —
recorded per-instance, never as a default, per the 2026-08-14 ruling.

Pre-registered expectations: n/a (bookkeeping).

Result: merged with `--no-ff`, clean. 3 files, 209 insertions, 0
deletions. Verified on main: `--permission-mode auto` present at
`scripts/air-nightly.sh:39` under its load-bearing comment; `bash -n`
clean; pytest **213 passed / 3 skipped**. No eval, grid, or pipeline file
touched, so no suite re-run was owed and `evals/baseline.json` still
describes main exactly. Third owner-directed merge of the marathon; the
pattern is now routine enough that its disclosure is the only thing
keeping it from becoming a default, which is the point of writing it
each time.

Timing was the whole reason for the merge: the fix had to be on `main`
before 02:00, because the nightly job's first act is `git pull --ff-only
origin main` and it would otherwise re-run the 2026-08-19 failure
verbatim. This is the second consecutive merge driven by what an
unattended session needs to read rather than by review readiness — worth
noticing as a pattern, since it means `main` is now serving as the
loop's instruction channel and not only its record.

Regressions and classifications: none.

Lesson (durable, one paragraph): A fix for an unattended runner is not
finished when it is correct, it is finished when it is on the branch the
runner reads — the gap between "fixed" and "in effect" was six hours and
one merge, and nothing in the repo would have marked the difference if
the deadline had been missed. Where a scheduled agent pulls `main` before
acting, merge timing is an operational act, not bookkeeping, and belongs
in the ledger with its deadline stated.

Status: BLESSED (owner-directed merge, 2026-08-19) as *bookkeeping*; the
W0 review and amendments A1–A6 remain PROPOSED for owner decision, as do
the three increments awaiting the first weekly batch review.

## 2026-08-19 · rung M · agent/state-of-play · local (stock-taking: owner briefing)

Attempted: A plain-language state-of-the-project briefing for the owner,
`docs/research/state-of-play-2026-08-19.md` — a stock-taking session, not
a marathon increment. No workstream advanced; no pipeline code changed.
Local `main` was 28 commits behind `origin/main` at session start and was
fast-forwarded to `034d226` before anything was read (no push).
Pre-registered expectations: n/a (briefing, not experiment).
Result: Briefing written and committed. Every headline claim was checked
against artifacts where possible and labelled ✓ (verified) or (relayed)
in the document. Verified this session: 30 grid files, 28 `provisional:
false`, 25 anchored / 3 from_scratch, 33 `silent_beat` + 6 `free_time`
regions on 4 clips, 28 at format 2; blessed baseline tier-1 tempo 17/30,
meter_triple 11/29, counts 12/28 (+7 abstained), by-style slices as
recorded; stage1 verified 28 clips P 0.334 R 0.449 F 0.383; rung-2
kill-test tables and per-clip from-scratch cohort (R@tac 0.561/0.691/
0.694, P_lc 0.242/0.452/0.490) from the committed JSON; scorer requires
meter AND tempo ±8% AND subdivision, so only 2 of the 7 wrong non-4/4
rows are meter-flippable (blessed non-4/4 truth is 2-of-9);
`evals/cases/rig-numbers-2-4-120-clean.yaml:12` still `accompanied:
false`; `scripts/air-nightly.sh:39` carries `--permission-mode auto`;
`analyze.py` does not reference the acoustic extractor (not wired in).
Live: pytest 213 passed / 3 skipped; `evals run --suite
tier0,tier1,stage1` prints "no outcome changes vs baseline". Noted: the
DEV rig MP3s ARE present on this (owner's) machine, so the vocables
listen is blocked only on the Air. The `video/youtube/Ballet Barre 1`
directory exists here and was deliberately not opened (HELD-OUT may live
in it). Briefing length ~2,000 words incl. headers, slightly over brief.
Regressions and classifications: none — `git diff --stat main` shows
only this ledger and the briefing; 0 files under `evals/cases/`,
`evals/traces/`, `evals/grids/`, `evals/baseline.json`, or
`src/musical_perception/evals/`. No `evals bless` run. `main` not pushed.
Lesson (durable, one paragraph): A briefing that cites the ledger has to
re-derive the ledger's claims from the files, because the ledger is a
record of what sessions believed, not of what is on disk — and the
cheapest way to keep the two honest is to label every sentence with
which one it came from.
Status: PROPOSED (owner: read the briefing; its §7 is the owner queue).
