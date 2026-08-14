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

Status: PROPOSED (rung 1.5 substantially complete: 28/30 verified, 2 declined
with recorded reasons). Two deliverables remain before the CURRENT RUNG
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
