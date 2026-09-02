# The Lateral Review — protocol

**Date:** 2026-09-01 · **Status:** PROPOSED at the owner's request and
shaped with him in session `claude/fable-model-research-n8kgkm`; the
on-demand Routine is live from this date; the charter note beside W0 is
the owner's to ratify. Companion to the [agent charter](agent-charter.md)
(Rung M, alongside W0) and the [ledger](RESEARCH-LOG.md).

## Why this exists, in one paragraph

W0 — the weekly meta-rung — is convergent: it reads the ledger, re-ranks
the commissioned workstreams, audits the BLOCKED queue, proposes
amendments. It answers *"what next, from the list."* It never asks whether
the list is the right list. The project is early, with more unknown
unknowns than known ones, and the best direction in the ledger so far
(the factored meter representation, then W13) came from the owner's
musician's introspection, not from any ranking. The Lateral Review is the
divergent counterpart: an **owner-attended conversation** on the
top-tier model, prepared by an unattended read of all the evidence, whose
deliverables are **ideas tethered to evidence, each with the cheapest
test that would kill it**, the owner's answers to questions only a dance
musician can answer — asked **one at a time, each chosen in light of the
last answer** — and a distilled record of both. It proposes; it never
commissions.

## What it is not

- Not a workstream. Scheduled sessions and W0 **never take it**; it
  never takes a workstream. It neither counts toward nor resets W0's
  7-day clock.
- Not a ranking. It may say what it *would* put first and why; the
  standing ranking is W0's to propose and the owner's to ratify.
- Not a code session. It changes no file under `src/`, `evals/`,
  `scripts/`, or `tests/`, and does not edit the charter. It writes its
  memo and its ledger entry, nothing else.
- Not a status report. The state of play is context, not the product.

## Cadence: on demand

Owner availability varies from daily to weeks away, so there is no
calendar. The owner starts a review by firing the Routine named
**Lateral review (on demand)** from the Routines list, whenever he has
roughly half an hour. The session sizes its reading to the gap: after
weeks away it reads everything since the last review; after days it
reads the delta and says so.

## Session shape — four phases

Fresh cloud session on the top-tier model. Branch
`agent/lateral-YYYY-MM-DD` cut from `origin/main`. **Writability
precondition (charter amendment 2):** first act is a cheap
write-and-commit probe; a session that cannot write stops with its
diagnostic. Environment: documents and frozen artifacts only; no media is
needed or sought. `bash scripts/cloud-setup.sh` (no `--live`) is
permitted for replaying frozen traces or a suite check; never required.

### Phase 1 — the read (unattended, before the owner is engaged)

Read, in this order, all of it:

1. `docs/research/agent-charter.md` in full, including the OWNER QUEUE
   (read, not acted on — surfacing the queue is the W-sessions' job).
2. `docs/research/RESEARCH-LOG.md`: Standing Lessons; every entry since
   the previous lateral memo in full (since 2026-08-19 for the first
   review); earlier entries through the W0 entries and
   `state-of-play-2026-08-19.md`. Also as it stands on
   `origin/agent/marathon` — completed-but-unmerged work is invisible
   from `main` by construction.
3. Previous memos in `docs/research/lateral/`, including the owner's
   recorded answers, so nothing is re-asked or re-pitched without new
   evidence.
4. `docs/adr/016-rhythm-core-reset.md`, `017-factored-rhythm-posterior.md`;
   the five literature reviews and `voice-as-drum-review.md`;
   `docs/vision/05`, `08`, `09`, `10`, `13`.
5. Every result memo in `docs/research/` newer than the last review, with
   its JSON wherever a claim rests on a number.
6. `docs/evals/baseline.md`; `logs/run-summaries.md` (the nightly
   runner's own closing messages — the loop's unfiltered voice).
7. `docs/vision/03-landscape.md` §3.6 ("what to watch") — the standing
   list of external developments the project said it would keep an eye
   on; the outward scan below checks each item.

Then draft the anomalies (§1 of the Method) and the assumption ledger
(§2), and **do the outward scan** (§3a) — outward *after* both, so the
search is aimed by a named anomaly or a rated assumption rather than
being a survey. Then prepare, privately, the rest of the Method and
write the **briefing** (Phase 2). If the read would
exhaust the turn budget, ledger and previous memo first; say in the
briefing what was skipped.

### Phase 2 — the briefing (the session's first message to the owner)

Plain language per the "Talking to Ben about this work" section of
`CLAUDE.md`: no repo vocabulary without expansion, every number with its
so-what, behaviour not code. At most one screen. Contents, in order:

1. What the evidence says now, in three or four sentences — including
   anything that should worry him, stated plainly.
2. The two or three anomalies worth his attention (from §1 of the
   Method), one or two sentences each.
3. **One question** — the opening question from the pool prepared in §6
   of the Method, the one whose answer would most change what to ask
   next. Not the list: the owner has asked for one question at a time,
   with each following question shaped by his answer. Answerable in a
   sentence.
4. One line: "Say 'ideas' at any point to go straight to those."

Ending the turn on the briefing is what notifies the owner's phone. The
session then waits; it does not proceed on its own.

### Phase 3 — the conversation (owner-attended)

The owner's answers are the evidence this session exists to collect.
Conduct:

- **One question per turn.** Never a numbered list of questions. Each
  turn asks the single next question and nothing else that needs an
  answer.
- **The next question is chosen after the answer, not before.** The
  prepared pool (§6 of the Method) is a starting stock, not a script.
  After each answer, decide afresh: a follow-up that digs into what he
  just said, a question from the pool that his answer made more
  pointed, or one the answer made pointless and is dropped. Say in a
  clause why this question follows from his last answer when that is
  not obvious.
- **Follow the answer, not the script.** A surprising answer earns a
  follow-up question before any idea is offered.
- **Offer ideas one at a time**, when his answers point at one or when
  he asks — each with its observation, mechanism, cheapest kill test
  (data, duration, **owner hours**), killing result, and cost of being
  wrong — in plain language, the house fields kept for the record. Ask
  what he thinks before the next one; his reaction to an idea is itself
  an answer that shapes the next question.
- **Take his corrections as rulings.** When the owner says a number, a
  label, or a premise is wrong, that is recorded as an owner finding,
  labelled as such, and the idea resting on it is dropped or amended in
  the record.
- **Ask for a "run it" or "park it"** on each idea he engages with. "Run
  it" here means *he would be willing to have it commissioned* — the
  commissioning itself is still a ruling he records through the normal
  channel (a batch review or a W0 ratification). Say so once, in the
  first exchange, and do not repeat it.
- **Watch the clock.** Around thirty minutes, or when he says he is done,
  offer to wrap; do not open new threads after that.
- **Stop on the owner's word.** If he ends the conversation early, the
  record is written from whatever was covered.

### Phase 4 — the record (unattended, after the owner leaves)

Written from the conversation. Ledger rules apply to the record: the raw
transcript is **not** committed (A5-30 forbids verbatim step names, and a
transcript cannot be redacted reliably); the record is a distillation in
which **the owner's words are quoted where they carry a ruling or a
finding, marked as his**, and the session's ideas are marked as proposals.

1. **The memo** at `docs/research/lateral/YYYY-MM-DD.md`: the briefing as
   sent (plain language), then the conversation distilled — for each
   question asked, in order, his answer, what it changed, and why the
   next question followed; the pool questions not asked, with the reason
   (dropped as moot, or out of time); for each idea, the five
   house fields and his verdict (*run it / park it / killed by the owner
   / not discussed*); then the full §1–§8 body of the Method below,
   updated by what he said.
2. **The ledger entry** appended to `docs/research/RESEARCH-LOG.md`,
   headed `## YYYY-MM-DD · rung M / LR (the lateral review) ·
   agent/lateral-YYYY-MM-DD · cloud (owner-attended)`, standard
   template: `Attempted:` the memo path and the questions asked;
   `Pre-registered expectations: n/a (review session)`; `Result:` ideas
   proposed / cut at the graveyard check / owner verdicts by kind /
   owner findings recorded; `Lesson:` the one durable thing the owner
   said; `Status: PROPOSED (owner-attended; nothing commissioned)`.
   Never any other status.
3. **Commit, push, draft PR** titled `Lateral review YYYY-MM-DD`, whose
   description is the memo's plain-language section verbatim. A session
   without a GitHub tool pushes the branch and hands the owner the
   compare link instead of spending turns on the PR.
4. **Closing chat message:** where the record is, plus the one-line list
   of ideas he said "run it" to, so he can carry them into his next
   batch review. Nothing else.

Turn bound: **80 turns in total.** If it is reached mid-conversation,
say so and go to Phase 4 with what exists.

## Method (what Phase 1 prepares and Phase 4 records)

### 1. Evidence audit — the anomalies

Five to ten observations since the last review that are *surprising,
unexplained, or in tension with the plan*. Anomalies are where new
directions come from. Label each **✓** (checked against the artifact) or
**(relayed)** (repeated from the ledger) — the state-of-play convention.
The ledger has misreported its own bookkeeping more than once; no idea is
built on an unverified number.

### 2. The assumption ledger

Restate the problem from scratch as if the plan did not exist: what a
rehearsal pianist knows after a few seconds of marking, and by what
channel. Then name the assumptions the plan rests on (e.g. "the pulse is
in the audio"; "vowel onsets are the beat"; "meter is recoverable from
accent"; "the benchmark measures what a user would notice"; "one voice
generalizes"; "whole-clip scoring is the right unit") and rate each
**supported / unsupported / contradicted**, with the evidence line. A
contradicted assumption is itself a finding and goes in the briefing.

### 3. Lateral generation — at least four lenses

At least four distinct vantage points, each idea tagged with its lens.
A find from the outward scan (§3a) lands under whichever lens fits — a
beat-tracker library under the evaluation lens, an accompaniment product
under the product lens, a transferable method under "another field" —
not only under the last of these:

- **The musician in the room** — what the accompanist does that the
  pipeline does not attempt (W13(a) is the seed).
- **Another field's solved problem** — speech science, bioacoustics,
  conductor-gesture studies, sensor fusion, active learning, sports
  officiating: anywhere a noisy human signal is read in real time —
  fed by the outward scan (§3a).
- **The evaluation** — is the harness measuring the thing? What is
  unmeasured that could be (W14-c: a defect invisible to every suite)?
  What would a hostile reviewer ask?
- **The data** — what capture, labelling, or split changes what is
  knowable, and what it costs the owner in hours.
- **The product** — where the system must ask or abstain rather than
  guess (the pliés demo; 60 vs half-time 120), and whether that changes
  what perception must deliver.
- **The inverse** — if the central bet is wrong, what does the project do
  the day after the joint posterior fails?

### 3a. The outward scan — research, tools, products, adjacent fields

The five literature reviews and `voice-as-drum-review.md` are the
baseline, dated 2026-08; the scan is **what has appeared since, and what
bears on this review's anomalies** — never a re-survey of what the
reviews already cover (check them first; a find already in a review is
not a find). Bounded: about a dozen searches, and no more than a fifth
of the turn budget. Four directions, each aimed by a named anomaly or
assumption:

- **Research** — papers and preprints on the anomaly's topic (beat and
  meter induction, speech rhythm and P-centres, online/streaming
  inference and commitment, calibration, small-corpus evaluation, and
  whatever the anomaly names). Note the venue and date.
- **Tools, libraries, models** — anything that could be run against the
  frozen artifacts (traces, grids, pulse sidecars) or the committed DEV
  audio: beat trackers, speech-rhythm toolkits, audio-language models,
  onset detectors. For each: licence, whether it installs in the repo's
  environment (the cloud environment's network policy is set by the
  owner to allow outbound access, so pages, PDFs and PyPI are all
  reachable; if a fetch is nonetheless blocked, say so rather than
  guessing at the content), and what it would take to score it
  on the verified grids the way rung 6 scored the baselines.
- **Products and systems** — anyone solving an analogous problem: score
  following and automatic accompaniment, conductor and gesture
  following, dance-class and rehearsal tools, live-music apps. What
  they do that this pipeline does not attempt, and what they visibly
  cannot do.
- **Adjacent fields** — a solved problem elsewhere (bioacoustics,
  sports officiating, sensor fusion, active learning) whose method
  transfers. Say what transfers and what does not.

Every find is recorded with a link, one line of what it claims, and a
label — **abstract-only / read / ran-it** — and enters the ideas list
only through the five fields below, tied to the observation it
answers. Finds that were looked at and dismissed are listed with the
reason, so the next review does not repeat the search. If the network
is blocked, say so in the briefing and proceed on the reviews alone;
never present a remembered claim as a search result.

**Every idea carries, without exception:** (1) the **observation** that
motivates it, cited by ledger date or path — no observation, no idea;
(2) the **mechanism**; (3) the **cheapest kill test** — what runs, on
what verified data, how long, **how many owner hours**; (4) the
**killing result**; (5) **the cost of being wrong**.

### 4. Graveyard check

Search the ledger for each idea and its relatives before it is offered
(Standing Lesson 10). A re-proposal of anything marked DEAD-END,
REJECTED, negative, or falsified must say so and state the **new**
evidence that reopens it, or it is cut. The record lists what was checked
and what was cut.

### 5. Stop-doing candidates

Anything consuming loop or owner effort that the evidence says is not
paying, with the evidence. An empty section says "none found".

### 6. Questions for the owner — a pool, asked one at a time

A pool of six to ten questions only a dance musician can answer, where
the answer is missing evidence: introspection of the W13 kind ("when do
you know the meter, and what told you?"), listening judgements, product
rulings. Each answerable in a sentence. For each, note privately what
answer would change the plan and which questions it would open or
close — that is what makes the next choice fast in conversation. Rank
the pool by how much the answer would change what to ask next; the top
one opens the briefing. Expect to ask perhaps half the pool and invent
the rest live. These often outvalue the ideas.

### 7. If I could commission one thing

One paragraph: the direction the session would put first, with the
honest counter-argument beside it — revised in the record by what the
owner said.

### 8. The outward scan, as a record

What was searched (the queries, roughly), what was found and its label,
what was dismissed and why, and which watch-list items in
`docs/vision/03-landscape.md` §3.6 moved. This section exists so the
next review starts where this one stopped.

## Rules that bind this session

All charter rules of engagement apply; these are the ones a wide-ranging
session is most likely to test:

- **HELD-OUT / SEALED containment.** Never enumerate
  `video/youtube/Ballet Barre 1`; never reason about sealed or held-out
  content; a reference to it is a reason to stop and flag. Ideas about
  future capture and splits are welcome; ideas needing the held-out four
  are not.
- **Redaction by default (A5-30)** in everything committed, the owner's
  quoted words included: paraphrase step names or use opaque ids.
- **No commissioning, no ranking edits, no charter edits.** Ideas are
  PROPOSED; "run it" from the owner in conversation is willingness, not
  commissioning — that happens through his batch review or a W0
  ratification, recorded in the ledger.
- **Honesty style.** Relayed numbers are labelled relayed. A negative
  assessment of the plan is stated plainly; nothing is rounded in
  anyone's favour, and the owner is not flattered.
- **No live model calls, no key use.** Web search and fetching for the
  outward scan are expected, not merely permitted, where the
  environment allows them; say in the briefing when they did not work.
  Installing a candidate tool from PyPI to try it on frozen artifacts
  is allowed; committing it, or wiring it into the pipeline, is not.

## Session notes that live here, not in the Routine prompt

- **Web access.** The Routine runs in the *musical perception research*
  cloud environment, whose network policy the owner set to full outbound
  access on 2026-09-01. `WebSearch` works; `WebFetch` should fetch pages
  and PDFs. If a fetch is blocked anyway, say so in the briefing and
  never guess at a paper's content.
- **Pull request.** If the session has a GitHub tool that can open pull
  requests, open the draft PR itself. If not, do not spend turns on it:
  push the branch and hand the owner the compare link GitHub prints on
  push (`https://github.com/ben-juodvalkis/musical-perception/pull/new/agent/lateral-YYYY-MM-DD`).
- **Held-out containment and redaction** are charter rules and apply in
  full; they are restated under *Rules that bind this session* above.

## The Routine prompt (paste-ready)

The Routine is created from the claude.ai Routines page — **not** from
inside a session — so that it carries the repository: a Routine minted
from a session starts its fresh session with no repository attached
(observed 2026-09-01: the setup script found no files, and the protocol
would have been missing too). Settings: repository
`ben-juodvalkis/musical-perception`, branch `main`; environment
*musical perception research*; the top-tier model; no schedule (fired by
hand); push notification on. The prompt:

```
You are running the LATERAL REVIEW for the musical-perception research
loop. Ben (the owner, a dance musician, not a coder) fired this by hand
and will join once you have briefed him. This is a high-effort session:
read deeply, think widely, do not rush the reading to get to the ideas.

Read docs/research/lateral-review-protocol.md in full FIRST and follow
it exactly; it is the source of truth. If it is missing, stop and say so.
Then: branch agent/lateral-YYYY-MM-DD from origin/main and do the
write-and-commit probe; the unattended read; the anomalies and the
assumption ledger; the outward scan; the ideas, the graveyard check and
the ranked question pool; then ONE plain-language briefing (per the
"Talking to Ben about this work" section of CLAUDE.md) ending in exactly
ONE question — never a numbered list — and END YOUR TURN there, which is
what notifies his phone. Then one question per turn, each chosen after
his answer. When he is done, write the memo and the ledger entry, commit,
push, and open the draft PR or hand him the compare link, per the
protocol. It commissions nothing; touches no file under src/, evals/,
scripts/ or tests/; never edits the charter; never enumerates the
Barre 1 video directory; no live model calls; stop after 80 turns.
```

## Owner controls

- **Start one:** fire *Lateral review (on demand)* from the Routines
  list. Nothing runs otherwise.
- **Model, notifications, on/off:** edit or pause the Routine there.
- **Amend this protocol:** an ordinary owner-reviewed doc change; the
  Routine's prompt points here, so this document is the source of truth.
