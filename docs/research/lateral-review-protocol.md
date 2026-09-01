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
musician can answer, and a distilled record of both. It proposes; it
never commissions.

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

Then prepare, privately, the material of §Method below, and write the
**briefing** (Phase 2). If the read would exhaust the turn budget, ledger
and previous memo first; say in the briefing what was skipped.

### Phase 2 — the briefing (the session's first message to the owner)

Plain language per the "Talking to Ben about this work" section of
`CLAUDE.md`: no repo vocabulary without expansion, every number with its
so-what, behaviour not code. At most one screen. Contents, in order:

1. What the evidence says now, in three or four sentences — including
   anything that should worry him, stated plainly.
2. The two or three anomalies worth his attention (from §1 of the
   Method), one or two sentences each.
3. The **questions for him** (§6 of the Method) — three to five, each
   answerable in a sentence, numbered so he can answer by number.
4. One line: "Ideas are ready when you want them; answer any of the
   questions first, or say 'ideas' to go straight there."

Ending the turn on the briefing is what notifies the owner's phone. The
session then waits; it does not proceed on its own.

### Phase 3 — the conversation (owner-attended)

The owner's answers are the evidence this session exists to collect.
Conduct:

- **Follow the answer, not the script.** A surprising answer earns a
  follow-up question before any idea is offered.
- **Offer ideas one or two at a time**, each with its observation,
  mechanism, cheapest kill test (data, duration, **owner hours**),
  killing result, and cost of being wrong — in plain language, the house
  fields kept for the record. Ask what he thinks before the next one.
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
   question, his answer and what it changes; for each idea, the five
   house fields and his verdict (*run it / park it / killed by the owner
   / not discussed*); then the full §1–§7 body of the Method below,
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

At least four distinct vantage points, each idea tagged with its lens:

- **The musician in the room** — what the accompanist does that the
  pipeline does not attempt (W13(a) is the seed).
- **Another field's solved problem** — speech science, bioacoustics,
  conductor-gesture studies, sensor fusion, active learning, sports
  officiating: anywhere a noisy human signal is read in real time.
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

### 6. Questions for the owner

Three to five questions only a dance musician can answer, where the
answer is missing evidence: introspection of the W13 kind ("when do you
know the meter, and what told you?"), listening judgements, product
rulings. Each answerable in a sentence. These often outvalue the ideas.

### 7. If I could commission one thing

One paragraph: the direction the session would put first, with the
honest counter-argument beside it — revised in the record by what the
owner said.

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
- **No live model calls, no key use.** Web search for literature is
  permitted where the environment allows; say when it did not.

## Owner controls

- **Start one:** fire *Lateral review (on demand)* from the Routines
  list. Nothing runs otherwise.
- **Model, notifications, on/off:** edit or pause the Routine there.
- **Amend this protocol:** an ordinary owner-reviewed doc change; the
  Routine's prompt points here, so this document is the source of truth.
