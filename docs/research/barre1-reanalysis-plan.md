# Barre 1 — retired 2026-09-01, and how to bring it back

**Status:** the 22 case files and 22 frozen traces were removed from the
repository on 2026-09-01 by owner ruling. **The media is untouched** on
the owner's main machine under `video/youtube/Ballet Barre 1`.

## Why it was retired

The owner's judgement, on being shown what the Barre 6 session produced:
*"Barre 1 clips and analysis and markings are a lot less rigorous than
Barre 6."* Concretely:

- **19 of its 22 cases carried `expect: {}`** — no tempo, no meter, no
  counts. They were scored and reported and could never gate anything.
- **No beat grids**, so no `stage1` pulse coverage either.
- **Same teacher and same pianist as Barre 6** (verified: "thanks, Rex"
  appears in the Barre 1 transcripts), so it adds no teacher-level or
  accompanist-level diversity. Barre 6 covers the same ground with real
  labels.
- Its boundaries were never owner-verified by ear. The Barre 6 session
  established that **both** automatic boundary methods fail — piano
  energy misses quiet playing, and spoken cues mis-cut silent balances —
  so boundaries authored without an owner listening pass are not
  trustworthy.

It also carried standing overhead: the enumeration ban, and W11-b, a
whole commissioned workstream existing only to give it pulse sidecars.

## What was NOT removed

- The media, on the owner's machine.
- The **4 HELD-OUT exercises**, which were moved off-repo before any
  ingestion and were never in the repository to begin with.
- The findings the batch produced. They stand in the ledger: the
  count-phrase-not-bar observation (F2 of the W4 entry), the
  no-full-eight finding (F1), and the containment lesson that produced
  the enumeration ban.

## How to bring it back, if it is ever worth it

Do **not** restore the deleted traces. They were frozen against
boundaries authored by the methods the Barre 6 session falsified;
re-analysis means re-cutting, which makes them worthless.

Follow the Barre 6 method instead, which is the one that worked:

1. **Cut from the teacher's spoken cues**, not audio energy — he
   announces each exercise, calls each side change, and closes each take.
   Then have the owner check the boundaries **by ear**; every one of the
   Barre 6 takes needed correction, and three were cut mid-exercise.
2. **Watch for the silent balance.** It is the failure both detectors
   share: the teacher goes quiet for 10-20s while a balance is held, and
   a cue detector reads that as the end of the exercise.
3. **Label while analysing, not afterwards.** Tempo, meter and counts per
   clip, cross-checked against clip duration as you go — that check
   caught a metric-level ambiguity, a two-tempo combination and three
   mis-cut boundaries in the Barre 6 session, none of which any detector
   saw.
4. **The demo is the case, the take is one realization** (owner ruling).
5. **One tempo per case** (owner ruling): split tails at a different
   tempo into their own clips, and drop any clip whose tempo drifts
   inside it.

## The open question the retirement raises

Barre 1 was split 8 DEV / 4 HELD-OUT at the exercise level, and the
**enumeration ban** exists because listing the DEV directory names the
held-out four by complement. With the DEV cases retired, that split is
dormant rather than resolved. **Owner's to rule:** does the ban still
stand (it costs nothing and the media is still there), or does a future
re-analysis treat the class as a whole? Until he rules, **the ban stands**
— an agent must never enumerate `video/youtube/Ballet Barre 1`.
