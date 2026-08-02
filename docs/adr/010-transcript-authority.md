# ADR-010: Transcript Authority and Deterministic Classification

**Date:** 2026-08-01
**Status:** Accepted

## Context

The first verified end-to-end run (`grande battement.mov`, 39.5s, clear audio,
ground truth: ~102 BPM in 4/4) exposed three compounding weaknesses in the
word-marker path:

1. **ASR degrades ballet French.** Whisper `base.en` heard "grand babma" for
   *grand battement* and "polybra" for *port de bras* — on every occurrence.
   An A/B across model sizes showed size alone fixes nothing: even
   `large-v3-turbo` without domain context still produced "grand babma"
   (0/10 correct). A ballet-vocabulary `initial_prompt` fixed 10/10 on every
   model size tested.

2. **The merge amplified ASR errors.** Gemini transcribed the audio itself and
   classified its *own* words; `analyze.py` then paired classifications to
   Whisper timestamps by sequential text matching. Any disagreement between
   the two transcriptions silently dropped markers — the first run kept only
   18 beats at 0% tempo confidence.

3. **Sampling noise made single runs unrepeatable.** The Gemini call used
   default temperature. Two runs on the identical clip returned 18 beats /
   55.5 BPM, then 3 beats / 98.1 BPM. Combined with onset-section sensitivity
   to shifted word boundaries, the pipeline's headline BPM flipped from a
   correct 101.6 to a wrong 74.2 between runs — with no code change that
   should have affected it.

## Decision

### 1. Ballet vocabulary prompt, on by default (`perception/whisper.py`)

`BALLET_VOCABULARY_PROMPT` — counting phrases plus ~45 step names — is passed
as Whisper's `initial_prompt` by default in `load_model()` (both the WhisperX
and plain-Whisper paths). Pass `initial_prompt=None` to disable.

### 2. Default ASR model: `large-v3-turbo`

Best transcript quality at acceptable speed (~14s for a 40s clip on M1 Max
CPU). `analyze()`, `load_model()`, and the CLI default to it; the CLI gains
`--model NAME` to downshift (`base.en` etc.) when speed matters more.

### 3. Index-keyed classification (transcript authority)

`analyze()` now passes Whisper's word list to `analyze_media()`, which embeds
it in the prompt as an indexed transcript (`[0] all [1] right …`) and requires
Gemini to classify **those exact tokens**, returning each word's transcript
index (schema gains a required `index` field when a transcript is provided).

The merge becomes a lookup: `_pair_markers_by_index()` anchors each
classification to `whisper_words[index]`'s timestamp. Out-of-range indices are
dropped. Gemini still receives the audio/video and judges each word's rhythmic
role by ear — but the *tokenization* has exactly one owner: Whisper, which is
also the timestamp owner. Transcription disagreement can no longer cost
markers.

The legacy text-matching merge (`_merge_gemini_with_timestamps`) remains as
the fallback for `analyze_media()` calls without a transcript (audio-only
scripts, older traces).

### 4. `temperature=0` for the Gemini call

Word classification, meter, and quality extraction are measurement, not
generation. Deterministic-as-possible output makes runs comparable and frozen
traces meaningful (ADR-009).

## Consequences

- Markers survive transcription drift; the marker path's recall is now bounded
  by Whisper's word coverage, not by two models agreeing on spelling.
- One tokenization authority also means one failure mode: a word Whisper never
  emits can never become a marker, even if Gemini hears it. The better default
  model narrows this; the onset path (ADR-006) and the planned
  accent-periodicity module (Vision 05 §5.3) cover the rest.
- What this does **not** fix, observed on the same clip: numbers spoken as
  *quantities* in explanation speech ("we take **two** grand battement front")
  still classify as beats at irregular spacing, which is why marker tempo
  stays low-confidence on explanation-heavy clips. That is the interleaved-
  explanation problem (Vision 08 §8.4 matrix), not a merge defect.
- Run-to-run variance drops but does not vanish (Gemini `temperature=0` is
  not a strict determinism guarantee; Whisper beam search ties can differ
  across hardware). Frozen traces remain the real reproducibility mechanism.
