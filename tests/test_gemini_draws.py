"""W6-a: the consumption path for ensembled semantics.

Two things are under test: that a distribution reduces to the hard
label when it is one-hot (the refactor is a refactor), and that it does
NOT reduce when the draws disagree (the socket is live, not
decorative). The sidecar's job is to refuse the one mistake it can
detect — draws pointing at a transcript that is not this one.
"""

import json

import pytest

from musical_perception.evals.gemini_draws import (
    GeminiDraw,
    SIDECAR_NAME,
    beliefs_from_draws,
    load_gemini_draws,
    transcript_fingerprint,
)
from musical_perception.evals.pulse_sidecar import SidecarError
from musical_perception.precision.posterior import (
    _map_times,
    _weighted_stream,
    beliefs_from_markers,
    estimate_rhythm,
)
from musical_perception.types import (
    MarkerBelief,
    MarkerType,
    TimedMarker,
    TimestampedWord,
)

PERIOD = 0.6                     # 100 BPM
COUNT = ["one", "two", "three", "four", "five", "six", "seven", "eight"]


def _clip(n_beats=16, period=PERIOD):
    """A clean counted clip: eight-count cycles, one word per beat."""
    words, markers = [], []
    for i in range(n_beats):
        t = 1.0 + i * period
        w = COUNT[i % 8]
        words.append(TimestampedWord(word=w, start=round(t, 3), end=round(t + 0.2, 3)))
        markers.append(TimedMarker(
            marker_type=MarkerType.BEAT, beat_number=i % 8 + 1,
            timestamp=round(t, 3), raw_word=w,
        ))
    return words, markers


def _draw(labels, *, draw_id="d0", model="test-model", beat_numbers=None):
    return GeminiDraw(
        draw_id=draw_id, model=model, params={"temperature": 1.0},
        labels=labels, beat_numbers=beat_numbers or {},
    )


class TestOneHotReduction:
    """One draw must be indistinguishable from the hard label."""

    def test_beliefs_reproduce_the_hard_label_streams(self):
        words, markers = _clip()
        beliefs = beliefs_from_markers(words, markers)
        beat_t, beat_w = _weighted_stream(beliefs, ("beat",))
        assert list(beat_t) == [m.timestamp for m in markers]
        assert set(beat_w) == {1.0}
        assert list(_map_times(beliefs, ("beat",))) == list(beat_t)

    def test_explicit_one_hot_beliefs_give_the_same_answer(self):
        words, markers = _clip()
        baseline = estimate_rhythm(words, markers)
        passed_in = estimate_rhythm(
            words, markers,
            marker_beliefs=beliefs_from_markers(words, markers),
        )
        assert baseline is not None
        assert passed_in == baseline

    def test_single_draw_matches_the_marker_path_stream_for_stream(self):
        words, markers = _clip()
        labels = {i: "beat" for i in range(len(words))}
        beat_numbers = {i: i % 8 + 1 for i in range(len(words))}
        from_draw = beliefs_from_draws(
            [_draw(labels, beat_numbers=beat_numbers)], words
        )
        from_markers = beliefs_from_markers(words, markers)
        for classes in (("beat",), ("and", "ah"), ("none",)):
            a_t, a_w = _weighted_stream(from_draw, classes)
            b_t, b_w = _weighted_stream(from_markers, classes)
            assert list(a_t) == list(b_t)
            assert list(a_w) == list(b_w)
        nums = lambda bs: [b.beat_number for b in bs if b.map_class == "beat"]
        assert nums(from_draw) == nums(from_markers)

    def test_e_class_tokens_belong_to_no_stream(self):
        """MarkerType.E is excluded from beat, sub AND word alike —
        the reason `e` is a belief class rather than folded into
        `none` (2026-08-30 pre-registration)."""
        words = [TimestampedWord("e", 1.0, 1.1)]
        markers = [TimedMarker(MarkerType.E, None, 1.0, "e")]
        beliefs = beliefs_from_markers(words, markers)
        assert len(beliefs) == 1
        for classes in (("beat",), ("and", "ah"), ("none",)):
            assert len(_weighted_stream(beliefs, classes)[0]) == 0


class TestDistributionsAreLive:
    """A split distribution must be able to change the answer — and the
    measured way it changes it is the finding W6-b has to know about."""

    @staticmethod
    def _split_clip(p_beat, n=32, period=PERIOD):
        """Every half-period is spoken. The on-beats are certain; the
        offbeats carry `p_beat` of belief that they are beats too and
        the rest that they are subdivisions. Nothing enters the word
        stream, so the level decision is the marker channel's alone."""
        beliefs, words = [], []
        for i in range(n):
            t = round(1.0 + i * period / 2, 3)
            words.append(TimestampedWord(
                COUNT[i // 2 % 8] if i % 2 == 0 else "and", t, t + 0.15))
            if i % 2 == 0:
                beliefs.append(MarkerBelief(
                    t, {"beat": 1.0}, i // 2 % 8 + 1, COUNT[i // 2 % 8]))
            else:
                beliefs.append(MarkerBelief(
                    t, {"beat": p_beat, "and": 1.0 - p_beat}, None, "and"))
        return words, beliefs

    def _bpm(self, p_beat, n=32):
        words, beliefs = self._split_clip(p_beat, n)
        result = estimate_rhythm(words, [], marker_beliefs=beliefs)
        assert result is not None
        return result.bpm

    def test_the_mixture_is_neither_of_its_votes(self):
        """Unanimous-slow reads the beat; unanimous-fast reads the
        half-beat; a 1-in-5 minority is enough to buy the fast reading,
        which is the point of the socket and also its hazard."""
        assert self._bpm(0.0) == pytest.approx(100.2, abs=0.5)
        assert self._bpm(1.0) > 150
        assert self._bpm(0.2) > 150        # one dissenting draw in five

    def test_fractional_mass_aggregates_across_tokens(self):
        """The flip point falls as the number of tokens carrying the
        minority mass rises: belief is spent per token and summed, so a
        minority spread wide is not a minority in likelihood terms.
        Measured 2026-08-30: 0.132 (24 offbeats), 0.159 (16), 0.185
        (12), 0.237 (8)."""
        flips = [self._flip_point(n) for n in (32, 16)]
        assert flips[0] < flips[1]
        assert 0.10 < flips[0] < 0.20 and 0.20 < flips[1] < 0.30

    def _flip_point(self, n):
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = (lo + hi) / 2
            if self._bpm(mid, n) > 150:
                hi = mid
            else:
                lo = mid
        return hi

    def test_fractional_mass_is_expected_support_not_a_vote(self):
        words = [TimestampedWord("one", 1.0, 1.2)]
        half = beliefs_from_draws(
            [_draw({0: "beat"}), _draw({0: "none"})], words
        )
        assert half[0].p("beat") == pytest.approx(0.5)
        assert _weighted_stream(half, ("beat",))[1][0] == pytest.approx(0.5)

    def test_draw_disagreement_becomes_fractional_mass(self):
        """Five draws, three seeing the beat where two see nothing."""
        words = [TimestampedWord("one", 1.0, 1.2)]
        draws = [_draw({0: "beat"}, draw_id=f"y{k}") for k in range(3)]
        draws += [_draw({0: "none"}, draw_id=f"n{k}") for k in range(2)]
        belief = beliefs_from_draws(draws, words)[0]
        assert belief.p("beat") == pytest.approx(0.6)
        assert belief.map_class == "beat"


class TestSidecar:
    def _trace(self, tmp_path, *, media_sha="abc123", **payload):
        d = tmp_path / "clip-1"
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"media_sha256": media_sha}))
        (d / SIDECAR_NAME).write_text(json.dumps(payload))
        return d

    def test_absent_sidecar_is_none_not_an_error(self, tmp_path):
        d = tmp_path / "clip-0"
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"media_sha256": "abc123"}))
        assert load_gemini_draws(d) is None

    def test_round_trip(self, tmp_path):
        words, _ = _clip(n_beats=8)
        d = self._trace(
            tmp_path,
            sidecar_format=1,
            media_sha256="abc123",
            transcript_sha256=transcript_fingerprint(words),
            draws=[
                {"draw_id": "flash#0", "model": "gemini-2.5-flash",
                 "params": {"temperature": 1.0},
                 "words": [{"index": 0, "marker_type": "beat", "beat_number": 1}]},
                {"draw_id": "pro#0", "model": "gemini-2.5-pro",
                 "params": {"temperature": 1.0},
                 "words": [{"index": 0, "marker_type": "none"}]},
            ],
        )
        side = load_gemini_draws(d, words=words)
        assert side.n_draws == 2
        assert side.models == ["gemini-2.5-flash", "gemini-2.5-pro"]
        assert side.draws[0].params["temperature"] == 1.0
        beliefs = beliefs_from_draws(side.draws, words)
        assert beliefs[0].p("beat") == pytest.approx(0.5)

    def test_media_hash_mismatch_raises(self, tmp_path):
        d = self._trace(tmp_path, media_sha256="deadbeef", draws=[])
        with pytest.raises(SidecarError, match="recorded against media"):
            load_gemini_draws(d)

    def test_transcript_mismatch_raises(self, tmp_path):
        words, _ = _clip(n_beats=8)
        d = self._trace(
            tmp_path, media_sha256="abc123",
            transcript_sha256="not-this-transcript", draws=[],
        )
        with pytest.raises(SidecarError, match="different token sequence"):
            load_gemini_draws(d, words=words)

    def test_index_outside_the_transcript_raises(self, tmp_path):
        words, _ = _clip(n_beats=4)
        d = self._trace(
            tmp_path, media_sha256="abc123",
            transcript_sha256=transcript_fingerprint(words),
            draws=[{"draw_id": "x", "model": "m",
                    "words": [{"index": 99, "marker_type": "beat"}]}],
        )
        with pytest.raises(SidecarError, match="outside a 4-token transcript"):
            load_gemini_draws(d, words=words)

    def test_unindexed_word_raises(self, tmp_path):
        d = self._trace(
            tmp_path, media_sha256="abc123",
            draws=[{"draw_id": "x", "model": "m",
                    "words": [{"marker_type": "beat"}]}],
        )
        with pytest.raises(SidecarError, match="carries no index"):
            load_gemini_draws(d)

    def test_unknown_class_raises(self, tmp_path):
        d = self._trace(
            tmp_path, media_sha256="abc123",
            draws=[{"draw_id": "x", "model": "m",
                    "words": [{"index": 0, "marker_type": "downbeat"}]}],
        )
        with pytest.raises(SidecarError, match="unknown class"):
            load_gemini_draws(d)
