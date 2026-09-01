"""Pulse sidecars (W11): the checksum contract and the second stage-1
pulse source. Hardcoded data — no media, no models, no extractor run."""

import json

import pytest

from musical_perception.evals.pulse_sidecar import (
    SIDECAR_NAME,
    PulseSidecar,
    SidecarError,
    load_pulse_sidecar,
    verify_media,
)
from musical_perception.evals.stage1 import (
    PULSE_SOURCE,
    PULSE_SOURCE_PEAKRATE,
    predicted_pulse_from_trace,
    run_stage1,
)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

needs_yaml = pytest.mark.skipif(not HAS_YAML, reason="needs pyyaml")

MEDIA_BYTES = b"not really audio, but it hashes"
MEDIA_SHA = "9c2ea9e3b0b6b6e1c1e2e0aa1e7c15f9d05a0e5f2e0d3d0f8e5a1c1e1d0e2f3a"


def _write_trace(trace_dir, *, media_sha=MEDIA_SHA, n_words=4):
    trace_dir.mkdir(parents=True)
    words = [
        {"word": f"w{i}", "start": 1.0 + 0.5 * i, "end": 1.2 + 0.5 * i}
        for i in range(n_words)
    ]
    (trace_dir / "whisper.json").write_text(json.dumps({"words": words}))
    (trace_dir / "meta.json").write_text(json.dumps({
        "media": "audio/fake.mp3", "media_sha256": media_sha,
        "analyze_flags": {},
    }))
    (trace_dir / "gemini.json").write_text(json.dumps(
        {"model": "m", "raw_response": "{}", "inputs": {}}
    ))


def _write_sidecar(trace_dir, events, *, media_sha=MEDIA_SHA):
    (trace_dir / SIDECAR_NAME).write_text(json.dumps({
        "sidecar_format": 1,
        "extractor": "acoustic-pulse/1",
        "media": "audio/fake.mp3",
        "media_sha256": media_sha,
        "params": {"events_per_nucleus": "all"},
        "events": events,
    }))


# --- the checksum contract ----------------------------------------------

def test_verify_media_accepts_a_matching_file(tmp_path):
    import hashlib

    trace = tmp_path / "clip"
    media = tmp_path / "fake.mp3"
    media.write_bytes(MEDIA_BYTES)
    real_sha = hashlib.sha256(MEDIA_BYTES).hexdigest()
    _write_trace(trace, media_sha=real_sha)
    assert verify_media(trace, media) == real_sha


def test_verify_media_refuses_a_different_file(tmp_path):
    trace = tmp_path / "clip"
    media = tmp_path / "fake.mp3"
    media.write_bytes(MEDIA_BYTES)
    _write_trace(trace, media_sha=MEDIA_SHA)  # deliberately not the real hash
    with pytest.raises(SidecarError, match="refusing to record"):
        verify_media(trace, media)


def test_verify_media_refuses_when_media_is_absent(tmp_path):
    trace = tmp_path / "clip"
    _write_trace(trace)
    with pytest.raises(SidecarError, match="not on this runner"):
        verify_media(trace, tmp_path / "missing.mp3")


# --- load-time verification ---------------------------------------------

def test_load_returns_none_without_a_sidecar(tmp_path):
    trace = tmp_path / "clip"
    _write_trace(trace)
    assert load_pulse_sidecar(trace) is None


def test_load_reads_events_and_provenance(tmp_path):
    trace = tmp_path / "clip"
    _write_trace(trace)
    _write_sidecar(trace, [1.0, 2.0, 3.0])
    sidecar = load_pulse_sidecar(trace)
    assert isinstance(sidecar, PulseSidecar)
    assert sidecar.events == [1.0, 2.0, 3.0]
    assert sidecar.n_events == 3
    assert sidecar.media_sha256 == MEDIA_SHA


def test_load_raises_when_the_sidecar_drifted_from_its_trace(tmp_path):
    """A sidecar recorded against other audio is not this clip's evidence.

    Guessing which side is right would silently feed a scorer events from
    a file nobody pinned, so this is an error rather than a warning.
    """
    trace = tmp_path / "clip"
    _write_trace(trace, media_sha="a" * 64)
    _write_sidecar(trace, [1.0], media_sha="b" * 64)
    with pytest.raises(SidecarError, match="re-record"):
        load_pulse_sidecar(trace)


# --- stage-1 source selection -------------------------------------------

def test_default_source_is_still_word_starts(tmp_path):
    trace = tmp_path / "clip"
    _write_trace(trace, n_words=3)
    _write_sidecar(trace, [9.0, 9.5])
    assert predicted_pulse_from_trace(trace) == [1.0, 1.5, 2.0]
    assert predicted_pulse_from_trace(trace, PULSE_SOURCE) == [1.0, 1.5, 2.0]


def test_peakrate_source_reads_the_sidecar(tmp_path):
    trace = tmp_path / "clip"
    _write_trace(trace, n_words=3)
    _write_sidecar(trace, [9.0, 9.5])
    assert predicted_pulse_from_trace(trace, PULSE_SOURCE_PEAKRATE) == [9.0, 9.5]


def test_missing_sidecar_is_an_error_not_an_empty_stream(tmp_path):
    trace = tmp_path / "clip"
    _write_trace(trace)
    with pytest.raises(FileNotFoundError):
        predicted_pulse_from_trace(trace, PULSE_SOURCE_PEAKRATE)


def test_unknown_source_is_rejected(tmp_path):
    trace = tmp_path / "clip"
    _write_trace(trace)
    with pytest.raises(ValueError, match="unknown pulse source"):
        predicted_pulse_from_trace(trace, "madmom")
    with pytest.raises(ValueError, match="unknown pulse source"):
        run_stage1(tmp_path, pulse_source="madmom")


@needs_yaml
def test_run_stage1_scores_each_source_separately(tmp_path):
    from musical_perception.annotation.grids import BeatGrid, save_grid

    (tmp_path / "cases").mkdir()
    (tmp_path / "cases" / "clip-a.yaml").write_text(yaml.safe_dump({
        "id": "clip-a",
        "input": {"trace": "traces/clip-a/"},
        "expect": {"marking_bpm": 104},
    }))
    _write_trace(tmp_path / "traces" / "clip-a", n_words=4)
    beats = [1.0, 1.5, 2.0, 2.5]
    save_grid(
        BeatGrid(clip="clip-a", provisional=True, beats=beats, onsets=beats),
        tmp_path / "grids",
    )
    # A sidecar that matches only the last two beats: the two sources must
    # produce visibly different rows, not the same numbers twice.
    _write_sidecar(tmp_path / "traces" / "clip-a", [2.0, 2.5])

    words = run_stage1(tmp_path)
    peaks = run_stage1(tmp_path, pulse_source=PULSE_SOURCE_PEAKRATE)
    assert words["pulse_source"] == PULSE_SOURCE
    assert peaks["pulse_source"] == PULSE_SOURCE_PEAKRATE
    assert words["clips"][0]["matched"] == 4
    assert peaks["clips"][0]["matched"] == 2
    assert peaks["clips"][0]["precision"] == 1.0
    assert peaks["clips"][0]["recall"] == 0.5


def test_run_suites_exposes_the_peakrate_suite_by_name():
    from musical_perception.evals.runner import run_suites

    with pytest.raises(ValueError, match="stage1-peakrate"):
        run_suites(["stage1-nope"], "evals")


# --- W11-b: the opaque media reference and checksum-directed lookup ------

def test_the_sidecar_records_the_traces_reference_not_the_path_given(tmp_path):
    """A sidecar may never name a file its trace did not already name.

    The Barre-1 traces pin `offrepo:<case-id>`; recording their real
    filenames would name the held-out exercises by complement.
    """
    import hashlib

    from musical_perception.evals.pulse_sidecar import record_pulse_sidecar

    trace = tmp_path / "clip"
    media = tmp_path / "secret-exercise-7.wav"
    media.write_bytes(MEDIA_BYTES)
    real_sha = hashlib.sha256(MEDIA_BYTES).hexdigest()
    _write_trace(trace, media_sha=real_sha)
    # the trace's own reference is opaque
    meta_path = trace / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["media"] = "offrepo:clip"
    meta_path.write_text(json.dumps(meta))

    def fake_events(y, sr, params):
        return [1.0, 2.0]

    import musical_perception.precision.pulse as pulse_mod
    import musical_perception.annotation.__main__ as ann_mod
    real_events = pulse_mod.acoustic_pulse_events
    real_load = ann_mod._load_audio
    pulse_mod.acoustic_pulse_events = fake_events
    ann_mod._load_audio = lambda p, sr: [0.0]
    try:
        record_pulse_sidecar(trace, media)
    finally:
        pulse_mod.acoustic_pulse_events = real_events
        ann_mod._load_audio = real_load

    payload = json.loads((trace / SIDECAR_NAME).read_text())
    assert payload["media"] == "offrepo:clip"
    assert "secret-exercise-7" not in json.dumps(payload)
    assert payload["media_sha256"] == real_sha


def test_resolve_by_checksum_returns_only_wanted_files(tmp_path):
    """Non-matching files are discarded on the spot — the containment
    property, not an optimisation: the return value is the only thing a
    caller can print, so it must not contain a held-out filename."""
    import hashlib

    from musical_perception.evals.pulse_sidecar import resolve_media_by_checksum

    root = tmp_path / "media"
    (root / "nested").mkdir(parents=True)
    wanted_bytes = b"the dev take"
    (root / "nested" / "dev-take.mp4").write_bytes(wanted_bytes)
    (root / "held-out-exercise.mp4").write_bytes(b"the held-out take")

    digest = hashlib.sha256(wanted_bytes).hexdigest()
    found = resolve_media_by_checksum(root, {digest: "clip-a"})

    assert set(found) == {"clip-a"}
    assert found["clip-a"].name == "dev-take.mp4"
    assert all("held-out" not in str(p) for p in found.values())


def test_resolve_by_checksum_reports_nothing_when_nothing_matches(tmp_path):
    from musical_perception.evals.pulse_sidecar import resolve_media_by_checksum

    root = tmp_path / "media"
    root.mkdir()
    (root / "held-out-exercise.mp4").write_bytes(b"not wanted")
    assert resolve_media_by_checksum(root, {"a" * 64: "clip-a"}) == {}


def test_trace_media_ref_reads_the_committed_reference(tmp_path):
    from musical_perception.evals.pulse_sidecar import trace_media_ref

    trace = tmp_path / "clip"
    _write_trace(trace)
    assert trace_media_ref(trace) == "audio/fake.mp3"
