"""W15 — tests for the stated-structure typing rules.

The channel is REPORTED-ONLY research code over frozen transcripts, so these
tests pin its *semantics* — what a spoken number is taken to be a quantity of
— not any pipeline outcome.  The disambiguation is the deliverable, so the
rejections are tested as hard as the acceptances.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "w15", ROOT / "scripts" / "w15-stated-structure.py")
w15 = importlib.util.module_from_spec(SPEC)
sys.modules["w15"] = w15          # dataclasses resolve annotations via sys.modules
SPEC.loader.exec_module(w15)


def _toks(text):
    """Whitespace-split text into the token stream the parser consumes."""
    words = [{"word": w, "start": float(i)} for i, w in enumerate(text.split())]
    return w15.tokenize(words)


def _type_of(text, index):
    toks = _toks(text)
    return w15.classify(toks, index, "test")


def test_counting_runs_generate_no_candidates():
    toks = _toks("one, two, three, four, five, six, seven, eight.")
    assert len(w15.counting_run_members(toks)) == 8


def test_a_two_long_ascent_is_not_a_counting_run():
    toks = _toks("we go seven and eight, to the front")
    assert w15.counting_run_members(toks) == set()


def test_n_counts_is_a_step_duration_never_beats_per_bar():
    c = _type_of("then four counts petit battement", 1)
    assert (c.rule, c.quantity, c.value) == ("R3", "step_duration", 4)
    assert c.naive_bpb == 4          # what the naive reading would have said


def test_in_n_counts_is_still_a_step_duration():
    c = _type_of("all the way around in four counts", 5)
    assert (c.frame, c.quantity) == ("in_N_counts", "step_duration")


def test_n_eights_is_a_phrase_length_not_a_bar_count():
    c = _type_of("two eights, please", 0)
    assert (c.rule, c.quantity, c.value) == ("R4", "phrases", 2)


def test_hyphenated_numeral_unit_pair_is_split():
    c = _type_of("coupé, two-eighths, two balance", 1)
    assert (c.quantity, c.value) == ("phrases", 2)


def test_bar_homonym_is_rejected_when_a_spatial_frame_owns_it():
    # "... to the bar" — furniture.  A numeral must quantify it directly.
    c = _type_of("port de bras to the four bars", 5)
    assert (c.rule, c.quantity) == ("R2-reject", "unknown")


def test_bar_is_a_musical_bar_when_a_numeral_quantifies_it():
    c = _type_of("we hold for four bars", 3)
    assert (c.rule, c.quantity, c.value) == ("R2", "bars", 4)


def test_bare_time_is_not_a_claim():
    c = _type_of("stretch at the same time", 3)
    assert (c.rule, c.quantity) == ("R1-reject", "unknown")


def test_one_more_time_is_a_repetition():
    c = _type_of("up to the front one more time", 4)
    assert (c.rule, c.quantity) == ("R1", "repetitions")


def test_n_more_without_a_unit_is_a_repetition():
    c = _type_of("then we do four more", 3)
    assert (c.rule, c.quantity, c.value) == ("R6", "repetitions", 4)


def test_in_n_at_a_boundary_is_an_entry_point():
    c = _type_of("we'll go in at three please", 4)
    assert (c.rule, c.quantity, c.value) == ("R5", "entry_point", 3)


def test_on_n_quantifying_a_noun_is_rejected():
    c = _type_of("demi-plié on two legs", 2)
    assert (c.rule, c.quantity) == ("R5-reject", "unknown")


def test_in_n_continuing_an_ascent_is_counting_not_an_entry_point():
    c = _type_of("and in, two, three, up", 2)
    assert (c.rule, c.quantity) == ("R5-reject", "unknown")


def test_explicit_meter_frame_is_the_only_route_to_beats_per_bar():
    c = _type_of("it is three beats to the bar", 2)
    assert (c.rule, c.quantity, c.value) == ("R7", "beats_per_bar", 3)


def test_unmatched_numerals_abstain():
    c = _type_of("we take two tendus to the front", 2)
    assert (c.rule, c.quantity) == ("R8", "unknown")


def test_folding_never_puts_a_step_duration_into_the_four_type_vocabulary():
    # The pre-registered vocabulary cannot hold "N counts of a step"; folding
    # it to `bars` would be exactly the mis-typing W15 exists to prevent.
    assert w15.FOLD["step_duration"] == "unknown"
    assert w15.FOLD["beats_per_bar"] == "beats-per-bar"


def test_every_gold_key_still_matches_a_generated_candidate():
    """The audit set is keyed by (case_id, time); a parser change that moves a
    candidate must move the gold with it, not silently drop it."""
    per_clip = w15.run()
    summary = w15.score(per_clip)
    assert summary["P4_type_precision"]["gold_keys_not_matched"] == []
    assert summary["P4_type_precision"]["audited"] == len(w15.GOLD)
