"""The iterative repair loop — offline, with scripted repair responses.

One pass left roughly a third of flagged claims standing, so the loop is the fix. Its risk
is the opposite one: rewriting is not monotonic, and a loop that trusts each rewrite can
hand back something worse than it started with.
"""

import taza_rag.factiva.answer as amod
import taza_rag.factiva.verify as vmod
from taza_rag.factiva.answer import _verify_and_repair

EVIDENCE = {
    "c1": "Deutsche Bank said quarterly profit rose to 1.2 billion euros, helped by lower provisions.",
    "c2": "The bank cut costs across its investment banking division.",
}


class _ScriptedRepair:
    """Replays scripted repairs for whichever repair path the loop reaches for.

    The loop tries sentence-level repair first and falls back to a whole-answer rewrite, so a
    stub that only answers one shape silently exercises the other path with a missing key.
    Each scripted entry is the sentence to substitute for the flagged one; `None` makes the
    sentence path defer so the fallback runs with the same entry as a whole answer.
    """

    def __init__(self, replacements):
        self.replacements = list(replacements)
        self.calls = 0
        self.sentence_calls = 0
        self.rewrite_calls = 0

    def __call__(self, system, user, **kwargs):
        self.calls += 1
        if not self.replacements:
            raise AssertionError("repair called more times than scripted")
        value = self.replacements.pop(0)
        if "ONE sentence" in system:
            self.sentence_calls += 1
            return {"sentence": "" if value is None else value}
        self.rewrite_calls += 1
        return {"answer": value or "", "abstain": False, "used_citations": ["c1"]}


def _entailment_all_supported(system, user, **kwargs):
    """Stub the entailment call so these tests exercise the loop, not the network.

    Without this the call fails, verify_answer swallows the error, and the tests pass for
    the wrong reason while still reaching out over the wire.
    """
    return {"verdicts": []}


def _run(initial, repairs, *, max_rounds=3):
    script = _ScriptedRepair(repairs)
    original_answer, original_verify = amod.chat_json, vmod.chat_json
    amod.chat_json = script
    vmod.chat_json = _entailment_all_supported
    try:
        text, abstained, _json, report = _verify_and_repair(
            "q", "sources", initial, False, {}, EVIDENCE, max_rounds=max_rounds
        )
    finally:
        amod.chat_json = original_answer
        vmod.chat_json = original_verify
    return text, report, script.calls


CLEAN = "Profit rose to 1.2 billion euros [c1]."
BAD_FIGURE = "Profit rose to 9.9 billion euros [c1]."
WORSE = "Profit rose to 9.9 billion euros and costs fell 42% [c1]."


def _surgical(answer, evidence, sentence_reply):
    """Run only the sentence-level repair, recording which sentences it was asked about."""
    from taza_rag.factiva.answer import _repair_sentences

    asked = []

    def fake(system, user, **kwargs):
        asked.append(user.split("Sentence:\n")[1].split("\n")[0])
        return {"sentence": sentence_reply(asked[-1])}

    original_answer, original_verify = amod.chat_json, vmod.chat_json
    amod.chat_json = fake
    vmod.chat_json = _entailment_all_supported
    try:
        report = vmod.verify_answer(answer, evidence, check_entailment=False)
        out = _repair_sentences(answer, report, evidence)
    finally:
        amod.chat_json = original_answer
        vmod.chat_json = original_verify
    return out, asked


GOOD_A = "Profit rose to 1.2 billion euros [c1]."
GOOD_B = "The bank cut costs in its investment bank [c2]."
BAD = "Revenue hit 9.9 billion euros [c1]."
SURGICAL_EV = {
    "c1": "Deutsche Bank said profit rose to 1.2 billion euros, helped by lower provisions.",
    "c2": "The bank cut costs in its investment bank.",
}


def test_surgical_repair_only_touches_the_flagged_sentence():
    """A whole-answer rewrite can damage claims that passed, which is why broader answers
    lost Accuracy faster than they gained Completeness."""
    answer = f"{GOOD_A} {BAD} {GOOD_B}"
    out, asked = _surgical(answer, SURGICAL_EV, lambda s: "")
    assert asked == [BAD], f"asked about unflagged sentences: {asked}"
    assert GOOD_A in out and GOOD_B in out, "a passing sentence was altered"
    assert "9.9" not in out


def test_surgical_repair_can_weaken_rather_than_delete():
    answer = f"{GOOD_A} {BAD}"
    weaker = "Revenue was not disclosed in the cited source [c1]."
    out, _asked = _surgical(answer, SURGICAL_EV, lambda s: weaker)
    assert GOOD_A in out and weaker in out


def test_surgical_repair_leaves_no_double_spacing_behind():
    answer = f"{GOOD_A} {BAD} {GOOD_B}"
    out, _ = _surgical(answer, SURGICAL_EV, lambda s: "")
    assert "  " not in out and out == out.strip()


def test_surgical_repair_defers_when_the_model_is_unreachable():
    """Returning None lets the caller fall back to a whole-answer rewrite."""
    from taza_rag.factiva.answer import _repair_sentences

    def boom(system, user, **kwargs):
        raise amod.LLMError("no credits")

    original_answer, original_verify = amod.chat_json, vmod.chat_json
    amod.chat_json = boom
    vmod.chat_json = _entailment_all_supported
    try:
        report = vmod.verify_answer(f"{GOOD_A} {BAD}", SURGICAL_EV, check_entailment=False)
        assert _repair_sentences(f"{GOOD_A} {BAD}", report, SURGICAL_EV) is None
    finally:
        amod.chat_json = original_answer
        vmod.chat_json = original_verify


def _run_flagging_abstain(initial, repaired_answer):
    """Repair returns abstain=true alongside a full answer, as the real model does."""
    def script(system, user, **kwargs):
        return {"answer": repaired_answer, "abstain": True, "used_citations": ["c1"]}

    original_answer, original_verify = amod.chat_json, vmod.chat_json
    amod.chat_json = script
    vmod.chat_json = _entailment_all_supported
    try:
        text, abstained, _json, report = _verify_and_repair(
            "q", "sources", initial, False, {}, EVIDENCE, max_rounds=1
        )
    finally:
        amod.chat_json = original_answer
        vmod.chat_json = original_verify
    return text, abstained


def test_a_repaired_answer_with_real_content_is_not_marked_a_refusal():
    """The repair prompt invites abstain=true when material is dropped, and the model sets
    it while still answering. Trusting the flag marked 13 of 52 answerable queries as
    refusals, every one carrying a full cited answer."""
    text, abstained = _run_flagging_abstain(BAD_FIGURE, CLEAN)
    assert text == CLEAN
    assert abstained is False, "a cited, factual answer is not an abstention"


def test_a_genuine_refusal_from_repair_is_still_respected():
    refusal = "The sources do not provide the requested figure."
    text, abstained = _run_flagging_abstain(BAD_FIGURE, refusal)
    assert text == refusal
    assert abstained is True, "a real refusal must stay an abstention"


def test_a_clean_answer_is_never_rewritten():
    text, report, calls = _run(CLEAN, [])
    assert text == CLEAN
    assert calls == 0
    assert report["repairs_applied"] == 0
    assert report["resolved"] is True


def test_repair_stops_as_soon_as_the_checks_are_clean():
    """Budget is 3 rounds, but a fix on the first must not spend the other two."""
    text, report, calls = _run(BAD_FIGURE, [CLEAN, CLEAN, CLEAN])
    assert text == CLEAN
    assert calls == 1
    assert report["repairs_applied"] == 1
    assert report["resolved"] is True


def test_it_keeps_trying_across_rounds_while_it_makes_progress():
    two_bad = "Profit rose to 9.9 billion euros and costs fell 42% [c1]."
    one_bad = "Profit rose to 9.9 billion euros [c1]."
    text, report, calls = _run(two_bad, [one_bad, CLEAN])
    assert text == CLEAN
    assert calls == 2
    assert report["repairs_applied"] == 2
    assert report["resolved"] is True


def test_a_rewrite_that_makes_things_worse_is_discarded():
    """The guard that matters: enabling the loop must never degrade an answer."""
    text, report, calls = _run(BAD_FIGURE, [WORSE, CLEAN])
    assert text == BAD_FIGURE, "kept the worse rewrite"
    assert calls == 1, "should stop once a rewrite regresses"
    assert report["resolved"] is False
    assert report["final"]["problems"] == {"unsupported_figure": 1}


def test_no_progress_stops_the_loop_rather_than_burning_the_budget():
    text, report, calls = _run(BAD_FIGURE, [BAD_FIGURE, CLEAN])
    assert text == BAD_FIGURE
    assert calls == 1


def test_the_round_budget_is_respected():
    ladder = [
        "A rose to 9.9 billion euros and costs fell 42% and staff fell 15% [c1].",
        "A rose to 9.9 billion euros and costs fell 42% [c1].",
        "A rose to 9.9 billion euros [c1].",
    ]
    text, report, calls = _run(
        "A rose to 9.9 billion and costs fell 42% and staff fell 15% and fees fell 3% [c1].",
        ladder,
        max_rounds=2,
    )
    assert calls == 2
    assert report["repairs_applied"] == 2
    assert report["resolved"] is False


def test_an_unreachable_model_leaves_the_original_answer_intact():
    def boom(system, user, **kwargs):
        raise amod.LLMError("no credits")

    original_answer, original_verify = amod.chat_json, vmod.chat_json
    amod.chat_json = boom
    vmod.chat_json = _entailment_all_supported
    try:
        text, _abstained, _json, report = _verify_and_repair(
            "q", "sources", BAD_FIGURE, False, {}, EVIDENCE, max_rounds=3
        )
    finally:
        amod.chat_json = original_answer
        vmod.chat_json = original_verify
    assert text == BAD_FIGURE
    assert report["repairs_applied"] == 0
    assert report["resolved"] is False


def test_every_attempt_is_recorded_for_audit():
    _text, report, _calls = _run(
        "Profit rose to 9.9 billion euros and costs fell 42% [c1].", ["Profit rose to 9.9 billion euros [c1].", CLEAN]
    )
    assert len(report["rounds"]) == 3
    assert report["rounds"][0]["problems"] == {"unsupported_figure": 2}
    assert report["rounds"][-1]["problems"] == {}
