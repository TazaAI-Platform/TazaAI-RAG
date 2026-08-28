"""The stopping rule is the claim this agent makes, so it is the part most worth pinning down.

Each exit has to be reachable, distinguishable, and reported honestly: a run cut off by its
budget must never be labelled as one that converged.
"""

from taza_rag.agent.models import Budget, Cost, Finding, ResearchPlan, RoundRecord, SubQuestion
from taza_rag.agent.sufficiency import aspect_gaps, assess, coverage_by_sub
from taza_rag.agent.text import covers
from taza_rag.models import SearchIntent


def _plan(*specs):
    subs = [
        SubQuestion(id=f"s{i}", question=q, intent=SearchIntent.ENTITY_INVESTIGATION, aspects=list(a))
        for i, (q, a) in enumerate(specs, start=1)
    ]
    return ResearchPlan(
        question="How exposed is SoftBank to its AI bets?",
        intent=SearchIntent.ENTITY_INVESTIGATION,
        entities=["SoftBank"],
        sub_questions=subs,
        method="llm",
    )


def _finding(text, sub="s1", label="c1"):
    return Finding(sub_question_id=sub, text=text, label=label)


def test_an_aspect_counts_as_covered_on_a_majority_of_its_own_terms():
    assert covers("record retail bond issuance", "SoftBank plans a record issuance of retail bonds.")
    assert not covers("record retail bond issuance", "SoftBank reported quarterly net profit.")


def test_a_single_common_term_does_not_mark_an_aspect_covered():
    """Otherwise "profit" alone would satisfy every aspect that mentions profit."""
    assert not covers("analyst consensus net profit figure", "Profit rose.")


def test_coverage_is_judged_against_every_fact_because_the_pool_is_shared():
    """A passage found while asking about borrowing can legitimately answer the profit step."""
    plan = _plan(
        ("SoftBank quarterly profit", ["net profit figure"]),
        ("SoftBank borrowing", ["OpenAI stake borrowing"]),
    )
    findings = [
        _finding("SoftBank borrowed against its OpenAI stake, disclosing the net profit figure.", sub="s2")
    ]
    sub_cov = coverage_by_sub(plan, findings)
    assert sub_cov["s1"] == 1.0
    assert sub_cov["s2"] == 1.0


def test_uncovered_aspects_are_reported_as_gaps_against_the_step_that_asked():
    plan = _plan(("SoftBank quarterly profit", ["net profit figure", "analyst consensus poll"]))
    gaps = aspect_gaps(plan, [_finding("SoftBank posted a net profit figure of 347 billion yen.")])
    assert [g.aspect for g in gaps] == ["analyst consensus poll"]
    assert gaps[0].sub_question_id == "s1"


def test_reaching_target_coverage_stops_the_run():
    plan = _plan(("SoftBank quarterly profit", ["net profit figure"]))
    findings = [_finding("SoftBank posted a net profit figure of 347.33 billion yen.")]
    verdict = assess(plan, findings, [RoundRecord(index=0, new_findings=1)], Cost(), Budget())
    assert verdict.stop
    assert verdict.reason == "target_coverage"
    assert verdict.coverage == 1.0


def test_a_round_that_buys_passages_and_learns_nothing_stops_the_run():
    """Plateau: further retrieval has stopped changing the answer."""
    plan = _plan(("SoftBank quarterly profit", ["net profit figure", "analyst consensus poll"]))
    findings = [_finding("SoftBank posted a net profit figure of 347 billion yen.")]
    rounds = [RoundRecord(index=0, new_findings=4), RoundRecord(index=1, new_chunks=6, new_findings=0)]
    verdict = assess(plan, findings, rounds, Cost(unique_chunks=12), Budget(max_rounds=5))
    assert verdict.stop
    assert verdict.reason == "plateau"


def test_an_incomplete_run_keeps_going_and_asks_only_for_what_is_missing():
    plan = _plan(
        ("SoftBank quarterly profit", ["net profit figure", "analyst consensus poll"]),
        ("SoftBank borrowing", ["OpenAI stake borrowing"]),
    )
    findings = [_finding("SoftBank posted a net profit figure of 347 billion yen.")]
    verdict = assess(plan, findings, [RoundRecord(index=0, new_findings=3)], Cost(), Budget())
    assert not verdict.stop
    assert verdict.reason == "continue"
    queries = [q for _s, q in verdict.refinements]
    # Refinements target the missing aspects, anchored on the entity, and never re-ask the
    # aspect that is already covered.
    assert any("analyst consensus poll" in q for q in queries)
    assert any("OpenAI stake borrowing" in q for q in queries)
    assert all(q.startswith("SoftBank") for q in queries)
    assert not any("net profit figure" in q for q in queries)


def test_a_round_full_of_facts_that_move_no_aspect_also_counts_as_plateau():
    """Coverage flat while the bill grows is the failure a live run actually showed."""
    plan = _plan(("SoftBank quarterly profit", ["net profit figure", "analyst consensus poll"]))
    findings = [_finding("SoftBank posted a net profit figure of 347 billion yen.")]
    rounds = [
        RoundRecord(index=0, new_findings=4, coverage=0.5),
        RoundRecord(index=1, new_chunks=6, new_findings=9, coverage=0.5),
    ]
    verdict = assess(plan, findings, rounds, Cost(unique_chunks=12), Budget(max_rounds=5))
    assert verdict.stop
    assert verdict.reason == "plateau"


def test_the_round_cap_is_reported_as_a_cap_not_as_convergence():
    plan = _plan(("SoftBank quarterly profit", ["net profit figure", "analyst consensus poll"]))
    rounds = [RoundRecord(index=i, new_findings=2) for i in range(3)]
    verdict = assess(plan, [], rounds, Cost(), Budget(max_rounds=3))
    assert verdict.reason == "round_cap"
    assert verdict.coverage < 0.8


def test_the_chunk_budget_is_reported_as_a_budget_exit():
    plan = _plan(("SoftBank quarterly profit", ["net profit figure", "analyst consensus poll"]))
    verdict = assess(
        plan,
        [],
        [RoundRecord(index=0, new_findings=2)],
        Cost(unique_chunks=40),
        Budget(max_rounds=5, max_unique_chunks=40),
    )
    assert verdict.reason == "budget_chunks"


def test_a_query_already_issued_is_not_paid_for_twice():
    plan = _plan(("SoftBank quarterly profit", ["analyst consensus poll"]))
    verdict = assess(
        plan,
        [],
        [RoundRecord(index=0, new_findings=2)],
        Cost(),
        Budget(),
        issued={"SoftBank analyst consensus poll"},
    )
    assert verdict.stop
    assert verdict.reason == "no_new_query"


def test_a_heuristic_plan_without_aspects_scores_on_whether_a_step_returned_anything():
    plan = _plan(("SoftBank Group", []), ("SoftBank Group strategy results", []))
    plan.method = "heuristic"
    sub_cov = coverage_by_sub(plan, [_finding("Anything at all.", sub="s1")])
    assert sub_cov == {"s1": 1.0, "s2": 0.0}
