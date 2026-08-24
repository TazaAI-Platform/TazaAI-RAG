"""Grounding verification — the deterministic paths need no network or key.

Cases are drawn from real failures the gpt-5 judge found: a figure that appears in no
source, an uncited claim, and an attribution the excerpt does not make.
"""

import taza_rag.factiva.verify as vmod
from taza_rag.factiva.verify import (
    check_citations,
    check_figures,
    check_support,
    figures,
    split_claims,
    strip_labels,
    verify_answer,
)

EVIDENCE = {
    "c1": "Deutsche Bank cuts costs\nDeutsche Bank said quarterly profit rose to 1.2 billion "
    "euros in the second quarter of 2026, helped by lower provisions.",
    "c2": "SoftBank Vision Fund\nSoftBank reported a Vision Funds gain of 451.39 billion yen, "
    "against 5.43 billion yen a year earlier.",
    "c3": "ADIA and NIIF\nADIA agreed to explore an investment of up to $1 billion alongside "
    "NIIF, people familiar with the matter said.",
}


def test_claims_keep_their_citation_labels():
    claims = split_claims("Profit rose to 1.2 billion euros [c1]. Costs fell sharply [c1][c2].")
    assert len(claims) == 2
    assert claims[0].labels == ["c1"]
    assert claims[1].labels == ["c1", "c2"]


def test_decimals_do_not_split_sentences():
    claims = split_claims("The gain was 451.39 billion yen [c2]. That reversed a loss [c2].")
    assert len(claims) == 2
    assert "451.39" in claims[0].text


def test_comma_separated_labels_are_parsed():
    claims = split_claims("Both banks cut costs [c1, c2].")
    assert claims[0].labels == ["c1", "c2"]


def test_a_fabricated_figure_is_caught():
    """'record profits of 3.8 billion' appears in no source — the real failure mode."""
    claims = split_claims("Deutsche Bank posted profit of 3.8 billion euros [c1].")
    problems = check_figures(claims, EVIDENCE)
    kinds = {p.kind for p in problems}
    assert "unsupported_figure" in kinds
    assert any("3.8" in p.detail for p in problems)


def test_a_correct_figure_passes():
    claims = split_claims("Profit rose to 1.2 billion euros [c1].")
    assert check_figures(claims, EVIDENCE) == []


def test_figures_present_but_wrongly_attributed_are_distinguished():
    """A real number cited to the wrong chunk is a different defect from invention."""
    claims = split_claims("The Vision Funds gain was 451.39 billion yen [c1].")
    problems = check_figures(claims, EVIDENCE)
    assert [p.kind for p in problems] == ["miscited_figure"]


def test_thousands_separators_and_currency_do_not_cause_false_alarms():
    evidence = {"c1": "Revenue reached $1,234.5 million for the period."}
    claims = split_claims("Revenue reached $1,234.5 million in the quarter [c1].")
    assert check_figures(claims, evidence) == []


def test_a_year_is_reported_but_never_blocks_a_rewrite():
    claims = split_claims("The bank restructured during 2019 to simplify the group [c1].")
    problems = check_figures(claims, EVIDENCE)
    assert [p.kind for p in problems] == ["unverified_year"]
    report = verify_answer(
        "The bank restructured during 2019 to simplify the group [c1].",
        EVIDENCE,
        check_entailment=False,
    )
    assert report.problems and report.blocking == []


def test_an_uncited_claim_is_caught():
    claims = split_claims("Deutsche Bank is widely expected to keep cutting headcount.")
    problems = check_citations(claims, set(EVIDENCE))
    assert [p.kind for p in problems] == ["uncited"]


def test_a_citation_to_a_nonexistent_source_is_caught():
    claims = split_claims("Profit rose to 1.2 billion euros [c9].")
    problems = check_citations(claims, set(EVIDENCE))
    assert [p.kind for p in problems] == ["invalid_label"]


def test_a_refusal_is_not_treated_as_an_uncited_claim():
    claims = split_claims("The sources do not provide information on the 2027 figure.")
    assert check_citations(claims, set(EVIDENCE)) == []


def test_entailment_flags_an_overstated_commitment():
    """'explore up to $1bn' became 'committed to invest' in a real answer."""
    captured = {}

    def fake(system, user, model=None, **kw):
        captured["user"] = user
        return {
            "verdicts": [
                {"index": 1, "supported": False, "reason": "excerpt says explore, not commit"}
            ]
        }

    vmod.chat_json = fake
    claims = split_claims("ADIA committed to invest $1 billion with NIIF [c3].")
    problems = check_support(claims, EVIDENCE)
    assert [p.kind for p in problems] == ["unsupported_claim"]
    assert "explore" in problems[0].detail
    # The checker must see the cited excerpt, not just the claim
    assert "agreed to explore" in captured["user"]


def test_support_check_is_skipped_when_nothing_is_cited():
    calls = []
    vmod.chat_json = lambda *a, **k: calls.append(1) or {"verdicts": []}
    assert check_support(split_claims("An uncited sentence about banks."), EVIDENCE) == []
    assert not calls


def test_verification_survives_an_llm_failure():
    """A provider outage must not discard the deterministic findings."""
    from taza_rag.llm import LLMError

    def boom(*a, **k):
        raise LLMError("provider down")

    vmod.chat_json = boom
    report = verify_answer("Profit was 9.9 billion euros [c1].", EVIDENCE)
    assert report.checked_support is False
    assert any(p.kind == "unsupported_figure" for p in report.problems)


def test_labels_are_stripped_before_number_extraction():
    assert figures("Profit rose [c1] to 1.2 billion [c2].") == ["1.2"]
    assert "c1" not in strip_labels("text [c1] here")


def test_report_summary_counts_by_kind():
    report = verify_answer(
        "Profit was 9.9 billion euros [c1]. Costs are expected to fall further.",
        EVIDENCE,
        check_entailment=False,
    )
    summary = report.summary()
    assert summary["problems"]["unsupported_figure"] == 1
    assert summary["problems"]["uncited"] == 1
    assert summary["claims"] == 2


def test_an_uncited_claim_is_not_also_reported_for_every_figure():
    """Duplicating one root cause across each number buries the actual problem."""
    claims = split_claims("ADIA is marking its 50th anniversary after 1976 and 3 decades.")
    assert check_figures(claims, EVIDENCE) == []
    report = verify_answer(
        "ADIA is marking its 50th anniversary after 1976 and 3 decades.",
        EVIDENCE,
        check_entailment=False,
    )
    assert [p.kind for p in report.problems] == ["uncited"]
