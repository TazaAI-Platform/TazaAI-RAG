"""Conflict detection must not invent disputes.

A fabricated disagreement is worse than a missed one: it puts a factual error in front of the
reader and it is exactly the distortion the A1 rubric fails an answer for. So the cases that
look like conflicts but are not — rounding, currency conversion, percentage against absolute —
are tested as carefully as the real ones.
"""

from taza_rag.agent.conflict import blocking_conflicts, describe_conflicts, detect_conflicts
from taza_rag.agent.models import EvidenceItem, Finding
from taza_rag.models import Chunk, RetrievedChunk


def _item(label, doc_id, source, published_at="2026-08-06", authority=1.0, text="excerpt"):
    return EvidenceItem(
        label=label,
        hit=RetrievedChunk(
            chunk=Chunk(
                chunk_id=f"{doc_id}::p000",
                doc_id=doc_id,
                text=text,
                title=f"{source} story",
                source=source,
                published_at=published_at,
            ),
            score=1.0,
            rank=1,
            scores={"authority": authority},
        ),
    )


def _finding(label, text, sub="s1"):
    return Finding(sub_question_id=sub, text=text, label=label)


def test_two_sources_with_materially_different_figures_are_a_disagreement():
    findings = [
        _finding("c1", "SoftBank booked an $8.2 billion valuation gain on its Intel stake."),
        _finding("c2", "SoftBank booked an $8.9 billion valuation gain on its Intel stake."),
    ]
    by_label = {
        "c1": _item("c1", "d1", "Dow Jones Newswires", authority=1.10),
        "c2": _item("c2", "d2", "Aggregator Wire", authority=0.95),
    }
    conflicts = detect_conflicts(findings, by_label)
    assert len(blocking_conflicts(conflicts)) == 1
    conflict = conflicts[0]
    assert conflict.kind == "disagreement"
    # Authority decides which side leads, and the reason has to name it.
    assert conflict.preferred_label == "c1"
    assert "Dow Jones" in conflict.reason


def test_the_same_figure_to_different_precision_is_rounding_not_a_dispute():
    findings = [
        _finding("c1", "Net profit fell 18% from a year earlier."),
        _finding("c2", "Net profit fell 17.7% from a year earlier."),
    ]
    by_label = {"c1": _item("c1", "d1", "Kyodo News"), "c2": _item("c2", "d2", "Dow Jones")}
    conflicts = detect_conflicts(findings, by_label)
    assert [c.kind for c in conflicts] == ["rounding"]
    assert blocking_conflicts(conflicts) == []


def test_the_same_amount_in_two_currencies_is_not_a_conflict():
    findings = [
        _finding("c1", "SoftBank posted a net profit of 347.33 billion yen for the quarter."),
        _finding("c2", "SoftBank posted a net profit of $2.2 billion for the quarter."),
    ]
    by_label = {"c1": _item("c1", "d1", "Kyodo News"), "c2": _item("c2", "d2", "Times of India")}
    assert detect_conflicts(findings, by_label) == []


def test_a_percentage_and_an_absolute_amount_are_different_measures():
    findings = [
        _finding("c1", "Quarterly net profit fell 18%."),
        _finding("c2", "Quarterly net profit was 347.33 billion yen."),
    ]
    by_label = {"c1": _item("c1", "d1", "Kyodo News"), "c2": _item("c2", "d2", "Dow Jones")}
    assert blocking_conflicts(detect_conflicts(findings, by_label)) == []


def test_facts_about_different_subjects_are_never_compared():
    findings = [
        _finding("c1", "SoftBank borrowed $10 billion against its OpenAI stake."),
        _finding("c2", "Nvidia reported quarterly revenue of $46 billion."),
    ]
    by_label = {"c1": _item("c1", "d1", "WSJ"), "c2": _item("c2", "d2", "Dow Jones")}
    assert detect_conflicts(findings, by_label) == []


def test_two_passages_of_one_article_are_not_two_conflicting_sources():
    findings = [
        _finding("c1", "The Intel stake produced an $8.2 billion gain."),
        _finding("c2", "The Intel stake produced an $8.9 billion gain."),
    ]
    same_doc = {
        "c1": _item("c1", "d1", "Dow Jones"),
        "c2": _item("c2", "d1", "Dow Jones"),
    }
    assert detect_conflicts(findings, same_doc) == []


def test_two_different_companies_with_the_same_phrasing_are_not_in_dispute():
    """A comparative question produced nine fabricated disagreements before this guard.

    Strip the digits from "Airbus delivered 60 aircraft" and "Boeing delivered 45 aircraft"
    and every remaining word matches, so subject overlap alone reads them as one subject.
    """
    findings = [
        _finding("c1", "Airbus delivered 60 aircraft in July."),
        _finding("c2", "Boeing delivered 45 aircraft in July."),
    ]
    by_label = {"c1": _item("c1", "d1", "Reuters"), "c2": _item("c2", "d2", "Dow Jones")}
    assert detect_conflicts(findings, by_label) == []


def test_the_same_company_from_two_outlets_is_still_compared():
    """The actor guard must not switch conflict detection off entirely."""
    findings = [
        _finding("c1", "Airbus delivered 60 aircraft in July."),
        _finding("c2", "Airbus delivered 71 aircraft in July."),
    ]
    by_label = {"c1": _item("c1", "d1", "Reuters"), "c2": _item("c2", "d2", "Dow Jones")}
    assert [c.kind for c in detect_conflicts(findings, by_label)] == ["disagreement"]


def test_facts_naming_no_entity_still_fall_through_to_the_overlap_test():
    findings = [
        _finding("c1", "Quarterly deliveries reached 60 aircraft."),
        _finding("c2", "Quarterly deliveries reached 71 aircraft."),
    ]
    by_label = {"c1": _item("c1", "d1", "Reuters"), "c2": _item("c2", "d2", "Dow Jones")}
    assert [c.kind for c in detect_conflicts(findings, by_label)] == ["disagreement"]


def test_a_year_mismatch_alone_is_not_a_disagreement():
    """"This year" is routinely paraphrased, so a year is weak evidence of a dispute."""
    findings = [
        _finding("c1", "The bank began the overhaul in 2025 across its private bank."),
        _finding("c2", "The bank began the overhaul in 2026 across its private bank."),
    ]
    by_label = {"c1": _item("c1", "d1", "WSJ"), "c2": _item("c2", "d2", "Reuters")}
    assert detect_conflicts(findings, by_label) == []


def test_equal_authority_falls_back_to_the_later_report():
    findings = [
        _finding("c1", "Costs are expected to rise by 100 million euros."),
        _finding("c2", "Costs are expected to rise by 140 million euros."),
    ]
    by_label = {
        "c1": _item("c1", "d1", "Paper A", published_at="2026-07-01", authority=1.0),
        "c2": _item("c2", "d2", "Paper B", published_at="2026-08-01", authority=1.0),
    }
    conflict = detect_conflicts(findings, by_label)[0]
    assert conflict.kind == "disagreement"
    assert conflict.preferred_label == "c2"
    assert "later report" in conflict.reason


def test_the_composer_is_told_both_sides_and_which_leads():
    findings = [
        _finding("c1", "SoftBank booked an $8.2 billion gain on Intel."),
        _finding("c2", "SoftBank booked an $8.9 billion gain on Intel."),
    ]
    by_label = {
        "c1": _item("c1", "d1", "Dow Jones Newswires", authority=1.10),
        "c2": _item("c2", "d2", "Aggregator Wire", authority=0.95),
    }
    text = describe_conflicts(detect_conflicts(findings, by_label), by_label)
    assert "8.2" in text and "8.9" in text
    assert "Dow Jones Newswires" in text and "Aggregator Wire" in text
    assert "report both" in text


def test_rounding_restatements_are_kept_out_of_the_composer_brief():
    findings = [
        _finding("c1", "Profit fell 18%."),
        _finding("c2", "Profit fell 17.7%."),
    ]
    by_label = {"c1": _item("c1", "d1", "Kyodo"), "c2": _item("c2", "d2", "Dow Jones")}
    assert describe_conflicts(detect_conflicts(findings, by_label), by_label) == ""
