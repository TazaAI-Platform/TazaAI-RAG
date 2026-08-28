"""Aspects are the agent's completion criteria, so an unsatisfiable one is a stuck run.

A first live run asked for an "official comment" and a "dissenting view". Those words never
appear in a news story, so lexical matching scored them zero forever: coverage sat at 0.44
while the agent spent every remaining round re-asking for material it already had. These
tests hold the line on both sides — abstract aspects must be checkable, and concrete ones
must not be waved through by a loose predicate.
"""

from taza_rag.agent.aspects import classify, satisfied

PROFIT = "SoftBank Group posted a net profit of 347.33 billion yen, down 17.7 percent."
QUOTE = 'Chief executive Masayoshi Son said the company would keep investing in AI.'
CAUTION = "However, analysts warned the concentration of risk in one stake was growing."
BOND = "SoftBank plans a record retail bond issuance to fund AI investments."
BLAND = "The group operates across several segments."


def test_a_concrete_aspect_is_matched_lexically():
    assert classify("record retail bond issuance") == "lexical"
    assert satisfied("record retail bond issuance", [BOND])
    assert not satisfied("record retail bond issuance", [PROFIT])


def test_an_abstract_figure_aspect_is_satisfied_by_any_figure():
    assert classify("key figures") == "figure"
    assert satisfied("key figures", [PROFIT])
    assert not satisfied("key figures", [BLAND])


def test_an_official_comment_aspect_is_satisfied_by_an_attribution():
    assert classify("official comment") == "attribution"
    assert satisfied("official comment", [QUOTE])
    # A figure alone is not somebody saying something.
    assert not satisfied("official comment", [BLAND])


def test_a_dissenting_view_aspect_is_satisfied_by_a_contrary_statement():
    assert classify("dissenting view") == "contrary view"
    assert satisfied("dissenting view", [CAUTION])
    assert not satisfied("dissenting view", [BOND])


def test_a_timing_aspect_is_satisfied_by_a_date_or_quarter():
    assert classify("timing") == "timing"
    assert satisfied("timing", ["The deal closed in August 2026."])
    assert not satisfied("timing", [BLAND])


def test_a_distinctive_term_keeps_the_lexical_path_even_beside_generic_words():
    """"OpenAI stake borrowing" must not be downgraded to "any number will do"."""
    assert classify("OpenAI stake borrowing figures") == "lexical"
    assert not satisfied("OpenAI stake borrowing figures", [PROFIT])
    assert satisfied(
        "OpenAI stake borrowing figures",
        ["SoftBank borrowed 10 billion dollars against its OpenAI stake."],
    )


def test_no_facts_means_nothing_is_satisfied():
    assert not satisfied("key figures", [])
    assert not satisfied("record retail bond issuance", [])
