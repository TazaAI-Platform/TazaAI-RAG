"""Fact extraction is the coverage path that does not ask the writer to invent.

A fact whose figure is absent from its cited excerpt must never reach composition: that is
the Accuracy failure the coverage prompt bought last time.
"""

import taza_rag.factiva.facts as fmod
from taza_rag.factiva.facts import (
    Fact,
    compose_from_facts,
    filter_facts,
    format_fact_list,
    generate_from_facts,
    parse_facts,
)

EVIDENCE = {
    "c1": "SoftBank posted net profit of 347.33 billion yen, an 18% decline.",
    "c2": "The Intel stake produced an $8.2 billion valuation gain.",
}


def test_parse_facts_normalises_bare_and_bracketed_labels():
    raw = {
        "facts": [
            {"text": "Profit fell 18%.", "citation": "C1"},
            {"text": "Intel gained $8.2 billion.", "citation": "[c2]"},
            {"text": "A third point.", "citation": "3"},
            {"text": "", "citation": "c1"},
            {"text": "No label."},
        ]
    }
    facts = parse_facts(raw)
    assert [f.citation for f in facts] == ["c1", "c2", "c3"]


def test_an_invented_figure_is_dropped_before_composition():
    facts = [
        Fact("Profit was 347.33 billion yen.", "c1"),
        Fact("Profit was 999.9 billion yen.", "c1"),
        Fact("Intel gained $8.2 billion.", "c2"),
        Fact("Staff fell 15%.", "c2"),
    ]
    kept = filter_facts(facts, EVIDENCE)
    assert [f.text for f in kept] == [
        "Profit was 347.33 billion yen.",
        "Intel gained $8.2 billion.",
    ]


def test_a_fact_citing_a_missing_label_is_dropped():
    kept = filter_facts([Fact("Anything at all.", "c9")], EVIDENCE)
    assert kept == []


def test_qualitative_facts_with_no_figures_survive():
    kept = filter_facts([Fact("SoftBank is increasing AI investment.", "c1")], EVIDENCE)
    assert len(kept) == 1


def test_compose_uses_only_the_fact_list():
    seen = {}

    def stub(system, user, **kwargs):
        seen["user"] = user
        return {
            "answer": "Profit was 347.33 billion yen [c1].",
            "abstain": False,
            "used_citations": ["c1"],
        }

    original = fmod.chat_json
    fmod.chat_json = stub
    try:
        out = compose_from_facts("SoftBank", [Fact("Profit was 347.33 billion yen.", "c1")])
    finally:
        fmod.chat_json = original
    assert "347.33" in seen["user"]
    assert "Sources:" not in seen["user"]
    assert out["answer"].startswith("Profit")


def test_empty_extraction_falls_back_so_the_one_shot_path_still_runs():
    def empty(system, user, **kwargs):
        return {"facts": []}

    original = fmod.chat_json
    fmod.chat_json = empty
    try:
        assert generate_from_facts("q", "ctx", EVIDENCE) is None
    finally:
        fmod.chat_json = original


def test_format_fact_list_carries_the_citation_the_writer_must_use():
    text = format_fact_list([Fact("Profit fell 18%.", "c1")])
    assert text == "1. Profit fell 18%. [c1]"
