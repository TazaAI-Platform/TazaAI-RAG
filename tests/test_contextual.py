"""Contextual retrieval over Factiva articles — offline, no API key needed."""

from taza_rag.factiva.contextual import (
    heuristic_context,
    salient_entities,
    semantic_scores,
    to_passages,
)
from taza_rag.models import Chunk, RetrievedChunk, SearchIntent
from taza_rag.retrieve.features import build_query_plan, entity_signal, topic_signal
from taza_rag.retrieve.quality import rank_candidates

LONG_BODY = "\n\n".join(
    [
        "Deutsche Bank confirmed a fresh buyback on Tuesday, its second this year.",
        " ".join(["Analysts welcomed the capital return decision."] * 30),
        " ".join(["Separately the lender said the overhaul would cut more jobs."] * 30),
        " ".join(["Regulators continue to monitor the German market."] * 30),
    ]
)


def _article(doc_id: str = "d1", title: str = "Deutsche Bank confirms buyback") -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"{doc_id}::p000",
        doc_id=doc_id,
        text=LONG_BODY,
        title=title,
        source="wsj.com",
        source_tier="premium",
        published_at="2026-07-29",
        chunk_index=0,
        metadata={"source_code": "wsjo"},
    )
    return RetrievedChunk(
        chunk=chunk, score=1.0, rank=2, method="factiva_retrieve", scores={"api_rank": 2.0}
    )


def test_article_is_split_into_contextualized_passages():
    passages = to_passages([_article()])
    assert len(passages) > 1
    for i, p in enumerate(passages):
        assert p.chunk.chunk_index == i
        assert p.chunk.contextualized_text is not None
        # The situating prefix must precede the passage text, not replace it
        assert p.chunk.contextualized_text.endswith(p.chunk.text)
        assert p.chunk.index_text == p.chunk.contextualized_text


def test_passage_ids_are_stable_across_variants():
    """Rank fusion can only merge a passage retrieved twice if its id is positional."""
    first = to_passages([_article()])
    second = to_passages([_article()])
    assert [p.chunk.chunk_id for p in first] == [p.chunk.chunk_id for p in second]
    assert first[0].chunk.chunk_id == "d1::p000"


def test_context_prefix_carries_document_anchors():
    ctx = heuristic_context(
        "Deutsche Bank confirms buyback", "wsj.com", "2026-07-29", ["Deutsche Bank"], 0, 3
    )
    assert "wsj.com" in ctx
    assert "2026-07-29" in ctx
    assert "Deutsche Bank" in ctx


def test_salient_entities_skips_stopwords_and_roles():
    ents = salient_entities("The CEO said Deutsche Bank would act. Deutsche Bank confirmed it.")
    assert "Deutsche Bank" in ents
    assert not any(e.lower() in {"the", "ceo"} for e in ents)


def test_lead_signal_only_applies_to_the_opening_passage():
    """A topic buried in paragraph nine must not get headline-level credit."""
    plan = build_query_plan("Deutsche Bank restructuring", SearchIntent.ENTITY_INVESTIGATION)
    passages = to_passages([_article()])
    later = [p for p in passages if p.chunk.chunk_index > 0]
    assert later, "expected more than one passage"
    for p in later:
        assert topic_signal(plan, p)["topic_lead"] == 0.0
        assert entity_signal(plan, p)["entity_lead"] == 0.0
    assert topic_signal(plan, passages[0])["topic_lead"] == 0.0  # opening is about buybacks


def test_ranking_returns_one_passage_per_document():
    plan = build_query_plan("Deutsche Bank restructuring", SearchIntent.ENTITY_INVESTIGATION)
    passages = to_passages([_article("d1"), _article("d2", "Deutsche Bank plans overhaul")])
    ranked = rank_candidates(plan, [passages], top_k=10)
    doc_ids = [h.chunk.doc_id for h in ranked]
    assert len(doc_ids) == len(set(doc_ids))


def test_later_passages_are_penalized_by_position():
    plan = build_query_plan("Deutsche Bank overhaul", SearchIntent.ENTITY_INVESTIGATION)
    passages = to_passages([_article()])
    ranked = rank_candidates(plan, [passages], top_k=10, one_per_doc=False)
    by_index = {h.chunk.chunk_index: h for h in ranked}
    assert by_index[0].scores["position"] == 0.0
    later = max(i for i in by_index if i > 0)
    assert by_index[later].scores["position"] > 0.0


def test_semantic_scoring_is_skipped_without_a_key():
    """The lexical stack must never depend on an API key being present."""
    passages = to_passages([_article()])
    assert semantic_scores("Deutsche Bank restructuring", passages) == []


def test_contextual_retrieval_finds_topic_outside_the_lead():
    """The overhaul sentence sits deep in the article; passages surface it as evidence."""
    plan = build_query_plan("Deutsche Bank overhaul", SearchIntent.ENTITY_INVESTIGATION)
    article = _article()
    whole = rank_candidates(plan, [[article]], top_k=5)
    passages = rank_candidates(plan, [to_passages([article])], top_k=5)
    assert whole and passages
    # The passage that actually discusses the overhaul becomes the cited evidence
    assert "overhaul" in passages[0].chunk.text.lower()
    assert len(passages[0].chunk.text) < len(whole[0].chunk.text)
