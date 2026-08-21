"""Unit tests for retrieval quality helpers — no network, no OpenAI."""

from taza_rag.factiva.strategy import detect_intent, expand_queries, normalize_query
from taza_rag.models import Chunk, RetrievedChunk, SearchIntent
from taza_rag.retrieve.quality import fuse_and_rerank, lexical_overlap


def _hit(doc_id: str, title: str, text: str, source_code: str = "djdn", published="2025-06-01", rank=1):
    chunk = Chunk(
        chunk_id=f"{doc_id}::0",
        doc_id=doc_id,
        text=text,
        title=title,
        source="Dow Jones",
        source_tier="premium",
        published_at=published,
        metadata={"source_code": source_code},
    )
    return RetrievedChunk(chunk=chunk, score=1.0, rank=rank, method="t", scores={"api_rank": float(rank)})


def test_normalize_deutsche():
    assert "Deutsche" in normalize_query("Deutche Bank restructuring")


def test_expand_entity_variants():
    variants = expand_queries("SoftBank Group", SearchIntent.ENTITY_INVESTIGATION)
    assert variants[0] == "SoftBank Group"
    assert len(variants) >= 2


def test_detect_risk_intent():
    assert detect_intent("key risks in private credit covenants") == SearchIntent.RISK_COMPLIANCE


def test_fuse_prefers_overlapping_authority():
    q = "SoftBank Group AI"
    a = [_hit("a", "SoftBank expands AI bets", "SoftBank Group AI infrastructure", "djdn", rank=1)]
    b = [_hit("b", "Unrelated retail note", "mall traffic rose", "wire", published="2020-01-01", rank=1)]
    # SoftBank also appears as rank 2 in second list
    b.append(_hit("a", "SoftBank expands AI bets", "SoftBank Group AI infrastructure", "djdn", rank=2))
    fused = fuse_and_rerank(q, [a, b], top_k=2)
    assert fused[0].chunk.doc_id == "a"
    assert lexical_overlap(q, fused[0]) > 0.3
