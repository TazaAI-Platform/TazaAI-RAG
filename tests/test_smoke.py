"""Offline smoke checks that do not need API keys (still need package deps installed)."""

from taza_rag.ingest.chunk import chunk_document
from taza_rag.ingest.contextualize import contextualize_chunk
from taza_rag.models import Document, SearchIntent, INTENT_PRIORITIES
from taza_rag.eval.retrieval import recall_at_k
from taza_rag.models import Chunk, RetrievedChunk


def test_chunking_produces_overlap_windows():
    body = "\n\n".join([f"Paragraph {i} with enough tokens to fill space." * 3 for i in range(12)])
    doc = Document(doc_id="d1", title="T", body=body, source="WSJ")
    chunks = chunk_document(doc, target_tokens=80, overlap_tokens=20)
    assert len(chunks) >= 2
    assert chunks[0].chunk_id.endswith("::c0000")


def test_heuristic_contextualize():
    doc = Document(
        doc_id="d1",
        title="SoftBank news",
        body="SoftBank expands AI bets.",
        source="DJ",
        published_at="2025-01-01",
        entities=["SoftBank"],
    )
    chunk = Chunk(
        chunk_id="d1::c0000",
        doc_id="d1",
        text="SoftBank expands AI bets.",
        title=doc.title,
        source=doc.source,
    )
    out = contextualize_chunk(doc, chunk, use_llm=False)
    assert out.contextualized_text is not None
    assert "SoftBank news" in out.contextualized_text


def test_intent_priors_sum_near_one():
    total = sum(INTENT_PRIORITIES.values())
    assert 0.7 < total < 1.01  # long-tail intents omitted from full 100%
    assert SearchIntent.ENTITY_INVESTIGATION in INTENT_PRIORITIES


def test_recall_at_k():
    chunks = [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id="a::0",
                doc_id="docA",
                text="x",
                title="t",
                source="s",
            ),
            score=1.0,
            rank=1,
        )
    ]
    assert recall_at_k(chunks, ["docA"], k=1) == 1.0
    assert recall_at_k(chunks, ["docB"], k=1) == 0.0
