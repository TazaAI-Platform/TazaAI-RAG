"""Contextual retrieval over Factiva articles (Anthropic-style).

The Retrieval API hands back whole articles. Ranking them as single blobs has two
costs: lexical signals get diluted across paragraphs that have nothing to do with the
query, and the evidence pack can only cite an entire story rather than the sentences
that support a claim.

This module splits each article into passages and prepends a situating context to
each one — source, date, headline, resolved subject — so a passage carries its own
document anchors into both retrieval and the citation. The context is heuristic by
default (no API key, no latency); an LLM-written variant is available behind a flag.
"""

from __future__ import annotations

import re
from collections import Counter

from taza_rag.config import settings
from taza_rag.ingest.chunk import chunk_document, estimate_tokens
from taza_rag.models import Chunk, Document, RetrievedChunk
from taza_rag.retrieve.features import ROLE_WORDS, STOPWORDS, content_terms

CONTEXT_PROMPT = """You situate a passage within its source news article for retrieval.
Write ONE sentence (max 30 words) naming the publication, the date, the subject
organisation or person, and what this passage covers. Do not quote the passage.
Output plain text only."""

_CAP_SPAN = re.compile(r"\b[A-Z][\w&’'\.\-]*(?:\s+[A-Z][\w&’'\.\-]*)*")

PASSAGE_TOKENS = 180
PASSAGE_OVERLAP = 40
MAX_PASSAGES = 6


def salient_entities(text: str, limit: int = 4) -> list[str]:
    """Most frequent capitalized spans — the passage's apparent subjects."""
    counts: Counter[str] = Counter()
    for span in _CAP_SPAN.findall(text or ""):
        span = span.strip()
        terms = content_terms(span)
        if not terms:
            continue
        if all(t in STOPWORDS or t in ROLE_WORDS for t in terms):
            continue
        if len(span) < 3:
            continue
        counts[span] += 1
    return [span for span, _ in counts.most_common(limit)]


def heuristic_context(
    title: str,
    source: str,
    published_at: str | None,
    entities: list[str],
    passage_index: int,
    passage_count: int,
) -> str:
    """Document anchors a passage would otherwise lose when read in isolation."""
    where = f"{source}" if source else "Factiva"
    when = f", {published_at[:10]}" if published_at else ""
    subject = f" Subject: {', '.join(entities)}." if entities else ""
    position = (
        " Opening of the article."
        if passage_index == 0
        else f" Passage {passage_index + 1} of {passage_count}."
    )
    return f"From '{title}' ({where}{when}).{subject}{position}"


def llm_context(title: str, source: str, published_at: str | None, body: str, passage: str) -> str:
    from taza_rag.llm import chat_text

    user = (
        f"<article>\nHeadline: {title}\nSource: {source}\n"
        f"Published: {published_at or 'unknown'}\n\n{body[:6000]}\n</article>\n\n"
        f"<passage>\n{passage}\n</passage>"
    )
    return chat_text(
        CONTEXT_PROMPT, user, model=settings.contextualize_model, temperature=0.0
    ).strip()


def _as_document(hit: RetrievedChunk) -> Document:
    c = hit.chunk
    return Document(
        doc_id=c.doc_id,
        title=c.title,
        body=c.text,
        source=c.source,
        source_tier=c.source_tier,
        published_at=c.published_at,
        url=c.url,
        metadata=c.metadata or {},
    )


def to_passages(
    hits: list[RetrievedChunk],
    *,
    target_tokens: int = PASSAGE_TOKENS,
    overlap_tokens: int = PASSAGE_OVERLAP,
    max_passages: int = MAX_PASSAGES,
    use_llm: bool = False,
) -> list[RetrievedChunk]:
    """Split each retrieved article into contextualized passages.

    Passage ids are positional and therefore stable across query variants, which is
    what lets rank fusion recognise the same passage retrieved by two different
    queries.
    """
    out: list[RetrievedChunk] = []
    for hit in hits:
        doc = _as_document(hit)
        if not doc.body.strip():
            out.append(hit)
            continue

        pieces = chunk_document(doc, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
        pieces = pieces[:max_passages]
        entities = salient_entities(f"{doc.title}\n{doc.body}")

        for i, piece in enumerate(pieces):
            if use_llm:
                context = llm_context(doc.title, doc.source, doc.published_at, doc.body, piece.text)
            else:
                context = heuristic_context(
                    doc.title, doc.source, doc.published_at, entities, i, len(pieces)
                )
            passage = Chunk(
                chunk_id=f"{doc.doc_id}::p{i:03d}",
                doc_id=doc.doc_id,
                text=piece.text,
                contextualized_text=f"{context}\n\n{piece.text}",
                title=doc.title,
                source=doc.source,
                source_tier=doc.source_tier,
                published_at=doc.published_at,
                url=doc.url,
                chunk_index=i,
                token_estimate=estimate_tokens(piece.text),
                metadata={
                    **(hit.chunk.metadata or {}),
                    "passage_count": len(pieces),
                    "context": context,
                },
            )
            out.append(
                RetrievedChunk(
                    chunk=passage,
                    score=hit.score,
                    rank=hit.rank,
                    method="factiva_contextual",
                    scores=dict(hit.scores),
                )
            )
    return out


def semantic_scores(query: str, hits: list[RetrievedChunk]) -> list[float]:
    """Cosine similarity over contextualized passages; empty list when unavailable.

    Returns [] rather than raising so the lexical stack stays the default and the
    pipeline never depends on an API key.
    """
    if not hits or not settings.openai_api_key:
        return []
    try:
        import numpy as np

        from taza_rag.llm import embed_texts

        vectors = embed_texts([query] + [h.chunk.index_text for h in hits])
    except Exception:
        return []
    if len(vectors) != len(hits) + 1:
        return []

    matrix = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    sims = matrix[1:] @ matrix[0]
    return [float(s) for s in sims]
