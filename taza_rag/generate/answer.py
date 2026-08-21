from __future__ import annotations

import time
from typing import Any

from taza_rag.config import settings
from taza_rag.index.store import HybridIndex
from taza_rag.llm import chat_json
from taza_rag.models import AnswerResult, Citation, RetrievedChunk
from taza_rag.retrieve.hybrid import hybrid_retrieve, simple_rerank

ANSWER_SYSTEM = """You are a Dow Jones / Factiva-aligned research assistant.
Answer ONLY using the provided source chunks. Rules:
- Every significant claim must include a citation marker like [c1], [c2] matching chunk labels.
- Do not invent facts, numbers, names, or dates not present in sources.
- Prefer higher-authority / premium sources when details conflict; mention contradictions explicitly.
- Be direct, salient, and professionally journalistic (Dow Jones Voice).
- If evidence is insufficient, set abstain=true and explain what is missing.
Return JSON with keys: answer (string), abstain (boolean), used_citations (list of chunk labels like "c1").
"""


def _pack_context(hits: list[RetrievedChunk], max_chunks: int) -> tuple[str, list[RetrievedChunk]]:
    selected = hits[:max_chunks]
    blocks = []
    for i, h in enumerate(selected, start=1):
        c = h.chunk
        blocks.append(
            f"[c{i}] doc_id={c.doc_id} | {c.source} | {c.published_at or 'n/a'} | {c.title}\n{c.text}"
        )
    return "\n\n".join(blocks), selected


def answer_query(
    index: HybridIndex,
    query: str,
    config_name: str = "contextual_hybrid",
    use_rerank: bool = True,
    marketplace_weights: bool = True,
) -> AnswerResult:
    t0 = time.perf_counter()
    hits = hybrid_retrieve(index, query, apply_marketplace_weights=marketplace_weights)
    t1 = time.perf_counter()
    if use_rerank:
        hits = simple_rerank(query, hits)
    t2 = time.perf_counter()

    context, selected = _pack_context(hits, settings.answer_max_chunks)
    user = f"Question: {query}\n\nSources:\n{context}"
    raw: dict[str, Any] = chat_json(ANSWER_SYSTEM, user, temperature=0.0)
    t3 = time.perf_counter()

    label_to_chunk = {f"c{i}": h.chunk for i, h in enumerate(selected, start=1)}
    citations: list[Citation] = []
    for label in raw.get("used_citations") or []:
        chunk = label_to_chunk.get(str(label))
        if not chunk:
            continue
        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=chunk.title,
                source=chunk.source,
                published_at=chunk.published_at,
                url=chunk.url,
                excerpt=chunk.text[:280],
            )
        )

    return AnswerResult(
        query=query,
        answer=str(raw.get("answer") or ""),
        citations=citations,
        retrieved=selected,
        abstained=bool(raw.get("abstain")),
        latency_ms={
            "retrieve": (t1 - t0) * 1000,
            "rerank": (t2 - t1) * 1000,
            "generate": (t3 - t2) * 1000,
            "total": (t3 - t0) * 1000,
        },
        config_name=config_name,
    )
