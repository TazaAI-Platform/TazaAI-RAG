from __future__ import annotations

import time
from typing import Any

from taza_rag.config import settings
from taza_rag.factiva.retrieve import FactivaRetrievalClient, hits_to_citations
from taza_rag.llm import chat_json
from taza_rag.models import AnswerResult, RetrievedChunk

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


def answer_with_factiva(
    query: str,
    *,
    limit: int = 10,
    days_range: str = "Last6Months",
    config_name: str = "factiva_retrieve",
) -> AnswerResult:
    """Retrieve via Factiva RAG API, then ground an answer with citations."""
    client = FactivaRetrievalClient()
    t0 = time.perf_counter()
    hits = client.retrieve(query, limit=limit, days_range=days_range)
    t1 = time.perf_counter()

    if not hits:
        return AnswerResult(
            query=query,
            answer="Insufficient evidence in Factiva retrieval results to answer.",
            citations=[],
            retrieved=[],
            abstained=True,
            latency_ms={"retrieve": (t1 - t0) * 1000, "total": (t1 - t0) * 1000},
            config_name=config_name,
        )

    context, selected = _pack_context(hits, settings.answer_max_chunks)
    user = f"Question: {query}\n\nSources:\n{context}"
    raw: dict[str, Any] = chat_json(ANSWER_SYSTEM, user, temperature=0.0)
    t2 = time.perf_counter()

    label_to_hit = {f"c{i}": h for i, h in enumerate(selected, start=1)}
    used = []
    for label in raw.get("used_citations") or []:
        hit = label and label_to_hit.get(str(label))
        if hit:
            used.append(hit)

    citations = hits_to_citations(used or selected[:3])
    return AnswerResult(
        query=query,
        answer=str(raw.get("answer") or ""),
        citations=citations,
        retrieved=selected,
        abstained=bool(raw.get("abstain")),
        latency_ms={
            "retrieve": (t1 - t0) * 1000,
            "generate": (t2 - t1) * 1000,
            "total": (t2 - t0) * 1000,
        },
        config_name=config_name,
    )
