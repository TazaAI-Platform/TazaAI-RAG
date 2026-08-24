from __future__ import annotations

import time
from typing import Any

from taza_rag.config import settings
from taza_rag.factiva.pipeline import QualityRetriever
from taza_rag.factiva.retrieve import FactivaRetrievalClient, hits_to_citations
from taza_rag.llm import chat_json
from taza_rag.models import AnswerResult, RetrievedChunk, SearchIntent

ANSWER_SYSTEM = """You are a Dow Jones / Factiva-aligned research assistant.
Answer ONLY using the provided source chunks. Rules:
- Every significant claim must include a citation marker like [c1], [c2] matching chunk labels.
- Do not invent facts, numbers, names, or dates not present in sources.
- Prefer higher-authority / premium sources when details conflict; mention contradictions explicitly.
- Be direct, salient, and professionally journalistic (Dow Jones Voice).
- Carry over the concrete figures the sources give (amounts, percentages, dates, counterparties)
  rather than describing them in general terms.
- Where the sources disagree or include a dissenting, cautionary or contrary view, say so and
  attribute it. Never manufacture a disagreement that the sources do not contain.
- State the significance of the facts only so far as the sources support it; do not speculate.
- If evidence is insufficient, set abstain=true and explain what is missing.
Return JSON with keys: answer (string), abstain (boolean), used_citations (list of chunk labels like "c1").
"""


def _pack_context(
    hits: list[RetrievedChunk], max_chunks: int, max_tokens: int
) -> tuple[str, list[RetrievedChunk]]:
    """Fill the context to a token budget, not a chunk count.

    Passages are roughly half the size of a whole article, so a fixed chunk count
    silently hands the generator half the evidence and costs Completeness. Budgeting
    by tokens lets the passage path spend its saving on more distinct sources instead.
    """
    selected: list[RetrievedChunk] = []
    used = 0
    for h in hits[:max_chunks]:
        tokens = len((h.chunk.text or "").split())
        if selected and used + tokens > max_tokens:
            break
        selected.append(h)
        used += tokens

    blocks = []
    for i, h in enumerate(selected, start=1):
        c = h.chunk
        blocks.append(
            f"[c{i}] doc_id={c.doc_id} | {c.source} | {c.published_at or 'n/a'} | {c.title}\n{c.text}"
        )
    return "\n\n".join(blocks), selected


def _abstain(query: str, config_name: str, elapsed_ms: float) -> AnswerResult:
    return AnswerResult(
        query=query,
        answer="Insufficient evidence in Factiva retrieval results to answer.",
        citations=[],
        retrieved=[],
        abstained=True,
        latency_ms={"retrieve": elapsed_ms, "total": elapsed_ms},
        config_name=config_name,
    )


def answer_with_factiva(
    query: str,
    *,
    top_k: int = 8,
    days_range: str | None = None,
    intent: SearchIntent | None = None,
    raw: bool = False,
    contextual: bool = True,
    semantic: bool = False,
    config_name: str | None = None,
) -> AnswerResult:
    """Retrieve from Factiva, then ground an answer with citations.

    Retrieval defaults to the full quality stack, so A1 scores describe the system
    that is actually shipped. `raw=True` drops to a single Factiva call in API order,
    which is the baseline the ranking work has to beat at the answer level too.
    """
    t0 = time.perf_counter()
    if raw:
        hits = FactivaRetrievalClient().retrieve(
            query, limit=top_k, days_range=days_range or "Last6Months"
        )
        used_config = config_name or "factiva_raw"
    else:
        run = QualityRetriever().retrieve(
            query,
            top_k=top_k,
            intent=intent,
            days_range=days_range,
            contextual=contextual,
            semantic=semantic,
        )
        hits = run.hits
        used_config = config_name or run.config
    t1 = time.perf_counter()

    if not hits:
        return _abstain(query, used_config, (t1 - t0) * 1000)

    context, selected = _pack_context(
        hits, settings.answer_max_chunks, settings.answer_context_tokens
    )
    user = f"Question: {query}\n\nSources:\n{context}"
    raw_json: dict[str, Any] = chat_json(ANSWER_SYSTEM, user, temperature=0.0)
    t2 = time.perf_counter()

    label_to_hit = {f"c{i}": h for i, h in enumerate(selected, start=1)}
    used = []
    for label in raw_json.get("used_citations") or []:
        hit = label and label_to_hit.get(str(label))
        if hit:
            used.append(hit)

    citations = hits_to_citations(used or selected[:3])
    return AnswerResult(
        query=query,
        answer=str(raw_json.get("answer") or ""),
        citations=citations,
        retrieved=selected,
        context=context,
        abstained=bool(raw_json.get("abstain")),
        latency_ms={
            "retrieve": (t1 - t0) * 1000,
            "generate": (t2 - t1) * 1000,
            "total": (t2 - t0) * 1000,
        },
        config_name=used_config,
    )
