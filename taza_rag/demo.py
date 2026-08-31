"""Hosted playground over the sample corpus — no Factiva account, no OpenAI key.

The product loop is unchanged: query is free, packages are opaque, transact pays,
fetch_content is the only way to read a body. Ranking is the same quality stack as
live Factiva, pointed at `data/sample_corpus/articles.jsonl`. Answers are extractive:
cited lead sentences, nothing paraphrased.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from taza_rag.factiva.contextual import to_passages
from taza_rag.factiva.pipeline import RetrievalRun, _source_cap
from taza_rag.factiva.strategy import detect_intent, expand_queries, normalize_query
from taza_rag.ingest.corpus import load_corpus_jsonl
from taza_rag.models import AnswerResult, Chunk, Citation, RetrievedChunk, SearchIntent
from taza_rag.retrieve.features import build_query_plan, words
from taza_rag.retrieve.quality import diversity_cap, mmr_diversify, rank_candidates
from taza_rag.ui.serialize import answer_payload, research_payload, run_payload

DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / "data" / "sample_corpus" / "articles.jsonl"


def _article_hit(doc, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=doc.doc_id,
            doc_id=doc.doc_id,
            text=doc.body,
            title=doc.title,
            source=doc.source,
            source_tier=doc.source_tier,
            published_at=doc.published_at,
            url=doc.url,
            chunk_index=0,
            token_estimate=max(1, len((doc.body or "").split())),
            metadata={"entities": doc.entities, "doc_kind": "article"},
        ),
        score=1.0,
        rank=rank,
        method="demo",
        scores={"api_rank": float(rank)},
    )


class DemoIndex:
    """BM25 over contextualized sample passages, then the live quality ranker."""

    def __init__(self, corpus: Path | None = None) -> None:
        path = corpus or DEFAULT_CORPUS
        docs = load_corpus_jsonl(path)
        articles = [_article_hit(doc, i) for i, doc in enumerate(docs, start=1)]
        self.passages = to_passages(articles)
        self._bm25 = BM25Okapi([words(p.chunk.index_text) for p in self.passages])

    def search(
        self, query: str, *, top_k: int = 10, intent: SearchIntent | None = None
    ) -> list[RetrievedChunk]:
        return self.retrieve(query, top_k=top_k, intent=intent).hits

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        intent: SearchIntent | None = None,
        raw: bool = False,
        max_variants: int = 3,
        diversity: bool = True,
    ) -> RetrievalRun:
        intent = intent or detect_intent(query)
        variants = [query] if raw else expand_queries(query, intent, max_variants=max_variants)
        plan = build_query_plan(normalize_query(query), intent)
        t0 = time.perf_counter()
        rankings = [self._rank_variant(v) for v in variants]
        rankings = [r for r in rankings if r]
        if not rankings:
            return RetrievalRun(
                query=query,
                intent=intent,
                variants=variants,
                hits=[],
                plan=plan,
                candidates=len({p.chunk.doc_id for p in self.passages}),
                passages=len(self.passages),
                config="demo_sample+empty",
                latency_ms={"total": (time.perf_counter() - t0) * 1000},
            )
        ranked = rank_candidates(plan, rankings, top_k=max(top_k * 3, top_k), entity_gate=not raw)
        if diversity and not raw:
            ranked = diversity_cap(ranked, max_per_source=_source_cap(intent))
            ranked = mmr_diversify(ranked, top_k=top_k)
        ranked = ranked[:top_k]
        for i, hit in enumerate(ranked, start=1):
            hit.rank = i
        elapsed = (time.perf_counter() - t0) * 1000
        return RetrievalRun(
            query=query,
            intent=intent,
            variants=variants,
            hits=ranked,
            plan=plan,
            candidates=len({p.chunk.doc_id for p in self.passages}),
            passages=len(self.passages),
            latency_ms={"rank": elapsed, "total": elapsed},
            config="demo_sample+ctx" if not raw else "demo_sample+raw",
        )

    def _rank_variant(self, variant: str) -> list[RetrievedChunk]:
        tokens = words(variant)
        if not tokens:
            return []
        scores = list(self._bm25.get_scores(tokens))
        order = sorted(range(len(scores)), key=lambda i: -float(scores[i]))
        out: list[RetrievedChunk] = []
        for rank, idx in enumerate(order, start=1):
            if scores[idx] <= 0:
                continue
            base = self.passages[idx]
            hit = base.model_copy(deep=True)
            hit.rank = rank
            hit.score = float(scores[idx])
            hit.method = "demo"
            hit.scores = {**dict(base.scores), "api_rank": float(rank)}
            out.append(hit)
        return out[:40]


def extractive_answer(query: str, hits: list[RetrievedChunk]) -> dict[str, Any]:
    from taza_rag.factiva.facts import extractive_compose, extractive_facts

    evidence = {
        f"c{i}": f"{h.chunk.title}\n{h.chunk.text}" for i, h in enumerate(hits, start=1)
    }
    facts = extractive_facts(evidence)
    composed = extractive_compose(facts)
    text = str(composed.get("answer") or "").strip()
    abstained = bool(composed.get("abstain")) or not text
    if abstained:
        text = "Insufficient evidence in the sample corpus to answer this question."
    used = set(composed.get("used_citations") or [])
    citations: list[Citation] = []
    for i, hit in enumerate(hits, start=1):
        label = f"c{i}"
        if used and label not in used:
            continue
        c = hit.chunk
        citations.append(
            Citation(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                title=c.title,
                source=c.source,
                published_at=c.published_at,
                url=c.url,
                excerpt=(c.text or "")[:280],
            )
        )
    result = AnswerResult(
        query=query,
        answer=text,
        citations=citations,
        retrieved=hits,
        abstained=abstained,
        config_name="demo_extractive",
    )
    payload = answer_payload(result)
    payload["usage"]["llm_calls"] = 0
    payload["usage"]["cited"] = len(citations)
    return payload


def _clamp(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def demo_handlers(corpus: Path | None = None) -> dict[str, Any]:
    """Wire the UI/MCP marketplace onto the sample corpus."""
    from taza_rag.agent.gather import MarketBackend
    from taza_rag.agent.loop import research as run_research
    from taza_rag.agent.models import Budget
    from taza_rag.market import Market

    index = DemoIndex(corpus)
    market = Market(search=index.search)

    def retrieve_fn(query: str, *, top_k: int, raw: bool) -> dict[str, Any]:
        return run_payload(index.retrieve(query, top_k=top_k, raw=raw))

    def write_fn(query: str, hits: list[RetrievedChunk]) -> dict[str, Any]:
        return extractive_answer(query, hits)

    def research_fn(query: str, body: dict[str, Any]) -> dict[str, Any]:
        budget = Budget(
            max_rounds=_clamp(body.get("max_rounds"), 1, 6, 3),
            max_unique_chunks=_clamp(body.get("max_chunks"), 4, 200, 40),
            max_sub_questions=_clamp(body.get("max_sub"), 1, 8, 5),
            top_k_per_query=_clamp(body.get("top_k"), 1, 20, 6),
            purchase_gate=bool(body.get("purchase_gate", True)),
        )
        result = run_research(
            query,
            backend=MarketBackend(market=market),
            budget=budget,
            verify=False,
            use_llm_plan=False,
            extractive=True,
        )
        return research_payload(result)

    return {
        "market": market,
        "retrieve_fn": retrieve_fn,
        "write_fn": write_fn,
        "research_fn": research_fn,
    }
