from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from taza_rag.factiva.contextual import PASSAGE_TOKENS, semantic_scores, to_passages
from taza_rag.factiva.retrieve import FactivaRetrievalClient, FactivaRetrieveError
from taza_rag.factiva.strategy import (
    default_days_range,
    detect_intent,
    expand_queries,
    normalize_query,
)
from taza_rag.models import RetrievedChunk, SearchIntent
from taza_rag.retrieve.features import QueryPlan, build_query_plan
from taza_rag.retrieve.quality import diversity_cap, mmr_diversify, rank_candidates


@dataclass
class RetrievalRun:
    query: str
    intent: SearchIntent
    variants: list[str]
    hits: list[RetrievedChunk]
    plan: QueryPlan | None = None
    candidates: int = 0
    passages: int = 0
    failed_variants: list[str] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    config: str = "factiva_quality_v2"


def _source_cap(intent: SearchIntent) -> int:
    """Survey-style intents need breadth; entity asks tolerate depth from one outlet."""
    if intent in {
        SearchIntent.TOPICAL_EXPLORATION,
        SearchIntent.GEOGRAPHIC_ASSESSMENT,
        SearchIntent.COMPETITIVE_INTEL,
    }:
        return 2
    return 3


class QualityRetriever:
    """Retrieval-quality pipeline over Factiva — no OpenAI required."""

    def __init__(self, client: FactivaRetrievalClient | None = None) -> None:
        self.client = client or FactivaRetrievalClient()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        per_variant_limit: int = 20,
        intent: SearchIntent | None = None,
        days_range: str | None = None,
        max_variants: int = 3,
        diversity: bool = True,
        max_per_source: int | None = None,
        entity_gate: bool = True,
        contextual: bool = True,
        llm_context: bool = False,
        semantic: bool = False,
        passage_tokens: int = PASSAGE_TOKENS,
    ) -> RetrievalRun:
        intent = intent or detect_intent(query)
        variants = expand_queries(query, intent, max_variants=max_variants)
        window = days_range or default_days_range(intent)
        # Plan from the normalized text, otherwise "Deutche Bank" never matches
        # "Deutsche" in the documents and the entity signals silently go to zero.
        plan = build_query_plan(normalize_query(query), intent)

        # Warm the token once so parallel calls do not each run the OAuth exchange.
        self.client.auth.get_access_token()

        t0 = time.perf_counter()
        rankings: list[list[RetrievedChunk]] = []
        failed: list[str] = []
        with ThreadPoolExecutor(max_workers=min(len(variants), 4)) as pool:
            futures = {
                pool.submit(
                    self.client.retrieve, v, limit=per_variant_limit, days_range=window
                ): v
                for v in variants
            }
            for future, variant in futures.items():
                try:
                    rankings.append(future.result())
                except FactivaRetrieveError:
                    # One flaky variant must not sink the whole pack.
                    failed.append(variant)
        t1 = time.perf_counter()

        if not rankings:
            raise FactivaRetrieveError(
                f"All {len(variants)} query variants failed for {query!r}"
            )

        candidate_ids = {h.chunk.doc_id for r in rankings for h in r}

        # Contextual retrieval: rank contextualized passages rather than whole articles.
        if contextual:
            rankings = [
                to_passages(r, use_llm=llm_context, target_tokens=passage_tokens)
                for r in rankings
            ]
        passage_ids = {h.chunk.chunk_id for r in rankings for h in r}
        t_ctx = time.perf_counter()

        semantic_used = False
        if semantic:
            unique: dict[str, RetrievedChunk] = {}
            for r in rankings:
                for h in r:
                    unique.setdefault(h.chunk.chunk_id, h)
            sims = semantic_scores(query, list(unique.values()))
            if sims:
                by_key = dict(zip(unique.keys(), sims))
                for r in rankings:
                    for h in r:
                        h.scores["semantic_pre"] = by_key.get(h.chunk.chunk_id, 0.0)
                semantic_used = True

        ranked = rank_candidates(
            plan,
            rankings,
            top_k=top_k * 3,
            entity_gate=entity_gate,
        )
        if diversity:
            cap = max_per_source if max_per_source is not None else _source_cap(intent)
            ranked = diversity_cap(ranked, max_per_source=cap)
            ranked = mmr_diversify(ranked, top_k=top_k)
        ranked = ranked[:top_k]
        for i, h in enumerate(ranked, start=1):
            h.rank = i
        t2 = time.perf_counter()

        config = "factiva_quality_v2"
        if contextual:
            config += "+ctx_llm" if llm_context else "+ctx"
        if semantic_used:
            config += "+semantic"

        return RetrievalRun(
            query=query,
            intent=intent,
            variants=variants,
            hits=ranked,
            plan=plan,
            candidates=len(candidate_ids),
            passages=len(passage_ids),
            failed_variants=failed,
            latency_ms={
                "factiva_multi": (t1 - t0) * 1000,
                "contextualize": (t_ctx - t1) * 1000,
                "rank": (t2 - t_ctx) * 1000,
                "total": (t2 - t0) * 1000,
            },
            config=config,
        )
