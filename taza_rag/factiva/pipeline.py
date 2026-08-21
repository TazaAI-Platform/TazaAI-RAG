from __future__ import annotations

import time
from dataclasses import dataclass, field

from taza_rag.factiva.retrieve import FactivaRetrievalClient
from taza_rag.factiva.strategy import default_days_range, detect_intent, expand_queries
from taza_rag.models import RetrievedChunk, SearchIntent
from taza_rag.retrieve.quality import diversity_cap, fuse_and_rerank


@dataclass
class RetrievalRun:
    query: str
    intent: SearchIntent
    variants: list[str]
    hits: list[RetrievedChunk]
    latency_ms: dict[str, float] = field(default_factory=dict)
    config: str = "factiva_quality_v1"


class QualityRetriever:
    """Retrieval-quality pipeline over Factiva — no OpenAI required."""

    def __init__(self, client: FactivaRetrievalClient | None = None) -> None:
        self.client = client or FactivaRetrievalClient()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        per_variant_limit: int = 10,
        intent: SearchIntent | None = None,
        days_range: str | None = None,
        max_variants: int = 3,
        diversity: bool = True,
    ) -> RetrievalRun:
        intent = intent or detect_intent(query)
        variants = expand_queries(query, intent)[:max_variants]
        window = days_range or default_days_range(intent)

        t0 = time.perf_counter()
        rankings: list[list[RetrievedChunk]] = []
        for variant in variants:
            hits = self.client.retrieve(
                variant,
                limit=per_variant_limit,
                days_range=window,
            )
            rankings.append(hits)
        t1 = time.perf_counter()

        fused = fuse_and_rerank(query, rankings, top_k=top_k * 2)
        if diversity:
            fused = diversity_cap(fused, max_per_source=3)
        fused = fused[:top_k]
        t2 = time.perf_counter()

        return RetrievalRun(
            query=query,
            intent=intent,
            variants=variants,
            hits=fused,
            latency_ms={
                "factiva_multi": (t1 - t0) * 1000,
                "fuse_rerank": (t2 - t1) * 1000,
                "total": (t2 - t0) * 1000,
            },
        )
