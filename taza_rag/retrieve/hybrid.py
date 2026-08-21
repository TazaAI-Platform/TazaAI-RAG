from __future__ import annotations

from collections import defaultdict

from taza_rag.config import settings
from taza_rag.index.store import HybridIndex
from taza_rag.models import RetrievedChunk

SOURCE_TIER_WEIGHT = {"premium": 1.08, "standard": 1.0, "wire": 0.95}


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking, start=1):
            scores[idx] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def _freshness_boost(published_at: str | None) -> float:
    if not published_at:
        return 1.0
    # Soft preference for newer content; ISO dates sort lexicographically for YYYY-MM-DD
    try:
        year = int(published_at[:4])
        if year >= 2025:
            return 1.06
        if year >= 2024:
            return 1.03
        if year >= 2022:
            return 1.0
        return 0.97
    except ValueError:
        return 1.0


def hybrid_retrieve(
    index: HybridIndex,
    query: str,
    dense_k: int | None = None,
    sparse_k: int | None = None,
    fuse_k: int | None = None,
    apply_marketplace_weights: bool = True,
) -> list[RetrievedChunk]:
    dense_k = dense_k or settings.retrieve_dense_k
    sparse_k = sparse_k or settings.retrieve_sparse_k
    fuse_k = fuse_k or settings.retrieve_fuse_k

    dense = index.dense_search(query, k=dense_k)
    sparse = index.sparse_search(query, k=sparse_k)
    fused = reciprocal_rank_fusion([dense, sparse])[:fuse_k]

    dense_map = {i: s for i, s in dense}
    sparse_map = {i: s for i, s in sparse}

    results: list[RetrievedChunk] = []
    for rank, (idx, rrf) in enumerate(fused, start=1):
        chunk = index.chunks[idx]
        score = rrf
        if apply_marketplace_weights:
            score *= SOURCE_TIER_WEIGHT.get(chunk.source_tier, 1.0)
            score *= _freshness_boost(chunk.published_at)
        results.append(
            RetrievedChunk(
                chunk=chunk,
                score=score,
                rank=rank,
                method="hybrid_rrf",
                scores={
                    "rrf": rrf,
                    "dense": dense_map.get(idx, 0.0),
                    "bm25": sparse_map.get(idx, 0.0),
                },
            )
        )
    results.sort(key=lambda r: -r.score)
    for i, r in enumerate(results, start=1):
        r.rank = i
    return results


def simple_rerank(query: str, hits: list[RetrievedChunk], top_k: int | None = None) -> list[RetrievedChunk]:
    """Lightweight lexical overlap rerank (placeholder until Cohere/Voyage reranker wired)."""
    top_k = top_k or settings.rerank_top_k
    q_tokens = set(query.lower().split())

    def overlap(h: RetrievedChunk) -> float:
        t = set(h.chunk.index_text.lower().split())
        if not q_tokens:
            return 0.0
        return len(q_tokens & t) / len(q_tokens)

    rescored = []
    for h in hits:
        lex = overlap(h)
        new_score = 0.75 * h.score + 0.25 * lex
        rescored.append(
            RetrievedChunk(
                chunk=h.chunk,
                score=new_score,
                rank=h.rank,
                method="hybrid+lex_rerank",
                scores={**h.scores, "lex_overlap": lex},
            )
        )
    rescored.sort(key=lambda r: -r.score)
    out = rescored[:top_k]
    for i, r in enumerate(out, start=1):
        r.rank = i
    return out
