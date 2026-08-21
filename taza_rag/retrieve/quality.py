from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, datetime

from taza_rag.models import RetrievedChunk

# Soft authority prior for Dow Jones / Factiva ecosystem (tunable)
SOURCE_AUTHORITY: dict[str, float] = {
    "wsj": 1.12,
    "wsjo": 1.12,
    "j": 1.10,  # Wall Street Journal often coded J
    "djdn": 1.10,
    "dj": 1.08,
    "ft": 1.08,
    "bloomberg": 1.07,
    "reuters": 1.05,
    "barrons": 1.06,
    "marketwatch": 1.03,
}

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-\.]{1,}", re.I)


def tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text or "")}


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]],
    k: int = 60,
) -> dict[str, float]:
    """RRF over chunk/doc keys across query variants."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = hit.chunk.doc_id
            scores[key] += 1.0 / (k + rank)
    return scores


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def freshness_score(published_at: str | None, today: date | None = None) -> float:
    today = today or date.today()
    d = _parse_date(published_at)
    if not d:
        return 1.0
    age_days = max(0, (today - d).days)
    if age_days <= 7:
        return 1.12
    if age_days <= 30:
        return 1.08
    if age_days <= 90:
        return 1.04
    if age_days <= 180:
        return 1.0
    if age_days <= 365:
        return 0.96
    return 0.92


def authority_score(hit: RetrievedChunk) -> float:
    code = str((hit.chunk.metadata or {}).get("source_code") or "").lower()
    if code in SOURCE_AUTHORITY:
        return SOURCE_AUTHORITY[code]
    name = (hit.chunk.source or "").lower()
    for key, weight in SOURCE_AUTHORITY.items():
        if key in name.replace(" ", ""):
            return weight
    if hit.chunk.source_tier == "premium":
        return 1.05
    if hit.chunk.source_tier == "wire":
        return 0.95
    return 1.0


def lexical_overlap(query: str, hit: RetrievedChunk) -> float:
    q = tokenize(query)
    if not q:
        return 0.0
    blob = f"{hit.chunk.title}\n{hit.chunk.text}"
    t = tokenize(blob)
    return len(q & t) / len(q)


def dedupe_by_doc(hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
    best: dict[str, RetrievedChunk] = {}
    for h in hits:
        key = h.chunk.doc_id
        if key not in best or h.score > best[key].score:
            best[key] = h
    return list(best.values())


def fuse_and_rerank(
    query: str,
    rankings: list[list[RetrievedChunk]],
    *,
    top_k: int = 10,
    w_rrf: float = 1.0,
    w_lex: float = 0.35,
    w_auth: float = 0.25,
    w_fresh: float = 0.20,
) -> list[RetrievedChunk]:
    """Merge multi-query Factiva rankings with quality-oriented rerank (no LLM)."""
    flat: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for h in ranking:
            prev = flat.get(h.chunk.doc_id)
            if prev is None or h.score > prev.score:
                flat[h.chunk.doc_id] = h

    rrf = reciprocal_rank_fusion(rankings)
    scored: list[RetrievedChunk] = []
    for doc_id, base in flat.items():
        rrf_s = rrf.get(doc_id, 0.0)
        lex = lexical_overlap(query, base)
        auth = authority_score(base)
        fresh = freshness_score(base.chunk.published_at)
        # Normalize soft factors around 1.0
        final = (
            w_rrf * rrf_s
            + w_lex * lex
            + w_auth * (auth - 1.0)
            + w_fresh * (fresh - 1.0)
        )
        scored.append(
            RetrievedChunk(
                chunk=base.chunk,
                score=final,
                rank=0,
                method="factiva_quality_fuse",
                scores={
                    "rrf": rrf_s,
                    "lex": lex,
                    "authority": auth,
                    "freshness": fresh,
                    "api_rank": base.scores.get("api_rank", 0.0),
                },
            )
        )

    scored.sort(key=lambda h: (-h.score, h.chunk.published_at or ""))
    out = scored[:top_k]
    for i, h in enumerate(out, start=1):
        h.rank = i
    return out


def diversity_cap(hits: list[RetrievedChunk], max_per_source: int = 3) -> list[RetrievedChunk]:
    """Avoid one wire flooding the top-k — helps Completeness / source mix."""
    counts: dict[str, int] = defaultdict(int)
    kept: list[RetrievedChunk] = []
    for h in hits:
        src = str((h.chunk.metadata or {}).get("source_code") or h.chunk.source)
        if counts[src] >= max_per_source:
            continue
        counts[src] += 1
        kept.append(h)
    for i, h in enumerate(kept, start=1):
        h.rank = i
    return kept
