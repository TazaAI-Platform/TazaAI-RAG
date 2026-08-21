from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime

from rank_bm25 import BM25Okapi

from taza_rag.models import RetrievedChunk, SearchIntent
from taza_rag.retrieve.features import (
    NEWS_INTENTS,
    QueryPlan,
    jaccard,
    build_query_plan,
    content_terms,
    doc_kind,
    entity_signal,
    kind_penalty,
    near_duplicate,
    topic_signal,
    words,
)

# Soft authority prior for the Dow Jones / Factiva ecosystem (tunable)
SOURCE_AUTHORITY: dict[str, float] = {
    "wsj": 1.12,
    "wsjo": 1.12,
    "j": 1.10,  # WSJ print edition is often coded J
    "djdn": 1.10,
    "dj": 1.08,
    "ft": 1.08,
    "bloomberg": 1.07,
    "barrons": 1.06,
    "reuters": 1.05,
    "marketwatch": 1.03,
}

# Composite score weights (within-tier ordering)
W_RRF = 0.9
W_BM25 = 1.0
W_ENTITY = 1.0
W_TOPIC = 1.2
W_AUTH = 0.8
W_FRESH = 0.6
W_SEMANTIC = 0.9
# Evidence from the top of an article outranks the same wording buried further down.
W_POSITION = 0.06
MAX_POSITION_PENALTY = 4


def tokenize(text: str) -> set[str]:
    return set(words(text))


def candidate_key(hit: RetrievedChunk) -> str:
    """Fusion key. Passage ids are positional, so they are stable across variants."""
    return hit.chunk.chunk_id or hit.chunk.doc_id


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]],
    k: int = 60,
) -> dict[str, float]:
    """RRF over candidates across query variants."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[candidate_key(hit)] += 1.0 / (k + rank)
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
    q = set(content_terms(query))
    if not q:
        return 0.0
    t = tokenize(f"{hit.chunk.title}\n{hit.chunk.text}")
    return len(q & t) / len(q)


def dedupe_by_doc(hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
    best: dict[str, RetrievedChunk] = {}
    for h in hits:
        key = h.chunk.doc_id
        if key not in best or h.score > best[key].score:
            best[key] = h
    return list(best.values())


TIER_HEADLINE = 0  # entity present and the topic is in the headline/lead
TIER_BODY = 1  # entity present, topic only mentioned deeper in the body
TIER_ENTITY_ONLY = 2  # entity present, topic absent
TIER_OFF_ENTITY = 3  # entity absent


def relevance_tier(plan: QueryPlan, entity: dict[str, float], topic: dict[str, float]) -> int:
    """Rank by where the answer to the ask actually lives.

    Tiering keeps ordering faithful to the query: for "Deutsche Bank restructuring" a
    buyback story names the entity but answers a different question, and an HQ-raid
    story that happens to mention an overhaul in paragraph nine is weaker evidence
    than a headline about the overhaul itself.
    """
    if plan.entity_tokens and entity["entity_any"] <= 0.0:
        return TIER_OFF_ENTITY

    # A query naming two entities is only fully answered by evidence naming both, so
    # missing one costs the top tier even when the topic is in the headline.
    multi_entity = len(plan.entity_tokens) > 1
    floor = TIER_BODY if multi_entity and entity["entity_all"] <= 0.0 else TIER_HEADLINE

    if not plan.topics:
        return floor
    if max(topic["topic_title"], topic["topic_lead"]) > 0.0:
        return floor
    if topic["topic_strong"] > 0.0:
        return max(TIER_BODY, floor)
    return TIER_ENTITY_ONLY


def _bm25_scores(plan: QueryPlan, candidates: list[RetrievedChunk]) -> list[float]:
    """BM25 over the candidate pool; title weighted by repetition.

    Uses `index_text`, so when contextual retrieval is on the situating prefix
    contributes its document anchors (source, date, subject) to lexical matching.
    """
    corpus = []
    for h in candidates:
        title_tokens = content_terms(h.chunk.title)
        body_tokens = content_terms(h.chunk.index_text)
        corpus.append(title_tokens * 3 + body_tokens)
    if not corpus:
        return []
    query_tokens = [t for t in plan.search_terms if t]
    if not query_tokens:
        return [0.0] * len(candidates)
    bm25 = BM25Okapi(corpus)
    raw = list(bm25.get_scores(query_tokens))
    top = max(raw) if raw else 0.0
    if top <= 0:
        return [0.0] * len(candidates)
    return [s / top for s in raw]


def rank_candidates(
    plan: QueryPlan,
    rankings: list[list[RetrievedChunk]],
    *,
    top_k: int = 10,
    entity_gate: bool = True,
    drop_near_duplicates: bool = True,
    one_per_doc: bool = True,
) -> list[RetrievedChunk]:
    """Score, gate, and order Factiva candidates for retrieval quality.

    Candidates may be whole articles or contextualized passages; `one_per_doc` keeps
    only each document's best passage so the pack does not spend its budget quoting
    the same story twice.
    """
    pool: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for h in ranking:
            key = candidate_key(h)
            prev = pool.get(key)
            if prev is None or (h.scores.get("api_rank", 99) < prev.scores.get("api_rank", 99)):
                pool[key] = h

    candidates = list(pool.values())
    if not candidates:
        return []

    rrf = reciprocal_rank_fusion(rankings)
    rrf_top = max(rrf.values()) if rrf else 0.0
    bm25 = _bm25_scores(plan, candidates)

    scored: list[RetrievedChunk] = []
    for i, base in enumerate(candidates):
        ent = entity_signal(plan, base)
        top = topic_signal(plan, base)
        kind = doc_kind(base)
        auth = authority_score(base)
        fresh = freshness_score(base.chunk.published_at)

        key = candidate_key(base)
        rrf_n = (rrf.get(key, 0.0) / rrf_top) if rrf_top else 0.0
        bm25_n = bm25[i] if i < len(bm25) else 0.0
        # Attached upstream when embeddings are enabled; absent by default.
        sem = base.scores.get("semantic_pre", 0.0)
        position_penalty = W_POSITION * min(base.chunk.chunk_index, MAX_POSITION_PENALTY)

        # Entity presence matters most in the title, then the lead paragraph.
        entity_component = (
            0.60 * ent["entity_title"]
            + 0.30 * ent["entity_lead"]
            + 0.10 * ent["entity_body"]
            + 0.15 * ent["entity_extra"]
        )
        topic_component = 0.6 * top["topic_title"] + 0.4 * top["topic_body"]

        penalty = kind_penalty(kind, plan.intent)
        tier = relevance_tier(plan, ent, top)

        score = (
            W_RRF * rrf_n
            + W_BM25 * bm25_n
            + W_SEMANTIC * sem
            + W_ENTITY * entity_component
            + W_TOPIC * topic_component
            + W_AUTH * (auth - 1.0)
            + W_FRESH * (fresh - 1.0)
            - penalty
            - position_penalty
        )

        scored.append(
            RetrievedChunk(
                chunk=base.chunk,
                score=score,
                rank=0,
                method="factiva_quality_v2",
                scores={
                    "tier": float(tier),
                    "rrf": rrf_n,
                    "bm25": bm25_n,
                    "semantic": sem,
                    "entity": entity_component,
                    "topic": topic_component,
                    "authority": auth,
                    "freshness": fresh,
                    "penalty": penalty,
                    "position": position_penalty,
                    "api_rank": base.scores.get("api_rank", 0.0),
                },
            )
        )
        # doc kind is useful for eval/debugging, keep it on the chunk metadata
        scored[-1].chunk.metadata = {**(base.chunk.metadata or {}), "doc_kind": kind}

    # Tier first (does it answer the ask), then composite score inside the tier.
    scored.sort(key=lambda h: (h.scores["tier"], -h.score, h.chunk.published_at or ""))

    # Keep each document's strongest passage; the rest is redundant evidence.
    if one_per_doc:
        best: dict[str, RetrievedChunk] = {}
        for h in scored:
            if h.chunk.doc_id not in best:
                best[h.chunk.doc_id] = h
        scored = sorted(
            best.values(),
            key=lambda h: (h.scores["tier"], -h.score, h.chunk.published_at or ""),
        )

    # Entity gate: for entity-style asks, evidence that never names the entity is noise.
    if entity_gate and plan.entity_tokens and plan.intent in NEWS_INTENTS:
        on_entity = [h for h in scored if h.scores["tier"] < TIER_OFF_ENTITY]
        if len(on_entity) >= min(top_k, 3):
            scored = on_entity

    if drop_near_duplicates:
        kept: list[RetrievedChunk] = []
        for h in scored:
            if any(near_duplicate(h, k) for k in kept):
                continue
            kept.append(h)
            if len(kept) >= top_k:
                break
        scored = kept

    out = scored[:top_k]
    for i, h in enumerate(out, start=1):
        h.rank = i
    return out


def fuse_and_rerank(
    query: str,
    rankings: list[list[RetrievedChunk]],
    *,
    top_k: int = 10,
    intent: SearchIntent | None = None,
) -> list[RetrievedChunk]:
    """Convenience wrapper that builds a query plan from the raw query string."""
    from taza_rag.factiva.strategy import detect_intent

    resolved = intent or detect_intent(query)
    plan = build_query_plan(query, resolved)
    return rank_candidates(plan, rankings, top_k=top_k)


def _content_signature(hit: RetrievedChunk) -> set[str]:
    return set(content_terms(f"{hit.chunk.title} {hit.chunk.text[:220]}"))


def mmr_diversify(
    hits: list[RetrievedChunk],
    *,
    top_k: int = 10,
    lambda_relevance: float = 0.5,
) -> list[RetrievedChunk]:
    """Maximal Marginal Relevance within each relevance tier.

    Broad topical asks otherwise fill up with the same angle (e.g. five stories on the
    same regional market). MMR trades a little score for narrative breadth, which is
    what the Completeness dimension rewards.
    """
    if not hits:
        return []

    tiers: dict[float, list[RetrievedChunk]] = defaultdict(list)
    order: list[float] = []
    for h in hits:
        tier = h.scores.get("tier", 0.0)
        if tier not in tiers:
            order.append(tier)
        tiers[tier].append(h)

    selected: list[RetrievedChunk] = []
    for tier in sorted(order):
        group = tiers[tier]
        scores = [h.score for h in group]
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        remaining = list(group)
        sigs = {id(h): _content_signature(h) for h in remaining}
        chosen: list[RetrievedChunk] = []
        while remaining and len(selected) + len(chosen) < top_k:
            best, best_value = None, float("-inf")
            for h in remaining:
                norm = (h.score - lo) / span
                redundancy = max(
                    (jaccard(sigs[id(h)], sigs[id(c)]) for c in chosen),
                    default=0.0,
                )
                value = lambda_relevance * norm - (1.0 - lambda_relevance) * redundancy
                if value > best_value:
                    best, best_value = h, value
            if best is None:
                break
            chosen.append(best)
            remaining.remove(best)
        selected.extend(chosen)
        if len(selected) >= top_k:
            break

    out = selected[:top_k]
    for i, h in enumerate(out, start=1):
        h.rank = i
    return out


def diversity_cap(hits: list[RetrievedChunk], max_per_source: int = 3) -> list[RetrievedChunk]:
    """Avoid one source flooding the top-k — supports Completeness / source mix."""
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
