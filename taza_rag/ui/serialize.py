"""JSON shapes the UI renders. Kept next to the ranker so the labels cannot drift."""

from __future__ import annotations

from typing import Any

from taza_rag.factiva.pipeline import RetrievalRun
from taza_rag.factiva.strategy import default_days_range, detect_intent, expand_queries, normalize_query
from taza_rag.models import AnswerResult, RetrievedChunk, SearchIntent
from taza_rag.retrieve.features import build_query_plan
from taza_rag.retrieve.quality import TIER_BODY, TIER_ENTITY_ONLY, TIER_HEADLINE, TIER_OFF_ENTITY

TIER_LABELS = {
    TIER_HEADLINE: "headline",
    TIER_BODY: "body",
    TIER_ENTITY_ONLY: "entity only",
    TIER_OFF_ENTITY: "off-entity",
}

TIER_HELP = {
    TIER_HEADLINE: "Entity and topic both in the title or lead — this passage answers the ask.",
    TIER_BODY: "Entity is present; the topic only appears deeper in the body.",
    TIER_ENTITY_ONLY: "Names the entity but answers a different question.",
    TIER_OFF_ENTITY: "Never names the query entity. Gated out of the pack when possible.",
}

# Shown as tooltips on the score meters. Values are 0–1 unless noted.
SCORE_LEGEND: list[dict[str, Any]] = [
    {
        "key": "entity",
        "label": "Entity",
        "kind": "unit",
        "help": "Where the named company or person appears. Title scores highest, then lead, then body.",
    },
    {
        "key": "topic",
        "label": "Topic",
        "kind": "unit",
        "help": "Whether the rest of the ask (the event, the theme) is in the headline or only later.",
    },
    {
        "key": "bm25",
        "label": "BM25",
        "kind": "unit",
        "help": "Lexical overlap with the query, scaled so the closest candidate in this pool is 1.00.",
    },
    {
        "key": "rrf",
        "label": "RRF",
        "kind": "unit",
        "help": "How often and how high this passage ranked across the parallel query variants.",
    },
    {
        "key": "authority",
        "label": "Authority",
        "kind": "factor",
        "help": "Source prior. WSJ 1.12, Dow Jones Newswires 1.10, unknown outlets 1.00.",
    },
    {
        "key": "freshness",
        "label": "Freshness",
        "kind": "factor",
        "help": "Recency band from the published date. Last 7 days 1.12, 8–30 days 1.08, older lower.",
    },
    {
        "key": "penalty",
        "label": "Penalty",
        "kind": "cost",
        "help": "Subtracted for digests, vendor profiles and aggregators. Original reporting pays 0.",
    },
]


def plan_payload(query: str, *, max_variants: int = 3) -> dict[str, Any]:
    """Local, no Factiva: what the pipeline will ask before it spends a retrieve."""
    q = (query or "").strip()
    intent = detect_intent(q)
    normalized = normalize_query(q)
    plan = build_query_plan(normalized, intent)
    return {
        "query": q,
        "normalized": normalized,
        "intent": intent.value,
        "entities": plan.entities,
        "topics": plan.topics,
        "expanded_topics": plan.expanded_topics,
        "variants": expand_queries(q, intent, max_variants=max_variants),
        "days_range": default_days_range(intent),
    }


def hit_payload(hit: RetrievedChunk) -> dict[str, Any]:
    c = hit.chunk
    meta = c.metadata or {}
    scores = {k: _num(v) for k, v in (hit.scores or {}).items()}
    tier = int(scores.get("tier", TIER_HEADLINE))
    return {
        "rank": hit.rank,
        "score": round(float(hit.score), 3),
        "label": f"c{hit.rank}",
        "title": c.title,
        "source": c.source,
        "published_at": c.published_at,
        "doc_id": c.doc_id,
        "chunk_id": c.chunk_id,
        "url": c.url,
        "kind": meta.get("doc_kind") or "article",
        "passage": {
            "index": int(c.chunk_index) + 1,
            "of": int(meta.get("passage_count") or 1),
        },
        "tier": tier,
        "tier_label": TIER_LABELS.get(tier, "unknown"),
        "tier_help": TIER_HELP.get(tier, ""),
        "scores": scores,
        "text": c.text or "",
    }


def run_payload(run: RetrievalRun) -> dict[str, Any]:
    plan = run.plan
    return {
        "query": run.query,
        "intent": run.intent.value if isinstance(run.intent, SearchIntent) else str(run.intent),
        "entities": list(plan.entities) if plan else [],
        "topics": list(plan.topics) if plan else [],
        "expanded_topics": list(plan.expanded_topics) if plan else [],
        "variants": list(run.variants),
        "failed_variants": list(run.failed_variants),
        "config": run.config,
        "days_range": default_days_range(run.intent) if isinstance(run.intent, SearchIntent) else "",
        "candidates": run.candidates,
        "passages": run.passages,
        "latency_ms": {k: round(v) for k, v in run.latency_ms.items()},
        "hits": [hit_payload(h) for h in run.hits],
    }


def answer_payload(result: AnswerResult) -> dict[str, Any]:
    return {
        "query": result.query,
        "answer": result.answer,
        "abstained": result.abstained,
        "config": result.config_name,
        "latency_ms": {k: round(v) for k, v in result.latency_ms.items()},
        "verification": _verification(result.verification),
        "citations": [
            {
                "doc_id": c.doc_id,
                "title": c.title,
                "source": c.source,
                "published_at": c.published_at,
                "excerpt": c.excerpt,
            }
            for c in result.citations
        ],
        "hits": [hit_payload(h) for h in result.retrieved],
    }


def health_payload(*, factiva: bool, openai: bool) -> dict[str, Any]:
    """Booleans only. Client ids and keys must never appear here."""
    return {"factiva": factiva, "openai": openai}


def _verification(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    if not raw:
        return None
    initial = raw.get("initial") or {}
    final = raw.get("final") or {}
    return {
        "repairs_applied": int(raw.get("repairs_applied") or 0),
        "resolved": bool(raw.get("resolved")),
        "initial_problems": _problem_count(initial),
        "final_problems": _problem_count(final),
        "detail": (final.get("detail") or [])[:12],
    }


def _problem_count(summary: dict[str, Any]) -> int:
    problems = summary.get("problems") or {}
    if isinstance(problems, dict):
        return int(sum(int(v) for v in problems.values()))
    if isinstance(problems, list):
        return len(problems)
    return 0


def _num(value: Any) -> float:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return 0.0
