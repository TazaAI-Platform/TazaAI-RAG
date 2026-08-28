"""Marketplace loop that matches the product MCP protocol.

app.tazalabs.ai sells content as a bid, not a retriever:

    query (free) → priced packages → transact (pays) → grant → fetch_content (bodies)

No LLM sits on this path. Ranking is the existing Factiva quality stack; packaging
is deterministic. The research agent stays a client of this loop, not a tool inside
it — that is the product's "no LLMs behind the MCP boundary" rule.

The priced unit here is a passage, not a dollar. A second corpus should keep the
same handles (bid, package, grant) and swap the search function.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from taza_rag.models import RetrievedChunk
from taza_rag.ui.serialize import hit_payload, usage_payload

TRADEOFF_CHEAPEST = "cheapest"
TRADEOFF_DENSEST = "densest"
TRADEOFF_TOKEN = "token_constrained"
TRADEOFF_THOROUGH = "most_thorough"
TRADEOFF_BALANCED = "balanced"
TRADEOFF_BUDGET = "budget_constrained"

# Closed vocab, same labels the product playground renders.
TRADEOFF_LABELS = (
    TRADEOFF_CHEAPEST,
    TRADEOFF_DENSEST,
    TRADEOFF_TOKEN,
    TRADEOFF_THOROUGH,
    TRADEOFF_BALANCED,
    TRADEOFF_BUDGET,
)

BID_TTL = timedelta(minutes=15)


class MarketError(ValueError):
    """Structured failure an MCP tool maps to a tool error, not a protocol crash."""


def _tokens(hit: RetrievedChunk) -> int:
    n = int(hit.chunk.token_estimate or 0)
    if n > 0:
        return n
    return max(1, len(hit.chunk.text or "") // 4)


def preview_of(hit: RetrievedChunk) -> dict[str, Any]:
    """Catalog row a buyer can see before paying: metadata, never the body."""
    c = hit.chunk
    return {
        "title": c.title,
        "source": c.source,
        "published_at": c.published_at,
        "doc_id": c.doc_id,
        "score": round(float(hit.score), 3),
        "token_count": _tokens(hit),
        "authority": round(float(hit.scores.get("authority", 1.0)), 3),
        "freshness": round(float(hit.scores.get("freshness", 1.0)), 3),
    }


def _density(hit: RetrievedChunk) -> float:
    return float(hit.score) / _tokens(hit)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Package:
    package_id: str
    tradeoff_label: str
    chunk_ids: tuple[str, ...]
    price: int
    token_count: int
    document_count: int
    mean_relevance: float
    recommended_read_count: int

    def offer(self, hits: list[RetrievedChunk] | None = None) -> dict[str, Any]:
        """What a caller sees before paying: a labelled quote plus catalog, not bodies."""
        by_id = {h.chunk.chunk_id: h for h in hits or []}
        contents = [preview_of(by_id[cid]) for cid in self.chunk_ids if cid in by_id]
        return {
            "package_id": self.package_id,
            "tradeoff_label": self.tradeoff_label,
            "price": {"amount": self.price, "unit": "chunks"},
            "summary_stats": {
                "document_count": self.document_count,
                "chunk_count": self.price,
                "token_count": self.token_count,
                "mean_relevance": round(self.mean_relevance, 3),
            },
            "recommended_read_count": self.recommended_read_count,
            "contents": contents,
        }


@dataclass
class Bid:
    bid_id: str
    query: str
    hits: list[RetrievedChunk]
    packages: dict[str, Package]
    expires_at: datetime
    resolved: bool = False
    rejected: bool = False
    transacted_package_id: str | None = None

    def expired(self, now: datetime | None = None) -> bool:
        return (now or _now()) >= self.expires_at


@dataclass
class Grant:
    grant_id: str
    bid_id: str
    package_id: str
    chunk_ids: tuple[str, ...]
    fetched_ids: set[str] = field(default_factory=set)


def assemble_packages(
    hits: list[RetrievedChunk],
    *,
    token_budget: int | None = None,
) -> list[Package]:
    """Build labelled packages from a ranked pool. Identical sets collapse."""
    if not hits:
        return []

    by_id = {h.chunk.chunk_id: h for h in hits}
    budget = token_budget if token_budget and token_budget > 0 else None

    def take(ordered: list[RetrievedChunk], *, limit: int | None = None) -> list[RetrievedChunk]:
        chosen: list[RetrievedChunk] = []
        used = 0
        for hit in ordered:
            cost = _tokens(hit)
            if budget is not None and used + cost > budget and chosen:
                break
            chosen.append(hit)
            used += cost
            if limit is not None and len(chosen) >= limit:
                break
        return chosen or ordered[:1]

    candidates: list[tuple[str, list[RetrievedChunk]]] = [
        (TRADEOFF_CHEAPEST, take(sorted(hits, key=lambda h: (_tokens(h), h.rank)), limit=1)),
        (TRADEOFF_DENSEST, take(sorted(hits, key=_density, reverse=True), limit=4)),
        (TRADEOFF_TOKEN, take(sorted(hits, key=lambda h: h.rank), limit=2)),
        (TRADEOFF_THOROUGH, take(sorted(hits, key=lambda h: h.rank))),
        (TRADEOFF_BALANCED, take(sorted(hits, key=lambda h: h.rank), limit=max(1, min(5, len(hits))))),
    ]
    if budget is not None:
        candidates.append((TRADEOFF_BUDGET, take(sorted(hits, key=lambda h: h.rank))))

    packages: list[Package] = []
    seen: set[frozenset[str]] = set()
    for label, group in candidates:
        ids = tuple(h.chunk.chunk_id for h in group)
        key = frozenset(ids)
        if not ids or key in seen:
            continue
        seen.add(key)
        tokens = sum(_tokens(by_id[i]) for i in ids)
        docs = {by_id[i].chunk.doc_id for i in ids}
        mean = sum(by_id[i].score for i in ids) / len(ids)
        packages.append(
            Package(
                package_id=str(uuid.uuid4()),
                tradeoff_label=label,
                chunk_ids=ids,
                price=len(ids),
                token_count=tokens,
                document_count=len(docs),
                mean_relevance=mean,
                recommended_read_count=len(ids),
            )
        )
    return packages


class Market:
    """Process-local bid/grant book. One instance per MCP session or UI server."""

    def __init__(self, search: Callable[..., list[RetrievedChunk]] | None = None) -> None:
        self._search = search
        self._bids: dict[str, Bid] = {}
        self._grants: dict[str, Grant] = {}
        self._packages: dict[str, str] = {}  # package_id → bid_id
        self._lock = threading.Lock()

    def query(
        self,
        text_query: str,
        *,
        top_k: int = 10,
        token_budget: int | None = None,
        max_packages_returned: int = 5,
        intent: Any = None,
    ) -> dict[str, Any]:
        q = (text_query or "").strip()
        if not q:
            raise MarketError("text_query is required")
        hits = list(self._run_search(q, top_k=top_k, intent=intent))
        packages = assemble_packages(hits, token_budget=token_budget)[: max(1, max_packages_returned)]
        bid = Bid(
            bid_id=str(uuid.uuid4()),
            query=q,
            hits=hits,
            packages={p.package_id: p for p in packages},
            expires_at=_now() + BID_TTL,
        )
        with self._lock:
            self._bids[bid.bid_id] = bid
            for package in packages:
                self._packages[package.package_id] = bid.bid_id
        related = {
            "suggested_tags": [],
            "related_topics": [],
            "adjacent_sources": sorted({h.chunk.source for h in hits if h.chunk.source})[:5],
        }
        return {
            "bid_id": bid.bid_id,
            "bid_expires_at": bid.expires_at.isoformat(),
            "packages": [p.offer(hits) for p in packages],
            "related": related,
            "usage": usage_payload(
                offered=len(hits),
                bought=0,
                refused=0,
                retrieval_calls=1,
            ),
        }

    def transact(self, package_id: str) -> dict[str, Any]:
        pid = (package_id or "").strip()
        if not pid:
            raise MarketError("package_id is required")
        with self._lock:
            bid_id = self._packages.get(pid)
            if not bid_id:
                raise MarketError("unknown_package")
            bid = self._bids[bid_id]
            if bid.expired():
                raise MarketError("bid_expired")
            if bid.rejected:
                raise MarketError("bid_already_resolved")
            if bid.resolved:
                raise MarketError("bid_already_resolved")
            package = bid.packages[pid]
            bid.resolved = True
            bid.transacted_package_id = pid
            grant = Grant(
                grant_id=str(uuid.uuid4()),
                bid_id=bid.bid_id,
                package_id=pid,
                chunk_ids=package.chunk_ids,
            )
            self._grants[grant.grant_id] = grant
            offered = len(bid.hits)
            bought = package.price
        return {
            "grant_id": grant.grant_id,
            "package_id": pid,
            "bid_id": bid.bid_id,
            "scope_kind": "specific_components",
            "purpose": "query_use",
            "caps": {"tokens": package.token_count, "chunk_count": bought, "document_count": package.document_count},
            "covered_summary": {
                "document_count": package.document_count,
                "chunk_count": bought,
            },
            "settlement": {"amount": bought, "unit": "chunks"},
            "usage": usage_payload(
                offered=offered,
                bought=bought,
                refused=max(0, offered - bought),
                retrieval_calls=1,
            ),
        }

    def fetch_content(self, grant_id: str, *, items: list[str] | None = None) -> dict[str, Any]:
        gid = (grant_id or "").strip()
        if not gid:
            raise MarketError("grant_id is required")
        with self._lock:
            grant = self._grants.get(gid)
            if grant is None:
                raise MarketError("unknown_grant")
            bid = self._bids[grant.bid_id]
            wanted = list(items) if items else list(grant.chunk_ids)
            by_id = {h.chunk.chunk_id: h for h in bid.hits}
            fetched: list[RetrievedChunk] = []
            outcomes: list[dict[str, Any]] = []
            for cid in wanted:
                if cid not in grant.chunk_ids:
                    outcomes.append({"content_id": cid, "outcome": "not_in_scope"})
                    continue
                hit = by_id.get(cid)
                if hit is None:
                    outcomes.append({"content_id": cid, "outcome": "not_in_scope"})
                    continue
                grant.fetched_ids.add(cid)
                fetched.append(hit)
                outcomes.append({"content_id": cid, "outcome": "covered"})
            package = bid.packages[grant.package_id]
            offered = len(bid.hits)
            bought = package.price
        return {
            "grant_id": gid,
            "items": [hit_payload(h) for h in fetched],
            "outcomes": outcomes,
            "usage": usage_payload(
                offered=offered,
                bought=bought,
                refused=max(0, offered - bought),
                cited=0,
                retrieval_calls=1,
            ),
        }

    def reject_bid(self, bid_id: str, *, reason: str = "other") -> dict[str, Any]:
        bid_id = (bid_id or "").strip()
        if not bid_id:
            raise MarketError("bid_id is required")
        with self._lock:
            bid = self._bids.get(bid_id)
            if bid is None:
                raise MarketError("unknown_bid")
            if bid.resolved or bid.rejected:
                raise MarketError("bid_already_resolved")
            bid.rejected = True
            bid.resolved = True
            offered = len(bid.hits)
        return {
            "bid_id": bid_id,
            "reason": reason,
            "usage": usage_payload(offered=offered, bought=0, refused=offered, retrieval_calls=1),
        }

    def _run_search(
        self, query: str, *, top_k: int, intent: Any = None
    ) -> list[RetrievedChunk]:
        if self._search is not None:
            try:
                return list(self._search(query, top_k=top_k, intent=intent))
            except TypeError:
                return list(self._search(query, top_k=top_k))
        from taza_rag.factiva.pipeline import QualityRetriever

        return QualityRetriever().retrieve(query, top_k=top_k, intent=intent).hits
