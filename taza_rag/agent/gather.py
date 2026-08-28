"""Retrieve for many sub-questions at once, into one shared, labelled evidence pool.

Two things here are load-bearing.

**Failure isolation.** One sub-question failing upstream must not sink the run. The live
corpus fails a few percent of calls even after retries, and a four-step plan that aborts on
the first 502 is useless. Failures are recorded and the run continues on what it has.

**One pool, one label space.** Sub-questions overlap heavily — ask about SoftBank's profit
and its borrowing and the same earnings story answers both. The pool keeps one entry per
passage, records every sub-question that found it, and assigns `c1..cN` once for the whole
run so a citation in the final answer resolves to exactly one source. Reuse is also the
cost signal: a passage returned twice was paid for twice.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

from taza_rag.agent.models import EvidenceItem, SubQuestion
from taza_rag.agent.purchase import MIN_VALUE, score_package, package_price
from taza_rag.factiva.pipeline import QualityRetriever
from taza_rag.factiva.retrieve import FactivaRetrieveError
from taza_rag.market import Market, MarketError
from taza_rag.models import RetrievedChunk, SearchIntent
from taza_rag.ui.serialize import hit_from_payload


class SearchBackend(Protocol):
    """The agent's only door to the corpus.

    Narrow on purpose: the agent decides what to ask and when to stop, the backend decides
    how to rank. Tests and offline evals supply a fixture backend, which is what keeps the
    loop verifiable without the network.
    """

    def search(
        self, query: str, *, top_k: int, intent: SearchIntent | None = None
    ) -> list[RetrievedChunk]:
        ...


class FactivaSearch:
    """Live ranking backend. Used as the search function inside Market, not as
    the agent's door: paid bodies come from transact → fetch_content.
    """

    settles_purchase = False

    def __init__(
        self,
        retriever: QualityRetriever | None = None,
        *,
        contextual: bool = True,
        semantic: bool = False,
    ) -> None:
        self.retriever = retriever or QualityRetriever()
        self.contextual = contextual
        self.semantic = semantic

    def search(
        self, query: str, *, top_k: int, intent: SearchIntent | None = None
    ) -> list[RetrievedChunk]:
        run = self.retriever.retrieve(
            query,
            top_k=top_k,
            intent=intent,
            contextual=self.contextual,
            semantic=self.semantic,
        )
        return run.hits


class MarketBackend:
    """The agent's live door: query is free, bodies cost a transact.

    Fixture tests keep injecting `FixtureSearch`. This is the default for CLI/UI,
    and it only ever sees bodies that came back from `fetch_content`.
    """

    settles_purchase = True

    def __init__(
        self,
        market: Market | None = None,
        search: SearchBackend | None = None,
    ) -> None:
        ranking = search or FactivaSearch()
        self.market = market or Market(search=ranking.search)


@dataclass
class TaskResult:
    sub_question_id: str
    query: str
    hits: list[RetrievedChunk] = field(default_factory=list)
    error: str = ""
    licensed: bool = False
    offered: int = 0
    refused: int = 0
    tradeoff_label: str = ""
    package_id: str = ""


@dataclass
class GatherOutcome:
    results: list[TaskResult] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def chunks_returned(self) -> int:
        return sum((r.offered or len(r.hits)) for r in self.results)

    @property
    def failures(self) -> list[TaskResult]:
        return [r for r in self.results if r.error]


class EvidencePool:
    """Deduplicated, globally labelled evidence for one run."""

    def __init__(self) -> None:
        self._items: dict[str, EvidenceItem] = {}

    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> list[EvidenceItem]:
        return list(self._items.values())

    def key(self, hit: RetrievedChunk) -> str:
        # Passage ids are positional and stable, so the same passage found by two
        # sub-questions collapses; different passages of one article stay distinct because
        # they are genuinely different evidence.
        return hit.chunk.chunk_id or hit.chunk.doc_id

    def add(self, hits: list[RetrievedChunk], sub_id: str, round_index: int) -> list[EvidenceItem]:
        """Merge hits into the pool. Returns only the entries that are new."""
        fresh: list[EvidenceItem] = []
        for hit in hits:
            k = self.key(hit)
            existing = self._items.get(k)
            if existing is not None:
                if sub_id not in existing.found_by:
                    existing.found_by.append(sub_id)
                continue
            item = EvidenceItem(
                label=f"c{len(self._items) + 1}",
                hit=hit,
                found_by=[sub_id],
                first_round=round_index,
            )
            self._items[k] = item
            fresh.append(item)
        return fresh

    def for_sub(self, sub_id: str) -> list[EvidenceItem]:
        return [i for i in self._items.values() if sub_id in i.found_by]

    def by_label(self) -> dict[str, EvidenceItem]:
        return {i.label: i for i in self._items.values()}

    def evidence_by_label(self) -> dict[str, str]:
        """The mapping the verifier and fact filter expect."""
        return {i.label: i.text for i in self._items.values()}

    def evidence_tokens(self) -> int:
        return sum(len((i.hit.chunk.text or "").split()) for i in self._items.values())

    def context_for(self, items: list[EvidenceItem], *, max_tokens: int = 3000) -> str:
        """Label-prefixed evidence block for fact extraction, capped by token budget."""
        blocks: list[str] = []
        used = 0
        for i in items:
            c = i.hit.chunk
            tokens = len((c.text or "").split())
            if blocks and used + tokens > max_tokens:
                break
            blocks.append(
                f"[{i.label}] doc_id={c.doc_id} | {c.source} | "
                f"{c.published_at or 'n/a'} | {c.title}\n{c.text}"
            )
            used += tokens
        return "\n\n".join(blocks)


def gather(
    backend: SearchBackend,
    tasks: list[tuple[SubQuestion, str]],
    *,
    top_k: int,
    workers: int = 4,
    wanted: list[str] | None = None,
    budget_left: int | None = None,
    purchase_gate: bool = True,
    min_value: float = MIN_VALUE,
    pooled_doc_ids: set[str] | None = None,
) -> GatherOutcome:
    """Run one wave of searches concurrently.

    `tasks` pairs a sub-question with the query text to issue for it, which is not always
    the sub-question's own wording: a later round searches only the aspects still missing.

    When the backend settles purchase (`MarketBackend`), the wave queries for free,
    picks packages against a shared budget, then transacts. Ranking backends that
    return unpaid hits keep the older search-then-gate path.
    """
    if getattr(backend, "settles_purchase", False):
        return shop_wave(
            backend,  # type: ignore[arg-type]
            tasks,
            top_k=top_k,
            workers=workers,
            wanted=wanted or [],
            budget_left=0 if budget_left is None else max(0, budget_left),
            purchase_gate=purchase_gate,
            min_value=min_value,
            pooled_doc_ids=pooled_doc_ids or set(),
        )

    outcome = GatherOutcome()
    if not tasks:
        return outcome

    t0 = time.perf_counter()

    def run(task: tuple[SubQuestion, str]) -> TaskResult:
        sub, query = task
        try:
            hits = backend.search(query, top_k=top_k, intent=sub.intent)
            return TaskResult(sub_question_id=sub.id, query=query, hits=hits)
        except FactivaRetrieveError as e:
            return TaskResult(sub_question_id=sub.id, query=query, error=f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        outcome.results = list(pool.map(run, tasks))

    outcome.latency_ms = (time.perf_counter() - t0) * 1000
    return outcome


def shop_wave(
    backend: MarketBackend,
    tasks: list[tuple[SubQuestion, str]],
    *,
    top_k: int,
    workers: int,
    wanted: list[str],
    budget_left: int,
    purchase_gate: bool,
    min_value: float,
    pooled_doc_ids: set[str],
) -> GatherOutcome:
    """Query in parallel, pick packages against one budget, then fetch licensed bodies."""
    outcome = GatherOutcome()
    if not tasks:
        return outcome

    t0 = time.perf_counter()
    market = backend.market

    def query_one(task: tuple[SubQuestion, str]) -> tuple[SubQuestion, str, dict, str]:
        sub, query = task
        try:
            bid = market.query(query, top_k=top_k, intent=sub.intent)
            return sub, query, bid, ""
        except (FactivaRetrieveError, MarketError) as e:
            return sub, query, {}, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        queried = list(pool.map(query_one, tasks))

    # One package per bid, total price ≤ budget. Gate value first, then cheaper.
    picks: list[tuple[float, int, SubQuestion, str, dict, dict]] = []
    for sub, query, bid, error in queried:
        if error or not bid:
            continue
        for offer in bid.get("packages") or []:
            price = package_price(offer)
            if price <= 0 or price > budget_left:
                continue
            scored = score_package(
                offer,
                wanted=wanted,
                pooled_doc_ids=pooled_doc_ids,
                purchase_gate=purchase_gate,
                min_value=min_value,
            )
            if scored is None:
                continue
            picks.append((scored[0], price, sub, query, bid, offer))

    picks.sort(key=lambda row: (-row[0], row[1], row[2].id))
    spent = 0
    taken_bids: set[str] = set()
    taken_subs: set[str] = set()
    chosen_by_sub: dict[str, tuple[str, dict, dict]] = {}
    for value, price, sub, query, bid, offer in picks:
        bid_id = str(bid.get("bid_id") or "")
        if not bid_id or bid_id in taken_bids or sub.id in taken_subs:
            continue
        if spent + price > budget_left:
            continue
        spent += price
        taken_bids.add(bid_id)
        taken_subs.add(sub.id)
        chosen_by_sub[sub.id] = (query, bid, offer)

    def fulfill(row: tuple[SubQuestion, str, dict, str]) -> TaskResult:
        sub, query, bid, error = row
        if error:
            return TaskResult(sub_question_id=sub.id, query=query, error=error)
        offered = int((bid.get("usage") or {}).get("offered") or 0)
        pick = chosen_by_sub.get(sub.id)
        if pick is None:
            bid_id = str(bid.get("bid_id") or "")
            if bid_id:
                try:
                    market.reject_bid(bid_id, reason="below_value")
                except MarketError as e:
                    # Already resolved (a sibling thread transacted) is the state we wanted.
                    if "bid_already_resolved" not in str(e) and "unknown_bid" not in str(e):
                        return TaskResult(
                            sub_question_id=sub.id,
                            query=query,
                            error=f"{type(e).__name__}: {e}",
                            licensed=True,
                            offered=offered,
                            refused=offered,
                        )
            return TaskResult(
                sub_question_id=sub.id,
                query=query,
                licensed=True,
                offered=offered,
                refused=offered,
            )
        _query, _bid, offer = pick
        try:
            grant = market.transact(str(offer["package_id"]))
            fetched = market.fetch_content(str(grant["grant_id"]))
        except MarketError as e:
            return TaskResult(
                sub_question_id=sub.id,
                query=query,
                error=f"{type(e).__name__}: {e}",
                licensed=True,
                offered=offered,
            )
        hits = [hit_from_payload(item) for item in fetched.get("items") or []]
        bought = int((grant.get("usage") or {}).get("bought") or len(hits))
        return TaskResult(
            sub_question_id=sub.id,
            query=query,
            hits=hits,
            licensed=True,
            offered=offered,
            refused=max(0, offered - bought),
            tradeoff_label=str(offer.get("tradeoff_label") or ""),
            package_id=str(offer.get("package_id") or ""),
        )

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(queried) or 1))) as pool:
        outcome.results = list(pool.map(fulfill, queried))

    outcome.latency_ms = (time.perf_counter() - t0) * 1000
    return outcome
