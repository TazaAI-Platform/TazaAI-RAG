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
from taza_rag.factiva.pipeline import QualityRetriever
from taza_rag.factiva.retrieve import FactivaRetrieveError
from taza_rag.models import RetrievedChunk, SearchIntent


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
    """Live backend: the measured quality stack, one retriever shared across threads.

    Sharing matters — a retriever per sub-question would repeat the OAuth exchange for
    every step. `QualityRetriever` already issues its own query variants in parallel
    internally, so concurrent use is the access pattern it was built for.
    """

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


@dataclass
class TaskResult:
    sub_question_id: str
    query: str
    hits: list[RetrievedChunk] = field(default_factory=list)
    error: str = ""


@dataclass
class GatherOutcome:
    results: list[TaskResult] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def chunks_returned(self) -> int:
        return sum(len(r.hits) for r in self.results)

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
) -> GatherOutcome:
    """Run one wave of searches concurrently.

    `tasks` pairs a sub-question with the query text to issue for it, which is not always
    the sub-question's own wording: a later round searches only the aspects still missing.
    """
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
