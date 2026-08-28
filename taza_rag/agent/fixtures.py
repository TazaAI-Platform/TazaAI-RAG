"""A deterministic in-memory search backend.

The loop's interesting behaviour — refining only what is missing, stopping when a round adds
nothing, isolating a failed step — is behaviour under a sequence of retrievals. Testing that
against the live corpus would be slow, non-reproducible and metered, so the `SearchBackend`
boundary exists partly to let a fixed corpus stand in.

Scoring is whole-term overlap with a title bonus. It is not meant to imitate the ranker; it
only needs to be monotonic in relevance so that a better-targeted refinement query returns
different passages than the original ask.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from taza_rag.agent.text import term_set
from taza_rag.models import Chunk, RetrievedChunk, SearchIntent


@dataclass
class FixtureDoc:
    doc_id: str
    title: str
    text: str
    source: str = "Dow Jones Newswires"
    published_at: str = "2026-08-06"
    authority: float = 1.10


@dataclass
class FixtureSearch:
    docs: list[FixtureDoc]
    calls: list[str] = field(default_factory=list)
    fail_on: set[str] = field(default_factory=set)
    settles_purchase: bool = False

    def search(
        self, query: str, *, top_k: int, intent: SearchIntent | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append(query)
        if query in self.fail_on:
            from taza_rag.factiva.retrieve import FactivaRetrieveError

            raise FactivaRetrieveError(f"fixture failure for {query!r}")

        wanted = term_set(query)
        scored: list[tuple[float, FixtureDoc]] = []
        for doc in self.docs:
            title_hits = len(wanted & term_set(doc.title))
            body_hits = len(wanted & term_set(doc.text))
            score = 2.0 * title_hits + body_hits
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda pair: (-pair[0], pair[1].doc_id))

        hits: list[RetrievedChunk] = []
        for rank, (score, doc) in enumerate(scored[:top_k], start=1):
            hits.append(
                RetrievedChunk(
                    chunk=Chunk(
                        chunk_id=f"{doc.doc_id}::p000",
                        doc_id=doc.doc_id,
                        text=doc.text,
                        title=doc.title,
                        source=doc.source,
                        published_at=doc.published_at,
                        chunk_index=0,
                        metadata={"doc_kind": "article", "passage_count": 1},
                    ),
                    score=score,
                    rank=rank,
                    method="fixture",
                    scores={"tier": 0.0, "authority": doc.authority},
                )
            )
        return hits
