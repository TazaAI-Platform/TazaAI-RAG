"""Multi-step research agent over the Factiva retrieval stack.

The retrieval pipeline answers one question well. This package answers a question that no
single retrieval can cover: it decomposes the ask, searches the corpus in parallel, decides
whether what came back is enough, and only then writes.

Retrieval is treated as a metered marketplace behind `MarketBackend`, not as code the
agent owns. The agent decides what to ask and which labelled package to buy; bodies
arrive only after `transact`. Fixture tests still inject `FixtureSearch` so the loop
is verifiable without the network.
"""

from taza_rag.agent.models import (
    Budget,
    Conflict,
    EvidenceItem,
    Finding,
    ResearchPlan,
    ResearchResult,
    RoundRecord,
    SubQuestion,
)

__all__ = [
    "Budget",
    "Conflict",
    "EvidenceItem",
    "Finding",
    "ResearchPlan",
    "ResearchResult",
    "RoundRecord",
    "SubQuestion",
]
