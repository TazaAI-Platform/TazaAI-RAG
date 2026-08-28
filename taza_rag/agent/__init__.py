"""Multi-step research agent over the Factiva retrieval stack.

The retrieval pipeline answers one question well. This package answers a question that no
single retrieval can cover: it decomposes the ask, searches the corpus in parallel, decides
whether what came back is enough, and only then writes.

Retrieval is treated as a metered tool behind `SearchBackend`, not as code the agent owns.
That keeps the agent's decisions (what to ask, when to stop, what to trust) separable from
the ranking work, and it is what makes the loop testable without the network.
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
