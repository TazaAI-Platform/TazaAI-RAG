"""Types the agent passes between stages.

Every stage returns data rather than mutating shared state, so a run can be serialized,
replayed and evaluated. The eval harness and the UI read these objects; nothing reaches
into the loop's internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from taza_rag.models import RetrievedChunk, SearchIntent


@dataclass
class SubQuestion:
    """One retrievable step of the plan.

    `aspects` is the part that earns its keep: short noun phrases naming what a complete
    answer must contain. They are matched against retrieved text deterministically, which
    is what lets the agent judge its own coverage without asking a model whether it is done.
    """

    id: str
    question: str
    intent: SearchIntent
    aspects: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    rationale: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "intent": self.intent.value,
            "aspects": list(self.aspects),
            "depends_on": list(self.depends_on),
            "rationale": self.rationale,
        }


@dataclass
class ResearchPlan:
    question: str
    intent: SearchIntent
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    sub_questions: list[SubQuestion] = field(default_factory=list)
    # "llm" or "heuristic" — a run must say which, because the heuristic plan is a fallback
    # and its coverage numbers are not comparable to a planned decomposition.
    method: str = "heuristic"

    def by_id(self, sub_id: str) -> SubQuestion | None:
        return next((s for s in self.sub_questions if s.id == sub_id), None)

    def payload(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "intent": self.intent.value,
            "entities": list(self.entities),
            "topics": list(self.topics),
            "method": self.method,
            "sub_questions": [s.payload() for s in self.sub_questions],
        }


@dataclass
class EvidenceItem:
    """A passage in the shared pool, labelled once for the whole run.

    Labels are global. Per-section labelling was the first thing tried and it collided:
    two sections each had a `c1` pointing at different documents, so a citation in the
    final answer could not be resolved back to a source.
    """

    label: str
    hit: RetrievedChunk
    found_by: list[str] = field(default_factory=list)
    first_round: int = 0

    @property
    def doc_id(self) -> str:
        return self.hit.chunk.doc_id

    @property
    def text(self) -> str:
        c = self.hit.chunk
        return f"{c.title}\n{c.text}"

    def payload(self) -> dict[str, Any]:
        c = self.hit.chunk
        return {
            "label": self.label,
            "doc_id": c.doc_id,
            "chunk_id": c.chunk_id,
            "title": c.title,
            "source": c.source,
            "published_at": c.published_at,
            "score": round(float(self.hit.score), 3),
            "authority": round(float(self.hit.scores.get("authority", 1.0)), 3),
            "found_by": list(self.found_by),
            "first_round": self.first_round,
        }


@dataclass
class Finding:
    """A grounded fact, tied to the sub-question that asked for it."""

    sub_question_id: str
    text: str
    label: str
    round_index: int = 0

    def payload(self) -> dict[str, Any]:
        return {
            "sub_question_id": self.sub_question_id,
            "text": self.text,
            "label": self.label,
            "round": self.round_index,
        }


@dataclass
class Conflict:
    """Two sources that state a different value for the same thing.

    `kind` separates the two cases that look identical to a naive diff: `rounding` is one
    outlet writing 18% where another writes 17.7%, which must not be reported as a dispute,
    and `disagreement` is a genuine divergence that the answer has to attribute.
    """

    kind: str
    subject: str
    left: Finding
    right: Finding
    preferred_label: str = ""
    reason: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "subject": self.subject,
            "left": self.left.payload(),
            "right": self.right.payload(),
            "preferred_label": self.preferred_label,
            "reason": self.reason,
        }


@dataclass
class Gap:
    """An aspect the plan asked for that no grounded fact covers."""

    sub_question_id: str
    aspect: str

    def payload(self) -> dict[str, Any]:
        return {"sub_question_id": self.sub_question_id, "aspect": self.aspect}


@dataclass
class RoundRecord:
    """What one retrieval round cost and what it bought.

    The pair (`new_chunks`, `new_findings`) is the stopping signal: a round that pays for
    chunks and returns no new grounded fact is the point where further spending stops
    changing the answer.
    """

    index: int
    queries: list[str] = field(default_factory=list)
    chunks_returned: int = 0
    new_chunks: int = 0
    new_findings: int = 0
    coverage: float = 0.0
    coverage_delta: float = 0.0
    failed_queries: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "queries": list(self.queries),
            "chunks_returned": self.chunks_returned,
            "new_chunks": self.new_chunks,
            "new_findings": self.new_findings,
            "coverage": round(self.coverage, 3),
            "coverage_delta": round(self.coverage_delta, 3),
            "failed_queries": list(self.failed_queries),
            "latency_ms": round(self.latency_ms),
        }


@dataclass
class Budget:
    """Caps expressed in the units the marketplace charges for.

    Chunks, not tokens, are the priced unit in the brief, so the cap that matters is on
    unique passages pulled. Rounds and calls are here to bound latency and spend, not
    because they are billed.
    """

    max_rounds: int = 3
    max_unique_chunks: int = 40
    max_sub_questions: int = 5
    top_k_per_query: int = 6
    # Stop when a round adds fewer than this many new grounded facts.
    min_new_findings: int = 1
    # Coverage at which the plan counts as answered; 1.0 is rarely reachable on a live
    # corpus and chasing it buys noise.
    target_coverage: float = 0.8

    def payload(self) -> dict[str, Any]:
        return {
            "max_rounds": self.max_rounds,
            "max_unique_chunks": self.max_unique_chunks,
            "max_sub_questions": self.max_sub_questions,
            "top_k_per_query": self.top_k_per_query,
            "min_new_findings": self.min_new_findings,
            "target_coverage": self.target_coverage,
        }


@dataclass
class Cost:
    chunks_returned: int = 0
    unique_chunks: int = 0
    retrieval_calls: int = 0
    llm_calls: int = 0
    evidence_tokens: int = 0

    @property
    def reuse_rate(self) -> float:
        """Share of returned passages already in the pool.

        High reuse means sub-questions are overlapping, which is a planner defect: the run
        paid for the same passage more than once.
        """
        if not self.chunks_returned:
            return 0.0
        return 1.0 - (self.unique_chunks / self.chunks_returned)

    def payload(self) -> dict[str, Any]:
        return {
            "chunks_returned": self.chunks_returned,
            "unique_chunks": self.unique_chunks,
            "retrieval_calls": self.retrieval_calls,
            "llm_calls": self.llm_calls,
            "evidence_tokens": self.evidence_tokens,
            "reuse_rate": round(self.reuse_rate, 3),
        }


@dataclass
class ResearchResult:
    question: str
    answer: str = ""
    plan: ResearchPlan | None = None
    evidence: list[EvidenceItem] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    rounds: list[RoundRecord] = field(default_factory=list)
    coverage: float = 0.0
    sub_coverage: dict[str, float] = field(default_factory=dict)
    stop_reason: str = ""
    abstained: bool = False
    verification: dict[str, Any] | None = None
    cost: Cost = field(default_factory=Cost)
    budget: Budget = field(default_factory=Budget)
    latency_ms: dict[str, float] = field(default_factory=dict)
    config_name: str = "research_v1"
    errors: list[str] = field(default_factory=list)

    def payload(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "abstained": self.abstained,
            "config": self.config_name,
            "coverage": round(self.coverage, 3),
            "sub_coverage": {k: round(v, 3) for k, v in self.sub_coverage.items()},
            "stop_reason": self.stop_reason,
            "plan": self.plan.payload() if self.plan else None,
            "rounds": [r.payload() for r in self.rounds],
            "findings": [f.payload() for f in self.findings],
            "conflicts": [c.payload() for c in self.conflicts],
            "gaps": [g.payload() for g in self.gaps],
            "evidence": [e.payload() for e in self.evidence],
            "verification": self.verification,
            "cost": self.cost.payload(),
            "budget": self.budget.payload(),
            "latency_ms": {k: round(v) for k, v in self.latency_ms.items()},
            "errors": list(self.errors),
        }
