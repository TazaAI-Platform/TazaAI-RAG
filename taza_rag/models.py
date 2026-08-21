from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SearchIntent(str, Enum):
    """Factiva search intent categories (CEO reference)."""

    ENTITY_INVESTIGATION = "entity_investigation"
    TOPICAL_EXPLORATION = "topical_exploration"
    EXECUTIVE_PROFILING = "executive_profiling"
    GEOGRAPHIC_ASSESSMENT = "geographic_assessment"
    INDUSTRY_SCAN = "industry_scan"
    EVENT_TRACKING = "event_tracking"
    KNOWN_ITEM = "known_item"
    RISK_COMPLIANCE = "risk_compliance"
    COMPETITIVE_INTEL = "competitive_intel"
    BRAND_PERCEPTION = "brand_perception"


# Approximate Factiva Total mix — use when sampling gold sets
INTENT_PRIORITIES: dict[SearchIntent, float] = {
    SearchIntent.ENTITY_INVESTIGATION: 0.41,
    SearchIntent.TOPICAL_EXPLORATION: 0.15,
    SearchIntent.EXECUTIVE_PROFILING: 0.06,
    SearchIntent.GEOGRAPHIC_ASSESSMENT: 0.05,
    SearchIntent.INDUSTRY_SCAN: 0.05,
    SearchIntent.EVENT_TRACKING: 0.04,
    SearchIntent.KNOWN_ITEM: 0.03,
    SearchIntent.RISK_COMPLIANCE: 0.02,
    SearchIntent.COMPETITIVE_INTEL: 0.01,
    SearchIntent.BRAND_PERCEPTION: 0.01,
}


class Document(BaseModel):
    doc_id: str
    title: str
    body: str
    source: str = "unknown"
    source_tier: str = "standard"  # premium | standard | wire
    published_at: str | None = None  # ISO date
    url: str | None = None
    entities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    contextualized_text: str | None = None
    title: str
    source: str
    source_tier: str = "standard"
    published_at: str | None = None
    url: str | None = None
    section: str | None = None
    chunk_index: int = 0
    token_estimate: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def index_text(self) -> str:
        return self.contextualized_text or self.text


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    rank: int
    method: str = "hybrid"
    scores: dict[str, float] = Field(default_factory=dict)


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    source: str
    published_at: str | None = None
    url: str | None = None
    excerpt: str


class AnswerResult(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    abstained: bool = False
    latency_ms: dict[str, float] = Field(default_factory=dict)
    config_name: str = "default"


# --- Dow Jones A1 evaluation models ---


class AccuracyScores(BaseModel):
    factual_correctness: bool
    citation_integrity: bool
    no_hallucinations: bool
    contextual_integrity: bool

    @property
    def pass_(self) -> bool:
        return all(
            [
                self.factual_correctness,
                self.citation_integrity,
                self.no_hallucinations,
                self.contextual_integrity,
            ]
        )


class DimensionScore(BaseModel):
    score: int = Field(ge=1, le=3)
    notes: str = ""


class A1Judgment(BaseModel):
    """Dow Jones A1 Core Evaluation Criteria."""

    query_id: str
    accuracy: AccuracyScores
    relevance: DimensionScore
    completeness: DimensionScore
    clarity: DimensionScore
    failure_tags: list[str] = Field(default_factory=list)
    notes: str = ""

    @property
    def overall_pass(self) -> bool:
        return (
            self.accuracy.pass_
            and self.relevance.score >= 2
            and self.completeness.score >= 2
            and self.clarity.score >= 2
        )


class GoldExample(BaseModel):
    id: str
    query: str
    intent: SearchIntent
    must_include_doc_ids: list[str] = Field(default_factory=list)
    acceptable_doc_ids: list[str] = Field(default_factory=list)
    hard_negative_doc_ids: list[str] = Field(default_factory=list)
    # For live Factiva evals when gold doc IDs are unknown:
    must_include_terms: list[str] = Field(default_factory=list)
    nice_to_have_terms: list[str] = Field(default_factory=list)
    notes: str = ""
    reference_answer: str | None = None
