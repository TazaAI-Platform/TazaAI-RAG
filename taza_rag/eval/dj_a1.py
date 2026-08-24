from __future__ import annotations

import json
from typing import Any

from taza_rag.config import settings
from taza_rag.llm import chat_json
from taza_rag.models import A1Judgment, AccuracyScores, AnswerResult, DimensionScore

JUDGE_SYSTEM = """You are an expert Dow Jones evaluator applying the A1 Core Evaluation Criteria.
Score ONLY against the provided source excerpts (the evidence the system used). Be strict.

Accuracy (yes/no each — ALL must be true to pass Accuracy):
1) factual_correctness: every statement/data/name verifiable in cited sources
2) citation_integrity: significant claims have citations that match sources
3) no_hallucinations: no claims absent from sources
4) contextual_integrity: facts not misrepresented vs source meaning

Relevance (1-3): directness, saliency (market-moving/important), contextual depth
Completeness (1-3): narrative completeness, source weighting on conflicts, intellectual honesty (dissent)
Clarity (1-3): conciseness, structure, Dow Jones objective/professional/analytical tone

For every score below 3, state plainly what is wrong. For completeness, also list the
specific aspects a complete answer would have covered but this one did not — name them
concretely (a subtopic, a counterparty, a figure, a dissenting view), not as generic advice.

Return JSON:
{
  "factual_correctness": bool,
  "citation_integrity": bool,
  "no_hallucinations": bool,
  "contextual_integrity": bool,
  "relevance": 1|2|3,
  "relevance_notes": string,
  "completeness": 1|2|3,
  "completeness_notes": string,
  "missing_aspects": [string],
  "clarity": 1|2|3,
  "clarity_notes": string,
  "failure_tags": [string],
  "notes": string
}
Failure tags examples: hallucination, uncited_claim, context_distortion, off_question,
low_saliency, missing_narrative, ignored_dissent, wrong_authority, jargon, poor_structure
"""


def judge_model_name(model: str | None = None) -> str:
    return model or settings.judge_model or settings.chat_model


def judge_a1(
    query_id: str,
    result: AnswerResult,
    source_excerpts: str,
    model: str | None = None,
) -> A1Judgment:
    user = (
        f"Query ID: {query_id}\n"
        f"Question: {result.query}\n\n"
        f"System answer:\n{result.answer}\n\n"
        f"Abstained: {result.abstained}\n"
        f"Citations: {json.dumps([c.model_dump() for c in result.citations])}\n\n"
        f"Source excerpts provided to the model:\n{source_excerpts}"
    )
    raw: dict[str, Any] = chat_json(
        JUDGE_SYSTEM, user, model=judge_model_name(model), temperature=0.0
    )
    return A1Judgment(
        query_id=query_id,
        accuracy=AccuracyScores(
            factual_correctness=bool(raw.get("factual_correctness")),
            citation_integrity=bool(raw.get("citation_integrity")),
            no_hallucinations=bool(raw.get("no_hallucinations")),
            contextual_integrity=bool(raw.get("contextual_integrity")),
        ),
        relevance=DimensionScore(
            score=int(raw.get("relevance", 1)), notes=str(raw.get("relevance_notes") or "")
        ),
        completeness=DimensionScore(
            score=int(raw.get("completeness", 1)), notes=str(raw.get("completeness_notes") or "")
        ),
        clarity=DimensionScore(
            score=int(raw.get("clarity", 1)), notes=str(raw.get("clarity_notes") or "")
        ),
        missing_aspects=[str(a) for a in (raw.get("missing_aspects") or [])],
        failure_tags=list(raw.get("failure_tags") or []),
        notes=str(raw.get("notes") or ""),
    )
