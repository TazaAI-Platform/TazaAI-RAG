"""Extract grounded facts from the evidence pack, then compose the answer from those facts.

The generator previously saw ~2,500 words and wrote ~96, leaving a median six supported
facts unused. Asking it to cover more in one pass raised Completeness and cost Accuracy,
because citation integrity is a per-answer gate and each extra claim is another chance to
fail it.

This path does the coverage work before writing. Extraction lists the facts with their
labels; a deterministic filter drops any fact whose figures are not in the cited excerpt;
composition is only allowed to use what survived. The writer cannot invent a number that
extraction never produced, and it cannot drop a fact it was handed without that being
visible in the card list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from taza_rag.factiva.verify import _normalise_digits, figures
from taza_rag.llm import LLMError, chat_json

EXTRACT_SYSTEM = """Extract every distinct fact in the sources that bears on the question.

Rules:
- One fact per item. State only what the cited source says, not interpretation or significance.
- citation must be exactly one label from the sources, like "c1".
- Copy figures, names, dates and counterparties exactly as written. Do not round or combine.
- If two sources disagree, extract both facts rather than choosing.
- Skip colour, repetition, and anything that does not help answer the question.
- Prefer 6-14 facts. Fewer is fine when the sources are thin; do not pad.

Return JSON: {"facts": [{"text": string, "citation": string}]}
"""

COMPOSE_SYSTEM = """Write a Dow Jones / Factiva research answer using ONLY the numbered facts.

Rules:
- Every sentence that states a fact must end with that fact's citation marker, like [c1].
- You may combine two facts in one sentence only if they share a citation.
- Do not add numbers, names, dates, attributions or significance that are not in the facts.
- Do not drop a fact that answers the question. Omitting a handed fact is a defect.
- Lead with the direct answer, then the supporting specifics, then any contrary view.
- Professionally journalistic. Do not pad or repeat.

If the facts cannot answer the question, set abstain=true and say what is missing.
Return JSON: {"answer": string, "abstain": boolean, "used_citations": ["c1"]}
"""


@dataclass
class Fact:
    text: str
    citation: str


def parse_facts(raw: Any) -> list[Fact]:
    """Accept only well-formed cards. A malformed extractor must not leak junk downstream."""
    items = raw.get("facts") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return []
    out: list[Fact] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        citation = str(item.get("citation") or "").strip().lower()
        if citation.startswith("[") and citation.endswith("]"):
            citation = citation[1:-1].strip()
        if citation.isdigit():
            citation = f"c{int(citation)}"
        if not text or not re.fullmatch(r"c\d+", citation):
            continue
        key = f"{citation}:{text.lower()}"
        if key in seen:
            continue
        seen.add(key)
        out.append(Fact(text=text, citation=citation))
    return out


def fact_is_grounded(text: str, cited: str) -> bool:
    """Every figure in the fact must appear in the excerpt it cites.

    Years are included here even though the later verifier treats them as non-blocking:
    an extracted fact that invents a year should never reach the writer.
    """
    haystack = _normalise_digits(cited)
    for fig in figures(text):
        if not re.search(rf"(?<!\d){re.escape(fig)}(?!\d)", haystack):
            return False
    return True


def filter_facts(facts: list[Fact], evidence: dict[str, str]) -> list[Fact]:
    kept: list[Fact] = []
    for fact in facts:
        cited = evidence.get(fact.citation)
        if cited is None:
            continue
        if fact_is_grounded(fact.text, cited):
            kept.append(fact)
    return kept


def format_fact_list(facts: list[Fact]) -> str:
    lines = []
    for i, fact in enumerate(facts, start=1):
        lines.append(f"{i}. {fact.text} [{fact.citation}]")
    return "\n".join(lines)


def extract_facts(query: str, context: str, evidence: dict[str, str]) -> list[Fact]:
    raw = chat_json(
        EXTRACT_SYSTEM, f"Question: {query}\n\nSources:\n{context}", temperature=0.0
    )
    return filter_facts(parse_facts(raw), evidence)


def compose_from_facts(query: str, facts: list[Fact]) -> dict[str, Any] | None:
    if not facts:
        return None
    raw = chat_json(
        COMPOSE_SYSTEM,
        f"Question: {query}\n\nFacts:\n{format_fact_list(facts)}",
        temperature=0.0,
    )
    if not str(raw.get("answer") or "").strip():
        return None
    if not raw.get("used_citations"):
        raw["used_citations"] = [f.citation for f in facts]
    return raw


def generate_from_facts(
    query: str, context: str, evidence: dict[str, str]
) -> dict[str, Any] | None:
    """Extract → filter → compose. None means the caller should use the one-shot path."""
    try:
        facts = extract_facts(query, context, evidence)
    except LLMError:
        return None
    if not facts:
        return None
    try:
        return compose_from_facts(query, facts)
    except LLMError:
        return None
