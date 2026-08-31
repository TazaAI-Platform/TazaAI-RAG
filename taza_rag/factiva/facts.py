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

from taza_rag.config import settings
from taza_rag.factiva.verify import _labels_in, _normalise_digits, figures
from taza_rag.llm import LLMError, chat_json

_STOP = frozenset(
    """about after against among because before between during from there their
    which while would could should about after under over into than that this
    with have been were said also more than the and for""".split()
)

EXTRACT_SYSTEM = """Extract every distinct fact in the sources that bears on the question.

Rules:
- One fact per item. State only what the cited source says, not interpretation or significance.
- citation must be exactly one label from the sources, like "c1".
- Copy figures, names, dates and counterparties exactly as written. Do not round or combine.
- If two sources disagree, extract both facts rather than choosing.
- Skip colour, repetition, and anything that does not help answer the question.
- Cover, when present: the headline result, every concrete figure, the cause, counterparties,
  official or executive comment, timing, and any contrary or cautionary view.
- Prefer 8-16 facts. Fewer is fine when the sources are thin; do not pad.

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


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def first_sentences(text: str, n: int = 2) -> list[str]:
    """Lead sentences of a passage. Used when no writer model is configured."""
    blob = " ".join((text or "").split())
    if not blob:
        return []
    parts = [p.strip() for p in _SENTENCE.split(blob) if p.strip()]
    return parts[:n] or [blob[:280]]


def extractive_facts(evidence: dict[str, str], *, per_source: int = 2) -> list[Fact]:
    """Cite the lead of each source. Every figure already lives in the excerpt."""
    facts: list[Fact] = []
    for label, blob in evidence.items():
        body = blob.split("\n", 1)[-1] if blob else ""
        for sent in first_sentences(body, per_source):
            facts.append(Fact(text=sent, citation=label))
    return filter_facts(facts, evidence)


def extractive_compose(facts: list[Fact], *, gaps: list[str] | None = None) -> dict[str, Any]:
    """Join grounded sentences. No model, so nothing is paraphrased or invented."""
    if not facts:
        return {"answer": "", "abstain": True, "used_citations": []}
    parts = [f"{fact.text.rstrip(' .')} [{fact.citation}]." for fact in facts]
    answer = " ".join(parts)
    if gaps:
        missing = "; ".join(g for g in gaps if g)
        if missing:
            answer += f" The sources do not cover: {missing}."
    answer = splice_unused_facts(answer, facts)
    return {
        "answer": answer,
        "abstain": False,
        "used_citations": _labels_in(answer) or [f.citation for f in facts],
    }


def format_fact_list(facts: list[Fact]) -> str:
    lines = []
    for i, fact in enumerate(facts, start=1):
        lines.append(f"{i}. {fact.text} [{fact.citation}]")
    return "\n".join(lines)


def _answer_model() -> str:
    return settings.answer_model or settings.chat_model


def fact_is_used(fact: Fact, answer: str) -> bool:
    """Did the writer actually carry this fact, or only its citation label?"""
    haystack = _normalise_digits(answer)
    figs = figures(fact.text)
    if figs and all(re.search(rf"(?<!\d){re.escape(fig)}(?!\d)", haystack) for fig in figs):
        return True
    words = [
        w
        for w in re.findall(r"[a-z]{5,}", fact.text.lower())
        if w not in _STOP
    ]
    if len(words) >= 2:
        blob = answer.lower()
        hits = sum(1 for w in words if w in blob)
        return hits >= min(3, len(words))
    return False


def splice_unused_facts(answer: str, facts: list[Fact]) -> str:
    """Append grounded facts the writer dropped. No new numbers, no new sources.

    Completeness failed because extracted facts never reached the page, not because
    retrieval missed them. This is mechanical coverage, not a second creative pass.
    """
    extras: list[str] = []
    for fact in facts:
        if fact_is_used(fact, answer):
            continue
        text = fact.text.rstrip(" .")
        extras.append(f"{text} [{fact.citation}].")
    if not extras:
        return answer
    return answer.rstrip() + " " + " ".join(extras)


def extract_facts(query: str, context: str, evidence: dict[str, str]) -> list[Fact]:
    raw = chat_json(
        EXTRACT_SYSTEM,
        f"Question: {query}\n\nSources:\n{context}",
        model=_answer_model(),
        temperature=0.0,
    )
    return filter_facts(parse_facts(raw), evidence)


def compose_from_facts(query: str, facts: list[Fact]) -> dict[str, Any] | None:
    if not facts:
        return None
    raw = chat_json(
        COMPOSE_SYSTEM,
        f"Question: {query}\n\nFacts:\n{format_fact_list(facts)}",
        model=_answer_model(),
        temperature=0.0,
    )
    text = str(raw.get("answer") or "").strip()
    if not text:
        return None
    raw["answer"] = splice_unused_facts(text, facts)
    # Splice can add labels the writer omitted; the judge sees both the prose and
    # this list, so they have to agree.
    raw["used_citations"] = _labels_in(raw["answer"]) or [f.citation for f in facts]
    return raw


def generate_from_facts(
    query: str, context: str, evidence: dict[str, str]
) -> dict[str, Any] | None:
    """Extract → filter → compose → splice unused cards. None falls back to one-shot."""
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
