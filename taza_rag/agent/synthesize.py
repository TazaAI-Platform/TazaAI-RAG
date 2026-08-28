"""Combine results into one grounded answer.

The single-question path already established what works here and it is reused rather than
reinvented: extract cited facts first, drop any whose figures are absent from the excerpt
they cite, compose only from what survived, then splice back any grounded fact the writer
dropped. On the 52-query gold set that path took A1 Accuracy from 0.538 to 0.712 and was the
first change to move Completeness with it.

What is new at the research level is that facts arrive from several sub-questions at once, so
this module adds three things the single-question composer had no need for:

- extraction runs **per sub-question, in parallel**, against that step's own evidence, which
  keeps each extraction prompt small and on-topic;
- the composer is handed **declared disagreements** and told to attribute both sides rather
  than silently pick one;
- it is handed **declared gaps** and told to say plainly that the sources do not cover them,
  which is the honest alternative to padding and is what the rubric's intellectual-honesty
  dimension rewards.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from taza_rag.agent.conflict import blocking_conflicts, describe_conflicts
from taza_rag.agent.gather import EvidencePool
from taza_rag.agent.models import Conflict, Finding, Gap, ResearchPlan, SubQuestion
from taza_rag.config import settings
from taza_rag.factiva.facts import extract_facts, splice_unused_facts
from taza_rag.factiva.facts import Fact
from taza_rag.factiva.verify import _labels_in, _normalise_digits
from taza_rag.llm import LLMError, chat_json

RESEARCH_COMPOSE_SYSTEM = """You write a Dow Jones / Factiva research answer from a fact list.

You are given the question, the research plan, numbered facts, any DISAGREEMENTS between
sources, and anything the sources DO NOT COVER.

Rules:
- Open with a direct answer to the question in the first one or two sentences.
- Use ONLY the numbered facts. Do not add numbers, names, dates, attributions, causes or
  significance that are not in them.
- Every sentence that states a fact must end with that fact's citation marker, like [c1].
  Write markers in full: [c4], never [4].
- Do not drop a fact that bears on the question. Omitting a handed fact is a defect.
- For each item under DISAGREEMENTS: report both values and attribute each to its source.
  Lead with the side marked preferred. Never resolve a disagreement silently and never
  average two figures.
- For each item under DO NOT COVER: state plainly that the sources do not address it. Do
  not speculate, and do not pad the answer to compensate.
- Follow the plan's order as short paragraphs. No headings, no bullet lists.
- Professionally journalistic, Dow Jones voice. Do not repeat a fact you have already given.

If the facts cannot answer the question at all, set abstain=true and say what is missing.
Return JSON: {"answer": string, "abstain": boolean, "used_citations": ["c1"]}
"""


def _answer_model() -> str:
    return settings.answer_model or settings.chat_model


def _dedupe(findings: list[Finding]) -> list[Finding]:
    """One entry per (source, statement).

    Overlapping sub-questions extract the same sentence twice. Left alone it reaches the
    composer as two facts, and the composer dutifully writes it twice.
    """
    seen: set[tuple[str, str]] = set()
    out: list[Finding] = []
    for f in findings:
        key = (f.label, " ".join(_normalise_digits(f.text).lower().split()))
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def extract_findings(
    plan: ResearchPlan,
    pool: EvidencePool,
    tasks: list[SubQuestion],
    *,
    round_index: int,
    max_context_tokens: int = 1800,
    workers: int = 4,
) -> tuple[list[Finding], int, list[str]]:
    """Extract grounded facts for each sub-question concurrently.

    Returns the findings, the number of model calls made (the run reports its own spend),
    and any per-step errors. A step whose extraction fails is recorded and skipped: one
    failed extraction must not lose the facts the other steps produced.
    """
    evidence = pool.evidence_by_label()
    jobs = [(sub, pool.for_sub(sub.id)) for sub in tasks]
    jobs = [(sub, items) for sub, items in jobs if items]
    if not jobs:
        return [], 0, []

    def run(job: tuple[SubQuestion, list]) -> tuple[str, list[Fact], str]:
        sub, items = job
        context = pool.context_for(items, max_tokens=max_context_tokens)
        try:
            return sub.id, extract_facts(sub.question, context, evidence), ""
        except LLMError as e:
            return sub.id, [], f"{sub.id}: extract failed: {type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as executor:
        results = list(executor.map(run, jobs))

    findings: list[Finding] = []
    errors: list[str] = []
    for sub_id, facts, error in results:
        if error:
            errors.append(error)
        for fact in facts:
            findings.append(
                Finding(
                    sub_question_id=sub_id,
                    text=fact.text,
                    label=fact.citation,
                    round_index=round_index,
                )
            )
    return _dedupe(findings), len(jobs), errors


def format_findings(plan: ResearchPlan, findings: list[Finding]) -> str:
    """Group facts under the step that asked for them, numbered globally.

    The grouping is what lets the composer follow the plan's shape instead of writing one
    undifferentiated block, and the global numbering is what keeps a citation resolvable.
    """
    by_sub: dict[str, list[Finding]] = {}
    for f in findings:
        by_sub.setdefault(f.sub_question_id, []).append(f)

    lines: list[str] = []
    n = 0
    for sub in plan.sub_questions:
        group = by_sub.get(sub.id) or []
        if not group:
            continue
        lines.append(f"\n{sub.id}. {sub.question}")
        for f in group:
            n += 1
            lines.append(f"  {n}. {f.text} [{f.label}]")

    orphans = [f for f in findings if f.sub_question_id not in {s.id for s in plan.sub_questions}]
    if orphans:
        lines.append("\nadditional")
        for f in orphans:
            n += 1
            lines.append(f"  {n}. {f.text} [{f.label}]")
    return "\n".join(lines).strip()


def format_gaps(plan: ResearchPlan, gaps: list[Gap]) -> str:
    if not gaps:
        return ""
    lines = []
    for gap in gaps:
        sub = plan.by_id(gap.sub_question_id)
        where = f" (asked for by {sub.id})" if sub else ""
        lines.append(f"- {gap.aspect}{where}")
    return "\n".join(lines)


def compose(
    plan: ResearchPlan,
    findings: list[Finding],
    conflicts: list[Conflict],
    gaps: list[Gap],
    pool: EvidencePool,
) -> dict[str, Any] | None:
    """Write the answer from the fact list. None means the caller should abstain."""
    if not findings:
        return None

    by_label = pool.by_label()
    sections = [
        f"Question: {plan.question}",
        f"\nPLAN:\n" + "\n".join(f"{s.id}. {s.question}" for s in plan.sub_questions),
        f"\nFACTS:\n{format_findings(plan, findings)}",
    ]
    disagreements = describe_conflicts(conflicts, by_label)
    if disagreements:
        sections.append(f"\nDISAGREEMENTS:\n{disagreements}")
    gap_text = format_gaps(plan, gaps)
    if gap_text:
        sections.append(f"\nDO NOT COVER:\n{gap_text}")

    raw = chat_json(
        RESEARCH_COMPOSE_SYSTEM,
        "\n".join(sections),
        model=_answer_model(),
        temperature=0.0,
    )
    text = str(raw.get("answer") or "").strip()
    if not text:
        return None

    # Same mechanical coverage guard as the single-question path: a grounded fact the writer
    # dropped is appended with its own citation. No new numbers, no new sources.
    facts = [Fact(text=f.text, citation=f.label) for f in findings]
    raw["answer"] = splice_unused_facts(text, facts)
    raw["used_citations"] = _labels_in(raw["answer"]) or [f.label for f in findings]
    return raw


def conflict_note(conflicts: list[Conflict]) -> str:
    """One-line summary for the CLI and reports."""
    blocking = blocking_conflicts(conflicts)
    rounding = len(conflicts) - len(blocking)
    return f"{len(blocking)} disagreement(s), {rounding} rounding restatement(s)"
