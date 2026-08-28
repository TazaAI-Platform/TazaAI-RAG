"""The research loop: plan, gather, assess, refine, write, verify.

    question
      → plan (2-5 sub-questions, each with its own completion criteria)
      → gather in parallel into one labelled evidence pool
      → extract grounded facts per step, in parallel
      → assess coverage; refine only the aspects still missing, or stop and say why
      → compose from facts, attributing disagreements and declaring gaps
      → verify claims against the cited excerpts, repair what fails

The loop is bounded three ways — rounds, unique passages, and marginal gain — because an
unbounded research agent on a metered corpus is a way to spend money without improving an
answer. Every exit is labelled, and the per-round record keeps what each round cost next to
what it bought, so "it stopped too early" and "it stopped because it converged" are
distinguishable after the fact.
"""

from __future__ import annotations

import time
from typing import Any

from taza_rag.agent.conflict import detect_conflicts
from taza_rag.agent.gather import EvidencePool, FactivaSearch, SearchBackend, gather
from taza_rag.agent.models import (
    Budget,
    Cost,
    Finding,
    ResearchPlan,
    ResearchResult,
    RoundRecord,
    SubQuestion,
)
from taza_rag.agent.plan import execution_order, make_plan
from taza_rag.agent.sufficiency import assess
from taza_rag.agent.synthesize import compose, extract_findings
from taza_rag.config import settings
from taza_rag.factiva.answer import _verify_and_repair
from taza_rag.llm import LLMError

ABSTAIN_TEXT = "Insufficient evidence in the retrieved sources to answer this question."


def research(
    question: str,
    *,
    backend: SearchBackend | None = None,
    budget: Budget | None = None,
    plan: ResearchPlan | None = None,
    use_llm_plan: bool = True,
    verify: bool = True,
    workers: int = 4,
) -> ResearchResult:
    """Answer a complex question from the corpus, or explain what is missing."""
    budget = budget or Budget()
    result = ResearchResult(question=question, budget=budget)
    cost = result.cost
    t_start = time.perf_counter()

    result.plan = plan or make_plan(
        question, max_sub_questions=budget.max_sub_questions, use_llm=use_llm_plan
    )
    if result.plan.method == "llm":
        cost.llm_calls += 1
    t_plan = time.perf_counter()

    backend = backend or FactivaSearch()
    pool = EvidencePool()
    findings: list[Finding] = []
    issued: set[str] = set()

    # --- Round 0: the plan itself, in dependency waves -----------------------------------
    round0 = RoundRecord(index=0)
    for wave in execution_order(result.plan):
        tasks = [(sub, sub.question) for sub in wave]
        outcome = gather(backend, tasks, top_k=budget.top_k_per_query, workers=workers)
        cost.retrieval_calls += len(tasks)
        round0.queries.extend(q for _s, q in tasks)
        issued.update(q for _s, q in tasks)
        round0.chunks_returned += outcome.chunks_returned
        round0.latency_ms += outcome.latency_ms
        for task in outcome.results:
            if task.error:
                round0.failed_queries.append(task.query)
                result.errors.append(task.error)
                continue
            sub = result.plan.by_id(task.sub_question_id)
            if sub is None:
                continue
            round0.new_chunks += len(pool.add(task.hits, sub.id, 0))

    cost.chunks_returned += round0.chunks_returned
    cost.unique_chunks = len(pool)
    t_gather = time.perf_counter()

    if len(pool) == 0:
        result.rounds.append(round0)
        result.stop_reason = "no_evidence"
        result.answer = ABSTAIN_TEXT
        result.abstained = True
        result.latency_ms = _timings(t_start, t_plan, t_gather, t_gather, t_gather)
        return result

    new, calls, errors = extract_findings(result.plan, pool, result.plan.sub_questions, round_index=0)
    cost.llm_calls += calls
    result.errors.extend(errors)
    findings.extend(new)
    round0.new_findings = len(new)

    verdict = assess(result.plan, findings, [round0], cost, budget, issued=issued)
    round0.coverage = verdict.coverage
    round0.coverage_delta = verdict.coverage
    result.rounds.append(round0)

    # --- Refinement rounds: ask only for what is still missing ---------------------------
    while not verdict.stop:
        previous_coverage = verdict.coverage
        record = RoundRecord(index=len(result.rounds))
        tasks = verdict.refinements
        outcome = gather(backend, tasks, top_k=budget.top_k_per_query, workers=workers)
        cost.retrieval_calls += len(tasks)
        record.queries = [q for _s, q in tasks]
        issued.update(record.queries)
        record.chunks_returned = outcome.chunks_returned
        record.latency_ms = outcome.latency_ms

        touched: dict[str, SubQuestion] = {}
        for task in outcome.results:
            if task.error:
                record.failed_queries.append(task.query)
                result.errors.append(task.error)
                continue
            sub = result.plan.by_id(task.sub_question_id)
            if sub is None:
                continue
            record.new_chunks += len(pool.add(task.hits, sub.id, record.index))
            touched[sub.id] = sub

        cost.chunks_returned += record.chunks_returned
        cost.unique_chunks = len(pool)

        if record.new_chunks and touched:
            new, calls, errors = extract_findings(
                result.plan, pool, list(touched.values()), round_index=record.index
            )
            cost.llm_calls += calls
            result.errors.extend(errors)
            # Only genuinely new statements count towards marginal gain, otherwise a round
            # that re-extracted the same sentence would look like progress.
            known = {(f.label, f.text) for f in findings}
            fresh = [f for f in new if (f.label, f.text) not in known]
            findings.extend(fresh)
            record.new_findings = len(fresh)

        verdict = assess(result.plan, findings, result.rounds + [record], cost, budget, issued=issued)
        record.coverage = verdict.coverage
        record.coverage_delta = verdict.coverage - previous_coverage
        result.rounds.append(record)

    t_assess = time.perf_counter()

    # --- Combine ------------------------------------------------------------------------
    result.findings = findings
    result.coverage = verdict.coverage
    result.sub_coverage = verdict.sub_coverage
    result.gaps = verdict.gaps
    result.stop_reason = verdict.reason
    result.evidence = pool.items
    cost.evidence_tokens = pool.evidence_tokens()
    result.conflicts = detect_conflicts(findings, pool.by_label())

    raw: dict[str, Any] | None = None
    if findings:
        try:
            raw = compose(result.plan, findings, result.conflicts, result.gaps, pool)
            cost.llm_calls += 1
        except LLMError as e:
            result.errors.append(f"compose failed: {type(e).__name__}: {e}")

    if raw is None:
        result.answer = ABSTAIN_TEXT
        result.abstained = True
        result.latency_ms = _timings(t_start, t_plan, t_gather, t_assess, time.perf_counter())
        return result

    answer_text = str(raw.get("answer") or "")
    abstained = bool(raw.get("abstain"))

    if verify and answer_text and not abstained:
        evidence = pool.evidence_by_label()
        context = pool.context_for(pool.items, max_tokens=settings.answer_context_tokens)
        answer_text, abstained, raw, verification = _verify_and_repair(
            question,
            context,
            answer_text,
            abstained,
            raw,
            evidence,
            max_rounds=settings.verify_max_rounds,
        )
        result.verification = verification

    result.answer = answer_text
    result.abstained = abstained
    result.config_name = f"research_v1+{result.plan.method}" + ("+verified" if verify else "")
    result.latency_ms = _timings(t_start, t_plan, t_gather, t_assess, time.perf_counter())
    return result


def _timings(t0: float, t_plan: float, t_gather: float, t_assess: float, t_end: float) -> dict[str, float]:
    return {
        "plan": (t_plan - t0) * 1000,
        "gather": (t_gather - t_plan) * 1000,
        "assess": (t_assess - t_gather) * 1000,
        "write": (t_end - t_assess) * 1000,
        "total": (t_end - t0) * 1000,
    }
