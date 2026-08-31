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
from taza_rag.agent.gather import EvidencePool, MarketBackend, SearchBackend, gather
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
from taza_rag.agent.purchase import PurchaseDecision, select
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
    extractive: bool = False,
) -> ResearchResult:
    """Answer a complex question from the corpus, or explain what is missing."""
    budget = budget or Budget()
    result = ResearchResult(question=question, budget=budget)
    cost = result.cost
    t_start = time.perf_counter()

    result.plan = plan or make_plan(
        question,
        max_sub_questions=budget.max_sub_questions,
        use_llm=use_llm_plan and not extractive,
    )
    if result.plan.method == "llm":
        cost.llm_calls += 1
    t_plan = time.perf_counter()

    backend = backend or MarketBackend()
    pool = EvidencePool()
    findings: list[Finding] = []
    issued: set[str] = set()

    # --- Round 0: the plan itself, in dependency waves -----------------------------------
    # Nothing is covered yet, so every aspect the plan declared is what the gate should be
    # looking for in a candidate's headline.
    open_aspects = [a for sub in result.plan.sub_questions for a in sub.aspects]

    round0 = RoundRecord(index=0)
    for wave in execution_order(result.plan):
        tasks = [(sub, sub.question) for sub in wave]
        outcome = gather(
            backend,
            tasks,
            top_k=budget.top_k_per_query,
            workers=workers,
            wanted=open_aspects,
            budget_left=max(0, budget.max_unique_chunks - len(pool)),
            purchase_gate=budget.purchase_gate,
            min_value=budget.min_purchase_value,
            pooled_doc_ids={i.doc_id for i in pool.items},
        )
        cost.retrieval_calls += len(tasks)
        round0.queries.extend(q for _s, q in tasks)
        issued.update(q for _s, q in tasks)
        round0.chunks_returned += outcome.chunks_returned
        round0.latency_ms += outcome.latency_ms
        for task in outcome.results:
            if task.error:
                round0.failed_queries.append(task.query)
                result.errors.append(task.error)

        round0.new_chunks += _admit(
            outcome, pool, result, budget, cost, wanted=open_aspects, round_index=0
        )

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

    new, calls, errors = extract_findings(
        result.plan,
        pool,
        result.plan.sub_questions,
        round_index=0,
        extractive=extractive,
    )
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
        outcome = gather(
            backend,
            tasks,
            top_k=budget.top_k_per_query,
            workers=workers,
            wanted=[g.aspect for g in verdict.gaps],
            budget_left=max(0, budget.max_unique_chunks - len(pool)),
            purchase_gate=budget.purchase_gate,
            min_value=budget.min_purchase_value,
            pooled_doc_ids={i.doc_id for i in pool.items},
        )
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
            if sub is not None:
                touched[sub.id] = sub

        record.new_chunks += _admit(
            outcome,
            pool,
            result,
            budget,
            cost,
            wanted=[g.aspect for g in verdict.gaps],
            round_index=record.index,
        )

        cost.chunks_returned += record.chunks_returned
        cost.unique_chunks = len(pool)

        if record.new_chunks and touched:
            new, calls, errors = extract_findings(
                result.plan,
                pool,
                list(touched.values()),
                round_index=record.index,
                extractive=extractive,
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
            raw = compose(
                result.plan,
                findings,
                result.conflicts,
                result.gaps,
                pool,
                extractive=extractive,
            )
            if not extractive:
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

    if verify and not extractive and answer_text and not abstained:
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
    result.config_name = f"research_v1+{result.plan.method}" + (
        "+extractive" if extractive else "+verified" if verify else ""
    )
    result.latency_ms = _timings(t_start, t_plan, t_gather, t_assess, time.perf_counter())
    return result


def _admit(
    outcome: Any,
    pool: EvidencePool,
    result: ResearchResult,
    budget: Budget,
    cost: Cost,
    *,
    wanted: list[str],
    round_index: int,
) -> int:
    """Pool what this wave licensed or, for unpaid backends, run the passage gate."""
    licensed = any(getattr(task, "licensed", False) for task in outcome.results)
    if licensed:
        return _admit_licensed(outcome, pool, result, cost, round_index=round_index)

    candidates: list[tuple[str, Any]] = []
    for task in outcome.results:
        if task.error:
            continue
        candidates.extend((task.sub_question_id, hit) for hit in task.hits)
    return _buy(candidates, pool, result, budget, cost, wanted=wanted, round_index=round_index)


def _admit_licensed(
    outcome: Any,
    pool: EvidencePool,
    result: ResearchResult,
    cost: Cost,
    *,
    round_index: int,
) -> int:
    """Hits already came from fetch_content; do not score them a second time."""
    new = 0
    for task in outcome.results:
        if task.error:
            continue
        cost.candidates_rejected += int(task.refused or 0)
        for hit in task.hits:
            added = pool.add([hit], task.sub_question_id, round_index)
            new += len(added)
            labels = {pool.key(item.hit): item.label for item in pool.items}
            already = not added
            result.ledger.record(
                PurchaseDecision(
                    doc_id=hit.chunk.doc_id,
                    chunk_id=hit.chunk.chunk_id,
                    title=hit.chunk.title,
                    source=hit.chunk.source,
                    published_at=hit.chunk.published_at,
                    sub_question_id=task.sub_question_id,
                    round_index=round_index,
                    value=0.0 if already else 1.0,
                    admitted=True,
                    reason=(
                        "already held; no additional charge"
                        if already
                        else f"bought {task.tradeoff_label or 'package'}"
                    ),
                    label=labels.get(pool.key(hit), ""),
                )
            )
    return new


def _buy(
    candidates: list[tuple[str, Any]],
    pool: EvidencePool,
    result: ResearchResult,
    budget: Budget,
    cost: Cost,
    *,
    wanted: list[str],
    round_index: int,
) -> int:
    """Admit what is worth buying, record every decision, return how much is new.

    With the gate off, everything retrieval offered is pooled up to the passage budget. That
    is the honest baseline the gate has to beat, and it is what `--no-purchase-gate` runs.
    """
    if not candidates:
        return 0

    budget_left = max(0, budget.max_unique_chunks - len(pool))

    if not budget.purchase_gate:
        admitted = candidates[:budget_left] if budget_left else []
        new = 0
        for sub_id, hit in admitted:
            new += len(pool.add([hit], sub_id, round_index))
        return new

    admitted, ledger = select(
        candidates,
        wanted=wanted,
        pooled_chunk_ids={pool.key(i.hit) for i in pool.items},
        pooled_doc_ids={i.doc_id for i in pool.items},
        budget_left=budget_left,
        round_index=round_index,
        min_value=budget.min_purchase_value,
    )

    new = 0
    for sub_id, hit in admitted:
        new += len(pool.add([hit], sub_id, round_index))

    # Stamp the label the pool assigned, so a ledger line can be traced to a citation.
    labels = {pool.key(item.hit): item.label for item in pool.items}
    for decision in ledger.decisions:
        if decision.admitted:
            decision.label = labels.get(decision.chunk_id or decision.doc_id, "")
        result.ledger.record(decision)
    cost.candidates_rejected += len(ledger.rejected)
    return new


def _timings(t0: float, t_plan: float, t_gather: float, t_assess: float, t_end: float) -> dict[str, float]:
    return {
        "plan": (t_plan - t0) * 1000,
        "gather": (t_gather - t_plan) * 1000,
        "assess": (t_assess - t_gather) * 1000,
        "write": (t_end - t_assess) * 1000,
        "total": (t_end - t0) * 1000,
    }
