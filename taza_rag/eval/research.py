"""Answer-level evaluation for the research agent.

The deterministic metrics lead and the judge follows, for a reason established on the
single-question path: re-scoring the same 52 answers with a different judge moved A1 Accuracy
from 0.538 to 0.827 without changing a byte of the answers. A band that wide cannot grade the
effects being measured, so the numbers this harness leads with are string and rank
computations with no model in the loop.

Three of them are specific to a multi-step agent and are the ones worth reading:

- **calibration error** — the gap between the coverage the agent believed it had when it
  stopped and the coverage its answer actually delivers. This is the only number that says
  whether the stopping rule can be trusted, and it is the one an unbounded agent has no way
  to report at all.
- **plan disjointness** — the share of retrieved passages that only one sub-question found.
  A plan whose steps all return the same articles is not decomposing anything, and this is
  the deterministic way to see it.
- **cost per covered aspect** — unique passages bought per gold aspect actually delivered.
  In a marketplace where every chunk is billed, an answer is not better for costing more.

A fourth metric was removed after its first run rather than kept and explained away. "Plan
facet coverage" compared gold facet labels against sub-question wording and scored 0.194 on
plans that were plainly correct: the agent asked "What financial results has SoftBank Group
reported?" against a gold facet written as "quarterly earnings". The match is semantic, and no
lexical threshold recovers it, so the metric was measuring vocabulary agreement between my
shorthand and the planner's phrasing. `facets` stays in the gold file as a statement of intent;
plan quality is now read from disjointness and from what the answer delivers.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from taza_rag.agent.aspects import satisfied
from taza_rag.agent.loop import research
from taza_rag.agent.models import Budget, ResearchResult
from taza_rag.agent.text import overlap
from taza_rag.config import settings
from taza_rag.eval.dj_a1 import judge_a1, judge_model_name
from taza_rag.factiva.retrieve import FactivaRetrieveError, hits_to_citations
from taza_rag.llm import LLMError
from taza_rag.models import AnswerResult

console = Console()


@dataclass
class ResearchGold:
    id: str
    question: str
    facets: list[str] = field(default_factory=list)
    must_cover: list[str] = field(default_factory=list)
    notes: str = ""


def load_gold(path: Path) -> list[ResearchGold]:
    rows: list[ResearchGold] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        rows.append(
            ResearchGold(
                id=str(raw["id"]),
                question=str(raw["question"]),
                facets=[str(f) for f in raw.get("facets") or []],
                must_cover=[str(a) for a in raw.get("must_cover") or []],
                notes=str(raw.get("notes") or ""),
            )
        )
    return rows


def plan_disjointness(result: ResearchResult) -> float:
    """Share of pooled passages that only one sub-question found.

    The deterministic read on whether the decomposition did any work. Near 1.0 means the steps
    looked at genuinely different evidence; near 0.0 means they were paraphrases of each other
    and the run paid several times over for one set of articles.
    """
    if not result.evidence:
        return 0.0
    alone = sum(1 for item in result.evidence if len(item.found_by) == 1)
    return alone / len(result.evidence)


def answer_aspect_coverage(gold: ResearchGold, result: ResearchResult) -> tuple[float, list[str]]:
    """Share of the gold aspects the finished answer actually delivers.

    Judged against the answer text, not the evidence pool: retrieving the material and then
    failing to write it is the defect the single-question path spent most of its effort on.
    """
    if not gold.must_cover:
        return 0.0, []
    missing = [a for a in gold.must_cover if not satisfied(a, [result.answer])]
    return (len(gold.must_cover) - len(missing)) / len(gold.must_cover), missing


def _as_answer_result(result: ResearchResult) -> AnswerResult:
    """Adapt a research run for the existing A1 judge."""
    used = [item.hit for item in result.evidence if f"[{item.label}]" in result.answer]
    return AnswerResult(
        query=result.question,
        answer=result.answer,
        citations=hits_to_citations(used or [i.hit for i in result.evidence[:3]]),
        retrieved=[i.hit for i in result.evidence],
        context="",
        abstained=result.abstained,
        latency_ms=result.latency_ms,
        config_name=result.config_name,
    )


def _evidence_excerpts(result: ResearchResult) -> str:
    return "\n\n".join(f"[{i.label}] {i.text}" for i in result.evidence)


def _unsupported_after_repair(result: ResearchResult) -> int:
    final = (result.verification or {}).get("final") or {}
    problems = final.get("problems") or {}
    if isinstance(problems, dict):
        return int(sum(int(v) for v in problems.values()))
    return 0


def run_research_eval(
    gold_path: Path,
    report_path: Path,
    *,
    budget: Budget | None = None,
    limit: int | None = None,
    judge: bool = True,
    judge_model: str | None = None,
    verify: bool = True,
) -> dict[str, Any]:
    gold = load_gold(gold_path)
    if limit:
        gold = gold[:limit]
    budget = budget or Budget()

    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for ex in gold:
        try:
            result = research(ex.question, budget=budget, verify=verify)
        except (LLMError, FactivaRetrieveError) as e:
            # A provider failure is not an evaluation result.
            failures.append(f"{ex.id}: {type(e).__name__}: {e}")
            continue

        coverage, missing = answer_aspect_coverage(ex, result)
        covered_aspects = len(ex.must_cover) - len(missing)
        row: dict[str, Any] = {
            "id": ex.id,
            "question": ex.question,
            "config": result.config_name,
            "plan_method": result.plan.method if result.plan else "none",
            "sub_questions": [s.question for s in (result.plan.sub_questions if result.plan else [])],
            "plan_disjointness": plan_disjointness(result),
            "answer_aspect_coverage": coverage,
            "covered_aspects": covered_aspects,
            "missing_aspects": missing,
            "self_coverage": result.coverage,
            # Positive means the agent thought it had more than it delivered.
            "calibration_error": result.coverage - coverage,
            "stop_reason": result.stop_reason,
            "rounds": len(result.rounds),
            "findings": len(result.findings),
            "disagreements": sum(1 for c in result.conflicts if c.kind == "disagreement"),
            "rounding_restatements": sum(1 for c in result.conflicts if c.kind == "rounding"),
            "gaps_declared": len(result.gaps),
            "unsupported_after_repair": _unsupported_after_repair(result),
            "abstained": result.abstained,
            "answer_words": len(result.answer.split()),
            "cost": result.cost.payload(),
            "latency_ms": {k: round(v) for k, v in result.latency_ms.items()},
            "answer": result.answer,
            "errors": result.errors,
        }

        if judge and settings.openai_api_key and not result.abstained:
            try:
                judgment = judge_a1(
                    ex.id, _as_answer_result(result), _evidence_excerpts(result), model=judge_model
                )
                row["a1"] = judgment.model_dump()
                row["accuracy_pass"] = judgment.accuracy.pass_
                row["overall_pass"] = judgment.overall_pass
            except LLMError as e:
                failures.append(f"{ex.id} (judge): {type(e).__name__}: {e}")

        rows.append(row)

    summary = {
        "config": rows[0]["config"] if rows else "research_v1",
        "n": len(gold),
        "n_scored": len(rows),
        "generator_model": settings.answer_model or settings.chat_model,
        "judge_model": judge_model_name(judge_model) if judge else None,
        "judge_model_note": "LLM judge; scores vary between runs. Deterministic metrics lead.",
        "budget": budget.payload(),
        **_aggregate(rows),
        "errors": failures,
        "rows": rows,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, report_path.with_suffix(".md"))
    _print(summary)
    return summary


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    judged = [r for r in rows if "accuracy_pass" in r]
    unique = [r["cost"]["unique_chunks"] for r in rows]
    covered = [r["answer_aspect_coverage"] for r in rows]
    # Passages bought per aspect actually delivered, not per unit of coverage ratio.
    cost_per_aspect = [
        r["cost"]["unique_chunks"] / r["covered_aspects"]
        for r in rows
        if r["covered_aspects"] > 0
    ]

    stop_reasons: dict[str, int] = {}
    for r in rows:
        stop_reasons[r["stop_reason"]] = stop_reasons.get(r["stop_reason"], 0) + 1

    missing: dict[str, int] = {}
    for r in rows:
        for aspect in r["missing_aspects"]:
            missing[aspect] = missing.get(aspect, 0) + 1

    out: dict[str, Any] = {
        "mean_plan_disjointness": _mean([r["plan_disjointness"] for r in rows]),
        "mean_answer_aspect_coverage": _mean(covered),
        "mean_self_coverage": _mean([r["self_coverage"] for r in rows]),
        "mean_calibration_error": _mean([r["calibration_error"] for r in rows]),
        "mean_abs_calibration_error": _mean([abs(r["calibration_error"]) for r in rows]),
        "mean_rounds": _mean([float(r["rounds"]) for r in rows]),
        "mean_unique_chunks": _mean([float(u) for u in unique]),
        "mean_reuse_rate": _mean([r["cost"]["reuse_rate"] for r in rows]),
        "mean_cost_per_covered_aspect": _mean(cost_per_aspect),
        "mean_answer_words": _mean([float(r["answer_words"]) for r in rows]),
        "total_disagreements": sum(r["disagreements"] for r in rows),
        "total_rounding_restatements": sum(r["rounding_restatements"] for r in rows),
        "total_gaps_declared": sum(r["gaps_declared"] for r in rows),
        "unsupported_after_repair": sum(r["unsupported_after_repair"] for r in rows),
        "abstention_rate": sum(1 for r in rows if r["abstained"]) / len(rows),
        "stop_reasons": stop_reasons,
        "top_missing_aspects": sorted(missing.items(), key=lambda kv: -kv[1])[:12],
        "median_latency_ms": statistics.median([r["latency_ms"]["total"] for r in rows]),
    }
    if judged:
        out["n_judged"] = len(judged)
        out["accuracy_pass_rate"] = _mean([1.0 if r["accuracy_pass"] else 0.0 for r in judged])
        out["overall_pass_rate"] = _mean([1.0 if r["overall_pass"] else 0.0 for r in judged])
        for dim in ("relevance", "completeness", "clarity"):
            out[f"mean_{dim}"] = _mean([float(r["a1"][dim]["score"]) for r in judged])
    return out


def _print(summary: dict[str, Any]) -> None:
    table = Table(title=f"Research agent eval — {summary['config']}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    def row(label: str, key: str, fmt: str = "{:.3f}") -> None:
        if key in summary and summary[key] is not None:
            table.add_row(label, fmt.format(summary[key]))

    row("Plan disjointness", "mean_plan_disjointness")
    row("Answer aspect coverage", "mean_answer_aspect_coverage")
    row("Self-assessed coverage", "mean_self_coverage")
    row("Calibration error (signed)", "mean_calibration_error", "{:+.3f}")
    row("Calibration error (absolute)", "mean_abs_calibration_error")
    row("Rounds", "mean_rounds", "{:.2f}")
    row("Unique passages bought", "mean_unique_chunks", "{:.1f}")
    row("Passage reuse rate", "mean_reuse_rate")
    row("Cost per covered aspect", "mean_cost_per_covered_aspect", "{:.1f}")
    row("Answer words", "mean_answer_words", "{:.0f}")
    row("Accuracy pass rate", "accuracy_pass_rate")
    row("Overall pass rate", "overall_pass_rate")
    row("Completeness (1-3)", "mean_completeness", "{:.2f}")
    row("Median latency (ms)", "median_latency_ms", "{:.0f}")
    console.print(table)
    console.print(f"Stop reasons: {summary.get('stop_reasons')}")
    console.print(
        f"Disagreements surfaced: {summary.get('total_disagreements')}  "
        f"rounding restatements filtered: {summary.get('total_rounding_restatements')}  "
        f"gaps declared: {summary.get('total_gaps_declared')}"
    )
    if summary.get("errors"):
        console.print(f"[yellow]{len(summary['errors'])} failed:[/yellow] {summary['errors'][:3]}")


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Research agent eval — {summary['config']}",
        "",
        f"- questions: {summary['n']} (scored: {summary['n_scored']})",
        f"- generator: `{summary['generator_model']}`  judge: `{summary.get('judge_model')}`",
        f"- budget: {summary['budget']}",
        "",
        "## Deterministic metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| plan disjointness | {summary.get('mean_plan_disjointness', 0):.3f} |",
        f"| answer aspect coverage | {summary.get('mean_answer_aspect_coverage', 0):.3f} |",
        f"| self-assessed coverage | {summary.get('mean_self_coverage', 0):.3f} |",
        f"| calibration error (signed) | {summary.get('mean_calibration_error', 0):+.3f} |",
        f"| calibration error (absolute) | {summary.get('mean_abs_calibration_error', 0):.3f} |",
        f"| rounds | {summary.get('mean_rounds', 0):.2f} |",
        f"| unique passages bought | {summary.get('mean_unique_chunks', 0):.1f} |",
        f"| passage reuse rate | {summary.get('mean_reuse_rate', 0):.3f} |",
        f"| cost per covered aspect | {summary.get('mean_cost_per_covered_aspect', 0):.1f} |",
        f"| unsupported claims after repair | {summary.get('unsupported_after_repair', 0)} |",
        f"| median latency ms | {summary.get('median_latency_ms', 0):.0f} |",
        "",
        f"Stop reasons: `{summary.get('stop_reasons')}`",
        "",
    ]
    if "accuracy_pass_rate" in summary:
        lines += [
            "## A1 (judge)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Accuracy gate | {summary['accuracy_pass_rate']:.3f} |",
            f"| Relevance | {summary.get('mean_relevance', 0):.2f} |",
            f"| Completeness | {summary.get('mean_completeness', 0):.2f} |",
            f"| Clarity | {summary.get('mean_clarity', 0):.2f} |",
            f"| Overall pass | {summary['overall_pass_rate']:.3f} |",
            "",
        ]

    lines += ["## Per-question", ""]
    for r in summary["rows"]:
        lines += [
            f"### {r['id']} — {r['question']}",
            "",
            f"- plan ({r['plan_method']}): " + "; ".join(r["sub_questions"]),
            f"- disjointness {r['plan_disjointness']:.2f} | aspect coverage "
            f"{r['answer_aspect_coverage']:.2f} | self {r['self_coverage']:.2f} "
            f"| stop `{r['stop_reason']}` after {r['rounds']} round(s)",
            f"- cost {r['cost']}",
        ]
        if r["missing_aspects"]:
            lines.append(f"- missing: {r['missing_aspects']}")
        if r["disagreements"]:
            lines.append(f"- disagreements surfaced: {r['disagreements']}")
        lines += ["", "> " + (r["answer"] or "").replace("\n", "\n> "), ""]
    path.write_text("\n".join(lines), encoding="utf-8")
