"""Answer-level evaluation against the Dow Jones A1 Core Evaluation Criteria.

The retrieval eval measures the evidence pack. This measures the answer built from
it, which is the only place Accuracy (the rubric's hard gate) and Clarity can be
scored at all. Accuracy is four yes/no checks that must all hold; Relevance,
Completeness and Clarity are 1-3.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from taza_rag.config import settings
from taza_rag.eval.dj_a1 import judge_a1, judge_model_name
from taza_rag.eval.retrieval import load_gold
from taza_rag.factiva.answer import answer_with_factiva
from taza_rag.llm import LLMError
from taza_rag.models import A1Judgment, AnswerResult, Citation, GoldExample

console = Console()

GATES = ("factual_correctness", "citation_integrity", "no_hallucinations", "contextual_integrity")


def _excerpts(result: AnswerResult) -> str:
    """The judge must score against exactly the evidence the answer was given.

    Truncating here silently penalised Accuracy: the generator saw the full passage
    while the judge saw the first 900 characters, so supported claims were flagged as
    unsupported. Prefer the verbatim context the generator received.
    """
    if result.context:
        return result.context
    return "\n\n".join(
        f"[c{i}] {h.chunk.title} | {h.chunk.source} | {h.chunk.published_at or 'n/a'}\n"
        f"{h.chunk.text}"
        for i, h in enumerate(result.retrieved, start=1)
    )


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _score_one(
    ex: GoldExample,
    *,
    raw: bool,
    top_k: int,
    judge_model: str | None = None,
    verify: bool = True,
) -> tuple[AnswerResult, A1Judgment, str]:
    result = answer_with_factiva(
        ex.query, top_k=top_k, intent=ex.intent, raw=raw, verify=verify
    )
    evidence = _excerpts(result)
    judgment = judge_a1(ex.id, result, evidence, model=judge_model)
    return result, judgment, evidence


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Answer quality is only meaningful where an answer was wanted. Judging a correct
    # refusal against the answer rubric scores the right behaviour as a failure.
    scored = [r for r in rows if r.get("a1") and not r["expect_abstention"]]
    by_intent: dict[str, list[float]] = defaultdict(list)
    tags: Counter[str] = Counter()
    for r in scored:
        by_intent[r["intent"]].append(1.0 if r["accuracy_pass"] else 0.0)
        tags.update(r["a1"]["failure_tags"])

    missing: Counter[str] = Counter()
    for r in scored:
        missing.update(a.lower().strip() for a in r["a1"].get("missing_aspects", []))

    expected_abstain = [r for r in rows if r["expect_abstention"]]
    answerable = [r for r in rows if r.get("a1") and not r["expect_abstention"]]

    if not scored:
        # An abstention-only run has no answer to score; zeroes would read as failure.
        return {
            "n_scored": 0,
            "n_expected_abstention": len(expected_abstain),
            "accuracy_pass_rate": None,
            "gate_pass_rates": {},
            "mean_relevance": None,
            "mean_completeness": None,
            "mean_clarity": None,
            "overall_pass_rate": None,
            "abstention_rate": None,
            "abstention_recall": (
                _mean([1.0 if r["abstained"] else 0.0 for r in expected_abstain])
                if expected_abstain
                else None
            ),
            "accuracy_pass_by_intent": {},
            "top_failure_tags": [],
            "top_missing_aspects": [],
        }

    return {
        "n_scored": len(scored),
        "n_expected_abstention": len(expected_abstain),
        "accuracy_pass_rate": _mean([1.0 if r["accuracy_pass"] else 0.0 for r in scored]),
        "gate_pass_rates": {
            g: _mean([1.0 if r["a1"]["accuracy"][g] else 0.0 for r in scored]) for g in GATES
        },
        "mean_relevance": _mean([r["a1"]["relevance"]["score"] for r in scored]),
        "mean_completeness": _mean([r["a1"]["completeness"]["score"] for r in scored]),
        "mean_clarity": _mean([r["a1"]["clarity"]["score"] for r in scored]),
        "overall_pass_rate": _mean([1.0 if r["overall_pass"] else 0.0 for r in scored]),
        # Refusing an answerable query is its own failure mode, so track both directions.
        "abstention_rate": _mean([1.0 if r["abstained"] else 0.0 for r in answerable]),
        "abstention_recall": (
            _mean([1.0 if r["abstained"] else 0.0 for r in expected_abstain])
            if expected_abstain
            else None
        ),
        "accuracy_pass_by_intent": {k: _mean(v) for k, v in by_intent.items()},
        "top_failure_tags": tags.most_common(8),
        "top_missing_aspects": missing.most_common(12),
    }


def run_a1_eval(
    gold_path: Path,
    report_path: Path,
    *,
    top_k: int = 8,
    limit: int | None = None,
    compare_baseline: bool = False,
    judge_model: str | None = None,
    verify: bool = True,
) -> dict[str, Any]:
    gold = load_gold(gold_path)
    if limit:
        gold = gold[:limit]

    rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for ex in gold:
        try:
            result, judgment, evidence = _score_one(
                ex, raw=False, top_k=top_k, judge_model=judge_model, verify=verify
            )
        except LLMError as e:
            # A billing or provider failure is not an evaluation result; record and move on.
            failures.append(f"{ex.id}: {e}")
            continue

        row = {
            "id": ex.id,
            "intent": ex.intent.value,
            "query": ex.query,
            "config": result.config_name,
            "expect_abstention": ex.expect_abstention,
            "abstained": result.abstained,
            "accuracy_pass": judgment.accuracy.pass_,
            "overall_pass": judgment.overall_pass,
            "a1": judgment.model_dump(),
            "answer": result.answer,
            "citations": [c.doc_id for c in result.citations],
            # Full citation records, not just doc ids: re-judging with these dropped makes
            # the judge fail citation integrity for an answer that was properly cited.
            "citations_full": [c.model_dump() for c in result.citations],
            "verification": result.verification,
            "latency_ms": result.latency_ms,
            # Kept so a different judge can score this exact answer later; re-generating
            # would change the answer and confound a judge comparison.
            "evidence": evidence,
        }
        rows.append(row)

        if compare_baseline:
            try:
                b_result, b_judgment, _ = _score_one(
                    ex, raw=True, top_k=top_k, judge_model=judge_model, verify=verify
                )
                base_rows.append(
                    {
                        "id": ex.id,
                        "intent": ex.intent.value,
                        "expect_abstention": ex.expect_abstention,
                        "abstained": b_result.abstained,
                        "accuracy_pass": b_judgment.accuracy.pass_,
                        "overall_pass": b_judgment.overall_pass,
                        "a1": b_judgment.model_dump(),
                    }
                )
            except LLMError as e:
                failures.append(f"{ex.id} (baseline): {e}")

    summary: dict[str, Any] = {
        "config": rows[0]["config"] if rows else "factiva_quality_v2+ctx",
        "n": len(gold),
        "generator_model": settings.chat_model,
        "judge_model": judge_model_name(judge_model),
        "judge_model_note": "LLM judge; scores vary slightly between runs",
        **_aggregate(rows),
        "errors": failures,
        "rows": rows,
    }
    if base_rows:
        summary["baseline"] = _aggregate(base_rows)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, report_path.with_suffix(".md"))
    _print(summary)
    return summary


def rejudge_report(
    source_report: Path,
    report_path: Path,
    *,
    judge_model: str,
) -> dict[str, Any]:
    """Re-score the answers stored in a previous report with a different judge.

    Holding the answers fixed is the only way to attribute a score change to the judge
    rather than to generation variance or a shifting live corpus.
    """
    source = json.loads(source_report.read_text(encoding="utf-8"))
    old_rows = source.get("rows") or []
    missing_evidence = [r["id"] for r in old_rows if not r.get("evidence")]
    if missing_evidence:
        raise ValueError(
            f"{source_report} has no stored evidence for {missing_evidence[:3]}; "
            "re-run `eval-a1` to record it before re-judging."
        )
    # Without the original citations the judge scores a differently-shaped answer, and
    # citation integrity fails for reasons the system never caused.
    missing_citations = [
        r["id"] for r in old_rows if r.get("citations") and not r.get("citations_full")
    ]
    if missing_citations:
        raise ValueError(
            f"{source_report} stores only citation doc ids for {missing_citations[:3]}; "
            "re-run `eval-a1` to record full citations before re-judging."
        )

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    disagreements: list[dict[str, Any]] = []

    for old in old_rows:
        stub = AnswerResult(
            query=old["query"],
            answer=old["answer"],
            citations=[Citation(**c) for c in old.get("citations_full") or []],
            retrieved=[],
            abstained=old["abstained"],
            config_name=old["config"],
            context=old["evidence"],
        )
        try:
            judgment = judge_a1(old["id"], stub, old["evidence"], model=judge_model)
        except LLMError as e:
            failures.append(f"{old['id']}: {e}")
            continue

        row = {**old, "a1": judgment.model_dump()}
        row["accuracy_pass"] = judgment.accuracy.pass_
        row["overall_pass"] = judgment.overall_pass
        rows.append(row)

        before, after = old["a1"], row["a1"]
        changed = {
            k: [before[k]["score"], after[k]["score"]]
            for k in ("relevance", "completeness", "clarity")
            if before[k]["score"] != after[k]["score"]
        }
        if old["accuracy_pass"] != row["accuracy_pass"]:
            changed["accuracy"] = [old["accuracy_pass"], row["accuracy_pass"]]
        if changed:
            disagreements.append({"id": old["id"], "query": old["query"], "changed": changed})

    summary: dict[str, Any] = {
        "config": source.get("config", "unknown"),
        "n": len(old_rows),
        "generator_model": source.get("generator_model", "unknown"),
        "judge_model": judge_model,
        "rejudged_from": str(source_report),
        "previous_judge_model": source.get("judge_model", "unknown"),
        "judge_model_note": "same stored answers, different judge",
        **_aggregate(rows),
        "judge_disagreements": disagreements,
        "judge_agreement_rate": 1.0 - (len(disagreements) / len(rows) if rows else 0.0),
        "errors": failures,
        "rows": rows,
    }
    summary["previous"] = _aggregate(old_rows)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, report_path.with_suffix(".md"))
    _print_rejudge(summary)
    return summary


def _print_rejudge(summary: dict[str, Any]) -> None:
    prev = summary["previous"]
    table = Table(
        title=f"Judge swap — {summary['previous_judge_model']} → {summary['judge_model']}"
    )
    table.add_column("Metric")
    table.add_column(str(summary["previous_judge_model"]), justify="right")
    table.add_column(str(summary["judge_model"]), justify="right")
    table.add_column("Delta", justify="right")

    for label, key, fmt in (
        ("Accuracy pass rate", "accuracy_pass_rate", "{:.3f}"),
        ("Relevance (1-3)", "mean_relevance", "{:.2f}"),
        ("Completeness (1-3)", "mean_completeness", "{:.2f}"),
        ("Clarity (1-3)", "mean_clarity", "{:.2f}"),
        ("Overall pass rate", "overall_pass_rate", "{:.3f}"),
    ):
        a, b = prev.get(key), summary.get(key)
        if a is None or b is None:
            continue
        table.add_row(label, fmt.format(a), fmt.format(b), f"{b - a:+.3f}")
    console.print(table)
    console.print(
        f"Per-query agreement: {summary['judge_agreement_rate']:.3f} "
        f"({len(summary['judge_disagreements'])} of {len(summary['rows'])} scored differently)"
    )
    for d in summary["judge_disagreements"][:8]:
        console.print(f"  [{d['id']}] {d['query'][:52]} → {d['changed']}")
    if summary["errors"]:
        console.print(f"[yellow]{len(summary['errors'])} failed:[/yellow] {summary['errors'][:3]}")


def _print(summary: dict[str, Any]) -> None:
    table = Table(title=f"A1 answer eval — {summary['config']}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    if summary.get("baseline"):
        table.add_column("Baseline", justify="right")
        table.add_column("Delta", justify="right")

    def row(label: str, key: str, fmt: str = "{:.3f}") -> None:
        cur = summary.get(key)
        if cur is None:
            return
        cells = [label, fmt.format(cur)]
        if summary.get("baseline"):
            base = summary["baseline"].get(key)
            cells.append(fmt.format(base) if base is not None else "—")
            cells.append(f"{cur - base:+.3f}" if base is not None else "—")
        table.add_row(*cells)

    row("Accuracy pass rate", "accuracy_pass_rate")
    row("Relevance (1-3)", "mean_relevance")
    row("Completeness (1-3)", "mean_completeness")
    row("Clarity (1-3)", "mean_clarity")
    row("Overall pass rate", "overall_pass_rate")
    row("Abstention rate", "abstention_rate")
    row("Abstention recall", "abstention_recall")
    console.print(table)
    if summary["gate_pass_rates"]:
        console.print(
            "Accuracy gates:", {k: round(v, 3) for k, v in summary["gate_pass_rates"].items()}
        )
        console.print(
            "Accuracy by intent:",
            {k: round(v, 3) for k, v in summary["accuracy_pass_by_intent"].items()},
        )
    if summary["top_failure_tags"]:
        console.print("Failure tags:", summary["top_failure_tags"])
    if summary["errors"]:
        console.print(f"[yellow]{len(summary['errors'])} query/queries failed:[/yellow]")
        for e in summary["errors"][:5]:
            console.print(f"  {e}")


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    def num(key: str, fmt: str = "{:.3f}") -> str:
        v = summary.get(key)
        return fmt.format(v) if v is not None else "n/a"

    lines = [
        f"# A1 answer eval — {summary['config']}",
        "",
        f"- queries: {summary['n']} (answer-scored: {summary['n_scored']}, "
        f"expected refusals: {summary['n_expected_abstention']})",
        f"- Accuracy pass rate: {num('accuracy_pass_rate')}",
        f"- Relevance: {num('mean_relevance', '{:.2f}')} / 3",
        f"- Completeness: {num('mean_completeness', '{:.2f}')} / 3",
        f"- Clarity: {num('mean_clarity', '{:.2f}')} / 3",
        f"- Overall pass rate: {num('overall_pass_rate')}",
        f"- Abstention recall: {num('abstention_recall')}",
        "",
        "## Accuracy gates",
        "",
        "| Gate | Pass rate |",
        "|------|-----------|",
    ]
    for g, v in summary["gate_pass_rates"].items():
        lines.append(f"| `{g}` | {v:.3f} |")

    if summary.get("baseline"):
        b = summary["baseline"]
        lines += [
            "",
            "## Quality stack vs single-call baseline",
            "",
            "| Metric | Baseline | Quality | Delta |",
            "|--------|----------|---------|-------|",
            f"| Accuracy pass rate | {b['accuracy_pass_rate']:.3f} | "
            f"{summary['accuracy_pass_rate']:.3f} | "
            f"{summary['accuracy_pass_rate'] - b['accuracy_pass_rate']:+.3f} |",
            f"| Relevance | {b['mean_relevance']:.2f} | {summary['mean_relevance']:.2f} | "
            f"{summary['mean_relevance'] - b['mean_relevance']:+.2f} |",
            f"| Completeness | {b['mean_completeness']:.2f} | {summary['mean_completeness']:.2f} | "
            f"{summary['mean_completeness'] - b['mean_completeness']:+.2f} |",
            f"| Clarity | {b['mean_clarity']:.2f} | {summary['mean_clarity']:.2f} | "
            f"{summary['mean_clarity'] - b['mean_clarity']:+.2f} |",
        ]

    lines += ["", "## Per-query", ""]
    for r in summary["rows"]:
        a1 = r["a1"]
        lines += [
            f"### {r['id']} — {r['query']}",
            "",
            f"- intent: `{r['intent']}` | abstained: {r['abstained']} "
            f"(expected: {r['expect_abstention']})",
            f"- Accuracy: {'PASS' if r['accuracy_pass'] else 'FAIL'} | "
            f"R={a1['relevance']['score']} C={a1['completeness']['score']} "
            f"Cl={a1['clarity']['score']}",
            f"- failure tags: {a1['failure_tags'] or 'none'}",
            f"- missing: {a1.get('missing_aspects') or 'nothing flagged'}",
            f"- citations: {r['citations']}",
            "",
            f"> {r['answer'][:700]}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
