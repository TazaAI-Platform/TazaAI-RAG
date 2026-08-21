from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from taza_rag.eval.dj_a1 import judge_a1
from taza_rag.eval.retrieval import (
    hard_negative_rate,
    intent_counts,
    load_gold,
    precision_at_k,
    recall_at_k,
)
from taza_rag.generate.answer import answer_query
from taza_rag.index.store import HybridIndex
from taza_rag.retrieve.hybrid import hybrid_retrieve, simple_rerank

console = Console()


def run_eval(
    index: HybridIndex,
    gold_path: Path,
    report_path: Path,
    judge: bool = True,
    config_name: str = "contextual_hybrid",
) -> dict[str, Any]:
    gold = load_gold(gold_path)
    rows: list[dict[str, Any]] = []

    recall10_all: list[float] = []
    prec5_all: list[float] = []
    hn5_all: list[float] = []
    a1_pass = 0
    a1_n = 0
    by_intent: dict[str, list[float]] = defaultdict(list)

    for ex in gold:
        hits = hybrid_retrieve(index, ex.query)
        hits = simple_rerank(ex.query, hits)
        relevant = ex.must_include_doc_ids or ex.acceptable_doc_ids
        r10 = recall_at_k(hits, ex.must_include_doc_ids, k=10)
        p5 = precision_at_k(hits, relevant, k=5)
        hn = hard_negative_rate(hits, ex.hard_negative_doc_ids, k=5)
        recall10_all.append(r10)
        prec5_all.append(p5)
        hn5_all.append(hn)
        by_intent[ex.intent.value].append(r10)

        row: dict[str, Any] = {
            "id": ex.id,
            "intent": ex.intent.value,
            "query": ex.query,
            "recall@10": r10,
            "precision@5": p5,
            "hard_negative@5": hn,
            "top_docs": [h.chunk.doc_id for h in hits[:5]],
        }

        if judge:
            result = answer_query(index, ex.query, config_name=config_name)
            excerpts = "\n\n".join(
                f"[{h.chunk.doc_id}] {h.chunk.title}\n{h.chunk.text[:600]}" for h in result.retrieved
            )
            judgment = judge_a1(ex.id, result, excerpts)
            row["a1"] = judgment.model_dump()
            row["a1_pass"] = judgment.overall_pass
            row["answer_preview"] = result.answer[:400]
            a1_n += 1
            if judgment.overall_pass:
                a1_pass += 1

        rows.append(row)

    summary = {
        "config": config_name,
        "n": len(gold),
        "intent_mix": intent_counts(gold),
        "mean_recall@10": sum(recall10_all) / max(1, len(recall10_all)),
        "mean_precision@5": sum(prec5_all) / max(1, len(prec5_all)),
        "mean_hard_negative@5": sum(hn5_all) / max(1, len(hn5_all)),
        "recall@10_by_intent": {k: sum(v) / len(v) for k, v in by_intent.items()},
        "a1_pass_rate": (a1_pass / a1_n) if a1_n else None,
        "rows": rows,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_summary(summary)
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    table = Table(title=f"Eval — {summary['config']}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("n", str(summary["n"]))
    table.add_row("mean Recall@10", f"{summary['mean_recall@10']:.3f}")
    table.add_row("mean Precision@5", f"{summary['mean_precision@5']:.3f}")
    table.add_row("mean HardNeg@5", f"{summary['mean_hard_negative@5']:.3f}")
    if summary["a1_pass_rate"] is not None:
        table.add_row("A1 pass rate", f"{summary['a1_pass_rate']:.3f}")
    console.print(table)
    console.print("Intent mix:", summary["intent_mix"])
    console.print("Recall@10 by intent:", summary["recall@10_by_intent"])
