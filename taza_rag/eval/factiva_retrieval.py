from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from taza_rag.eval.retrieval import intent_counts, load_gold
from taza_rag.factiva.pipeline import QualityRetriever
from taza_rag.models import RetrievedChunk

console = Console()


def _blob(hits: list[RetrievedChunk], k: int) -> str:
    parts = []
    for h in hits[:k]:
        parts.append(f"{h.chunk.title}\n{h.chunk.text}")
    return "\n".join(parts).lower()


def term_hit_rate(hits: list[RetrievedChunk], terms: list[str], k: int = 10) -> float:
    if not terms:
        return 1.0
    blob = _blob(hits, k)
    hits_n = 0
    for term in terms:
        if term.lower() in blob:
            hits_n += 1
    return hits_n / len(terms)


def salience_proxy(query: str, hits: list[RetrievedChunk], k: int = 5) -> float:
    """Cheap proxy: fraction of query tokens appearing in top-k titles/snippets."""
    q_terms = [t for t in re.findall(r"[a-z0-9]{3,}", query.lower()) if t not in {"the", "and", "for", "with"}]
    if not q_terms:
        return 1.0
    blob = _blob(hits, k)
    return sum(1 for t in q_terms if t in blob) / len(q_terms)


def authority_mix(hits: list[RetrievedChunk], k: int = 5) -> dict[str, int]:
    mix: dict[str, int] = defaultdict(int)
    for h in hits[:k]:
        code = str((h.chunk.metadata or {}).get("source_code") or h.chunk.source)
        mix[code] += 1
    return dict(mix)


def run_factiva_retrieval_eval(
    gold_path: Path,
    report_path: Path,
    *,
    top_k: int = 10,
    limit: int | None = None,
    config_name: str = "factiva_quality_v1",
) -> dict[str, Any]:
    """Offline-capable retrieval eval against live Factiva (no OpenAI)."""
    gold = load_gold(gold_path)
    if limit:
        gold = gold[:limit]

    retriever = QualityRetriever()
    rows: list[dict[str, Any]] = []
    term_rates: list[float] = []
    saliences: list[float] = []
    by_intent: dict[str, list[float]] = defaultdict(list)

    for ex in gold:
        run = retriever.retrieve(ex.query, top_k=top_k, intent=ex.intent)
        terms = ex.must_include_terms
        thr = term_hit_rate(run.hits, terms, k=top_k)
        sal = salience_proxy(ex.query, run.hits, k=min(5, top_k))
        term_rates.append(thr)
        saliences.append(sal)
        by_intent[ex.intent.value].append(thr)

        rows.append(
            {
                "id": ex.id,
                "intent": ex.intent.value,
                "query": ex.query,
                "variants": run.variants,
                "term_hit@k": thr,
                "salience@5": sal,
                "authority_mix@5": authority_mix(run.hits, k=5),
                "latency_ms": run.latency_ms,
                "top": [
                    {
                        "rank": h.rank,
                        "doc_id": h.chunk.doc_id,
                        "title": h.chunk.title,
                        "source": h.chunk.source,
                        "published_at": h.chunk.published_at,
                        "score": h.score,
                        "scores": h.scores,
                        "excerpt": h.chunk.text[:240],
                    }
                    for h in run.hits
                ],
                # Human A1 worksheet stubs (Accuracy N/A until answer stage)
                "a1_human": {
                    "relevance_1_3": None,
                    "completeness_1_3": None,
                    "notes": "",
                    "failure_tags": [],
                },
            }
        )

    summary = {
        "config": config_name,
        "n": len(gold),
        "intent_mix": intent_counts(gold),
        "mean_term_hit@k": sum(term_rates) / max(1, len(term_rates)),
        "mean_salience@5": sum(saliences) / max(1, len(saliences)),
        "term_hit_by_intent": {k: sum(v) / len(v) for k, v in by_intent.items()},
        "rows": rows,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Human-readable markdown worksheet for A1 Relevance/Completeness on retrieval packs
    md_path = report_path.with_suffix(".md")
    md_path.write_text(_to_markdown(summary), encoding="utf-8")

    _print(summary)
    console.print(f"JSON → {report_path}")
    console.print(f"Human worksheet → {md_path}")
    return summary


def _to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Factiva retrieval eval — `{summary['config']}`",
        "",
        f"- n={summary['n']}",
        f"- mean term_hit@k={summary['mean_term_hit@k']:.3f}",
        f"- mean salience@5={summary['mean_salience@5']:.3f}",
        "",
        "Score **Relevance (1–3)** and **Completeness (1–3)** on the retrieved pack (pre-answer).",
        "Accuracy applies after generation; here we judge whether the pack could support a DJ-quality answer.",
        "",
    ]
    for row in summary["rows"]:
        lines.append(f"## {row['id']} · {row['intent']}")
        lines.append(f"**Query:** {row['query']}")
        lines.append(f"**Variants:** {', '.join(row['variants'])}")
        lines.append(
            f"**Auto:** term_hit={row['term_hit@k']:.2f} salience={row['salience@5']:.2f}"
        )
        lines.append("")
        for t in row["top"][:8]:
            lines.append(
                f"{t['rank']}. [{t['source']} · {t['published_at']}] {t['title']}  "
                f"(score={t['score']:.4f})"
            )
            lines.append(f"   {t['excerpt'].replace(chr(10), ' ')}")
        lines.append("")
        lines.append("- Relevance (1–3): ")
        lines.append("- Completeness (1–3): ")
        lines.append("- Failure tags: ")
        lines.append("")
    return "\n".join(lines)


def _print(summary: dict[str, Any]) -> None:
    table = Table(title=f"Factiva retrieval — {summary['config']}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("n", str(summary["n"]))
    table.add_row("mean term_hit@k", f"{summary['mean_term_hit@k']:.3f}")
    table.add_row("mean salience@5", f"{summary['mean_salience@5']:.3f}")
    console.print(table)
    console.print("By intent:", summary["term_hit_by_intent"])
