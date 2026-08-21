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
from taza_rag.factiva.retrieve import FactivaRetrievalClient
from taza_rag.factiva.strategy import default_days_range
from taza_rag.models import RetrievedChunk
from taza_rag.retrieve.features import LEAD_CHARS, build_query_plan, doc_kind, entity_signal

console = Console()


def _blob(hits: list[RetrievedChunk], k: int) -> str:
    return "\n".join(f"{h.chunk.title}\n{h.chunk.text}" for h in hits[:k]).lower()


def term_hit_rate(hits: list[RetrievedChunk], terms: list[str], k: int = 10) -> float:
    if not terms:
        return 1.0
    blob = _blob(hits, k)
    return sum(1 for term in terms if term.lower() in blob) / len(terms)


def term_hit_lead(hits: list[RetrievedChunk], terms: list[str], k: int = 3) -> float:
    """Stricter than term_hit@k: required terms must reach the top-k headlines/leads.

    term_hit@k saturates at 1.0 because a long body mentions almost anything, so it
    cannot separate a good pack from a mediocre one. Position matters for the pack a
    downstream answer actually cites.
    """
    if not terms:
        return 1.0
    blob = "\n".join(
        f"{h.chunk.title}\n{h.chunk.text[:LEAD_CHARS]}" for h in hits[:k]
    ).lower()
    return sum(1 for term in terms if term.lower() in blob) / len(terms)


def aspect_coverage(hits: list[RetrievedChunk], aspects: list[str], k: int = 5) -> float:
    """How many of the substantive angles the pack surfaces (Completeness proxy).

    `must_include_terms` only checks the subject was found, which the API does almost
    by construction. The `nice_to_have_terms` are the angles a Dow Jones-quality answer
    would be expected to cover, so they discriminate between packs.
    """
    if not aspects:
        return 1.0
    blob = "\n".join(
        f"{h.chunk.title}\n{h.chunk.text[:LEAD_CHARS]}" for h in hits[:k]
    ).lower()
    return sum(1 for a in aspects if a.lower() in blob) / len(aspects)


def salience_proxy(query: str, hits: list[RetrievedChunk], k: int = 5) -> float:
    """Cheap proxy: fraction of query tokens appearing in top-k titles/snippets."""
    q_terms = [
        t
        for t in re.findall(r"[a-z0-9]{3,}", query.lower())
        if t not in {"the", "and", "for", "with", "what", "are", "recent", "news", "latest"}
    ]
    if not q_terms:
        return 1.0
    blob = _blob(hits, k)
    return sum(1 for t in q_terms if t in blob) / len(q_terms)


def noise_rate(hits: list[RetrievedChunk], k: int = 5) -> float:
    """Share of top-k that are digests or company profiles rather than reported stories."""
    top = hits[:k]
    if not top:
        return 0.0
    bad = sum(
        1
        for h in top
        if ((h.chunk.metadata or {}).get("doc_kind") or doc_kind(h)) in {"digest", "profile"}
    )
    return bad / len(top)


def entity_precision(query: str, intent, hits: list[RetrievedChunk], k: int = 5) -> float:
    """Share of top-k that actually name the query entity in title or lead."""
    plan = build_query_plan(query, intent)
    if not plan.entity_tokens:
        return 1.0
    top = hits[:k]
    if not top:
        return 0.0
    named = 0
    for h in top:
        sig = entity_signal(plan, h)
        if max(sig["entity_title"], sig["entity_lead"]) > 0:
            named += 1
    return named / len(top)


def authority_mix(hits: list[RetrievedChunk], k: int = 5) -> dict[str, int]:
    mix: dict[str, int] = defaultdict(int)
    for h in hits[:k]:
        code = str((h.chunk.metadata or {}).get("source_code") or h.chunk.source)
        mix[code] += 1
    return dict(mix)


def _metrics(
    query: str,
    intent,
    hits: list[RetrievedChunk],
    terms: list[str],
    aspects: list[str],
    top_k: int,
) -> dict[str, Any]:
    return {
        "term_hit@k": term_hit_rate(hits, terms, k=top_k),
        "term_hit_lead@3": term_hit_lead(hits, terms, k=3),
        "aspect_coverage@5": aspect_coverage(hits, aspects, k=min(5, top_k)),
        "salience@5": salience_proxy(query, hits, k=min(5, top_k)),
        "entity_precision@5": entity_precision(query, intent, hits, k=min(5, top_k)),
        "noise_rate@5": noise_rate(hits, k=min(5, top_k)),
    }


def run_factiva_retrieval_eval(
    gold_path: Path,
    report_path: Path,
    *,
    top_k: int = 10,
    limit: int | None = None,
    compare_baseline: bool = False,
    config_name: str = "factiva_quality_v2",
) -> dict[str, Any]:
    """Retrieval-quality eval against live Factiva (no OpenAI required)."""
    gold = load_gold(gold_path)
    if limit:
        gold = gold[:limit]

    retriever = QualityRetriever()
    baseline_client = FactivaRetrievalClient(auth=retriever.client.auth) if compare_baseline else None

    rows: list[dict[str, Any]] = []
    agg: dict[str, list[float]] = defaultdict(list)
    base_agg: dict[str, list[float]] = defaultdict(list)
    by_intent: dict[str, list[float]] = defaultdict(list)

    for ex in gold:
        run = retriever.retrieve(ex.query, top_k=top_k, intent=ex.intent)
        m = _metrics(
            ex.query,
            ex.intent,
            run.hits,
            ex.must_include_terms,
            ex.nice_to_have_terms,
            top_k,
        )
        for key, value in m.items():
            agg[key].append(value)
        by_intent[ex.intent.value].append(m["term_hit@k"])

        row: dict[str, Any] = {
            "id": ex.id,
            "intent": ex.intent.value,
            "query": ex.query,
            "entities": run.plan.entities if run.plan else [],
            "topics": run.plan.topics if run.plan else [],
            "variants": run.variants,
            "candidates": run.candidates,
            **m,
            "authority_mix@5": authority_mix(run.hits, k=5),
            "latency_ms": run.latency_ms,
            "top": [
                {
                    "rank": h.rank,
                    "doc_id": h.chunk.doc_id,
                    "title": h.chunk.title,
                    "source": h.chunk.source,
                    "published_at": h.chunk.published_at,
                    "doc_kind": (h.chunk.metadata or {}).get("doc_kind"),
                    "score": h.score,
                    "scores": h.scores,
                    "excerpt": h.chunk.text[:240],
                }
                for h in run.hits
            ],
            # Human A1 worksheet stubs (Accuracy applies at answer stage)
            "a1_human": {
                "relevance_1_3": None,
                "completeness_1_3": None,
                "notes": "",
                "failure_tags": [],
            },
        }

        if baseline_client is not None:
            base_hits = baseline_client.retrieve(
                ex.query, limit=top_k, days_range=default_days_range(ex.intent)
            )
            bm = _metrics(
                ex.query,
                ex.intent,
                base_hits,
                ex.must_include_terms,
                ex.nice_to_have_terms,
                top_k,
            )
            for key, value in bm.items():
                base_agg[key].append(value)
            row["baseline"] = bm

        rows.append(row)

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    summary: dict[str, Any] = {
        "config": config_name,
        "n": len(gold),
        "intent_mix": intent_counts(gold),
        "mean_term_hit@k": mean(agg["term_hit@k"]),
        "mean_term_hit_lead@3": mean(agg["term_hit_lead@3"]),
        "mean_aspect_coverage@5": mean(agg["aspect_coverage@5"]),
        "mean_salience@5": mean(agg["salience@5"]),
        "mean_entity_precision@5": mean(agg["entity_precision@5"]),
        "mean_noise_rate@5": mean(agg["noise_rate@5"]),
        "term_hit_by_intent": {k: mean(v) for k, v in by_intent.items()},
        "rows": rows,
    }
    if base_agg:
        summary["baseline"] = {
            "mean_term_hit@k": mean(base_agg["term_hit@k"]),
            "mean_term_hit_lead@3": mean(base_agg["term_hit_lead@3"]),
            "mean_aspect_coverage@5": mean(base_agg["aspect_coverage@5"]),
            "mean_salience@5": mean(base_agg["salience@5"]),
            "mean_entity_precision@5": mean(base_agg["entity_precision@5"]),
            "mean_noise_rate@5": mean(base_agg["noise_rate@5"]),
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

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
        f"- mean term_hit_lead@3={summary['mean_term_hit_lead@3']:.3f}",
        f"- mean aspect_coverage@5={summary['mean_aspect_coverage@5']:.3f}",
        f"- mean salience@5={summary['mean_salience@5']:.3f}",
        f"- mean entity_precision@5={summary['mean_entity_precision@5']:.3f}",
        f"- mean noise_rate@5={summary['mean_noise_rate@5']:.3f} (lower is better)",
        "",
        "Score **Relevance (1–3)** and **Completeness (1–3)** on each retrieved pack.",
        "Accuracy applies after generation; here we judge whether the pack could support",
        "a Dow Jones-quality answer.",
        "",
    ]
    if "baseline" in summary:
        b = summary["baseline"]
        lines += [
            "## Baseline (single Factiva call) vs quality stack",
            "",
            "| Metric | Baseline | Quality | Delta |",
            "|--------|----------|---------|-------|",
            _md_row("term_hit@k", b["mean_term_hit@k"], summary["mean_term_hit@k"]),
            _md_row(
                "term_hit_lead@3",
                b["mean_term_hit_lead@3"],
                summary["mean_term_hit_lead@3"],
            ),
            _md_row(
                "aspect_coverage@5",
                b["mean_aspect_coverage@5"],
                summary["mean_aspect_coverage@5"],
            ),
            _md_row("salience@5", b["mean_salience@5"], summary["mean_salience@5"]),
            _md_row(
                "entity_precision@5",
                b["mean_entity_precision@5"],
                summary["mean_entity_precision@5"],
            ),
            _md_row("noise_rate@5", b["mean_noise_rate@5"], summary["mean_noise_rate@5"]),
            "",
        ]

    for row in summary["rows"]:
        lines.append(f"## {row['id']} · {row['intent']}")
        lines.append(f"**Query:** {row['query']}")
        lines.append(f"**Entities:** {row['entities']}  **Topics:** {row['topics']}")
        lines.append(f"**Variants:** {', '.join(row['variants'])}")
        lines.append(
            f"**Auto:** term_hit={row['term_hit@k']:.2f} "
            f"term_hit_lead@3={row['term_hit_lead@3']:.2f} "
            f"aspect_cov={row['aspect_coverage@5']:.2f} "
            f"salience={row['salience@5']:.2f} "
            f"entity_prec={row['entity_precision@5']:.2f} "
            f"noise={row['noise_rate@5']:.2f}"
        )
        lines.append("")
        for t in row["top"][:8]:
            kind = t.get("doc_kind") or "article"
            lines.append(
                f"{t['rank']}. [{t['source']} · {t['published_at']} · {kind}] {t['title']}  "
                f"(score={t['score']:.3f})"
            )
            lines.append(f"   {t['excerpt'].replace(chr(10), ' ')}")
        lines.append("")
        lines.append("- Relevance (1–3): ")
        lines.append("- Completeness (1–3): ")
        lines.append("- Failure tags: ")
        lines.append("")
    return "\n".join(lines)


def _md_row(name: str, baseline: float, quality: float) -> str:
    return f"| {name} | {baseline:.3f} | {quality:.3f} | {quality - baseline:+.3f} |"


def _print(summary: dict[str, Any]) -> None:
    table = Table(title=f"Factiva retrieval — {summary['config']}")
    table.add_column("Metric")
    table.add_column("Quality", justify="right")
    if "baseline" in summary:
        table.add_column("Baseline", justify="right")
        table.add_column("Delta", justify="right")

    def add(label: str, key: str) -> None:
        q = summary[f"mean_{key}"]
        if "baseline" in summary:
            b = summary["baseline"][f"mean_{key}"]
            table.add_row(label, f"{q:.3f}", f"{b:.3f}", f"{q - b:+.3f}")
        else:
            table.add_row(label, f"{q:.3f}")

    table.add_row("n", str(summary["n"]))
    add("term_hit@k", "term_hit@k")
    add("term_hit_lead@3", "term_hit_lead@3")
    add("aspect_coverage@5", "aspect_coverage@5")
    add("salience@5", "salience@5")
    add("entity_precision@5", "entity_precision@5")
    add("noise_rate@5 (lower better)", "noise_rate@5")
    console.print(table)
    console.print("term_hit@k by intent:", summary["term_hit_by_intent"])
