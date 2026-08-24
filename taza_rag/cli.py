from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from taza_rag.config import settings
from taza_rag.eval.a1_factiva import run_a1_eval
from taza_rag.eval.factiva_retrieval import run_factiva_retrieval_eval
from taza_rag.eval.run import run_eval
from taza_rag.factiva.answer import answer_with_factiva
from taza_rag.factiva.auth import FactivaAuth
from taza_rag.factiva.pipeline import QualityRetriever
from taza_rag.factiva.strategy import detect_intent
from taza_rag.generate.answer import answer_query
from taza_rag.index.store import HybridIndex
from taza_rag.ingest import build_chunks
from taza_rag.ingest.corpus import load_corpus_jsonl
from taza_rag.llm import LLMError
from taza_rag.models import SearchIntent

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Taza RAG — retrieval quality first (Factiva). Generation is optional.",
)
console = Console()

_SECRET_KEYS = {
    "openai_api_key",
    "factiva_rag_password",
    "factiva_rag_portal_password",
    "factiva_feed_password",
    "factiva_feed_portal_password",
}


@app.command("retrieve")
def retrieve_cmd(
    q: str = typer.Argument(..., help="Research question"),
    top_k: int = typer.Option(10, min=1, max=50),
    days_range: Optional[str] = typer.Option(None, help="Override Factiva days_range"),
    intent: Optional[str] = typer.Option(None, help="Force Factiva intent name"),
    out: Optional[Path] = typer.Option(None, help="Write ranked JSON"),
    raw: bool = typer.Option(False, help="Single Factiva call, skip multi-query quality stack"),
    variants: int = typer.Option(3, min=1, max=6, help="Query variants to issue in parallel"),
    diversity: bool = typer.Option(
        True, "--diversity/--no-diversity", help="Ablation: source cap + MMR"
    ),
    entity_gate: bool = typer.Option(
        True, "--entity-gate/--no-entity-gate", help="Ablation: drop off-entity candidates"
    ),
    contextual: bool = typer.Option(
        True,
        "--contextual/--no-contextual",
        help="Contextual retrieval: rank contextualized passages, not whole articles",
    ),
    llm_context: bool = typer.Option(
        False, "--llm-context", help="Write passage context with an LLM (needs OPENAI_API_KEY)"
    ),
    semantic: bool = typer.Option(
        False, "--semantic", help="Add embedding similarity (needs OPENAI_API_KEY)"
    ),
    passage_tokens: int = typer.Option(180, help="Target passage size for contextual retrieval"),
) -> None:
    """Primary command: high-quality Factiva retrieval (no OpenAI)."""
    forced: SearchIntent | None = SearchIntent(intent) if intent else None
    if raw:
        from taza_rag.factiva.retrieve import FactivaRetrievalClient

        hits = FactivaRetrievalClient().retrieve(
            q, limit=top_k, days_range=days_range or "Last6Months"
        )
        console.print(
            f"[dim]baseline: single Factiva call, API order only (no quality scoring)[/dim]\n"
            f"intent≈{detect_intent(q).value}  n={len(hits)}\n"
        )
        payload = []
        for h in hits:
            _print_hit(h, show_scores=False)
            payload.append({"rank": h.rank, **h.chunk.model_dump()})
    else:
        run = QualityRetriever().retrieve(
            q,
            top_k=top_k,
            intent=forced,
            days_range=days_range,
            max_variants=variants,
            diversity=diversity,
            entity_gate=entity_gate,
            contextual=contextual,
            llm_context=llm_context,
            semantic=semantic,
            passage_tokens=passage_tokens,
        )
        plan = run.plan
        console.print(
            f"[bold]intent[/bold]={run.intent.value}  "
            f"[bold]entities[/bold]={plan.entities if plan else []}  "
            f"[bold]topics[/bold]={plan.topics if plan else []}"
        )
        console.print(
            f"[bold]config[/bold]={run.config}  "
            f"[bold]variants[/bold]={run.variants}"
        )
        console.print(
            f"[bold]articles[/bold]={run.candidates}  "
            f"[bold]passages[/bold]={run.passages}  "
            f"[bold]latency_ms[/bold]="
            f"{ {k: round(v) for k, v in run.latency_ms.items()} }"
        )
        if run.failed_variants:
            console.print(f"[yellow]variants failed upstream:[/yellow] {run.failed_variants}")
        console.print()
        payload = []
        for h in run.hits:
            _print_hit(h)
            payload.append(
                {
                    "rank": h.rank,
                    "score": h.score,
                    "scores": h.scores,
                    **h.chunk.model_dump(),
                }
            )

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"Wrote {out}")


def _print_hit(h, show_scores: bool = True) -> None:
    c = h.chunk
    kind = (c.metadata or {}).get("doc_kind")
    header = f"[bold]#{h.rank}[/bold]"
    if show_scores:
        header += f" ({h.score:.3f})"
    console.print(f"{header} {c.title}")
    tail = f" | {kind}" if kind else ""
    passages = (c.metadata or {}).get("passage_count")
    if passages:
        tail += f" | passage {c.chunk_index + 1}/{passages}"
    console.print(f"  {c.source} | {c.published_at} | {c.doc_id}{tail}")
    if show_scores and h.scores:
        semantic = h.scores.get("semantic", 0.0)
        line = (
            f"  [dim]entity={h.scores.get('entity', 0):.2f} "
            f"topic={h.scores.get('topic', 0):.2f} "
            f"bm25={h.scores.get('bm25', 0):.2f} "
            f"rrf={h.scores.get('rrf', 0):.2f} "
        )
        if semantic:
            line += f"semantic={semantic:.2f} "
        line += (
            f"auth={h.scores.get('authority', 1):.2f} "
            f"fresh={h.scores.get('freshness', 1):.2f} "
            f"penalty={h.scores.get('penalty', 0):.2f}[/dim]"
        )
        console.print(line)
    console.print(f"  {c.text[:300].replace(chr(10), ' ')}…\n")


@app.command("eval-retrieve")
def eval_retrieve_cmd(
    gold: Path = typer.Option(Path("evals/gold/factiva_live_v1.jsonl")),
    report: Path = typer.Option(Path("evals/reports/factiva_retrieve_latest.json")),
    top_k: int = typer.Option(10),
    limit: Optional[int] = typer.Option(None, help="Only first N gold rows"),
    compare: bool = typer.Option(
        False,
        "--compare/--no-compare",
        help="Also run the single-call baseline and report the delta (2x API calls)",
    ),
    contextual: bool = typer.Option(
        True, "--contextual/--no-contextual", help="Ablation: contextual passage retrieval"
    ),
    semantic: bool = typer.Option(
        False, "--semantic", help="Add embedding similarity (needs OPENAI_API_KEY)"
    ),
    passage_tokens: int = typer.Option(180, help="Target passage size for contextual retrieval"),
) -> None:
    """Retrieval-quality eval on Factiva gold set — no OpenAI."""
    run_factiva_retrieval_eval(
        gold,
        report,
        top_k=top_k,
        limit=limit,
        compare_baseline=compare,
        contextual=contextual,
        semantic=semantic,
        passage_tokens=passage_tokens,
    )


@app.command("factiva-auth")
def factiva_auth(
    account: str = typer.Option("rag", help="rag | feed"),
) -> None:
    """Verify Dow Jones OAuth for RAG or News Feed account."""
    if account not in {"rag", "feed"}:
        raise typer.BadParameter("account must be rag or feed")
    auth = FactivaAuth(account=account)  # type: ignore[arg-type]
    token = auth.get_access_token(force=True)
    console.print(f"[green]OK[/green] {account} AuthZ token acquired ({len(token)} chars)")


@app.command("answer")
def answer_cmd(
    q: str = typer.Argument(...),
    top_k: int = typer.Option(8),
    days_range: Optional[str] = typer.Option(None),
    raw: bool = typer.Option(False, "--raw", help="Baseline: single Factiva call, no quality stack"),
    semantic: bool = typer.Option(False, "--semantic", help="Add embedding similarity"),
) -> None:
    """Optional: retrieve + generate answer (needs OPENAI_API_KEY). Quality path is `retrieve`."""
    if not settings.openai_api_key:
        console.print(
            "[yellow]OPENAI_API_KEY not set.[/yellow] "
            "Use `taza-rag retrieve` for retrieval-only (recommended for quality work)."
        )
        raise typer.Exit(code=2)
    try:
        result = answer_with_factiva(
            q, top_k=top_k, days_range=days_range, raw=raw, semantic=semantic
        )
    except LLMError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=3) from e
    console.print(f"[bold]config[/bold]={result.config_name}  abstained={result.abstained}\n")
    console.print(result.answer)
    console.print("\nCitations:")
    for c in result.citations:
        console.print(f"- [{c.doc_id}] {c.title} ({c.source}, {c.published_at})")
    console.print(f"\nLatency ms: {result.latency_ms}")


@app.command("eval-a1")
def eval_a1_cmd(
    gold: Path = typer.Option(Path("evals/gold/factiva_live_v1.jsonl")),
    report: Path = typer.Option(Path("evals/reports/a1_latest.json")),
    top_k: int = typer.Option(8, min=1, max=20),
    limit: Optional[int] = typer.Option(None, help="Only the first N gold queries"),
    compare: bool = typer.Option(
        False, "--compare/--no-compare", help="Also score the single-call baseline (2x LLM calls)"
    ),
) -> None:
    """Answer-level A1 eval: Accuracy gate + Relevance / Completeness / Clarity."""
    if not settings.openai_api_key:
        console.print(
            "[yellow]OPENAI_API_KEY not set.[/yellow] "
            "A1 Accuracy and Clarity are answer-level and need a generated answer to score."
        )
        raise typer.Exit(code=2)
    run_a1_eval(gold, report, top_k=top_k, limit=limit, compare_baseline=compare)
    console.print(f"\nJSON → {report}")
    console.print(f"Worksheet → {report.with_suffix('.md')}")


# --- Local sample-index tools (ablations / offline) ---


@app.command()
def ingest(
    corpus: Path = typer.Option(...),
    out: Optional[Path] = typer.Option(None),
    contextualize: bool = typer.Option(True),
    llm_context: bool = typer.Option(False),
) -> None:
    """Build local hybrid index from JSONL (needs embeddings API if embedding)."""
    if llm_context and not settings.openai_api_key:
        console.print("llm_context requires OPENAI_API_KEY")
        raise typer.Exit(2)
    settings.ensure_dirs()
    out = out or settings.index_dir
    docs = load_corpus_jsonl(corpus)
    chunks = build_chunks(docs, contextualize=contextualize, use_llm_context=llm_context)
    console.print(f"{len(docs)} docs → {len(chunks)} chunks; embedding…")
    index = HybridIndex.build(chunks)
    index.save(out)
    console.print(f"Saved → {out}")


@app.command("eval-local")
def eval_local_cmd(
    gold: Path = typer.Option(Path("evals/gold/v1.jsonl")),
    index_dir: Optional[Path] = typer.Option(None),
    report: Path = typer.Option(Path("evals/reports/local_latest.json")),
    judge: bool = typer.Option(False, "--judge/--no-judge"),
) -> None:
    """Local-index retrieval metrics (judge optional; needs OpenAI if --judge)."""
    if judge and not settings.openai_api_key:
        console.print("--judge requires OPENAI_API_KEY; running retrieval metrics only.")
        judge = False
    index = HybridIndex.load(index_dir or settings.index_dir)
    run_eval(index, gold, report, judge=judge, config_name="local_hybrid")
    console.print(f"Wrote {report}")


@app.command("query-local")
def query_local_cmd(
    q: str = typer.Argument(...),
    index_dir: Optional[Path] = typer.Option(None),
) -> None:
    """Local index + answer (needs OPENAI_API_KEY)."""
    if not settings.openai_api_key:
        raise typer.Exit(code=2)
    result = answer_query(HybridIndex.load(index_dir or settings.index_dir), q)
    console.print(result.answer)


@app.command()
def show_config() -> None:
    data = settings.model_dump(mode="json")
    for k in _SECRET_KEYS:
        if data.get(k):
            data[k] = "***"
    console.print(json.dumps(data, indent=2, default=str))


# Back-compat aliases
@app.command("factiva-retrieve", hidden=True)
def factiva_retrieve_alias(
    q: str = typer.Argument(...),
    limit: int = typer.Option(10),
    days_range: str = typer.Option("Last6Months"),
    out: Optional[Path] = typer.Option(None),
) -> None:
    retrieve_cmd(q=q, top_k=limit, days_range=days_range, intent=None, out=out, raw=False)


if __name__ == "__main__":
    app()
