from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from taza_rag.config import settings
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
) -> None:
    """Primary command: high-quality Factiva retrieval (no OpenAI)."""
    forced: SearchIntent | None = SearchIntent(intent) if intent else None
    if raw:
        from taza_rag.factiva.retrieve import FactivaRetrievalClient

        hits = FactivaRetrievalClient().retrieve(
            q, limit=top_k, days_range=days_range or "Last6Months"
        )
        console.print(f"[dim]raw factiva[/dim] intent≈{detect_intent(q).value} n={len(hits)}\n")
        payload = []
        for h in hits:
            _print_hit(h)
            payload.append({"rank": h.rank, "score": h.score, **h.chunk.model_dump()})
    else:
        run = QualityRetriever().retrieve(
            q, top_k=top_k, intent=forced, days_range=days_range
        )
        console.print(
            f"[bold]intent[/bold]={run.intent.value}  "
            f"[bold]variants[/bold]={run.variants}  "
            f"[bold]latency_ms[/bold]={run.latency_ms}\n"
        )
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


def _print_hit(h) -> None:
    c = h.chunk
    console.print(f"[bold]#{h.rank}[/bold] ({h.score:.4f}) {c.title}")
    console.print(f"  {c.source} | {c.published_at} | {c.doc_id}")
    if h.scores:
        console.print(
            f"  [dim]rrf={h.scores.get('rrf', 0):.4f} "
            f"lex={h.scores.get('lex', 0):.2f} "
            f"auth={h.scores.get('authority', 1):.2f} "
            f"fresh={h.scores.get('freshness', 1):.2f}[/dim]"
        )
    console.print(f"  {c.text[:300].replace(chr(10), ' ')}…\n")


@app.command("eval-retrieve")
def eval_retrieve_cmd(
    gold: Path = typer.Option(Path("evals/gold/factiva_live_v1.jsonl")),
    report: Path = typer.Option(Path("evals/reports/factiva_retrieve_latest.json")),
    top_k: int = typer.Option(10),
    limit: Optional[int] = typer.Option(None, help="Only first N gold rows"),
) -> None:
    """Retrieval-quality eval on Factiva gold set — no OpenAI."""
    run_factiva_retrieval_eval(gold, report, top_k=top_k, limit=limit)


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
    days_range: str = typer.Option("Last6Months"),
) -> None:
    """Optional: retrieve + generate answer (needs OPENAI_API_KEY). Quality path is `retrieve`."""
    if not settings.openai_api_key:
        console.print(
            "[yellow]OPENAI_API_KEY not set.[/yellow] "
            "Use `taza-rag retrieve` for retrieval-only (recommended for quality work)."
        )
        raise typer.Exit(code=2)
    result = answer_with_factiva(q, limit=top_k, days_range=days_range)
    console.print(result.answer)
    console.print("\nCitations:")
    for c in result.citations:
        console.print(f"- [{c.doc_id}] {c.title} ({c.source}, {c.published_at})")
    console.print(f"\nLatency ms: {result.latency_ms}")


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
