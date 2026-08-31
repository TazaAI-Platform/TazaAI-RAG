from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape

from taza_rag.config import settings
from taza_rag.eval.a1_factiva import rejudge_report, run_a1_eval
from taza_rag.eval.factiva_retrieval import run_factiva_retrieval_eval
from taza_rag.eval.run import run_eval
from taza_rag.factiva.answer import answer_with_factiva
from taza_rag.factiva.auth import FactivaAuth
from taza_rag.factiva.pipeline import QualityRetriever
from taza_rag.factiva.retrieve import FactivaRetrieveError
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

# Client ids are not passwords, but they identify the trial account and pair with the
# passwords, so `show-config` output pasted into an issue or shown in a demo would give
# away half the credential set for nothing.
_SECRET_KEYS = {
    "openai_api_key",
    "factiva_rag_client_id",
    "factiva_rag_password",
    "factiva_rag_portal_password",
    "factiva_rag_username",
    "factiva_rag_portal_user",
    "factiva_feed_client_id",
    "factiva_feed_password",
    "factiva_feed_portal_password",
    "factiva_feed_username",
    "factiva_feed_portal_user",
    "factiva_metrics_user_id",
    "ui_share_token",
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
    console.print(f"{header} {escape(c.title or '')}")
    tail = f" | {kind}" if kind else ""
    passages = (c.metadata or {}).get("passage_count")
    if passages:
        tail += f" | passage {c.chunk_index + 1}/{passages}"
    console.print(f"  {c.source} | {c.published_at} | {c.doc_id}{tail}", markup=False)
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
    console.print(f"  {c.text[:300].replace(chr(10), ' ')}…\n", markup=False)


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
    verify: bool = typer.Option(
        True, "--verify/--no-verify", help="Ground-check claims and repair before returning"
    ),
    facts: bool = typer.Option(
        True,
        "--facts/--no-facts",
        help="Extract cited facts, then write the answer from that list",
    ),
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
            q,
            top_k=top_k,
            days_range=days_range,
            raw=raw,
            semantic=semantic,
            verify=verify,
            extract_facts=facts,
        )
    except LLMError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=3) from e
    console.print(f"[bold]config[/bold]={result.config_name}  abstained={result.abstained}")
    if result.verification:
        console.print(f"[dim]verification: {escape(str(result.verification))}[/dim]")
    console.print()
    # markup=False throughout: rich reads [c1] as a style tag and silently deletes it, so a
    # correctly cited answer printed as markup looks entirely uncited.
    console.print(result.answer, markup=False)
    console.print("\nCitations:")
    for c in result.citations:
        console.print(f"- [{c.doc_id}] {c.title} ({c.source}, {c.published_at})", markup=False)
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
    judge_model: Optional[str] = typer.Option(
        None, "--judge-model", help="Score with a different model than the generator"
    ),
    verify: bool = typer.Option(
        True, "--verify/--no-verify", help="Ground-check claims and repair before scoring"
    ),
    facts: bool = typer.Option(
        True,
        "--facts/--no-facts",
        help="Extract cited facts before writing; --no-facts is the previous one-shot path",
    ),
) -> None:
    """Answer-level A1 eval: Accuracy gate + Relevance / Completeness / Clarity."""
    if not settings.openai_api_key:
        console.print(
            "[yellow]OPENAI_API_KEY not set.[/yellow] "
            "A1 Accuracy and Clarity are answer-level and need a generated answer to score."
        )
        raise typer.Exit(code=2)
    run_a1_eval(
        gold,
        report,
        top_k=top_k,
        limit=limit,
        compare_baseline=compare,
        judge_model=judge_model,
        verify=verify,
        extract_facts=facts,
    )
    console.print(f"\nJSON → {report}")
    console.print(f"Worksheet → {report.with_suffix('.md')}")


@app.command("rejudge-a1")
def rejudge_a1_cmd(
    judge_model: str = typer.Option(..., "--judge-model", help="e.g. gpt-5, gpt-4.1, o3"),
    source: Path = typer.Option(Path("evals/reports/a1_latest.json"), "--source"),
    report: Optional[Path] = typer.Option(None, "--report"),
) -> None:
    """Re-score stored answers with a different judge, isolating the judge from generation."""
    if not settings.openai_api_key:
        console.print("[yellow]OPENAI_API_KEY not set.[/yellow]")
        raise typer.Exit(code=2)
    out = report or source.with_name(f"{source.stem}_judge_{judge_model.replace('.', '')}.json")
    try:
        rejudge_report(source, out, judge_model=judge_model)
    except (ValueError, LLMError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=3) from e
    console.print(f"\nJSON → {out}")
    console.print(f"Worksheet → {out.with_suffix('.md')}")


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
    console.print(result.answer, markup=False)


@app.command("research")
def research_cmd(
    q: str = typer.Argument(..., help="Complex research question"),
    top_k: int = typer.Option(6, min=1, max=20, help="Passages per sub-question query"),
    max_rounds: int = typer.Option(3, min=1, max=6, help="Retrieval rounds allowed"),
    max_chunks: int = typer.Option(40, min=4, max=200, help="Unique passage budget"),
    max_sub: int = typer.Option(5, min=1, max=8, help="Sub-questions the plan may hold"),
    target: float = typer.Option(0.8, help="Aspect coverage that counts as answered"),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="Ground-check and repair"),
    llm_plan: bool = typer.Option(
        True, "--llm-plan/--no-llm-plan", help="--no-llm-plan uses heuristic query expansion"
    ),
    purchase_gate: bool = typer.Option(
        True,
        "--purchase-gate/--no-purchase-gate",
        help="Buy only passages whose metadata suggests they close an open gap",
    ),
    out: Optional[Path] = typer.Option(None, help="Write the full run as JSON"),
) -> None:
    """Multi-step research agent: shop packages, judge sufficiency, answer."""
    if not settings.openai_api_key:
        console.print(
            "[yellow]OPENAI_API_KEY not set.[/yellow] "
            "The agent needs it to plan, extract facts and write. Use `taza-rag retrieve` "
            "for the key-free retrieval path."
        )
        raise typer.Exit(code=2)

    from taza_rag.agent.loop import research
    from taza_rag.agent.models import Budget
    from taza_rag.agent.synthesize import conflict_note

    budget = Budget(
        max_rounds=max_rounds,
        max_unique_chunks=max_chunks,
        max_sub_questions=max_sub,
        top_k_per_query=top_k,
        target_coverage=target,
        purchase_gate=purchase_gate,
    )
    try:
        result = research(q, budget=budget, verify=verify, use_llm_plan=llm_plan)
    except (LLMError, FactivaRetrieveError) as e:
        console.print(f"[red]{type(e).__name__}: {e}[/red]")
        raise typer.Exit(code=3) from e

    plan = result.plan
    if plan:
        console.print(
            f"[bold]plan[/bold]={plan.method}  [bold]intent[/bold]={plan.intent.value}  "
            f"[bold]entities[/bold]={plan.entities}"
        )
        for sub in plan.sub_questions:
            cov = result.sub_coverage.get(sub.id)
            cov_text = f"{cov:.2f}" if cov is not None else "n/a"
            console.print(f"  [dim]{sub.id}[/dim] coverage={cov_text}  {escape(sub.question)}")
            if sub.aspects:
                console.print(f"      [dim]aspects: {escape(', '.join(sub.aspects))}[/dim]")

    console.print(
        f"\n[bold]coverage[/bold]={result.coverage:.3f}  "
        f"[bold]stop[/bold]={result.stop_reason}  "
        f"[bold]rounds[/bold]={len(result.rounds)}"
    )
    for r in result.rounds:
        console.print(
            f"  [dim]round {r.index}: {len(r.queries)} query/queries, "
            f"+{r.new_chunks} new passages, +{r.new_findings} new facts, "
            f"coverage {r.coverage:.2f} ({r.coverage_delta:+.2f})[/dim]"
        )
    console.print(f"[dim]conflicts: {conflict_note(result.conflicts)}[/dim]")
    if result.gaps:
        console.print("[yellow]not covered by the sources:[/yellow]")
        for gap in result.gaps:
            console.print(f"  - {escape(gap.aspect)} [dim]({gap.sub_question_id})[/dim]")
    if result.errors:
        console.print(f"[yellow]{len(result.errors)} step error(s):[/yellow] {result.errors[:3]}")

    # markup=False: rich reads [c1] as a style tag and deletes it, which turns a correctly
    # cited answer into an apparently uncited one on screen.
    console.print(f"\n[bold]answer[/bold]  ({'abstained' if result.abstained else 'answered'})")
    console.print(result.answer, markup=False)

    console.print("\nCitations:")
    for item in result.evidence:
        if f"[{item.label}]" not in result.answer:
            continue
        c = item.hit.chunk
        console.print(
            f"- [{item.label}] {c.title} ({c.source}, {c.published_at}) {c.doc_id}",
            markup=False,
        )
    if result.ledger.decisions:
        led = result.ledger
        console.print(
            f"\n[bold]purchases[/bold] {len(led.charged)} bought of {len(led.decisions)} offered  "
            f"[dim]{len(led.admitted) - len(led.charged)} already held, "
            f"{len(led.rejected)} refused[/dim]"
        )
        for reason, count in sorted(led.rejection_reasons().items(), key=lambda kv: -kv[1]):
            console.print(f"  [dim]refused ×{count}: {escape(reason)}[/dim]")

    console.print(f"\nCost: {result.cost.payload()}")
    console.print(f"Latency ms: { {k: round(v) for k, v in result.latency_ms.items()} }")

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.payload(), indent=2), encoding="utf-8")
        console.print(f"Wrote {out}")


@app.command("eval-research")
def eval_research_cmd(
    gold: Path = typer.Option(Path("evals/gold/research_v1.jsonl")),
    report: Path = typer.Option(Path("evals/reports/research_latest.json")),
    limit: Optional[int] = typer.Option(None, help="Only the first N questions"),
    top_k: int = typer.Option(6, min=1, max=20),
    max_rounds: int = typer.Option(3, min=1, max=6),
    max_chunks: int = typer.Option(40, min=4, max=200),
    judge: bool = typer.Option(True, "--judge/--no-judge", help="Also score A1 on the answer"),
    judge_model: Optional[str] = typer.Option(None, "--judge-model"),
    verify: bool = typer.Option(True, "--verify/--no-verify"),
    purchase_gate: bool = typer.Option(True, "--purchase-gate/--no-purchase-gate"),
) -> None:
    """Research-agent eval: plan coverage, answer coverage, stopping calibration, cost."""
    if not settings.openai_api_key:
        console.print("[yellow]OPENAI_API_KEY not set.[/yellow] The agent cannot plan or write.")
        raise typer.Exit(code=2)

    from taza_rag.agent.models import Budget
    from taza_rag.eval.research import run_research_eval

    run_research_eval(
        gold,
        report,
        budget=Budget(
            max_rounds=max_rounds,
            max_unique_chunks=max_chunks,
            top_k_per_query=top_k,
            purchase_gate=purchase_gate,
        ),
        limit=limit,
        judge=judge,
        judge_model=judge_model,
        verify=verify,
    )
    console.print(f"\nJSON → {report}")
    console.print(f"Worksheet → {report.with_suffix('.md')}")


@app.command("mcp")
def mcp_cmd() -> None:
    """Serve marketplace MCP tools over stdio (query / transact / fetch_content)."""
    from taza_rag.mcp_server import serve

    # No console output: stdout is the protocol stream, and a stray line corrupts it.
    serve()


@app.command("ui")
def ui_cmd(
    host: Optional[str] = typer.Option(
        None, help="Bind address. Default 127.0.0.1; 0.0.0.0 if PORT is set (hosted)."
    ),
    port: Optional[int] = typer.Option(None, help="Port. Default 8765, or $PORT when hosted."),
    demo: Optional[bool] = typer.Option(
        None,
        "--demo/--live",
        help="Sample-corpus playground (default when Factiva credentials are missing).",
    ),
) -> None:
    """Query playground: ask the marketplace, pick a package, fetch licensed content."""
    import os

    from taza_rag.ui.server import serve

    env_port = os.environ.get("PORT")
    bind_host = host or os.environ.get("HOST") or ("0.0.0.0" if env_port else "127.0.0.1")
    bind_port = port or int(env_port or 8765)
    serve(bind_host, bind_port, demo=demo)


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
