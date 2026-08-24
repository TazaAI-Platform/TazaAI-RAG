from __future__ import annotations

import time
from typing import Any

from taza_rag.config import settings
from taza_rag.factiva.pipeline import QualityRetriever
from taza_rag.factiva.retrieve import FactivaRetrievalClient, hits_to_citations
from taza_rag.factiva.verify import (
    REPAIR_SYSTEM,
    VerificationReport,
    _has_factual_content,
    describe_problems,
    split_claims,
    verify_answer,
)
from taza_rag.llm import LLMError, chat_json
from taza_rag.models import AnswerResult, RetrievedChunk, SearchIntent

ANSWER_SYSTEM = """You are a Dow Jones / Factiva-aligned research assistant.
Answer ONLY using the provided source chunks.

Coverage — this is what separates an adequate answer from a good one:
- Before writing, identify every distinct substantive point in the sources that bears on the
  question: results and figures, causes, counterparties, timing, official or executive
  comment, consequences, and any contrary view.
- Cover each of those points. A point supported by the sources and left out is a defect, not
  concision.
- Give each point its own sentence carrying its own concrete figure or fact. Do not
  generalise several figures into one vague clause.
- Aim for 200-350 words for a normal question. Length is a consequence of covering the
  material, not a target in itself: never pad, repeat, or add commentary to reach it.

Accuracy — these constraints are absolute and outrank coverage:
- Every significant claim must include a citation marker like [c1], [c2] matching chunk labels.
- Do not invent facts, numbers, names, or dates not present in sources.
- Carry figures over exactly as the sources state them (amounts, percentages, dates,
  counterparties). Never estimate, round, or combine figures from different sources.
- State the significance of the facts only so far as the sources support it; do not speculate.
- Where the sources disagree or include a dissenting, cautionary or contrary view, say so and
  attribute it. Never manufacture a disagreement that the sources do not contain.
- Prefer higher-authority / premium sources when details conflict; mention contradictions
  explicitly.

Form: lead with the direct answer to the question, then the supporting specifics, then any
contrary view. Professionally journalistic (Dow Jones Voice).

If the sources genuinely do not address the question, set abstain=true and explain what is
missing. Do not abstain merely because the picture is partial — report what the sources do
support and note what is absent.
Return JSON with keys: answer (string), abstain (boolean), used_citations (list of chunk labels like "c1").
"""


def _evidence_by_label(selected: list[RetrievedChunk]) -> dict[str, str]:
    return {
        f"c{i}": f"{h.chunk.title}\n{h.chunk.text}" for i, h in enumerate(selected, start=1)
    }


def _pack_context(
    hits: list[RetrievedChunk], max_chunks: int, max_tokens: int
) -> tuple[str, list[RetrievedChunk]]:
    """Fill the context to a token budget, not a chunk count.

    Passages are roughly half the size of a whole article, so a fixed chunk count
    silently hands the generator half the evidence and costs Completeness. Budgeting
    by tokens lets the passage path spend its saving on more distinct sources instead.
    """
    selected: list[RetrievedChunk] = []
    used = 0
    for h in hits[:max_chunks]:
        tokens = len((h.chunk.text or "").split())
        if selected and used + tokens > max_tokens:
            break
        selected.append(h)
        used += tokens

    blocks = []
    for i, h in enumerate(selected, start=1):
        c = h.chunk
        blocks.append(
            f"[c{i}] doc_id={c.doc_id} | {c.source} | {c.published_at or 'n/a'} | {c.title}\n{c.text}"
        )
    return "\n\n".join(blocks), selected


def _abstain(query: str, config_name: str, elapsed_ms: float) -> AnswerResult:
    return AnswerResult(
        query=query,
        answer="Insufficient evidence in Factiva retrieval results to answer.",
        citations=[],
        retrieved=[],
        abstained=True,
        latency_ms={"retrieve": elapsed_ms, "total": elapsed_ms},
        config_name=config_name,
    )


def answer_with_factiva(
    query: str,
    *,
    top_k: int = 8,
    days_range: str | None = None,
    intent: SearchIntent | None = None,
    raw: bool = False,
    contextual: bool = True,
    semantic: bool = False,
    verify: bool = True,
    config_name: str | None = None,
) -> AnswerResult:
    """Retrieve from Factiva, then ground an answer with citations.

    Retrieval defaults to the full quality stack, so A1 scores describe the system
    that is actually shipped. `raw=True` drops to a single Factiva call in API order,
    which is the baseline the ranking work has to beat at the answer level too.
    """
    t0 = time.perf_counter()
    if raw:
        hits = FactivaRetrievalClient().retrieve(
            query, limit=top_k, days_range=days_range or "Last6Months"
        )
        used_config = config_name or "factiva_raw"
    else:
        run = QualityRetriever().retrieve(
            query,
            top_k=top_k,
            intent=intent,
            days_range=days_range,
            contextual=contextual,
            semantic=semantic,
        )
        hits = run.hits
        used_config = config_name or run.config
    t1 = time.perf_counter()

    if not hits:
        return _abstain(query, used_config, (t1 - t0) * 1000)

    context, selected = _pack_context(
        hits, settings.answer_max_chunks, settings.answer_context_tokens
    )
    user = f"Question: {query}\n\nSources:\n{context}"
    raw_json: dict[str, Any] = chat_json(ANSWER_SYSTEM, user, temperature=0.0)
    t2 = time.perf_counter()

    answer_text = str(raw_json.get("answer") or "")
    abstained = bool(raw_json.get("abstain"))
    verification: dict[str, Any] | None = None

    if verify and answer_text and not abstained:
        answer_text, abstained, raw_json, verification = _verify_and_repair(
            query,
            context,
            answer_text,
            abstained,
            raw_json,
            _evidence_by_label(selected),
            max_rounds=settings.verify_max_rounds,
        )
    t3 = time.perf_counter()

    label_to_hit = {f"c{i}": h for i, h in enumerate(selected, start=1)}
    used = []
    for label in raw_json.get("used_citations") or []:
        hit = label and label_to_hit.get(str(label))
        if hit:
            used.append(hit)

    citations = hits_to_citations(used or selected[:3])
    return AnswerResult(
        query=query,
        answer=answer_text,
        citations=citations,
        retrieved=selected,
        context=context,
        abstained=abstained,
        verification=verification,
        latency_ms={
            "retrieve": (t1 - t0) * 1000,
            "generate": (t2 - t1) * 1000,
            "verify": (t3 - t2) * 1000,
            "total": (t3 - t0) * 1000,
        },
        config_name=used_config + ("+verified" if verify else ""),
    )


def _is_substantive(text: str) -> bool:
    """Does this text actually answer, whatever flag came back with it?

    The repair prompt asks for `abstain=true` when little survives, and the model sets it
    while still returning a full cited answer — it reads the flag as "I removed something".
    Trusting it marked a quarter of answerable queries as refusals in a 52-query run, all
    of them carrying real answers. The text is the evidence, not the flag.
    """
    claims = split_claims(text)
    return any(c.effective_labels and _has_factual_content(c.text) for c in claims)


def _verify_and_repair(
    query: str,
    context: str,
    answer_text: str,
    abstained: bool,
    raw_json: dict[str, Any],
    evidence: dict[str, str],
    *,
    max_rounds: int,
) -> tuple[str, bool, dict[str, Any], dict[str, Any]]:
    """Repair until the checks are clean, the budget runs out, or progress stalls.

    A single pass left roughly a third of flagged claims standing, and it re-checked only
    the deterministic rules — so the entailment failures that dominate the remainder were
    never re-tested. Each round here re-runs the full check, which is what makes an
    unsupported-claim fix verifiable rather than assumed.

    Rewriting is not monotonic: a pass that drops a bad figure can overstate something
    else. So every attempt is scored and the best one is returned, which means enabling
    this can leave an answer unchanged but never degrade it.
    """
    report = verify_answer(answer_text, evidence)
    rounds = [report.summary()]
    best = (answer_text, abstained, raw_json, report)

    for _ in range(max_rounds):
        if not best[3].blocking:
            break
        repaired = _repair(query, context, best[0], best[3], attempt=len(rounds))
        if repaired is None:
            break
        text, new_abstain, new_json = repaired
        new_abstain = new_abstain and not _is_substantive(text)
        new_report = verify_answer(text, evidence)
        rounds.append(new_report.summary())
        # Ties keep the earlier answer: each rewrite thins the answer, and Completeness
        # is already the binding constraint.
        if len(new_report.blocking) >= len(best[3].blocking):
            break
        best = (text, new_abstain, new_json, new_report)

    verification = {
        "initial": rounds[0],
        "final": best[3].summary(),
        "rounds": rounds,
        "repairs_applied": len(rounds) - 1,
        "resolved": not best[3].blocking,
    }
    return best[0], best[1], best[2], verification


def _repair(
    query: str, context: str, answer: str, report: VerificationReport, *, attempt: int = 1
) -> tuple[str, bool, dict[str, Any]] | None:
    """One corrective pass. Returns None if the model could not be reached."""
    problems = describe_problems(report, split_claims(answer))
    if not problems:
        return None
    retry_note = (
        ""
        if attempt <= 1
        else (
            f"\n\nThis is correction attempt {attempt}. A previous rewrite still failed "
            "these checks, so remove the offending material rather than rephrasing it."
        )
    )
    user = (
        f"Question: {query}\n\nSources:\n{context}\n\n"
        f"Answer that failed verification:\n{answer}\n\n"
        f"Problems found:\n{problems}{retry_note}"
    )
    try:
        fixed: dict[str, Any] = chat_json(REPAIR_SYSTEM, user, temperature=0.0)
    except LLMError:
        return None
    text = str(fixed.get("answer") or "").strip()
    if not text:
        return None
    return text, bool(fixed.get("abstain")), fixed
