"""Decide which retrieved passages are worth buying, before reading them.

The brief names this as one of the hardest problems in the business: *can we predict whether a
source will materially improve an agent's result before the information itself is revealed?*
Retrieval hands back candidates; an agent that pools all of them has not made a decision, it
has just spent money.

So admission to the evidence pool is a decision with a price attached. Only admitted passages
are billed, enter the pool, and become citable. Every decision — admit or reject — is recorded
with the value it scored and the reason, so the request path stays auditable rather than being
a model's opinion about relevance.

**The score uses only metadata: title, source, date, the ranker's own composite, and the lead
snippet.** It never reads the passage body. That constraint is the point. It keeps the decision
honest about what a buyer can see before paying, and it means the same function transfers
unchanged to a metadata-first API where the body is a separately priced fetch.

What the score rewards, in order of weight:

1. **Targeting** — does the headline or lead address an aspect that is still uncovered? This is
   the closest available proxy for "will this change the answer", and it is why the gate is
   given the current gap list rather than the whole plan.
2. **Rank** — the retrieval stack's own composite, normalised within the offered batch.
3. **Authority and freshness** — the same priors the ranker uses, so a purchase decision is
   explainable in the same terms as a ranking decision.

And what it penalises: a passage from a document already in the pool, because a second angle
on a story we already hold rarely earns its price.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from taza_rag.agent.aspects import classify, satisfied
from taza_rag.market import TRADEOFF_BALANCED, TRADEOFF_THOROUGH
from taza_rag.models import Chunk, RetrievedChunk

# Targeting dominates: a passage that speaks to a live gap is worth more than a
# marginally better-ranked one that does not.
W_TARGET = 1.0
W_MULTI = 0.15
W_RANK = 0.5
W_AUTH = 0.6
W_FRESH = 0.4
P_SAME_DOC = 0.45

# Below this a passage is not worth its price, and the threshold sits deliberately above
# W_RANK. Set lower, rank alone clears it: `rank_score` is normalised within the offered batch,
# so whatever tops the batch scores 1.0 and a single-candidate batch always scores 1.0. The
# gate would then buy anything retrieval happened to return, which is not a decision. Above
# W_RANK, a purchase needs either a targeted gap or a premium, fresh source behind it.
MIN_VALUE = 0.55

# How much of the passage the gate is allowed to see. A Factiva lead is the first paragraph,
# which is what a metadata-first API would expose alongside the headline.
LEAD_CHARS = 220


@dataclass
class PurchaseDecision:
    """One admit-or-reject decision, kept for the audit trail."""

    doc_id: str
    chunk_id: str
    title: str
    source: str
    published_at: str | None
    sub_question_id: str
    round_index: int
    value: float
    admitted: bool
    reason: str
    label: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "source": self.source,
            "published_at": self.published_at,
            "sub_question_id": self.sub_question_id,
            "round": self.round_index,
            "value": round(self.value, 3),
            "admitted": self.admitted,
            "reason": self.reason,
            "label": self.label,
        }


@dataclass
class Ledger:
    """Every purchase decision in a run, in the order it was made."""

    decisions: list[PurchaseDecision] = field(default_factory=list)

    def record(self, decision: PurchaseDecision) -> None:
        self.decisions.append(decision)

    @property
    def admitted(self) -> list[PurchaseDecision]:
        return [d for d in self.decisions if d.admitted]

    @property
    def charged(self) -> list[PurchaseDecision]:
        """Admissions that actually cost something.

        Re-offers of a passage already held are admitted for free, so counting them as
        purchases overstates spend — the first UI read "12 bought of 12 offered" beside a
        pack of 4.
        """
        return [d for d in self.decisions if d.admitted and not d.reason.startswith("already held")]

    @property
    def rejected(self) -> list[PurchaseDecision]:
        return [d for d in self.decisions if not d.admitted]

    def admission_rate(self) -> float:
        """Share of offered passages that were bought, counting only charged purchases."""
        if not self.decisions:
            return 0.0
        return len(self.charged) / len(self.decisions)

    def rejection_reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.rejected:
            key = d.reason.split(";")[0].strip()
            counts[key] = counts.get(key, 0) + 1
        return counts

    def payload(self) -> dict[str, Any]:
        return {
            "offered": len(self.decisions),
            "admitted": len(self.admitted),
            "charged": len(self.charged),
            "already_held": len(self.admitted) - len(self.charged),
            "rejected": len(self.rejected),
            "admission_rate": round(self.admission_rate(), 3),
            "rejection_reasons": self.rejection_reasons(),
            "decisions": [d.payload() for d in self.decisions],
        }


def metadata_of(hit: RetrievedChunk) -> str:
    """Everything the gate is permitted to look at: headline plus lead snippet."""
    chunk = hit.chunk
    return f"{chunk.title or ''}\n{(chunk.text or '')[:LEAD_CHARS]}"


def purchasable(wanted: list[str]) -> list[str]:
    """Keep only the aspects that can discriminate between candidates.

    Structural aspects — "financial statements", "official comment" — are satisfied by any
    passage carrying a number or an attribution, which is very nearly every business article.
    They are the right way to score *coverage*, and useless for deciding what to *buy*: with one
    of them in the plan, the first live run admitted 36 of 36 candidates and the gate was
    decorative. Only lexical aspects name something specific enough to shop for.
    """
    return [a for a in wanted if classify(a) == "lexical"]


def score_candidate(
    hit: RetrievedChunk,
    *,
    wanted: list[str],
    pooled_doc_ids: set[str],
    best_rank_score: float,
) -> tuple[float, str]:
    """Expected value of admitting this passage, and the reason in plain words."""
    chunk = hit.chunk
    meta = metadata_of(hit)
    notes: list[str] = []

    wanted = purchasable(wanted)
    matched = [a for a in wanted if satisfied(a, [meta])]
    if wanted:
        targeting = 1.0 if matched else 0.0
        if matched:
            shown = ", ".join(repr(a) for a in matched[:2])
            notes.append(f"targets {shown}")
        else:
            notes.append("targets no open gap")
    else:
        # No lexical gap to shop for — a heuristic plan, every aspect covered, or only
        # structural aspects left. Targeting is unknowable rather than zero, so fall back to
        # rank and let the budget limit spend. Scoring it 0 would starve the run.
        targeting = 0.5
        notes.append("no specific open gap to target; ranked on retrieval score")

    rank_score = 0.0
    if best_rank_score > 0:
        rank_score = max(0.0, min(1.0, float(hit.score) / best_rank_score))

    authority = float(hit.scores.get("authority", 1.0))
    freshness = float(hit.scores.get("freshness", 1.0))

    value = (
        W_TARGET * targeting
        + W_MULTI * max(0, len(matched) - 1)
        + W_RANK * rank_score
        + W_AUTH * (authority - 1.0)
        + W_FRESH * (freshness - 1.0)
    )

    if chunk.doc_id in pooled_doc_ids:
        value -= P_SAME_DOC
        notes.append("another passage of this document is already held")

    if authority > 1.0:
        notes.append(f"{chunk.source} authority {authority:.2f}")

    return value, "; ".join(notes)


def select(
    candidates: list[tuple[str, RetrievedChunk]],
    *,
    wanted: list[str],
    pooled_chunk_ids: set[str],
    pooled_doc_ids: set[str],
    budget_left: int,
    round_index: int,
    min_value: float = MIN_VALUE,
) -> tuple[list[tuple[str, RetrievedChunk]], Ledger]:
    """Choose what to buy from one wave of candidates.

    Selection is global across the wave rather than first-come per sub-question, so a tight
    budget is spent on the best passages available anywhere in the plan instead of on whichever
    query happened to return first.

    A passage already in the pool costs nothing and is admitted without consuming budget: it is
    already paid for, and re-offering it is how the pool learns that two sub-questions both
    wanted it.
    """
    ledger = Ledger()
    if not candidates:
        return [], ledger

    best_rank = max((float(h.score) for _s, h in candidates), default=0.0)

    # Sub-questions overlap heavily, so the same passage arrives several times in one wave.
    # It is scored once and charged once; the later arrivals are recorded as free, because
    # charging per offer counted four bought passages as twelve.
    already: list[tuple[str, RetrievedChunk]] = []
    fresh: dict[str, tuple[RetrievedChunk, list[str]]] = {}
    for sub_id, hit in candidates:
        key = hit.chunk.chunk_id or hit.chunk.doc_id
        if key in pooled_chunk_ids:
            already.append((sub_id, hit))
            ledger.record(
                _decision(hit, sub_id, round_index, 0.0, True, "already held; no additional charge")
            )
            continue
        if key in fresh:
            fresh[key][1].append(sub_id)
        else:
            fresh[key] = (hit, [sub_id])

    scored: list[tuple[float, str, str, RetrievedChunk, list[str]]] = []
    for key, (hit, sub_ids) in fresh.items():
        value, reason = score_candidate(
            hit, wanted=wanted, pooled_doc_ids=pooled_doc_ids, best_rank_score=best_rank
        )
        scored.append((value, key, reason, hit, sub_ids))

    # Deterministic: value first, then doc id, so a tie does not depend on thread timing.
    scored.sort(key=lambda row: (-row[0], row[3].chunk.doc_id, row[3].chunk.chunk_id))

    admitted: list[tuple[str, RetrievedChunk]] = list(already)
    spent = 0
    for value, _key, reason, hit, sub_ids in scored:
        if spent >= budget_left:
            verdict = f"budget exhausted; {reason}" if reason else "budget exhausted"
        elif value < min_value:
            verdict = f"below value threshold; {reason}"
        else:
            verdict = ""

        if verdict:
            for sub_id in sub_ids:
                ledger.record(_decision(hit, sub_id, round_index, value, False, verdict))
            continue

        spent += 1
        for i, sub_id in enumerate(sub_ids):
            admitted.append((sub_id, hit))
            ledger.record(
                _decision(
                    hit,
                    sub_id,
                    round_index,
                    value if i == 0 else 0.0,
                    True,
                    reason if i == 0 else f"already held; also wanted by {sub_ids[0]}",
                )
            )

    return admitted, ledger


def hits_from_previews(contents: list[dict[str, Any]]) -> list[RetrievedChunk]:
    """Rebuild rankable hits from a package catalog. Text is empty on purpose."""
    hits: list[RetrievedChunk] = []
    for i, row in enumerate(contents or [], start=1):
        hits.append(
            RetrievedChunk(
                chunk=Chunk(
                    chunk_id=f"preview-{row.get('doc_id') or i}",
                    doc_id=str(row.get("doc_id") or f"preview-{i}"),
                    text="",
                    title=str(row.get("title") or ""),
                    source=str(row.get("source") or ""),
                    published_at=row.get("published_at"),
                    token_estimate=int(row.get("token_count") or 0),
                ),
                score=float(row.get("score") or 0.0),
                rank=i,
                scores={
                    "authority": float(row.get("authority") or 1.0),
                    "freshness": float(row.get("freshness") or 1.0),
                },
            )
        )
    return hits


def package_price(offer: dict[str, Any]) -> int:
    price = offer.get("price") or {}
    try:
        return int(price.get("amount") or 0)
    except (TypeError, ValueError):
        return 0


def score_package(
    offer: dict[str, Any],
    *,
    wanted: list[str],
    pooled_doc_ids: set[str] | None = None,
    purchase_gate: bool = True,
    min_value: float = MIN_VALUE,
) -> tuple[float, int] | None:
    """Expected value of buying this package, or None if it is not worth its price.

    Catalog only: titles, sources, scores. The body is not in the offer.
    """
    price = package_price(offer)
    if price <= 0:
        return None
    if not purchase_gate:
        label = str(offer.get("tradeoff_label") or "")
        bonus = 2.0 if label == TRADEOFF_THOROUGH else 1.0 if label == TRADEOFF_BALANCED else 0.0
        return (float(price) + bonus, price)

    previews = hits_from_previews(list(offer.get("contents") or []))
    if not previews:
        return None
    _admitted, ledger = select(
        [("shop", hit) for hit in previews],
        wanted=wanted,
        pooled_chunk_ids=set(),
        pooled_doc_ids=pooled_doc_ids or set(),
        budget_left=len(previews),
        round_index=0,
        min_value=min_value,
    )
    if not ledger.charged:
        return None
    return (sum(d.value for d in ledger.charged), price)


def choose_package(
    packages: list[dict[str, Any]],
    *,
    wanted: list[str],
    budget_left: int,
    pooled_doc_ids: set[str] | None = None,
    purchase_gate: bool = True,
    min_value: float = MIN_VALUE,
) -> dict[str, Any] | None:
    """Pick one package from a bid using only what query revealed.

    The commercial unit is the package, not the passage: you cannot buy two of five
    items. So this scores each offer on its catalog and either commits to the whole
    package or walks away.
    """
    ranked: list[tuple[float, int, dict[str, Any]]] = []
    for offer in packages:
        price = package_price(offer)
        if price <= 0 or price > budget_left:
            continue
        scored = score_package(
            offer,
            wanted=wanted,
            pooled_doc_ids=pooled_doc_ids,
            purchase_gate=purchase_gate,
            min_value=min_value,
        )
        if scored is None:
            continue
        ranked.append((scored[0], price, offer))
    if not ranked:
        return None
    ranked.sort(key=lambda row: (-row[0], row[1], str(row[2].get("tradeoff_label") or "")))
    return ranked[0][2]


def _decision(
    hit: RetrievedChunk,
    sub_id: str,
    round_index: int,
    value: float,
    admitted: bool,
    reason: str,
) -> PurchaseDecision:
    return PurchaseDecision(
        doc_id=hit.chunk.doc_id,
        chunk_id=hit.chunk.chunk_id,
        title=hit.chunk.title,
        source=hit.chunk.source,
        published_at=hit.chunk.published_at,
        sub_question_id=sub_id,
        round_index=round_index,
        value=value,
        admitted=admitted,
        reason=reason,
    )
