"""Decide whether the agent has enough — and stop paying when it does not help.

This is the question the brief singles out: *how does it know when it has enough
information?* Asking a model "are you done?" is not an answer. It has no way to know what it
has not seen, it is agreeable under pressure, and its verdict cannot be audited after the
fact.

So sufficiency here is measured, not asked:

- The plan declares its own completion criteria up front as **aspects**. Coverage is the
  share of those aspects that some grounded fact addresses, computed by whole-term
  stem-insensitive matching.
- The run stops on the first of five conditions. The interesting one is **plateau**: a round
  that pays for new passages and yields no new grounded fact is the point where further
  retrieval has stopped changing the answer. That is the empirical version of "enough", and
  it is also the cost discipline the marketplace demands — chunks that will not change the
  answer should not be bought.

Every stop is labelled with its reason, so a run that ended early because the budget ran out
can never be mistaken for one that ended because the question was answered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from taza_rag.agent.models import (
    Budget,
    Cost,
    Finding,
    Gap,
    ResearchPlan,
    RoundRecord,
    SubQuestion,
)
from taza_rag.agent.aspects import satisfied

# A refinement round that fans out as wide as the first one defeats the point; the whole
# reason to continue is that only a few aspects are missing.
_MAX_REFINEMENTS = 3


@dataclass
class Assessment:
    coverage: float = 0.0
    sub_coverage: dict[str, float] = field(default_factory=dict)
    gaps: list[Gap] = field(default_factory=list)
    stop: bool = True
    reason: str = ""
    refinements: list[tuple[SubQuestion, str]] = field(default_factory=list)


def aspect_gaps(plan: ResearchPlan, findings: list[Finding]) -> list[Gap]:
    """Aspects the plan asked for that no grounded fact addresses.

    Coverage is judged against every fact in the run, not only the facts retrieved for that
    sub-question: the pool is shared, so a passage found while asking about borrowing can
    legitimately answer an aspect of the earnings step. Attribution to a sub-question is
    kept because it is what tells a refinement round *what* to go and ask for.
    """
    texts = [f.text for f in findings]
    gaps: list[Gap] = []
    for sub in plan.sub_questions:
        for aspect in sub.aspects:
            if not satisfied(aspect, texts):
                gaps.append(Gap(sub_question_id=sub.id, aspect=aspect))
    return gaps


def coverage_by_sub(plan: ResearchPlan, findings: list[Finding]) -> dict[str, float]:
    texts = [f.text for f in findings]
    have_findings = {f.sub_question_id for f in findings}
    out: dict[str, float] = {}
    for sub in plan.sub_questions:
        if sub.aspects:
            hit = sum(1 for a in sub.aspects if satisfied(a, texts))
            out[sub.id] = hit / len(sub.aspects)
        else:
            # A heuristic plan carries no aspects, so the only signal available is whether
            # the step returned anything at all. Coarse, and marked as such in the report.
            out[sub.id] = 1.0 if sub.id in have_findings else 0.0
    return out


def _refinement_query(plan: ResearchPlan, sub: SubQuestion, aspect: str) -> str:
    """Search for the missing aspect, anchored so it stays on-entity.

    A bare aspect ("record retail bond issuance") retrieves the whole market. The sub-
    question's own entity is not always present in its wording, so the plan's primary
    entity is the anchor of last resort. Deterministic on purpose: a second model call to
    rephrase a gap costs money and adds a failure mode for no measured gain.
    """
    anchor = plan.entities[0] if plan.entities else ""
    if anchor and anchor.lower() not in aspect.lower():
        return f"{anchor} {aspect}".strip()
    return aspect.strip() or sub.question


def _plateaued(rounds: list[RoundRecord], coverage: float, budget: Budget) -> bool:
    """Has spending stopped moving the answer?

    Two ways to be stuck, and both matter. No new grounded facts is the obvious one. The
    subtler one is a round that returns plenty of facts which happen not to address any
    remaining aspect — coverage flat while the bill grows. A first live run did exactly that,
    so coverage is compared against the previous round rather than trusting fact counts.

    The comparison uses the previous round's recorded coverage, not the current record's
    delta, because the caller only fills that in after this decision is made.
    """
    if rounds[-1].new_findings < budget.min_new_findings:
        return True
    return len(rounds) >= 2 and coverage <= rounds[-2].coverage


def assess(
    plan: ResearchPlan,
    findings: list[Finding],
    rounds: list[RoundRecord],
    cost: Cost,
    budget: Budget,
    *,
    issued: set[str] | None = None,
) -> Assessment:
    """Score coverage and decide whether to spend another round."""
    issued = issued or set()
    sub_coverage = coverage_by_sub(plan, findings)
    coverage = sum(sub_coverage.values()) / len(sub_coverage) if sub_coverage else 0.0
    gaps = aspect_gaps(plan, findings)

    result = Assessment(
        coverage=coverage,
        sub_coverage=sub_coverage,
        gaps=gaps,
        stop=True,
    )

    # Candidate follow-ups: the uncovered aspects of the least-covered steps first, so a
    # capped round spends itself where the plan is weakest.
    ordered_gaps = sorted(gaps, key=lambda g: sub_coverage.get(g.sub_question_id, 0.0))
    refinements: list[tuple[SubQuestion, str]] = []
    for gap in ordered_gaps:
        sub = plan.by_id(gap.sub_question_id)
        if sub is None:
            continue
        query = _refinement_query(plan, sub, gap.aspect)
        if not query or query in issued:
            continue
        if any(query == q for _s, q in refinements):
            continue
        refinements.append((sub, query))
        if len(refinements) >= _MAX_REFINEMENTS:
            break

    # Order matters. Budget and round caps are reported before plateau so a run that was
    # cut off is never labelled as one that converged.
    if coverage >= budget.target_coverage:
        result.reason = "target_coverage"
        return result
    if len(rounds) >= budget.max_rounds:
        result.reason = "round_cap"
        return result
    if cost.unique_chunks >= budget.max_unique_chunks:
        result.reason = "budget_chunks"
        return result
    if rounds and _plateaued(rounds, coverage, budget):
        # Paid for passages and either learned nothing new, or learned things that do not
        # address what is still missing. More of the same will not help.
        result.reason = "plateau"
        return result
    if not refinements:
        result.reason = "nothing_left_to_ask" if not gaps else "no_new_query"
        return result

    result.stop = False
    result.reason = "continue"
    result.refinements = refinements
    return result
