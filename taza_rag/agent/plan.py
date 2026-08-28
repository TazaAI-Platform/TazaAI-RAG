"""Decide what work needs to happen.

A complex question is rarely one retrieval. "How exposed is SoftBank to its AI bets, and
what do its own numbers say?" needs the earnings line, the borrowing, the commitments and
any dissenting view — four different searches whose results have to be reconciled.

Two properties matter more than clever decomposition:

1. **Fewer sub-questions is better.** Every sub-question is a paid retrieval, so a plan
   that emits six paraphrases of one ask burns the budget and returns the same passages.
   Near-duplicate sub-questions are collapsed deterministically after planning, because a
   model asked not to repeat itself still does.
2. **Each sub-question carries its own aspects.** They are what the agent later measures
   coverage against, so the plan defines its own completion criteria up front rather than
   asking a model afterwards whether it feels finished.
"""

from __future__ import annotations

from taza_rag.agent.models import ResearchPlan, SubQuestion
from taza_rag.agent.text import overlap
from taza_rag.config import settings
from taza_rag.factiva.strategy import detect_intent, expand_queries, normalize_query
from taza_rag.llm import LLMError, chat_json
from taza_rag.models import SearchIntent
from taza_rag.retrieve.features import build_query_plan

PLAN_SYSTEM = """You turn a complex research question into a minimal research plan that will
be executed against a Dow Jones / Factiva news archive.

Rules:
- Emit 2-5 sub-questions. Fewer is better: every sub-question costs a paid retrieval.
- Each sub-question must stand alone as an archive search. Name the company, person or
  topic explicitly; never write "it", "they" or "the company".
- Sub-questions must not paraphrase each other. Two questions that would return the same
  articles are one question.
- Cover the distinct angles the question actually needs, which for a business question is
  usually: the headline result or event, the mechanism or cause, the counterparties or
  money involved, and any contrary or cautionary view.
- aspects: 2-4 short noun phrases naming what a complete answer must contain. These are
  matched against the words of retrieved articles, so write them the way the copy would,
  naming the thing itself rather than describing the shape of an answer.
  Good: "record retail bond issuance", "OpenAI stake borrowing", "Vision Fund writedown",
  "net profit for the quarter".
  Bad, because no article ever contains these words: "key figures", "official comment",
  "dissenting view", "relevant context", "financial details".
- depends_on: list a sub-question id only when this one genuinely cannot be searched until
  that one is answered, such as needing to learn an executive's name first. Most plans
  have no dependencies.

Return JSON:
{"sub_questions": [{"id": "s1", "question": str, "aspects": [str], "depends_on": [str],
"rationale": str}]}
"""

# Above this overlap two sub-questions retrieve the same articles, so the second is waste.
_DUPLICATE_OVERLAP = 0.7


def _planner_model() -> str:
    return settings.answer_model or settings.chat_model


def heuristic_plan(question: str, *, max_sub_questions: int = 5) -> ResearchPlan:
    """Plan without an LLM, so the agent still runs on the key-free path.

    This reuses the retrieval stack's own query expansion: the literal ask, an
    entity-anchored variant and topic paraphrases. It is a weaker plan — the variants are
    rephrasings rather than genuinely different angles, and they carry no aspects — so a
    run records `method="heuristic"` and its coverage is not comparable to a planned run.
    """
    normalized = normalize_query(question)
    intent = detect_intent(normalized)
    qp = build_query_plan(normalized, intent)
    variants = expand_queries(question, intent, max_variants=max_sub_questions)

    subs: list[SubQuestion] = []
    for i, variant in enumerate(variants, start=1):
        subs.append(
            SubQuestion(
                id=f"s{i}",
                question=variant,
                intent=intent,
                # Topics are the only aspect-like signal available without a model.
                aspects=list(qp.topics[:3]),
                rationale="heuristic query expansion",
            )
        )
    return ResearchPlan(
        question=question,
        intent=intent,
        entities=list(qp.entities),
        topics=list(qp.topics),
        sub_questions=subs,
        method="heuristic",
    )


def parse_plan(
    raw: object, question: str, intent: SearchIntent, *, max_sub_questions: int
) -> list[SubQuestion]:
    """Accept only well-formed steps; a malformed planner must not reach retrieval."""
    items = raw.get("sub_questions") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return []

    subs: list[SubQuestion] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("question") or "").strip()
        if len(text) < 8:
            continue
        aspects = [
            str(a).strip()
            for a in (item.get("aspects") or [])
            if isinstance(a, (str, int, float)) and str(a).strip()
        ]
        depends = [
            str(d).strip().lower()
            for d in (item.get("depends_on") or [])
            if isinstance(d, str) and d.strip()
        ]
        subs.append(
            SubQuestion(
                id=f"s{len(subs) + 1}",
                question=text,
                # Intent is detected from the sub-question's own wording, not inherited:
                # "what did Son say" is executive profiling even when the parent ask is an
                # entity investigation, and intent drives the retrieval date window.
                intent=detect_intent(text),
                aspects=aspects[:4],
                depends_on=depends,
                rationale=str(item.get("rationale") or "")[:200],
            )
        )
        if len(subs) >= max_sub_questions:
            break

    return _drop_near_duplicates(subs)


def _drop_near_duplicates(subs: list[SubQuestion]) -> list[SubQuestion]:
    """Collapse sub-questions that would retrieve the same articles.

    Aspects of a dropped step are merged into the one that survives, so removing a
    paraphrase never quietly lowers what the plan requires of the answer.
    """
    kept: list[SubQuestion] = []
    for sub in subs:
        twin = next((k for k in kept if overlap(k.question, sub.question) >= _DUPLICATE_OVERLAP), None)
        if twin is None:
            kept.append(sub)
            continue
        for aspect in sub.aspects:
            if aspect not in twin.aspects:
                twin.aspects.append(aspect)
    # Ids are positional and referenced by depends_on, so renumber and drop references to
    # steps that no longer exist rather than leaving a dangling dependency.
    old_to_new = {}
    for i, sub in enumerate(kept, start=1):
        old_to_new[sub.id] = f"s{i}"
    for i, sub in enumerate(kept, start=1):
        sub.id = f"s{i}"
    for sub in kept:
        sub.depends_on = [
            old_to_new[d] for d in sub.depends_on if d in old_to_new and old_to_new[d] != sub.id
        ]
    return kept


def make_plan(question: str, *, max_sub_questions: int = 5, use_llm: bool = True) -> ResearchPlan:
    """Plan the run, falling back to query expansion when no model is available."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question is empty")

    if not use_llm or not settings.openai_api_key:
        return heuristic_plan(question, max_sub_questions=max_sub_questions)

    normalized = normalize_query(question)
    intent = detect_intent(normalized)
    qp = build_query_plan(normalized, intent)
    try:
        raw = chat_json(
            PLAN_SYSTEM,
            f"Question: {question}\n\nNamed entities detected: {qp.entities or 'none'}",
            model=_planner_model(),
            temperature=0.0,
        )
    except LLMError:
        return heuristic_plan(question, max_sub_questions=max_sub_questions)

    subs = parse_plan(raw, question, intent, max_sub_questions=max_sub_questions)
    if not subs:
        return heuristic_plan(question, max_sub_questions=max_sub_questions)

    return ResearchPlan(
        question=question,
        intent=intent,
        entities=list(qp.entities),
        topics=list(qp.topics),
        sub_questions=subs,
        method="llm",
    )


def execution_order(plan: ResearchPlan) -> list[list[SubQuestion]]:
    """Group sub-questions into waves that can each run in parallel.

    Independent steps belong in one wave so they issue concurrently; a step that declares a
    dependency waits for the wave containing it. A dependency cycle would otherwise hang
    the run, so anything still unscheduled after the graph stops making progress is
    released as a final wave — a late search is better than a deadlock.
    """
    remaining = list(plan.sub_questions)
    done: set[str] = set()
    waves: list[list[SubQuestion]] = []

    while remaining:
        wave = [s for s in remaining if all(d in done for d in s.depends_on)]
        if not wave:
            waves.append(list(remaining))
            break
        waves.append(wave)
        done.update(s.id for s in wave)
        remaining = [s for s in remaining if s.id not in done]
    return waves
