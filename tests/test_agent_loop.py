"""The loop end to end, against a fixed corpus and scripted model calls.

Everything here is offline. The runner blocks sockets, so a stub that quietly reached the
network would fail loudly rather than turning an assertion into a coin flip.

The stubs are generic rather than hand-written per case: extraction reads whichever labels
the pool actually assigned and quotes the passage body, and composition echoes the fact list
it was handed. That way a label-assignment bug shows up as a broken citation instead of being
papered over by a stub that hardcoded the right answer.
"""

import re

import taza_rag.agent.plan as pmod
import taza_rag.agent.synthesize as smod
import taza_rag.factiva.facts as fmod
import taza_rag.factiva.verify as vmod
from taza_rag.agent.fixtures import FixtureDoc, FixtureSearch
from taza_rag.agent.loop import research
from taza_rag.agent.models import Budget

DOCS = [
    FixtureDoc(
        doc_id="d1",
        title="SoftBank Group net profit falls 18 percent",
        text=(
            "SoftBank Group posted a net profit of 347.33 billion yen in the quarter, "
            "down 17.7 percent from a year earlier as Vision Fund gains shrank."
        ),
        source="Dow Jones Newswires",
        authority=1.10,
    ),
    FixtureDoc(
        doc_id="d2",
        title="SoftBank borrows 10 billion against OpenAI stake",
        text=(
            "SoftBank Group borrowed 10 billion dollars from a group of banks against its "
            "OpenAI stake, raising the risk on its artificial intelligence bet."
        ),
        source="The Wall Street Journal",
        authority=1.12,
    ),
    FixtureDoc(
        doc_id="d3",
        title="SoftBank plans record retail bond issuance",
        text=(
            "SoftBank Group plans a record retail bond issuance of 6.3 billion dollars to "
            "partly fund its artificial intelligence investments."
        ),
        source="The Wall Street Journal",
        authority=1.12,
    ),
]

PLAN = {
    "sub_questions": [
        {
            "id": "s1",
            "question": "SoftBank Group quarterly net profit",
            "aspects": ["net profit figure"],
        },
        {
            "id": "s2",
            "question": "SoftBank Group borrowing against OpenAI stake",
            "aspects": ["OpenAI stake borrowing"],
        },
    ]
}

# The header line carries the label and the title; the passage body follows it.
_BLOCK = re.compile(r"\[(c\d+)\] doc_id=\S+[^\n]*\n(.*?)(?=\n\n\[c|\Z)", re.DOTALL)
_FACT_LINE = re.compile(r"^\s*\d+\.\s+(.*?)\s+\[(c\d+)\]\s*$", re.MULTILINE)


def _extract_stub(system, user, **kwargs):
    """Quote each cited passage back as a fact, using the label the pool assigned."""
    facts = []
    for label, body in _BLOCK.findall(user):
        sentence = body.strip().split(". ")[0].strip().rstrip(".")
        if sentence:
            facts.append({"text": f"{sentence}.", "citation": label})
    return {"facts": facts}


def _compose_stub(system, user, **kwargs):
    """Echo the handed fact list as cited sentences."""
    sentences = [
        f"{text.rstrip('.')} [{label}]." for text, label in _FACT_LINE.findall(user)
    ]
    return {
        "answer": " ".join(sentences),
        "abstain": False,
        "used_citations": sorted({label for _t, label in _FACT_LINE.findall(user)}),
    }


def _entailment_clean(system, user, **kwargs):
    return {"verdicts": []}


class _Harness:
    """Swaps every model call the loop can reach, and restores them."""

    def __init__(self, plan=PLAN, compose=_compose_stub):
        self.plan = plan
        self.compose = compose

    def __enter__(self):
        self._saved = (
            pmod.chat_json,
            fmod.chat_json,
            smod.chat_json,
            vmod.chat_json,
            pmod.settings.openai_api_key,
        )
        pmod.chat_json = lambda system, user, **kw: self.plan
        fmod.chat_json = _extract_stub
        smod.chat_json = self.compose
        vmod.chat_json = _entailment_clean
        pmod.settings.openai_api_key = "sk-test-offline"
        return self

    def __exit__(self, *exc):
        (
            pmod.chat_json,
            fmod.chat_json,
            smod.chat_json,
            vmod.chat_json,
            pmod.settings.openai_api_key,
        ) = self._saved
        return False


QUESTION = "How exposed is SoftBank Group to its AI bets, and what do its numbers say?"


def test_a_full_run_answers_from_the_corpus_with_resolvable_citations():
    backend = FixtureSearch(list(DOCS))
    with _Harness():
        result = research(QUESTION, backend=backend, budget=Budget(top_k_per_query=4))

    assert not result.abstained
    assert result.answer
    labels = set(re.findall(r"\[(c\d+)\]", result.answer))
    assert labels, result.answer
    known = {item.label for item in result.evidence}
    # Every citation in the prose must resolve to exactly one pooled passage.
    assert labels <= known, labels - known
    assert result.plan is not None and result.plan.method == "llm"


def test_the_plan_is_searched_in_parallel_and_each_step_is_issued_once():
    backend = FixtureSearch(list(DOCS))
    with _Harness():
        research(QUESTION, backend=backend, budget=Budget(top_k_per_query=4))

    first_round = [
        "SoftBank Group quarterly net profit",
        "SoftBank Group borrowing against OpenAI stake",
    ]
    assert backend.calls[:2] == first_round
    assert len(backend.calls) == len(set(backend.calls)), backend.calls


def test_a_passage_found_by_two_steps_is_pooled_once_and_recorded_against_both():
    """The reuse rate is the cost signal: paying twice for one passage is a planner defect."""
    overlapping = {
        "sub_questions": [
            {"question": "SoftBank Group artificial intelligence investments", "aspects": ["AI bet"]},
            {"question": "SoftBank Group artificial intelligence bond issuance", "aspects": ["bond issuance"]},
        ]
    }
    backend = FixtureSearch(list(DOCS))
    with _Harness(plan=overlapping):
        result = research(QUESTION, backend=backend, budget=Budget(top_k_per_query=4))

    chunk_ids = [item.hit.chunk.chunk_id for item in result.evidence]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert any(len(item.found_by) > 1 for item in result.evidence)
    assert result.cost.unique_chunks < result.cost.chunks_returned
    assert result.cost.reuse_rate > 0


def test_one_failing_step_does_not_sink_the_run():
    backend = FixtureSearch(list(DOCS), fail_on={"SoftBank Group quarterly net profit"})
    with _Harness():
        result = research(QUESTION, backend=backend, budget=Budget(top_k_per_query=4))

    assert result.errors, "the failure must be recorded, not swallowed"
    assert "SoftBank Group quarterly net profit" in result.rounds[0].failed_queries
    # The surviving step still produced an answer.
    assert not result.abstained
    assert result.answer


def test_an_empty_corpus_abstains_and_says_why():
    backend = FixtureSearch([])
    with _Harness():
        result = research(QUESTION, backend=backend)

    assert result.abstained
    assert result.stop_reason == "no_evidence"
    assert result.cost.unique_chunks == 0
    # Abstaining must not cost a compose or verify call.
    assert result.cost.llm_calls == 1


def test_a_missing_aspect_triggers_a_second_round_that_asks_only_for_it():
    plan = {
        "sub_questions": [
            {"question": "SoftBank Group quarterly net profit", "aspects": ["net profit figure"]},
            {
                "question": "SoftBank Group quarterly costs",
                "aspects": ["record retail bond issuance"],
            },
        ]
    }
    backend = FixtureSearch(list(DOCS))
    with _Harness(plan=plan):
        result = research(
            QUESTION, backend=backend, budget=Budget(top_k_per_query=2, max_rounds=3)
        )

    assert len(result.rounds) >= 2, result.stop_reason
    refinement = result.rounds[1].queries
    assert any("record retail bond issuance" in q for q in refinement), refinement
    # The refinement is entity-anchored so it does not retrieve the whole market.
    assert all(q.startswith("SoftBank") for q in refinement)


def test_every_round_records_what_it_cost_and_what_it_bought():
    backend = FixtureSearch(list(DOCS))
    with _Harness():
        result = research(QUESTION, backend=backend, budget=Budget(top_k_per_query=4))

    assert result.rounds
    for record in result.rounds:
        payload = record.payload()
        for key in ("chunks_returned", "new_chunks", "new_findings", "coverage"):
            assert key in payload
    assert result.stop_reason
    assert result.cost.retrieval_calls >= 2
    assert result.latency_ms["total"] >= 0


def test_the_run_serialises_without_leaking_credentials():
    backend = FixtureSearch(list(DOCS))
    with _Harness():
        result = research(QUESTION, backend=backend, budget=Budget(top_k_per_query=4))

    dumped = str(result.payload())
    assert "sk-test-offline" not in dumped
    assert "password" not in dumped


def test_a_composer_that_returns_nothing_abstains_rather_than_shipping_an_empty_answer():
    def empty(system, user, **kwargs):
        return {"answer": "   ", "abstain": False}

    backend = FixtureSearch(list(DOCS))
    with _Harness(compose=empty):
        result = research(QUESTION, backend=backend, budget=Budget(top_k_per_query=4))

    assert result.abstained
    assert result.answer
