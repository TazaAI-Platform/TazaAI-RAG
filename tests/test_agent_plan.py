"""Planning defects are expensive: every sub-question is a paid retrieval.

The two failures worth guarding are a plan that pays twice for the same search, and a
dependency graph that stops the run instead of ordering it.
"""

import taza_rag.agent.plan as pmod
from taza_rag.agent.plan import execution_order, heuristic_plan, make_plan, parse_plan
from taza_rag.models import SearchIntent


def _raw(*questions):
    return {
        "sub_questions": [
            {"question": q, "aspects": [f"aspect {i}"], "depends_on": []}
            for i, q in enumerate(questions, start=1)
        ]
    }


def test_paraphrased_sub_questions_are_collapsed_into_one_retrieval():
    """Two questions that would return the same articles are one question."""
    raw = _raw(
        "SoftBank Group first quarter profit",
        "SoftBank Group profit first quarter",
        "SoftBank Group borrowing against OpenAI stake",
    )
    subs = parse_plan(raw, "SoftBank", SearchIntent.ENTITY_INVESTIGATION, max_sub_questions=5)
    assert len(subs) == 2, [s.question for s in subs]
    assert [s.id for s in subs] == ["s1", "s2"]


def test_collapsing_a_paraphrase_keeps_what_it_asked_for():
    """Dropping a duplicate must not quietly lower the plan's completion criteria."""
    raw = {
        "sub_questions": [
            {"question": "SoftBank Group quarterly profit", "aspects": ["net profit figure"]},
            {"question": "SoftBank Group profit quarterly", "aspects": ["analyst consensus"]},
        ]
    }
    subs = parse_plan(raw, "SoftBank", SearchIntent.ENTITY_INVESTIGATION, max_sub_questions=5)
    assert len(subs) == 1
    assert subs[0].aspects == ["net profit figure", "analyst consensus"]


def test_malformed_steps_never_reach_retrieval():
    raw = {
        "sub_questions": [
            {"question": "  ", "aspects": []},
            {"question": "short"},
            {"aspects": ["no question at all"]},
            "not even an object",
            {"question": "SoftBank Group borrowing against its OpenAI stake", "aspects": ["$10bn"]},
        ]
    }
    subs = parse_plan(raw, "SoftBank", SearchIntent.ENTITY_INVESTIGATION, max_sub_questions=5)
    assert len(subs) == 1
    assert subs[0].question.startswith("SoftBank Group borrowing")


def test_the_budget_caps_how_many_steps_a_plan_may_hold():
    raw = _raw(
        "Nvidia export restrictions to China",
        "Tesla robotaxi rollout timeline",
        "HSBC restructuring in Asia",
        "Barclays investment banking overhaul",
        "Rheinmetall defence order backlog",
    )
    subs = parse_plan(raw, "q", SearchIntent.TOPICAL_EXPLORATION, max_sub_questions=2)
    assert len(subs) == 2


def test_each_sub_question_gets_its_own_intent_not_the_parent_s():
    """Intent drives the retrieval date window, so inheriting it loses recency."""
    raw = {
        "sub_questions": [
            {"question": "SoftBank Group quarterly net profit", "aspects": ["figure"]},
            {"question": "What has Masayoshi Son said about Arm Holdings?", "aspects": ["quote"]},
        ]
    }
    subs = parse_plan(raw, "SoftBank", SearchIntent.ENTITY_INVESTIGATION, max_sub_questions=5)
    intents = {s.id: s.intent for s in subs}
    assert intents["s2"] == SearchIntent.EXECUTIVE_PROFILING


def test_independent_steps_share_one_wave_so_they_issue_in_parallel():
    plan = heuristic_plan("SoftBank Group AI exposure")
    waves = execution_order(plan)
    assert len(waves) == 1
    assert len(waves[0]) == len(plan.sub_questions)


def test_a_declared_dependency_orders_the_waves():
    raw = {
        "sub_questions": [
            {"question": "Who runs SoftBank Group's Vision Fund?", "aspects": ["name"]},
            {"question": "What has that executive said about AI investments?",
             "aspects": ["comment"], "depends_on": ["s1"]},
        ]
    }
    subs = parse_plan(raw, "SoftBank", SearchIntent.ENTITY_INVESTIGATION, max_sub_questions=5)
    plan = heuristic_plan("SoftBank")
    plan.sub_questions = subs
    waves = execution_order(plan)
    assert [[s.id for s in w] for w in waves] == [["s1"], ["s2"]]


def test_a_dependency_cycle_releases_instead_of_hanging():
    """A late search beats a deadlock."""
    plan = heuristic_plan("SoftBank")
    subs = parse_plan(
        {
            "sub_questions": [
                {"question": "First step about SoftBank Group", "depends_on": ["s2"]},
                {"question": "Second step about ByteDance revenue", "depends_on": ["s1"]},
            ]
        },
        "q",
        SearchIntent.ENTITY_INVESTIGATION,
        max_sub_questions=5,
    )
    plan.sub_questions = subs
    waves = execution_order(plan)
    assert sum(len(w) for w in waves) == 2


def test_a_dropped_step_leaves_no_dangling_dependency():
    raw = {
        "sub_questions": [
            {"question": "SoftBank Group quarterly profit", "aspects": ["figure"]},
            {"question": "SoftBank Group profit quarterly", "aspects": ["consensus"]},
            {"question": "SoftBank Group Vision Fund writedowns", "depends_on": ["s2"]},
        ]
    }
    subs = parse_plan(raw, "q", SearchIntent.ENTITY_INVESTIGATION, max_sub_questions=5)
    ids = {s.id for s in subs}
    for sub in subs:
        for dep in sub.depends_on:
            assert dep in ids, f"{sub.id} depends on missing {dep}"


def test_planning_falls_back_to_query_expansion_when_the_model_fails():
    """No key, or a failed call, must still leave a runnable plan."""
    from taza_rag.llm import LLMError

    def boom(system, user, **kwargs):
        raise LLMError("no")

    original_key, original_chat = pmod.settings.openai_api_key, pmod.chat_json
    pmod.settings.openai_api_key = "sk-test"
    pmod.chat_json = boom
    try:
        plan = make_plan("Deutche Bank restructuring")
    finally:
        pmod.settings.openai_api_key = original_key
        pmod.chat_json = original_chat

    assert plan.method == "heuristic"
    assert plan.sub_questions
    # Normalisation still has to happen, or the entity signals go to zero downstream.
    assert any("Deutsche" in s.question for s in plan.sub_questions)


def test_an_empty_plan_response_falls_back_rather_than_returning_nothing():
    original_key, original_chat = pmod.settings.openai_api_key, pmod.chat_json
    pmod.settings.openai_api_key = "sk-test"
    pmod.chat_json = lambda system, user, **kw: {"sub_questions": []}
    try:
        plan = make_plan("SoftBank Group AI exposure")
    finally:
        pmod.settings.openai_api_key = original_key
        pmod.chat_json = original_chat
    assert plan.method == "heuristic"
    assert plan.sub_questions
