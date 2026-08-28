"""The UI must explain the pipeline without leaking credentials or inventing scores."""

from pathlib import Path

from taza_rag.factiva.pipeline import RetrievalRun
from taza_rag.models import Chunk, RetrievedChunk, SearchIntent
from taza_rag.ui.serialize import (
    SCORE_LEGEND,
    USAGE_FIELDS,
    health_payload,
    hit_payload,
    plan_payload,
    run_payload,
    usage_payload,
)
from taza_rag.ui.server import STATIC_DIR, STATIC_FILES


def test_plan_is_local_and_normalizes_the_misspelling():
    plan = plan_payload("Deutche Bank restructuring")
    assert plan["normalized"] == "Deutsche Bank restructuring"
    assert plan["intent"] == SearchIntent.ENTITY_INVESTIGATION.value
    assert "Deutsche Bank" in plan["entities"]
    assert plan["variants"][0] == "Deutsche Bank restructuring"
    assert plan["days_range"]


def test_hit_payload_uses_rank_as_citation_label():
    hit = RetrievedChunk(
        chunk=Chunk(
            chunk_id="d1::p000",
            doc_id="DJDN000020260806em86001dn",
            text="SoftBank beat expectations, helped by Intel.",
            title="SoftBank Group Reports Profit Beat",
            source="Dow Jones Newswires",
            published_at="2026-08-06",
            chunk_index=0,
            metadata={"doc_kind": "article", "passage_count": 1},
        ),
        score=3.695,
        rank=10,
        scores={
            "tier": 0,
            "entity": 1.0,
            "topic": 1.0,
            "bm25": 0.97,
            "rrf": 0.45,
            "authority": 1.10,
            "freshness": 1.08,
            "penalty": 0.0,
        },
    )
    payload = hit_payload(hit)
    assert payload["label"] == "c10"
    assert payload["tier_label"] == "headline"
    assert payload["kind"] == "article"
    assert payload["passage"] == {"index": 1, "of": 1}
    assert payload["scores"]["authority"] == 1.1


def test_run_payload_carries_the_funnel_not_credentials():
    run = RetrievalRun(
        query="SoftBank Group",
        intent=SearchIntent.ENTITY_INVESTIGATION,
        variants=["SoftBank Group"],
        hits=[],
        candidates=40,
        passages=72,
        config="factiva_quality_v2+ctx",
        latency_ms={"total": 1234.6},
    )
    payload = run_payload(run)
    dumped = str(payload)
    assert "password" not in dumped
    assert "api_key" not in dumped
    assert payload["candidates"] == 40
    assert payload["latency_ms"]["total"] == 1235
    assert payload["usage"]["offered"] == 40
    assert payload["usage"]["bought"] == 0
    assert tuple(payload["usage"]) == USAGE_FIELDS


def test_health_is_booleans_only():
    payload = health_payload(factiva=True, openai=False)
    assert payload == {"factiva": True, "openai": False}


def test_score_legend_names_every_meter_the_cli_prints():
    keys = {row["key"] for row in SCORE_LEGEND}
    assert keys == {"entity", "topic", "bm25", "rrf", "authority", "freshness", "penalty"}


def test_static_bundle_is_complete():
    for name in STATIC_FILES.values():
        path = STATIC_DIR / name
        assert path.is_file(), path
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "Packages" in html
    assert "Consumption" in html
    assert "/app.css" in html
    assert "/app.js" in html


def test_the_playground_hides_ranking_knobs_behind_advanced():
    """Anjana's review: send a task, get options, do not configure ranking up front."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    start = html.index('<details class="advanced" id="advanced">')
    end = html.index("</details>", start)
    advanced = html[start:end]
    for knob in ("top-k", "max-rounds", "raw", "purchase-gate"):
        assert f'id="{knob}"' in advanced, knob
    # Budget is the spend cap, so it stays on the research toolbar, not in Advanced.
    assert 'id="max-chunks"' not in advanced
    assert 'id="max-chunks"' in html
    assert 'id="query-label">Task<' in html
    assert 'id="retrieve-btn" class="retrieve-only"' in html
    assert "Ask the marketplace" in html
    # Default path does not pre-fill a ranking-lab entity query.
    assert 'value="SoftBank Group"' not in html


def test_the_ui_has_a_panel_for_every_stage_of_the_agent():
    """The UI exists to make the agent's decisions inspectable, so the panels are the contract."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for heading in ("Consumption", "Packages", "Research plan", "Rounds", "Purchases", "Source disagreements"):
        assert heading in html, heading
    for element in ("research-btn", "purchase-gate", "max-rounds", "max-chunks", "gaps", "usage", "packages"):
        assert f'id="{element}"' in html, element


def test_the_script_renders_the_usage_contract():
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    for key in ("offered", "bought", "refused", "cited"):
        assert f"u.{key}" in js, key
    assert 'metric(u.offered, "Offered")' in js
    assert "Ask the marketplace" in (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "data-package" in js


def test_every_element_the_script_reaches_for_exists_in_the_page():
    """A renamed id fails silently in the browser; here it fails loudly."""
    import re

    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    referenced = set(re.findall(r'\$\("([a-zA-Z-]+)"\)', js))
    present = set(re.findall(r'id="([a-zA-Z-]+)"', html))
    missing = sorted(referenced - present)
    assert not missing, f"app.js reaches for ids that do not exist: {missing}"


def _research_result():
    from taza_rag.agent.models import (
        Conflict,
        Cost,
        EvidenceItem,
        Finding,
        Gap,
        ResearchPlan,
        ResearchResult,
        RoundRecord,
        SubQuestion,
    )
    from taza_rag.agent.purchase import PurchaseDecision

    hit = RetrievedChunk(
        chunk=Chunk(
            chunk_id="d1::p000",
            doc_id="DJDN1",
            text="SoftBank posted a net profit of 347.33 billion yen.",
            title="SoftBank Group net profit falls",
            source="Dow Jones Newswires",
            published_at="2026-08-06",
            metadata={"doc_kind": "article", "passage_count": 1},
        ),
        score=3.5,
        rank=1,
        scores={"tier": 0, "entity": 1.0, "topic": 1.0, "authority": 1.10},
    )
    left = Finding(sub_question_id="s1", text="Profit was 347.33 billion yen.", label="c1")
    right = Finding(sub_question_id="s1", text="Profit was 340.00 billion yen.", label="c2")
    result = ResearchResult(
        question="How exposed is SoftBank to its AI bets?",
        answer="SoftBank posted a net profit of 347.33 billion yen [c1].",
        plan=ResearchPlan(
            question="How exposed is SoftBank to its AI bets?",
            intent=SearchIntent.ENTITY_INVESTIGATION,
            entities=["SoftBank"],
            sub_questions=[
                SubQuestion(
                    id="s1",
                    question="SoftBank quarterly net profit",
                    intent=SearchIntent.ENTITY_INVESTIGATION,
                    aspects=["net profit figure", "analyst consensus"],
                )
            ],
            method="llm",
        ),
        evidence=[EvidenceItem(label="c1", hit=hit, found_by=["s1"])],
        findings=[left],
        conflicts=[Conflict(kind="disagreement", subject="profit", left=left, right=right,
                            preferred_label="c1", reason="higher authority")],
        gaps=[Gap(sub_question_id="s1", aspect="analyst consensus")],
        rounds=[RoundRecord(index=0, queries=["SoftBank quarterly net profit"], chunks_returned=4,
                            new_chunks=1, new_findings=1, coverage=0.5)],
        coverage=0.5,
        sub_coverage={"s1": 0.5},
        stop_reason="plateau",
        cost=Cost(chunks_returned=4, unique_chunks=1, retrieval_calls=1, llm_calls=2),
        verification={"initial": {"problems": {"uncited": 2}}, "final": {"problems": {}},
                      "repairs_applied": 1, "resolved": True},
    )
    result.ledger.record(
        PurchaseDecision(
            doc_id="DJDN1", chunk_id="d1::p000", title="SoftBank Group net profit falls",
            source="Dow Jones Newswires", published_at="2026-08-06", sub_question_id="s1",
            round_index=0, value=1.8, admitted=True, reason="targets 'net profit figure'",
            label="c1",
        )
    )
    return result


def test_the_query_handler_returns_packages_without_bodies():
    from taza_rag.market import Market
    from taza_rag.ui.server import UiHandler

    from tests.test_market import POOL

    handler = UiHandler.__new__(UiHandler)
    handler.server = type("S", (), {
        "query_fn": None,
        "market": Market(search=lambda query, top_k: POOL),
    })()
    payload = handler._query({"query": "SoftBank profit"})
    assert payload["packages"]
    assert payload["usage"]["bought"] == 0
    assert "SECRET_BODY_A" not in str(payload)


def test_the_retrieve_handler_returns_usage_from_the_injected_backend():
    from taza_rag.ui.server import UiHandler

    handler = UiHandler.__new__(UiHandler)
    handler.server = type("S", (), {
        "retrieve_fn": staticmethod(
            lambda query, *, top_k, raw: run_payload(
                RetrievalRun(
                    query=query,
                    intent=SearchIntent.ENTITY_INVESTIGATION,
                    variants=[query],
                    hits=[],
                    candidates=12,
                    passages=20,
                    config="mock",
                )
            )
        ),
        "answer_fn": None,
        "research_fn": None,
    })()
    payload = handler._retrieve("SoftBank Group", top_k=10, raw=False)
    assert tuple(payload["usage"]) == USAGE_FIELDS
    assert payload["usage"]["offered"] == 12
    assert payload["usage"]["bought"] == 0


def test_the_research_handler_returns_usage_from_the_injected_backend():
    from taza_rag.ui.serialize import research_payload
    from taza_rag.ui.server import UiHandler

    handler = UiHandler.__new__(UiHandler)
    handler.server = type("S", (), {
        "retrieve_fn": None,
        "answer_fn": None,
        "research_fn": staticmethod(lambda query, body: research_payload(_research_result())),
    })()
    payload = handler._research("How exposed is SoftBank?", {"max_chunks": 40})
    assert tuple(payload["usage"]) == USAGE_FIELDS
    assert payload["usage"]["cited"] == 1
    assert payload["usage"]["bought"] == 1


def test_the_research_payload_carries_every_panel_the_ui_renders():
    from taza_rag.ui.serialize import research_payload

    data = research_payload(_research_result())
    for key in ("plan", "rounds", "ledger", "conflicts", "gaps", "evidence", "cost",
                "coverage", "sub_coverage", "stop_reason", "answer", "citations", "usage"):
        assert key in data, key
    assert data["ledger"]["admitted"] == 1
    assert data["conflicts"][0]["kind"] == "disagreement"
    assert tuple(data["usage"]) == USAGE_FIELDS
    assert data["usage"]["offered"] == 1
    assert data["usage"]["bought"] == 1
    assert data["usage"]["cited"] == 1
    assert data["usage"]["budget"] == 40


def test_research_evidence_keeps_the_pool_label_not_the_rank_label():
    """A rank-derived label would point a citation at the wrong source."""
    from taza_rag.ui.serialize import research_payload

    result = _research_result()
    result.evidence[0].label = "c7"
    result.answer = "Profit was 347.33 billion yen [c7]."
    data = research_payload(result)
    assert data["evidence"][0]["label"] == "c7"
    assert [c["label"] for c in data["citations"]] == ["c7"]


def test_research_verification_is_normalised_for_the_answer_panel():
    from taza_rag.ui.serialize import research_payload

    v = research_payload(_research_result())["verification"]
    assert v["resolved"] is True
    assert v["initial_problems"] == 2
    assert v["final_problems"] == 0


def test_usage_shape_is_identical_everywhere():
    """A second corpus does not get to invent a new consumption record."""
    empty = usage_payload()
    assert tuple(empty) == USAGE_FIELDS
    assert empty == {
        "offered": 0,
        "bought": 0,
        "refused": 0,
        "cited": 0,
        "retrieval_calls": 0,
        "llm_calls": 0,
        "budget": None,
    }


def test_the_research_payload_does_not_leak_credentials():
    from taza_rag.ui.serialize import research_payload

    dumped = str(research_payload(_research_result()))
    for secret in ("password", "api_key", "sk-"):
        assert secret not in dumped
