"""The UI must explain the pipeline without leaking credentials or inventing scores."""

from pathlib import Path

from taza_rag.factiva.pipeline import RetrievalRun
from taza_rag.models import Chunk, RetrievedChunk, SearchIntent
from taza_rag.ui.serialize import (
    SCORE_LEGEND,
    health_payload,
    hit_payload,
    plan_payload,
    run_payload,
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
    assert "Evidence pack" in html
    assert "/app.css" in html
    assert "/app.js" in html
