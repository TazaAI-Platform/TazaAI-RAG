"""A1 answer-level eval — offline, with the LLM and Factiva calls stubbed.

The point is that the whole scoring path is verifiable without spending credits or
hitting the live API, so a billing problem can never be mistaken for a broken judge.
"""

from pathlib import Path

import taza_rag.eval.a1_factiva as a1mod
import taza_rag.eval.dj_a1 as judge_mod
from taza_rag.eval.a1_factiva import GATES, _aggregate, run_a1_eval
from taza_rag.eval.dj_a1 import judge_a1
from taza_rag.models import (
    AnswerResult,
    Chunk,
    Citation,
    GoldExample,
    RetrievedChunk,
    SearchIntent,
)

PERFECT = {
    "factual_correctness": True,
    "citation_integrity": True,
    "no_hallucinations": True,
    "contextual_integrity": True,
    "relevance": 3,
    "completeness": 3,
    "clarity": 3,
    "failure_tags": [],
    "notes": "clean",
}
HALLUCINATED = {
    "factual_correctness": False,
    "citation_integrity": True,
    "no_hallucinations": False,
    "contextual_integrity": True,
    "relevance": 2,
    "completeness": 2,
    "clarity": 3,
    "failure_tags": ["hallucination", "uncited_claim"],
    "notes": "invented a figure",
}


def _result(answer: str = "Deutsche Bank cut costs [c1].", abstained: bool = False) -> AnswerResult:
    chunk = Chunk(
        chunk_id="d1::p000",
        doc_id="d1",
        text="Deutsche Bank announced cost cuts.",
        title="Deutsche Bank cuts costs",
        source="wsj.com",
        published_at="2026-07-01",
    )
    return AnswerResult(
        query="Deutsche Bank restructuring",
        answer=answer,
        citations=[
            Citation(
                chunk_id="d1::p000",
                doc_id="d1",
                title="Deutsche Bank cuts costs",
                source="wsj.com",
                published_at="2026-07-01",
                excerpt="cost cuts",
            )
        ],
        retrieved=[RetrievedChunk(chunk=chunk, score=1.0, rank=1)],
        abstained=abstained,
        config_name="factiva_quality_v2+ctx",
    )


def test_context_is_budgeted_by_tokens_not_chunk_count():
    """Passages are smaller, so a fixed chunk count would under-fill the context."""
    from taza_rag.factiva.answer import _pack_context

    def hit(i: int, words: int) -> RetrievedChunk:
        c = Chunk(
            chunk_id=f"d{i}::p000",
            doc_id=f"d{i}",
            text=" ".join(["word"] * words),
            title=f"Story {i}",
            source="wsj.com",
        )
        return RetrievedChunk(chunk=c, score=1.0, rank=i)

    small = [hit(i, 100) for i in range(10)]
    _, picked_small = _pack_context(small, max_chunks=16, max_tokens=550)
    assert len(picked_small) == 5

    large = [hit(i, 500) for i in range(10)]
    _, picked_large = _pack_context(large, max_chunks=16, max_tokens=550)
    assert len(picked_large) == 1

    # A single oversized chunk still gets through rather than yielding empty context
    huge = [hit(0, 5000)]
    text, picked_huge = _pack_context(huge, max_chunks=16, max_tokens=550)
    assert len(picked_huge) == 1 and text


def test_accuracy_is_a_hard_gate():
    """All four checks must hold; one failure fails Accuracy regardless of the rest."""
    judge_mod.chat_json = lambda *a, **k: dict(HALLUCINATED)
    j = judge_a1("q1", _result(), "excerpts")
    assert j.accuracy.pass_ is False
    assert j.overall_pass is False
    assert "hallucination" in j.failure_tags

    judge_mod.chat_json = lambda *a, **k: dict(PERFECT)
    j2 = judge_a1("q1", _result(), "excerpts")
    assert j2.accuracy.pass_ is True
    assert j2.overall_pass is True


def test_overall_pass_needs_two_or_better_on_every_dimension():
    weak = dict(PERFECT, clarity=1)
    judge_mod.chat_json = lambda *a, **k: dict(weak)
    j = judge_a1("q1", _result(), "excerpts")
    assert j.accuracy.pass_ is True
    assert j.overall_pass is False


def test_judge_sees_only_the_evidence_the_answer_was_given():
    captured = {}

    def fake(system, user, **kw):
        captured["user"] = user
        return dict(PERFECT)

    judge_mod.chat_json = fake
    judge_a1("q1", _result(), "[c1] Deutsche Bank cuts costs")
    assert "Deutsche Bank cuts costs" in captured["user"]
    # The gold file's expected terms must never leak into the judge prompt
    assert "must_include_terms" not in captured["user"]


def test_aggregate_reports_gate_level_pass_rates():
    rows = [
        {
            "intent": "entity_investigation",
            "expect_abstention": False,
            "abstained": False,
            "accuracy_pass": True,
            "overall_pass": True,
            "a1": {
                "accuracy": {g: True for g in GATES},
                "relevance": {"score": 3},
                "completeness": {"score": 2},
                "clarity": {"score": 3},
                "failure_tags": [],
            },
        },
        {
            "intent": "entity_investigation",
            "expect_abstention": True,
            "abstained": False,
            "accuracy_pass": False,
            "overall_pass": False,
            "a1": {
                "accuracy": {**{g: True for g in GATES}, "no_hallucinations": False},
                "relevance": {"score": 2},
                "completeness": {"score": 2},
                "clarity": {"score": 2},
                "failure_tags": ["hallucination"],
            },
        },
    ]
    agg = _aggregate(rows)
    # Only the answerable row counts toward answer quality
    assert agg["n_scored"] == 1
    assert agg["accuracy_pass_rate"] == 1.0
    assert agg["gate_pass_rates"]["factual_correctness"] == 1.0
    assert agg["mean_relevance"] == 3.0
    # The row that should have refused did not
    assert agg["abstention_recall"] == 0.0
    assert agg["n_expected_abstention"] == 1


def test_eval_runs_end_to_end_offline(tmp_path: Path = Path("/tmp/a1_offline")):
    gold = tmp_path / "gold.jsonl"
    tmp_path.mkdir(parents=True, exist_ok=True)
    gold.write_text(
        GoldExample(
            id="g1",
            query="Deutsche Bank restructuring",
            intent=SearchIntent.ENTITY_INVESTIGATION,
            must_include_terms=["Deutsche"],
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    a1mod.answer_with_factiva = lambda q, **kw: _result()
    judge_mod.chat_json = lambda *a, **k: dict(PERFECT)

    report = tmp_path / "report.json"
    summary = run_a1_eval(gold, report, top_k=3)
    assert summary["n_scored"] == 1
    assert summary["accuracy_pass_rate"] == 1.0
    assert summary["mean_clarity"] == 3.0
    assert report.exists() and report.with_suffix(".md").exists()
    assert "Accuracy gates" in report.with_suffix(".md").read_text()


def test_a_refusal_is_recorded_as_abstention():
    a1mod.answer_with_factiva = lambda q, **kw: _result(
        answer="Insufficient evidence.", abstained=True
    )
    judge_mod.chat_json = lambda *a, **k: dict(PERFECT)
    tmp = Path("/tmp/a1_abstain")
    tmp.mkdir(parents=True, exist_ok=True)
    gold = tmp / "gold.jsonl"
    gold.write_text(
        GoldExample(
            id="a1",
            query="What was the exact figure in November 2027?",
            intent=SearchIntent.ENTITY_INVESTIGATION,
            expect_abstention=True,
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    summary = run_a1_eval(gold, tmp / "report.json", top_k=3)
    assert summary["abstention_recall"] == 1.0
    # A refusal that was asked for must not be averaged into answer-quality scores
    assert summary["n_scored"] == 0
    assert summary["n_expected_abstention"] == 1
