"""The research agent is a marketplace client: bodies arrive only after transact."""

from taza_rag.agent.fixtures import FixtureDoc, FixtureSearch
from taza_rag.agent.gather import MarketBackend
from taza_rag.agent.loop import research
from taza_rag.agent.models import Budget
from taza_rag.market import Market
from tests.test_agent_loop import DOCS, QUESTION, _Harness


def test_query_catalogs_do_not_contain_passage_bodies():
    secret = "SECRET_BODY_PROFIT 347.33 billion yen"
    docs = [
        FixtureDoc(
            doc_id="d1",
            title="SoftBank Group net profit falls 18 percent",
            text=secret,
        )
    ]
    market = Market(search=FixtureSearch(docs).search)
    out = market.query("SoftBank Group quarterly net profit")
    dumped = str(out)
    assert out["packages"]
    assert "SECRET_BODY_PROFIT" not in dumped
    assert out["usage"]["bought"] == 0


def test_research_through_the_market_only_reads_bodies_from_a_grant():
    fixture = FixtureSearch(list(DOCS))
    market = Market(search=fixture.search)
    backend = MarketBackend(market=market)
    with _Harness():
        result = research(QUESTION, backend=backend, budget=Budget(top_k_per_query=4))

    assert not result.abstained
    assert result.evidence
    assert market._grants, "every pooled passage must have been fetched under a grant"
    texts = " ".join(item.hit.chunk.text for item in result.evidence)
    assert "347.33" in texts or "10 billion" in texts or "6.3 billion" in texts
    for decision in result.ledger.charged:
        assert "bought" in decision.reason


def test_a_tight_budget_buys_one_package_not_one_per_sub_question():
    fixture = FixtureSearch(list(DOCS))
    market = Market(search=fixture.search)
    backend = MarketBackend(market=market)
    with _Harness():
        result = research(
            QUESTION,
            backend=backend,
            budget=Budget(top_k_per_query=4, max_unique_chunks=1),
        )

    assert result.cost.unique_chunks <= 1
    assert len(market._grants) == 1


def test_a_failing_market_query_does_not_sink_the_run():
    fixture = FixtureSearch(list(DOCS), fail_on={"SoftBank Group quarterly net profit"})
    backend = MarketBackend(market=Market(search=fixture.search))
    with _Harness():
        result = research(QUESTION, backend=backend, budget=Budget(top_k_per_query=4))

    assert result.errors
    assert "SoftBank Group quarterly net profit" in result.rounds[0].failed_queries
    assert not result.abstained
    assert result.answer
