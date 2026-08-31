"""Sample-corpus playground: the hosted demo must work without Factiva or OpenAI."""

from taza_rag.demo import DemoIndex, demo_handlers, extractive_answer
from taza_rag.factiva.facts import extractive_compose, extractive_facts, first_sentences
from taza_rag.market import Market
from taza_rag.ui.server import UiHandler


def test_first_sentences_keeps_the_lead():
    text = "SoftBank Group is expanding AI bets. Vision Fund may exit mature holdings. Ignore this."
    assert first_sentences(text, 2) == [
        "SoftBank Group is expanding AI bets.",
        "Vision Fund may exit mature holdings.",
    ]


def test_extractive_facts_are_grounded_in_the_excerpt():
    evidence = {
        "c1": "SoftBank AI\nSoftBank Group is expanding its artificial intelligence investments. Arm remains central."
    }
    facts = extractive_facts(evidence)
    assert facts
    assert all(f.citation == "c1" for f in facts)
    composed = extractive_compose(facts)
    assert "[c1]" in composed["answer"]
    assert composed["abstain"] is False


def test_demo_retrieve_finds_softbank_and_normalises_deutsche():
    index = DemoIndex()
    run = index.retrieve("SoftBank Group", top_k=5)
    assert run.hits
    assert any("SoftBank" in (h.chunk.title or "") for h in run.hits)
    misspelled = index.retrieve("Deutche Bank restructuring", top_k=5)
    assert any("Deutsche Bank" in (h.chunk.title or "") for h in misspelled.hits)


def test_demo_market_query_does_not_leak_bodies():
    index = DemoIndex()
    market = Market(search=index.search)
    bid = market.query("SoftBank Group", top_k=8)
    assert bid["packages"]
    dumped = str(bid)
    assert "super optimistic" not in dumped
    densest = next(p for p in bid["packages"] if p["tradeoff_label"] == "densest")
    grant = market.transact(densest["package_id"])
    fetched = market.fetch_content(grant["grant_id"])
    assert fetched["items"]
    assert any("SoftBank" in (item.get("title") or "") for item in fetched["items"])


def test_extractive_answer_cites_licensed_hits():
    hits = DemoIndex().search("Jerome Powell", top_k=3)
    payload = extractive_answer("Jerome Powell", hits)
    assert payload["answer"]
    assert "[c1]" in payload["answer"]
    assert payload["citations"]
    assert payload["usage"]["llm_calls"] == 0


def test_demo_health_and_query_handlers_are_wired():
    handlers = demo_handlers()
    handler = UiHandler.__new__(UiHandler)
    handler.server = type(
        "S",
        (),
        {
            "demo": True,
            "market": handlers["market"],
            "query_fn": None,
            "retrieve_fn": handlers["retrieve_fn"],
            "write_fn": handlers["write_fn"],
            "research_fn": handlers["research_fn"],
        },
    )()
    from taza_rag.ui.serialize import health_payload

    health = health_payload(factiva=False, openai=False, demo=True)
    assert health["demo"] is True
    bid = handler._query({"query": "private credit market trends", "top_k": 6})
    assert bid["packages"]
    assert bid["usage"]["bought"] == 0


def test_extractive_research_answers_from_the_sample_corpus():
    handlers = demo_handlers()
    run = handlers["research_fn"](
        "How exposed is SoftBank Group to its AI bets?",
        {"top_k": 4, "max_rounds": 1, "max_chunks": 12, "purchase_gate": True},
    )
    assert run["plan"]["method"] == "heuristic"
    assert run["usage"]["llm_calls"] == 0
    assert run["evidence"]
    assert "[c" in (run.get("answer") or "")
