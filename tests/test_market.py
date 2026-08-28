"""The marketplace loop is the product contract: query is free, bodies cost a transact."""

from taza_rag.market import (
    TRADEOFF_CHEAPEST,
    Market,
    MarketError,
    assemble_packages,
)
from taza_rag.models import Chunk, RetrievedChunk


def _hit(chunk_id: str, *, rank: int, score: float, tokens: int, text: str, doc: str = "d1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=doc,
            text=text,
            title=f"Title {chunk_id}",
            source="Dow Jones Newswires",
            published_at="2026-08-06",
            token_estimate=tokens,
        ),
        score=score,
        rank=rank,
        scores={"tier": 0, "entity": 1.0},
    )


POOL = [
    _hit("c-a", rank=1, score=4.0, tokens=80, text="SECRET_BODY_A profit beat."),
    _hit("c-b", rank=2, score=3.0, tokens=200, text="SECRET_BODY_B longer recap.", doc="d2"),
    _hit("c-c", rank=3, score=2.0, tokens=40, text="SECRET_BODY_C brief wire.", doc="d3"),
]


def test_packages_are_labelled_and_collapse_identical_sets():
    packages = assemble_packages(POOL)
    labels = [p.tradeoff_label for p in packages]
    assert TRADEOFF_CHEAPEST in labels
    assert any(p.price > 1 for p in packages)
    sets = [frozenset(p.chunk_ids) for p in packages]
    assert len(sets) == len(set(sets))
    assert len(packages) >= 2


def test_cheapest_is_the_smallest_token_item():
    cheapest = next(p for p in assemble_packages(POOL) if p.tradeoff_label == TRADEOFF_CHEAPEST)
    assert cheapest.chunk_ids == ("c-c",)
    assert cheapest.price == 1


def test_query_is_free_and_does_not_reveal_bodies():
    market = Market(search=lambda query, top_k: POOL)
    out = market.query("SoftBank profit")
    dumped = str(out)
    assert out["usage"]["bought"] == 0
    assert out["usage"]["offered"] == 3
    assert out["packages"]
    assert "SECRET_BODY_A" not in dumped
    assert "text" not in str(out["packages"])
    for offer in out["packages"]:
        assert offer["tradeoff_label"]
        assert offer["package_id"]
        assert "chunk_ids" not in offer


def test_transact_then_fetch_is_the_only_way_to_read_a_body():
    market = Market(search=lambda query, top_k: POOL)
    bid = market.query("SoftBank profit")
    cheapest = next(p for p in bid["packages"] if p["tradeoff_label"] == TRADEOFF_CHEAPEST)
    grant = market.transact(cheapest["package_id"])
    assert grant["usage"]["bought"] == 1
    assert grant["usage"]["offered"] == 3
    fetched = market.fetch_content(grant["grant_id"])
    texts = [item["text"] for item in fetched["items"]]
    assert texts == ["SECRET_BODY_C brief wire."]
    assert fetched["usage"]["bought"] == 1


def test_a_second_transact_on_the_same_bid_is_refused():
    market = Market(search=lambda query, top_k: POOL)
    bid = market.query("SoftBank profit")
    first, second = bid["packages"][0], bid["packages"][1]
    market.transact(first["package_id"])
    try:
        market.transact(second["package_id"])
    except MarketError as e:
        assert "bid_already_resolved" in str(e)
    else:
        raise AssertionError("sibling package should not be transactable")


def test_fetch_outside_the_grant_is_not_in_scope():
    market = Market(search=lambda query, top_k: POOL)
    bid = market.query("SoftBank profit")
    cheapest = next(p for p in bid["packages"] if p["tradeoff_label"] == TRADEOFF_CHEAPEST)
    grant = market.transact(cheapest["package_id"])
    out = market.fetch_content(grant["grant_id"], items=["c-a"])
    assert out["items"] == []
    assert out["outcomes"][0]["outcome"] == "not_in_scope"


def test_rejecting_a_bid_buys_nothing():
    market = Market(search=lambda query, top_k: POOL)
    bid = market.query("SoftBank profit")
    rejected = market.reject_bid(bid["bid_id"], reason="too_expensive")
    assert rejected["usage"]["bought"] == 0
    assert rejected["usage"]["refused"] == 3
    try:
        market.transact(bid["packages"][0]["package_id"])
    except MarketError as e:
        assert "bid_already_resolved" in str(e)
    else:
        raise AssertionError("rejected bid should not transact")


def test_an_empty_corpus_returns_zero_packages_not_an_error():
    market = Market(search=lambda query, top_k: [])
    out = market.query("nothing")
    assert out["packages"] == []
    assert out["usage"]["offered"] == 0
