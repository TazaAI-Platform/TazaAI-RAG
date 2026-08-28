"""Buying decisions have to be defensible line by line, so they are tested that way.

The gate is the agent's answer to "value before access": score a candidate on what a buyer can
see before paying, buy only what should move the answer, and record why. Two properties matter
most — that it never reads the body it has not bought, and that it cannot starve a run by
refusing everything.
"""

from taza_rag.agent.purchase import (
    LEAD_CHARS,
    Ledger,
    metadata_of,
    purchasable,
    score_candidate,
    select,
)
from taza_rag.models import Chunk, RetrievedChunk


def _hit(doc_id, title, text, *, score=1.0, authority=1.0, freshness=1.0, source="Dow Jones"):
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"{doc_id}::p000",
            doc_id=doc_id,
            text=text,
            title=title,
            source=source,
            published_at="2026-08-06",
        ),
        score=score,
        rank=1,
        scores={"authority": authority, "freshness": freshness},
    )


BOND = _hit(
    "d1",
    "SoftBank plans record retail bond issuance",
    "SoftBank Group plans a record retail bond issuance of 6.3 billion dollars.",
    score=4.0,
)
UNRELATED = _hit(
    "d2",
    "Tokyo stocks close mixed as investors await data",
    "The Nikkei ended the session little changed on light volume.",
    score=1.0,
)


def _select(candidates, wanted, **kw):
    return select(
        candidates,
        wanted=wanted,
        pooled_chunk_ids=kw.get("pooled_chunk_ids", set()),
        pooled_doc_ids=kw.get("pooled_doc_ids", set()),
        budget_left=kw.get("budget_left", 10),
        round_index=kw.get("round_index", 0),
    )


def test_a_candidate_whose_headline_targets_an_open_gap_is_bought():
    admitted, ledger = _select([("s1", BOND)], ["record retail bond issuance"])
    assert [h.chunk.doc_id for _s, h in admitted] == ["d1"]
    assert ledger.admitted[0].admitted
    assert "targets" in ledger.admitted[0].reason


def test_a_candidate_that_targets_nothing_open_is_refused():
    admitted, ledger = _select([("s1", UNRELATED)], ["record retail bond issuance"])
    assert admitted == []
    assert ledger.rejected
    assert "below value threshold" in ledger.rejected[0].reason


def test_the_gate_only_reads_the_headline_and_lead_never_the_body():
    """The constraint is the whole point: a buyer cannot see what it has not paid for."""
    buried = _hit(
        "d9",
        "Tokyo stocks close mixed as investors await data",
        "Filler. " * 120 + "SoftBank announced a record retail bond issuance.",
    )
    meta = metadata_of(buried)
    assert "record retail bond issuance" not in meta
    assert len(meta) <= len(buried.chunk.title) + LEAD_CHARS + 1

    admitted, _ledger = _select([("s1", buried)], ["record retail bond issuance"])
    assert admitted == [], "the gate scored on text it should not have been able to see"


def test_the_passage_budget_caps_what_can_be_bought():
    a = _hit("d1", "SoftBank record retail bond issuance", "body", score=5.0)
    b = _hit("d2", "SoftBank record retail bond issuance again", "body", score=4.0)
    c = _hit("d3", "SoftBank record retail bond issuance third", "body", score=3.0)
    admitted, ledger = _select(
        [("s1", a), ("s1", b), ("s1", c)], ["record retail bond issuance"], budget_left=2
    )
    assert len(admitted) == 2
    assert len(ledger.rejected) == 1
    assert "budget exhausted" in ledger.rejected[0].reason
    # The budget is spent on the highest-value candidates, not the first to arrive.
    assert {h.chunk.doc_id for _s, h in admitted} == {"d1", "d2"}


def test_a_passage_already_held_costs_nothing_and_does_not_consume_budget():
    admitted, ledger = _select(
        [("s2", BOND)],
        ["record retail bond issuance"],
        pooled_chunk_ids={"d1::p000"},
        budget_left=0,
    )
    assert len(admitted) == 1
    assert ledger.admitted[0].value == 0.0
    assert "already held" in ledger.admitted[0].reason


def test_another_passage_of_a_document_already_held_is_worth_less():
    plain, _ = score_candidate(
        BOND, wanted=["record retail bond issuance"], pooled_doc_ids=set(), best_rank_score=4.0
    )
    duplicate, reason = score_candidate(
        BOND, wanted=["record retail bond issuance"], pooled_doc_ids={"d1"}, best_rank_score=4.0
    )
    assert duplicate < plain
    assert "already held" in reason


def test_authority_and_freshness_raise_the_price_a_candidate_can_justify():
    weak = _hit("d1", "SoftBank record retail bond issuance", "b", authority=0.95, freshness=0.92)
    strong = _hit("d2", "SoftBank record retail bond issuance", "b", authority=1.12, freshness=1.12)
    wanted = ["record retail bond issuance"]
    weak_value, _ = score_candidate(weak, wanted=wanted, pooled_doc_ids=set(), best_rank_score=1.0)
    strong_value, _ = score_candidate(strong, wanted=wanted, pooled_doc_ids=set(), best_rank_score=1.0)
    assert strong_value > weak_value


def test_a_structural_aspect_gives_no_purchasing_signal():
    """"financial statements" is satisfied by any number, so it cannot discriminate.

    Left in, one such aspect made the first live run admit 36 of 36 candidates.
    """
    assert purchasable(["financial statements", "official comment"]) == []
    assert purchasable(["record retail bond issuance"]) == ["record retail bond issuance"]

    market_wrap = _hit(
        "d3",
        "Tokyo stocks close mixed as investors await data",
        "The Nikkei ended little changed at 42,150 as a trader said volume was light.",
    )
    # Carries a figure and an attribution, so it satisfies both structural aspects — and must
    # still not be bought against them.
    admitted, ledger = _select([("s1", market_wrap)], ["financial statements", "official comment"])
    assert "no specific open gap" in ledger.decisions[0].reason


def test_with_no_open_gap_the_gate_falls_back_to_rank_instead_of_starving_the_run():
    """Scoring targeting as zero here would refuse everything and leave the agent blind."""
    admitted, ledger = _select([("s1", BOND), ("s1", UNRELATED)], [])
    assert admitted, "an empty gap list must not empty the evidence pool"
    assert "no specific open gap" in ledger.decisions[0].reason


def test_selection_is_deterministic_regardless_of_arrival_order():
    a = _hit("d1", "SoftBank record retail bond issuance", "b", score=5.0)
    b = _hit("d2", "SoftBank record retail bond issuance", "b", score=5.0)
    wanted = ["record retail bond issuance"]
    first, _ = _select([("s1", a), ("s1", b)], wanted, budget_left=1)
    second, _ = _select([("s1", b), ("s1", a)], wanted, budget_left=1)
    assert [h.chunk.doc_id for _s, h in first] == [h.chunk.doc_id for _s, h in second]


def test_the_ledger_accounts_for_every_candidate_it_was_offered():
    candidates = [("s1", BOND), ("s1", UNRELATED)]
    _admitted, ledger = _select(candidates, ["record retail bond issuance"])
    assert len(ledger.decisions) == len(candidates)
    assert len(ledger.admitted) + len(ledger.rejected) == len(candidates)
    assert 0.0 <= ledger.admission_rate() <= 1.0
    assert sum(ledger.rejection_reasons().values()) == len(ledger.rejected)


def test_an_empty_ledger_reports_zero_rather_than_dividing_by_zero():
    assert Ledger().admission_rate() == 0.0
    assert Ledger().payload()["offered"] == 0


def test_one_passage_wanted_by_two_steps_in_the_same_wave_is_charged_once():
    """Charging per offer read four bought passages as twelve."""
    _admitted, ledger = _select(
        [("s1", BOND), ("s2", BOND), ("s3", BOND)], ["record retail bond issuance"]
    )
    payload = ledger.payload()
    assert payload["offered"] == 3
    assert payload["charged"] == 1
    assert payload["already_held"] == 2
    # Every step that wanted it is still credited, so `found_by` stays complete.
    assert {d.sub_question_id for d in ledger.admitted} == {"s1", "s2", "s3"}


def test_a_refusal_is_recorded_for_every_step_that_asked_for_it():
    _admitted, ledger = _select([("s1", UNRELATED), ("s2", UNRELATED)], ["record retail bond issuance"])
    assert ledger.payload()["offered"] == 2
    assert ledger.payload()["rejected"] == 2
    assert ledger.payload()["charged"] == 0


def test_a_passage_already_held_is_not_counted_as_a_purchase():
    """Counting free re-offers as buys made the UI read "12 bought" beside a pack of 4."""
    _admitted, ledger = _select(
        [("s1", BOND), ("s2", BOND), ("s1", UNRELATED)],
        ["record retail bond issuance"],
        pooled_chunk_ids={"d1::p000"},
    )
    payload = ledger.payload()
    assert payload["offered"] == 3
    assert payload["already_held"] == 2
    assert payload["charged"] == 0
    assert payload["rejected"] == 1
    assert payload["admission_rate"] == 0.0


def _offer(label, contents, price):
    return {
        "package_id": label,
        "tradeoff_label": label,
        "price": {"amount": price, "unit": "chunks"},
        "contents": contents,
    }


def test_choose_package_buys_the_catalog_that_targets_the_gap():
    from taza_rag.agent.purchase import choose_package

    bond = {
        "title": "SoftBank plans record retail bond issuance",
        "source": "The Wall Street Journal",
        "doc_id": "d1",
        "score": 4.0,
        "authority": 1.12,
        "freshness": 1.0,
    }
    junk = {
        "title": "Tokyo stocks close mixed as investors await data",
        "source": "Dow Jones",
        "doc_id": "d2",
        "score": 3.0,
        "authority": 1.0,
        "freshness": 1.0,
    }
    picked = choose_package(
        [
            _offer("most_thorough", [bond, junk], 2),
            _offer("cheapest", [junk], 1),
        ],
        wanted=["record retail bond issuance"],
        budget_left=10,
    )
    assert picked is not None
    assert picked["package_id"] == "most_thorough"


def test_choose_package_walks_away_when_nothing_is_worth_buying():
    from taza_rag.agent.purchase import choose_package

    junk = {
        "title": "Tokyo stocks close mixed as investors await data",
        "source": "Dow Jones",
        "doc_id": "d2",
        "score": 3.0,
    }
    assert (
        choose_package(
            [_offer("cheapest", [junk], 1)],
            wanted=["record retail bond issuance"],
            budget_left=10,
        )
        is None
    )


def test_choose_package_will_not_overshoot_the_passage_budget():
    from taza_rag.agent.purchase import choose_package

    bond = {
        "title": "SoftBank plans record retail bond issuance",
        "source": "WSJ",
        "doc_id": "d1",
        "score": 4.0,
    }
    picked = choose_package(
        [_offer("most_thorough", [bond, bond, bond], 3), _offer("cheapest", [bond], 1)],
        wanted=["record retail bond issuance"],
        budget_left=1,
    )
    assert picked is not None
    assert picked["package_id"] == "cheapest"
