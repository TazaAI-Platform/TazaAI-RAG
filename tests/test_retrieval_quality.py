"""Unit tests for retrieval quality logic — no network, no OpenAI."""

from taza_rag.factiva.strategy import detect_intent, expand_queries, normalize_query
from taza_rag.models import Chunk, RetrievedChunk, SearchIntent
from taza_rag.retrieve.features import (
    build_query_plan,
    doc_kind,
    entity_signal,
    extract_entities,
    topic_signal,
)
from taza_rag.retrieve.quality import (
    diversity_cap,
    fuse_and_rerank,
    mmr_diversify,
    rank_candidates,
    relevance_tier,
)


def _hit(
    doc_id: str,
    title: str,
    text: str,
    source: str = "Dow Jones Newswires",
    source_code: str = "djdn",
    published: str = "2026-07-29",
    api_rank: int = 1,
) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"{doc_id}::0",
        doc_id=doc_id,
        text=text,
        title=title,
        source=source,
        source_tier="premium",
        published_at=published,
        metadata={"source_code": source_code},
    )
    return RetrievedChunk(
        chunk=chunk,
        score=1.0,
        rank=api_rank,
        method="factiva_retrieve",
        scores={"api_rank": float(api_rank)},
    )


def test_normalize_fixes_misspelling():
    assert "Deutsche" in normalize_query("Deutche Bank restructuring")


def test_extract_entities_prefers_longest_span():
    ents = extract_entities("Deutsche Bank restructuring")
    assert "Deutsche Bank" in ents
    assert "Deutsche" not in ents


def test_adjacent_entities_are_split_apart():
    """"Larry Fink BlackRock" is a phrase no document contains; as one span it drove
    every entity signal to zero."""
    assert extract_entities("Larry Fink BlackRock private markets") == [
        "Larry Fink",
        "BlackRock",
    ]
    assert extract_entities("Elon Musk Tesla robotaxi") == ["Elon Musk", "Tesla"]


def test_multi_word_organisation_names_survive_the_split():
    assert extract_entities("Deutsche Bank restructuring") == ["Deutsche Bank"]
    assert extract_entities("European Central Bank policy") == ["European Central Bank"]
    assert extract_entities("Goldman Sachs asset management") == ["Goldman Sachs"]


def test_acronyms_stay_attached_to_their_name():
    """An all-caps token is not a brand of its own — "Taza AI" is one entity."""
    assert extract_entities("Taza AI funding") == ["Taza AI"]


def test_missing_a_second_entity_costs_the_top_tier():
    """"Andy Jassy AWS growth strategy" is not answered by a Jassy story about retail."""
    plan = build_query_plan("Andy Jassy AWS growth strategy", SearchIntent.EXECUTIVE_PROFILING)
    both = _hit("d1", "Andy Jassy sets out AWS growth strategy", "AWS revenue accelerated.")
    only_person = _hit("d2", "Andy Jassy's India visit and Amazon's growth strategy", "Retail.")
    assert relevance_tier(plan, entity_signal(plan, both), topic_signal(plan, both)) < relevance_tier(
        plan, entity_signal(plan, only_person), topic_signal(plan, only_person)
    )


def test_single_entity_queries_keep_the_top_tier():
    plan = build_query_plan("Deutsche Bank restructuring", SearchIntent.ENTITY_INVESTIGATION)
    hit = _hit("d1", "Deutsche Bank plans restructuring", "Body.")
    assert relevance_tier(plan, entity_signal(plan, hit), topic_signal(plan, hit)) == 0


def test_split_entities_score_independently():
    plan = build_query_plan("Larry Fink BlackRock private markets", SearchIntent.EXECUTIVE_PROFILING)
    hit = _hit("d1", "Larry Fink defends BlackRock's private markets push", "Body text.")
    sig = entity_signal(plan, hit)
    assert sig["entity_any"] == 1.0
    assert sig["entity_extra"] > 0.0


def test_generic_role_words_are_not_entities():
    """"Larry Fink's annual letter to CEOs" must gate on Fink, not on "CEOs"."""
    ents = extract_entities("Larry Fink's annual letter to CEOs")
    assert ents == ["Larry Fink"]


def test_possessive_does_not_leak_into_topics():
    plan = build_query_plan("Larry Fink's annual letter to CEOs", SearchIntent.EXECUTIVE_PROFILING)
    assert plan.entities == ["Larry Fink"]
    assert "fink's" not in plan.topics
    assert "annual" in plan.topics and "letter" in plan.topics


def test_secondary_entity_alone_does_not_satisfy_gate():
    plan = build_query_plan(
        "Masayoshi Son and Arm Holdings", SearchIntent.EXECUTIVE_PROFILING
    )
    arm_only = _hit("arm", "Arm Holdings raises guidance", "Arm Holdings reported results.")
    sig = entity_signal(plan, arm_only)
    assert sig["entity_any"] == 0.0
    assert sig["entity_extra"] > 0.0


def test_misspelled_entity_still_scores_after_normalization():
    from taza_rag.factiva.strategy import normalize_query

    plan = build_query_plan(
        normalize_query("Deutche Bank restructuring"), SearchIntent.ENTITY_INVESTIGATION
    )
    hit = _hit("d", "Deutsche Bank announces overhaul", "Deutsche Bank confirmed job cuts.")
    assert entity_signal(plan, hit)["entity_title"] == 1.0


def test_query_plan_splits_entity_and_topic():
    plan = build_query_plan("Deutsche Bank restructuring", SearchIntent.ENTITY_INVESTIGATION)
    assert plan.entities == ["Deutsche Bank"]
    assert plan.topics == ["restructuring"]
    assert "job cuts" in plan.expanded_topics


def test_expansion_uses_distinct_paraphrase_groups():
    variants = expand_queries(
        "Deutsche Bank restructuring", SearchIntent.ENTITY_INVESTIGATION, max_variants=4
    )
    assert variants[0] == "Deutsche Bank restructuring"
    assert all(v.startswith("Deutsche Bank") for v in variants)
    # Paraphrase calls must use different vocabulary, not spelling variants
    assert any("overhaul" in v and "job cuts" in v for v in variants)
    assert any("cost cuts" in v and "divest" in v for v in variants)
    # Bare entity is the last-resort recall anchor
    assert "Deutsche Bank" in variants


def test_detect_intent_risk_and_org_not_person():
    assert detect_intent("key risks in private credit covenants") == SearchIntent.RISK_COMPLIANCE
    assert detect_intent("Deutsche Bank") == SearchIntent.ENTITY_INVESTIGATION
    assert detect_intent("Jerome Powell") == SearchIntent.EXECUTIVE_PROFILING


def test_doc_kind_detects_digest_and_profile():
    digest = _hit(
        "d1",
        "Dow Jones Top Financial Services Headlines at 12 AM ET: Robinhood to Cut 10% "
        "of Workforce in Restructuring | Germany ...",
        "Robinhood to Cut 10% of Workforce in Restructuring",
    )
    profile = _hit(
        "d2",
        "Deutsche Bank AG - History",
        "Deutsche Bank AG history of contracts",
        source="GlobalData Company Profiles",
        source_code="glomcp",
    )
    article = _hit("d3", "Deutsche Bank plans fresh buyback", "Deutsche Bank said it would buy back")
    assert doc_kind(digest) == "digest"
    assert doc_kind(profile) == "profile"
    assert doc_kind(article) == "article"


def test_entity_and_topic_signals():
    plan = build_query_plan("Deutsche Bank restructuring", SearchIntent.ENTITY_INVESTIGATION)
    on_topic = _hit(
        "d1",
        "Deutsche Bank to cut jobs in overhaul",
        "Deutsche Bank said the overhaul includes job cuts across its investment bank.",
    )
    off_entity = _hit("d2", "Robinhood to cut 10% of workforce", "Robinhood restructuring")

    assert entity_signal(plan, on_topic)["entity_title"] == 1.0
    assert entity_signal(plan, off_entity)["entity_any"] == 0.0
    assert topic_signal(plan, on_topic)["topic_title"] == 1.0


def test_ranking_demotes_digest_and_promotes_entity_story():
    plan = build_query_plan("Deutsche Bank restructuring", SearchIntent.ENTITY_INVESTIGATION)
    digest = _hit(
        "digest",
        "Dow Jones Top Financial Services Headlines at 12 AM ET: Robinhood to Cut 10% "
        "of Workforce in Restructuring | Germany Rejects UniCredit Bid | More",
        "Robinhood to Cut 10% of Workforce in Restructuring. Germany rejects UniCredit bid.",
        api_rank=1,
    )
    story = _hit(
        "story",
        "Deutsche Bank accelerates overhaul with fresh job cuts",
        "Deutsche Bank said its restructuring will bring further job cuts and cost savings.",
        source="wsj.com",
        source_code="wsjo",
        api_rank=4,
    )
    profile = _hit(
        "profile",
        "Deutsche Bank AG - Company Profile",
        "Deutsche Bank AG is a global financial institution.",
        source="MarketLine Company Profiles",
        source_code="datmon",
        api_rank=2,
    )

    ranked = rank_candidates(plan, [[digest, profile, story]], top_k=3)
    assert ranked[0].chunk.doc_id == "story"
    assert ranked[0].scores["penalty"] == 0.0
    kinds = [(h.chunk.metadata or {}).get("doc_kind") for h in ranked]
    assert "digest" in kinds or len(ranked) < 3


def test_entity_gate_drops_off_entity_candidates():
    plan = build_query_plan("Deutsche Bank restructuring", SearchIntent.ENTITY_INVESTIGATION)
    good = [
        _hit(f"g{i}", f"Deutsche Bank overhaul step {i}", "Deutsche Bank restructuring job cuts")
        for i in range(3)
    ]
    bad = _hit("bad", "Robinhood restructuring", "Robinhood cuts jobs")
    ranked = rank_candidates(plan, [good + [bad]], top_k=5)
    assert all(h.chunk.doc_id != "bad" for h in ranked)


def test_tiering_puts_topic_match_above_entity_only():
    plan = build_query_plan("Deutsche Bank restructuring", SearchIntent.ENTITY_INVESTIGATION)
    on_topic = _hit(
        "restructuring",
        "Deutsche Bank sells India retail unit in Hausbank overhaul",
        "Deutsche Bank will divest the business as part of its restructuring, sharpening focus.",
        source="zacks.com",
        source_code="zdcom",
        api_rank=9,
    )
    off_topic = _hit(
        "buyback",
        "Deutsche Bank Plans Fresh $569 Million Buyback After Earnings Beat Hopes",
        "Deutsche Bank said it would buy back another half a billion euros in shares.",
        source="wsj.com",
        source_code="wsjo",
        api_rank=2,
    )
    buried_topic = _hit(
        "raid",
        "Deutsche Bank headquarters raided for third time this year",
        "Latest search threatens efforts to rehabilitate the lender. "
        + ("Unrelated detail. " * 40)
        + "Analysts noted the overhaul announced last year.",
        source="Luxembourg Times",
        source_code="luxtim",
        api_rank=7,
    )
    ranked = rank_candidates(plan, [[off_topic, buried_topic, on_topic]], top_k=3)
    assert [h.chunk.doc_id for h in ranked] == ["restructuring", "raid", "buyback"]
    assert [h.scores["tier"] for h in ranked] == [0.0, 1.0, 2.0]


def test_mmr_breaks_up_a_repeated_angle():
    plan = build_query_plan("private credit market trends", SearchIntent.TOPICAL_EXPLORATION)
    india = [
        _hit(
            f"in{i}",
            f"India's private credit market doubles to $25 billion AUM report {i}",
            "India's private credit market has doubled over five years to $25 billion AUM",
            source=f"Indian Outlet {i}",
            source_code=f"ind{i}",
        )
        for i in range(3)
    ]
    other = _hit(
        "wsj",
        "Private Credit Is Under Growing Strain as Default Rates Hit Highs",
        "Private credit is showing increasing signs of stress across US direct lenders.",
        source="wsj.com",
        source_code="wsjo",
        api_rank=6,
    )
    ranked = rank_candidates(plan, [india + [other]], top_k=10)
    diversified = mmr_diversify(diversity_cap(ranked, max_per_source=2), top_k=3)
    assert "wsj" in [h.chunk.doc_id for h in diversified[:2]]


def test_inflected_headline_counts_as_topic_match():
    """"Deutsche Bank sells Indian assets" is divestment news even without "sell"."""
    plan = build_query_plan("Deutsche Bank restructuring", SearchIntent.ENTITY_INVESTIGATION)
    hit = _hit(
        "sale",
        "Deutsche Bank sells Indian assets but backs country's growth outlook",
        "The lender agreed to offload its retail arm.",
    )
    assert topic_signal(plan, hit)["topic_title"] == 1.0
    assert entity_signal(plan, hit)["entity_title"] == 1.0


def test_possessive_entity_form_matches():
    plan = build_query_plan("Deutsche Bank strategy", SearchIntent.ENTITY_INVESTIGATION)
    hit = _hit("nl", "Deutsche Bank's bankers learn AI", "Strategic priorities shift.")
    assert entity_signal(plan, hit)["entity_title"] == 1.0


def test_newsletter_roundup_is_treated_as_digest():
    hit = _hit(
        "nl",
        "Jane Fraser's 'damn tough' time; The latest investment banking news from Financial News",
        "This is an online version of Financial News's weekly investment banking newsletter. "
        "You can sign up to receive it via email first here.",
        source="fnlondon.com",
        source_code="lonfin",
    )
    assert doc_kind(hit) == "digest"


def test_fuse_and_rerank_prefers_higher_authority_on_a_tie():
    """The query-string wrapper still applies the source authority prior."""
    wire = _hit(
        "wire",
        "SoftBank Group profit rises on Arm gains",
        "SoftBank Group reported higher profit.",
        source="Some Local Outlet",
        source_code="localx",
    )
    premium = _hit(
        "premium",
        "SoftBank Group earnings climb as Vision Fund recovers",
        "SoftBank Group reported higher profit.",
        source="wsj.com",
        source_code="wsjo",
    )
    ranked = fuse_and_rerank("SoftBank Group", [[wire], [premium]], top_k=2)
    assert ranked[0].chunk.doc_id == "premium"
    assert ranked[0].scores["authority"] > ranked[1].scores["authority"]


def test_near_duplicates_are_collapsed():
    plan = build_query_plan("Deutsche Bank buyback", SearchIntent.ENTITY_INVESTIGATION)
    a = _hit("a", "Deutsche Bank Plans Fresh $569 Million Buyback After Earnings Beat Hopes", "text a")
    b = _hit("b", "Deutsche Bank Plans Fresh $569 Million Buyback After Earnings Beat Hopes Again", "text b")
    ranked = rank_candidates(plan, [[a, b]], top_k=5)
    assert len(ranked) == 1
