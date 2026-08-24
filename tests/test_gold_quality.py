"""The gold set is the instrument every other number depends on, so it gets linted.

Two classes of defect here are invisible in the reported metrics and flatter them:
a term that matches English rather than the subject, and an intent with too few rows to
mean anything while still contributing to the headline average.
"""

from pathlib import Path

from taza_rag.eval.factiva_retrieval import term_matches
from taza_rag.eval.retrieval import intent_counts, load_gold
from taza_rag.models import SearchIntent

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "evals/gold/factiva_live_v1.jsonl"
ABSTAIN = ROOT / "evals/gold/factiva_abstain_v1.jsonl"

# Words so common that a term containing one as a whole word is always matched, which
# would score for free on any pack.
DEGENERATE = {
    "said",
    "says",
    "year",
    "company",
    "market",
    "business",
    "new",
    "report",
    "news",
    "time",
    "people",
}

MIN_PER_INTENT = 3


def test_term_matching_is_whole_word():
    """The bug this replaced: substring matching scored these for free."""
    assert not term_matches("AI", "the spokesman said nothing")
    assert not term_matches("EV", "seven analysts")
    assert not term_matches("AWS", "under european laws")
    assert not term_matches("oil", "would spoil the deal")
    assert not term_matches("rate", "the corporate structure")
    assert not term_matches("Abel", "a private label brand")
    assert not term_matches("import", "an important announcement")


def test_term_matching_still_finds_real_mentions():
    assert term_matches("AI", "the AI Act applies")
    assert term_matches("AWS", "AWS revenue grew")
    assert term_matches("oil", "oil prices fell")
    assert term_matches("private credit", "the private credit market")
    assert term_matches("high-risk", "a high-risk system")
    assert term_matches("Nvidia", "NVIDIA said")


def test_spelling_alternation_counts_as_one_term():
    assert term_matches("defence|defense", "the defense budget")
    assert term_matches("defence|defense", "the defence budget")
    assert not term_matches("defence|defense", "the offence was minor")


def test_gold_ids_are_unique_across_both_sets():
    ids = [ex.id for ex in load_gold(LIVE)] + [ex.id for ex in load_gold(ABSTAIN)]
    assert len(ids) == len(set(ids)), "duplicate gold ids make rows impossible to trace"


def test_answerable_rows_declare_what_must_be_found():
    for ex in load_gold(LIVE):
        assert ex.must_include_terms, f"{ex.id} has no must_include_terms"
        assert not ex.expect_abstention, f"{ex.id} is in the answerable set"


def test_abstention_rows_all_expect_abstention():
    rows = load_gold(ABSTAIN)
    assert len(rows) >= 10, "too few rows to read abstention recall as a rate"
    for ex in rows:
        assert ex.expect_abstention, f"{ex.id} is in the abstention set but expects an answer"
        assert ex.notes, f"{ex.id} needs a note saying why it is unanswerable"


def test_no_term_is_a_degenerate_english_word():
    """Only bare common words are degenerate: a phrase like "market share" is specific
    even though "market" alone would not be."""
    for ex in load_gold(LIVE) + load_gold(ABSTAIN):
        for term in ex.must_include_terms + ex.nice_to_have_terms:
            for variant in term.lower().split("|"):
                words = variant.split()
                if len(words) == 1:
                    assert words[0] not in DEGENERATE, f"{ex.id}: {term!r} scores for free"


def test_alternation_variants_are_non_empty():
    for ex in load_gold(LIVE):
        for term in ex.must_include_terms + ex.nice_to_have_terms:
            if "|" in term:
                assert all(v.strip() for v in term.split("|")), f"{ex.id}: empty variant"


def test_every_factiva_intent_has_enough_rows_to_measure():
    """A stratum of one cannot support a per-intent number but still moves the mean."""
    counts = intent_counts(load_gold(LIVE))
    missing = {i.value: counts.get(i.value, 0) for i in SearchIntent}
    thin = {k: v for k, v in missing.items() if v < MIN_PER_INTENT}
    assert not thin, f"intents below {MIN_PER_INTENT} rows: {thin}"


def test_the_set_is_large_enough_to_move_less_than_a_point_per_query():
    rows = load_gold(LIVE)
    assert len(rows) >= 50, f"n={len(rows)}: one query is worth {100 / max(len(rows), 1):.1f} points"
