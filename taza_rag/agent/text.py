"""Deterministic text matching shared by coverage, conflict detection and the eval.

These helpers decide whether an aspect is covered and whether two facts are about the same
thing. They are deliberately model-free: the agent's stopping decision is only trustworthy
if the signal it stops on cannot itself hallucinate, and the eval is only trustworthy if it
does not reuse the generator to grade the generator.

Matching is stem-insensitive so "cuts costs" registers against "cost cutting", and it is
whole-term so "AI" cannot match inside "said" — the substring bug that once scored gold
terms for free in the retrieval eval.
"""

from __future__ import annotations

from taza_rag.retrieve.features import content_terms, stem


def terms(text: str) -> list[str]:
    """Content terms, stemmed, order preserved."""
    return [stem(t) for t in content_terms(text or "")]


def term_set(text: str) -> set[str]:
    return set(terms(text))


def overlap(a: str, b: str) -> float:
    """Jaccard over stemmed content terms."""
    sa, sb = term_set(a), term_set(b)
    if not sa or not sb:
        return 0.0
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


def covers(aspect: str, text: str, *, ratio: float = 0.6) -> bool:
    """Does `text` address `aspect`?

    An aspect is a short noun phrase ("record retail bond issuance", "analyst consensus
    figure"). Requiring every term is too strict — journalists reorder and inflect — and
    requiring one term is far too loose, because a single common term like "profit" would
    mark every aspect covered. A majority of the aspect's own terms is the compromise, and
    the threshold is a parameter so the eval can report sensitivity to it rather than
    hiding it.
    """
    wanted = term_set(aspect)
    if not wanted:
        return False
    have = term_set(text)
    if not have:
        return False
    hits = len(wanted & have)
    needed = max(1, round(len(wanted) * ratio))
    return hits >= needed


def covered_by_any(aspect: str, texts: list[str], *, ratio: float = 0.6) -> bool:
    return any(covers(aspect, t, ratio=ratio) for t in texts)
