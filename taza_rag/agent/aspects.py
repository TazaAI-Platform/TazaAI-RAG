"""Decide whether an aspect has been satisfied.

Most aspects are lexical: "record retail bond issuance" is wording that appears in the
copy, so whole-term overlap settles it.

Some are not, and a first live run made the cost of ignoring that obvious. The planner asked
for an "official comment" and a "dissenting view" — categories a complete answer genuinely
needs, and which the A1 rubric explicitly rewards — but those exact words never appear in a
news story. Judged lexically they can never be satisfied, so coverage sat at 0.44 while the
agent spent every remaining round re-asking for something it already had.

So an aspect made only of generic vocabulary is treated as a **structural** requirement and
checked by a predicate over the facts instead: is there a figure, is something attributed to
someone, is a contrary view present, is the timing given. An aspect carrying any distinctive
term keeps the lexical path, because that is the stronger signal when it is available.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from taza_rag.agent.text import covers, term_set
from taza_rag.factiva.verify import figures
from taza_rag.retrieve.features import stem


def _vocab(words: set[str]) -> frozenset[str]:
    """Stem the vocabulary, because the aspect it is compared against is stemmed too.

    Written unstemmed and compared stemmed, every one of these sets silently matched
    nothing: "figures" reaches the comparison as "figur".
    """
    return frozenset(stem(w) for w in words)


_FIGURE = {
    "figure", "figures", "number", "numbers", "amount", "amounts", "size", "value",
    "valuation", "financial", "financials", "percentage", "price", "cost", "costs",
    "total", "totals", "sum", "metric", "metrics",
}
_COMMENT = {
    "comment", "comments", "statement", "statements", "quote", "quotes", "remark",
    "remarks", "said", "says", "commentary", "response", "reaction", "guidance",
}
_DISSENT = {
    "dissent", "dissenting", "contrary", "opposing", "opposition", "criticism",
    "critical", "concern", "concerns", "caution", "cautionary", "risk", "risks",
    "skeptic", "sceptic", "skepticism", "pushback", "objection", "warning", "downside",
}
_TIMING = {"timing", "date", "dates", "when", "schedule", "timeline", "deadline", "period"}

# Words that carry no retrievable meaning of their own. An aspect built only from these
# describes the shape of an answer, not anything a journalist would write.
_FILLER = {
    "official", "executive", "executives", "company", "companies", "key", "main",
    "overall", "general", "specific", "detail", "details", "context", "information",
    "view", "views", "point", "points", "item", "items", "data", "figure",
}

_ATTRIBUTION = re.compile(
    r"\b(said|says|told|stated|announced|disclosed|confirmed|denied|wrote|reported|"
    r"according to|declined to comment|spokesman|spokeswoman|spokesperson)\b",
    re.I,
)
_CONTRAST = re.compile(
    r"\b(however|but|although|though|despite|nevertheless|critic|critics|criticised|"
    r"criticized|sceptic|skeptic|concern|concerns|warned|warning|risk|risks|caution|"
    r"downside|pushback|disputed|questioned|declined|fell short|worse than)\b",
    re.I,
)
_MONTH = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|"
    r"december|q[1-4]|first quarter|second quarter|third quarter|fourth quarter)\b",
    re.I,
)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def _has_figure(text: str) -> bool:
    return bool(figures(text))


def _has_attribution(text: str) -> bool:
    return bool(_ATTRIBUTION.search(text))


def _has_contrast(text: str) -> bool:
    return bool(_CONTRAST.search(text))


def _has_timing(text: str) -> bool:
    return bool(_MONTH.search(text) or _YEAR.search(text))


_STRUCTURAL: tuple[tuple[frozenset[str], Callable[[str], bool], str], ...] = (
    (_vocab(_FIGURE), _has_figure, "figure"),
    (_vocab(_COMMENT), _has_attribution, "attribution"),
    (_vocab(_DISSENT), _has_contrast, "contrary view"),
    (_vocab(_TIMING), _has_timing, "timing"),
)

_FILLER_STEMS = _vocab(_FILLER)


def classify(aspect: str) -> str:
    """"lexical", or the name of the structural requirement this aspect really states."""
    wanted = term_set(aspect)
    if not wanted:
        return "lexical"
    distinctive = {t for t in wanted if t not in _FILLER_STEMS}
    for vocab, _predicate, _name in _STRUCTURAL:
        distinctive = distinctive - vocab
    if distinctive:
        # Something corpus-specific to match on; the lexical signal is the stronger one.
        return "lexical"
    for vocab, _predicate, name in _STRUCTURAL:
        if wanted & vocab:
            return name
    return "lexical"


def satisfied(aspect: str, texts: Iterable[str], *, ratio: float = 0.6) -> bool:
    """Is this aspect covered by any of these facts?"""
    texts = list(texts)
    if not texts:
        return False
    kind = classify(aspect)
    if kind == "lexical":
        return any(covers(aspect, t, ratio=ratio) for t in texts)
    for vocab, predicate, name in _STRUCTURAL:
        if name == kind:
            return any(predicate(t) for t in texts)
    return False
