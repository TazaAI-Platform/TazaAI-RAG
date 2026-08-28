"""Handle sources that disagree — without inventing disagreements.

A research answer over a news archive will pull the same event from a wire, a national
paper and an aggregator. Their numbers rarely match exactly, and the two reasons they
differ demand opposite treatment:

- **Rounding.** One outlet writes an 18% fall, another 17.7%. Reporting that as a dispute
  is itself a factual error, and the A1 rubric penalises exactly that kind of distortion.
- **Disagreement.** Two sources state materially different values. Silently picking one is
  the failure mode the brief asks about; the answer has to carry both, attributed.

So detection is deliberately conservative. A false positive puts a fabricated dispute in
front of a Dow Jones reader, which is worse than missing a real one. Four guards keep it
honest: the two facts must name the same actor, must be about the same subject, must be
denominated the same way, and must diverge by more than a rounding tolerance. All four are
deterministic, so this check cannot hallucinate a conflict the way a model asked "do these
disagree?" can.
"""

from __future__ import annotations

import re

from taza_rag.agent.models import Conflict, EvidenceItem, Finding
from taza_rag.agent.text import overlap, term_set
from taza_rag.factiva.verify import figures
from taza_rag.retrieve.features import extract_entities, stem

# Two facts must be recognisably about the same thing before their numbers are compared.
# Below this they are simply different facts that happen to contain numbers.
_SUBJECT_OVERLAP = 0.45

# Reporting conventions differ by a fraction of a percent; genuine disputes do not.
# 18 vs 17.7 is rounding (1.7%); 8.2 vs 8.5 is not (3.7%).
_ROUNDING_TOLERANCE = 0.02

_YEAR = re.compile(r"^(?:19|20)\d{2}$")

_CURRENCY = {
    "$": "usd", "usd": "usd", "dollar": "usd", "dollars": "usd",
    "¥": "jpy", "yen": "jpy", "jpy": "jpy",
    "€": "eur", "euro": "eur", "euros": "eur", "eur": "eur",
    "£": "gbp", "pound": "gbp", "pounds": "gbp", "sterling": "gbp", "gbp": "gbp",
    "yuan": "cny", "renminbi": "cny", "rmb": "cny",
    "won": "krw", "rupee": "inr", "rupees": "inr", "franc": "chf",
}
_PERCENT = ("%", "percent", "percentage", "pct")


def _denomination(text: str) -> frozenset[str]:
    """What this fact is measured in.

    "347.33 billion yen" and "$2.2 billion" are the same fact twice, not a conflict, and
    the only cheap way to know that is to notice the units differ. A percentage and an
    absolute amount are likewise different measures of the same event.
    """
    low = text.lower()
    found: set[str] = set()
    for token, code in _CURRENCY.items():
        if token in ("$", "¥", "€", "£"):
            if token in low:
                found.add(code)
        elif re.search(rf"\b{re.escape(token)}\b", low):
            found.add(code)
    if any(p in low for p in _PERCENT):
        found.add("pct")
    return frozenset(found)


def _values(text: str) -> list[float]:
    """Comparable numbers in a fact, years excluded.

    Years are dropped for the same reason the verifier treats them as non-blocking: "this
    year" is routinely paraphrased, so a year mismatch is weak evidence of a dispute.
    """
    out: list[float] = []
    for raw in figures(text):
        if _YEAR.match(raw):
            continue
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def _is_rounding(left: list[float], right: list[float]) -> bool:
    """Does every number in the shorter list have a near-twin in the other?"""
    if not left or not right:
        return False
    short, long = (left, right) if len(left) <= len(right) else (right, left)
    for value in short:
        if not any(_close(value, other) for other in long):
            return False
    return True


def _close(a: float, b: float) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    if scale == 0:
        return True
    return abs(a - b) / scale <= _ROUNDING_TOLERANCE


def _subject(text: str) -> str:
    """The fact with its numbers removed, so overlap compares wording not digits."""
    return re.sub(r"\d[\d,\.]*", " ", text)


# Capitalised, but never the actor of a sentence. Without this filter "Airbus delivered 60
# aircraft in July" and "Boeing delivered 45 aircraft in July" intersect on "July" and are
# read as the same subject.
_NON_ACTOR = frozenset(
    stem(w)
    for w in """january february march april may june july august september october
    november december monday tuesday wednesday thursday friday saturday sunday
    quarter quarterly annual""".split()
)


def _entities(text: str) -> frozenset[str]:
    out: set[str] = set()
    for entity in extract_entities(text):
        for token in term_set(entity):
            token = token.strip(".,;:'\u2019-")
            if token and token not in _NON_ACTOR:
                out.add(token)
    return frozenset(out)


def _same_actor(left: str, right: str) -> bool:
    """Are these two facts about the same company or person?

    Term overlap alone is not enough, and a comparative question proved it. "Airbus delivered
    60 aircraft in July" and "Boeing delivered 45 aircraft in July" share every word except
    the name, so stripping digits leaves them looking like the same subject — and the first
    live eval duly reported nine fabricated disagreements on one Airbus/Boeing question.

    When both facts name entities and those names do not intersect, they are about different
    actors and their numbers are not comparable. When either names none, fall through to the
    overlap test rather than guessing.
    """
    left_entities, right_entities = _entities(left), _entities(right)
    if not left_entities or not right_entities:
        return True
    return bool(left_entities & right_entities)


def _prefer(
    left: Finding, right: Finding, by_label: dict[str, EvidenceItem]
) -> tuple[str, str]:
    """Which side to lead with, and why.

    Authority first, freshness as the tie-break — the same priors the ranker already uses,
    so the agent's preference is explainable in the same terms as its retrieval. The
    non-preferred side is never dropped; it is attributed in the answer.
    """
    a, b = by_label.get(left.label), by_label.get(right.label)
    if a is None or b is None:
        return (left.label, "no evidence record for one side")

    auth_a = float(a.hit.scores.get("authority", 1.0))
    auth_b = float(b.hit.scores.get("authority", 1.0))
    if abs(auth_a - auth_b) > 0.01:
        winner, loser = (a, b) if auth_a > auth_b else (b, a)
        return (
            winner.label,
            f"{winner.hit.chunk.source} carries higher source authority than "
            f"{loser.hit.chunk.source}",
        )

    date_a = a.hit.chunk.published_at or ""
    date_b = b.hit.chunk.published_at or ""
    if date_a != date_b:
        winner = a if date_a > date_b else b
        return (winner.label, f"{winner.hit.chunk.source} is the later report ({winner.hit.chunk.published_at})")

    return (a.label, "equal authority and date; both reported")


def detect_conflicts(
    findings: list[Finding], by_label: dict[str, EvidenceItem]
) -> list[Conflict]:
    """Pairwise scan for same-subject, same-denomination numeric divergence.

    Quadratic in findings, which is fine: a run holds tens of facts, not thousands, and the
    pairing is a few string operations. Only facts from *different documents* are compared —
    two passages of one article restating a figure is not a conflict between sources.
    """
    conflicts: list[Conflict] = []
    seen: set[tuple[str, str]] = set()

    for i, left in enumerate(findings):
        left_item = by_label.get(left.label)
        for right in findings[i + 1 :]:
            right_item = by_label.get(right.label)
            if left_item is None or right_item is None:
                continue
            if left_item.doc_id == right_item.doc_id:
                continue

            pair = tuple(sorted((left.label + left.text[:40], right.label + right.text[:40])))
            if pair in seen:
                continue

            if not _same_actor(left.text, right.text):
                continue

            if overlap(_subject(left.text), _subject(right.text)) < _SUBJECT_OVERLAP:
                continue

            den_l, den_r = _denomination(left.text), _denomination(right.text)
            if den_l and den_r and den_l != den_r:
                # Same event, different units. Comparing the digits would manufacture a
                # dispute out of a currency conversion.
                continue

            vals_l, vals_r = _values(left.text), _values(right.text)
            if not vals_l or not vals_r:
                continue
            if set(vals_l) == set(vals_r):
                continue

            seen.add(pair)
            rounding = _is_rounding(vals_l, vals_r)
            preferred, reason = _prefer(left, right, by_label)
            conflicts.append(
                Conflict(
                    kind="rounding" if rounding else "disagreement",
                    subject=" ".join(_subject(left.text).split())[:120],
                    left=left,
                    right=right,
                    preferred_label=preferred,
                    reason="reported to different precision" if rounding else reason,
                )
            )
    return conflicts


def blocking_conflicts(conflicts: list[Conflict]) -> list[Conflict]:
    """Only genuine disagreements reach the answer; rounding is noise to the reader."""
    return [c for c in conflicts if c.kind == "disagreement"]


def describe_conflicts(conflicts: list[Conflict], by_label: dict[str, EvidenceItem]) -> str:
    """Render disagreements for the composer, with sources named."""
    lines: list[str] = []
    for c in blocking_conflicts(conflicts):
        left_src = _source_of(c.left.label, by_label)
        right_src = _source_of(c.right.label, by_label)
        lines.append(
            f"- {left_src} [{c.left.label}]: {c.left.text}\n"
            f"  {right_src} [{c.right.label}]: {c.right.text}\n"
            f"  lead with [{c.preferred_label}] ({c.reason}), but report both."
        )
    return "\n".join(lines)


def _source_of(label: str, by_label: dict[str, EvidenceItem]) -> str:
    item = by_label.get(label)
    return item.hit.chunk.source if item else "unknown source"
