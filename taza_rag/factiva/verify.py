"""Verify a generated answer against the evidence it was given.

The generator is asked to cite everything and invent nothing, and then trusted. Under an
independent judge that trust does not hold: citation integrity fails on roughly half of
answers, through uncited claims, figures that appear nowhere in the sources, and
attributions the sources do not make.

This module checks the answer instead of trusting it. The figure and citation checks are
deterministic, so they cost nothing and cannot themselves hallucinate; only paraphrase-level
support needs a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from taza_rag.llm import LLMError, chat_json

_LABEL = re.compile(r"\[(c\d+(?:\s*,\s*c\d+)*)\]", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"“'(\[]?[A-Z0-9])")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# Bare years are routinely paraphrased ("this year"), so they are reported but never
# used to trigger a rewrite; a wrong monetary figure is a different matter.
_YEAR = re.compile(r"^(?:19|20)\d{2}$")

VERIFY_SYSTEM = """You check whether each claim is supported by its cited excerpt.

For each numbered claim, decide:
- supported: true only if the cited excerpt states the claim or directly implies it.
- Mark false if the claim adds attribution, causation, certainty, or magnitude the excerpt
  does not contain. Upgrading "explored a deal" to "committed to a deal", or attributing a
  statement to a person the excerpt does not quote, is NOT supported.
- Judge only against the cited excerpt text. Do not use outside knowledge.

Return JSON: {"verdicts": [{"index": int, "supported": bool, "reason": string}]}
"""

REPAIR_SYSTEM = """You are correcting a Dow Jones / Factiva research answer that failed
verification. You are given the sources, the answer, and the specific problems found.

Rules:
- Remove or correct every flagged claim. Do not defend it.
- Drop a figure that is absent from the sources; never substitute a guess.
- Where attribution was overstated, restate only what the source says.
- Every remaining significant claim must carry a citation marker like [c1] that matches a
  real source label.
- Keep the surviving analysis fluent and journalistic; do not leave stubs or fragments.
- If almost nothing survives, set abstain=true and say what evidence is missing.
Return JSON with keys: answer (string), abstain (boolean), used_citations (list like ["c1"]).
"""


@dataclass
class Claim:
    index: int
    text: str
    labels: list[str]


@dataclass
class Problem:
    claim_index: int
    kind: str
    detail: str


@dataclass
class VerificationReport:
    claims: int = 0
    problems: list[Problem] = field(default_factory=list)
    checked_support: bool = False

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def blocking(self) -> list[Problem]:
        """Problems concrete enough to justify rewriting the answer."""
        return [p for p in self.problems if p.kind != "unverified_year"]

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for p in self.problems:
            counts[p.kind] = counts.get(p.kind, 0) + 1
        return {
            "claims": self.claims,
            "problems": counts,
            "checked_support": self.checked_support,
            "detail": [
                {"claim": p.claim_index, "kind": p.kind, "detail": p.detail}
                for p in self.problems
            ],
        }


def _labels_in(text: str) -> list[str]:
    out: list[str] = []
    for group in _LABEL.findall(text):
        for part in group.split(","):
            label = part.strip().lower()
            if label and label not in out:
                out.append(label)
    return out


def strip_labels(text: str) -> str:
    return _LABEL.sub(" ", text)


def split_claims(answer: str) -> list[Claim]:
    """Split into sentence-level claims, keeping each sentence's citation labels."""
    claims: list[Claim] = []
    for block in answer.split("\n"):
        block = block.strip()
        if not block:
            continue
        # Bullets and headings carry claims too, but a bare heading has no verb to check.
        block = re.sub(r"^[\-\*\u2022]\s*", "", block)
        for piece in _SENTENCE_SPLIT.split(block):
            piece = piece.strip()
            if len(piece) < 12:
                continue
            claims.append(Claim(index=len(claims) + 1, text=piece, labels=_labels_in(piece)))
    return claims


def _has_factual_content(text: str) -> bool:
    """Filter framing sentences that assert nothing about the world."""
    bare = strip_labels(text).strip().lower()
    if len(bare) < 25:
        return False
    hedges = ("the sources do not", "no information", "insufficient evidence")
    return not any(h in bare for h in hedges)


def _normalise_digits(text: str) -> str:
    return text.replace(",", "").replace("\u2009", "").replace("\u00a0", " ")


def figures(text: str) -> list[str]:
    return [_normalise_digits(m) for m in _NUMBER.findall(_normalise_digits(strip_labels(text)))]


def check_citations(claims: list[Claim], valid_labels: set[str]) -> list[Problem]:
    problems: list[Problem] = []
    for c in claims:
        if not _has_factual_content(c.text):
            continue
        if not c.labels:
            problems.append(
                Problem(c.index, "uncited", f"no citation marker: {c.text[:90]!r}")
            )
            continue
        bad = [x for x in c.labels if x not in valid_labels]
        if bad:
            problems.append(
                Problem(c.index, "invalid_label", f"cites {bad} which do not exist")
            )
    return problems


def check_figures(claims: list[Claim], evidence_by_label: dict[str, str]) -> list[Problem]:
    """Every figure must appear in the text the claim cites.

    Deterministic, so it is the one grounding check that cannot itself hallucinate.
    """
    problems: list[Problem] = []
    all_evidence = _normalise_digits(" ".join(evidence_by_label.values()))
    for c in claims:
        if not c.labels:
            # check_citations already reports this; re-reporting every figure in the
            # sentence buries the actual cause under duplicates.
            continue
        cited = _normalise_digits(
            " ".join(evidence_by_label.get(label, "") for label in c.labels)
        )
        for fig in figures(c.text):
            pattern = re.compile(rf"(?<!\d){re.escape(fig)}(?!\d)")
            if pattern.search(cited):
                continue
            if _YEAR.match(fig):
                problems.append(Problem(c.index, "unverified_year", f"year {fig} not in cited"))
            elif pattern.search(all_evidence):
                problems.append(
                    Problem(c.index, "miscited_figure", f"{fig} is in the sources but not {c.labels}")
                )
            else:
                problems.append(
                    Problem(c.index, "unsupported_figure", f"{fig} appears in no source")
                )
    return problems


def check_support(
    claims: list[Claim], evidence_by_label: dict[str, str], model: str | None = None
) -> list[Problem]:
    """Paraphrase-level entailment, batched into one call."""
    checkable = [c for c in claims if c.labels and _has_factual_content(c.text)]
    if not checkable:
        return []

    blocks = []
    for c in checkable:
        cited = "\n".join(
            f"[{label}] {evidence_by_label.get(label, '(missing)')[:1200]}" for label in c.labels
        )
        blocks.append(f"Claim {c.index}: {strip_labels(c.text).strip()}\nCited excerpt(s):\n{cited}")

    raw = chat_json(VERIFY_SYSTEM, "\n\n".join(blocks), model=model, temperature=0.0)
    problems: list[Problem] = []
    for v in raw.get("verdicts") or []:
        try:
            idx = int(v.get("index"))
        except (TypeError, ValueError):
            continue
        if not v.get("supported", True):
            problems.append(
                Problem(idx, "unsupported_claim", str(v.get("reason") or "not entailed")[:160])
            )
    return problems


def verify_answer(
    answer: str,
    evidence_by_label: dict[str, str],
    *,
    check_entailment: bool = True,
    model: str | None = None,
) -> VerificationReport:
    claims = split_claims(answer)
    report = VerificationReport(claims=len(claims))
    valid = set(evidence_by_label)
    report.problems.extend(check_citations(claims, valid))
    report.problems.extend(check_figures(claims, evidence_by_label))
    if check_entailment:
        try:
            report.problems.extend(check_support(claims, evidence_by_label, model=model))
            report.checked_support = True
        except LLMError:
            # The deterministic findings still stand; record that support went unchecked.
            report.checked_support = False
    return report


def describe_problems(report: VerificationReport, claims: list[Claim]) -> str:
    by_index = {c.index: c for c in claims}
    lines = []
    for p in report.blocking:
        text = strip_labels(by_index[p.claim_index].text).strip() if p.claim_index in by_index else ""
        lines.append(f"- [{p.kind}] {p.detail}\n  in: {text[:200]}")
    return "\n".join(lines)
