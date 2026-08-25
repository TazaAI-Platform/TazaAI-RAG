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
# Generators sometimes drop the prefix and emit "[9]" beside "[c2]". The intent is
# unambiguous, but a strict pattern misses it twice over: the sentence looks uncited, and
# the bare number survives label-stripping to be scored as a figure that needs grounding.
_LOOSE_LABEL = re.compile(r"\[\s*c?\s*\d+(?:\s*,\s*c?\s*\d+)*\s*\]", re.IGNORECASE)
_LOOSE_NUMBER = re.compile(r"c?\s*(\d+)", re.IGNORECASE)
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

SENTENCE_REPAIR_SYSTEM = """You are correcting ONE sentence of a Dow Jones / Factiva research
answer that failed verification. You are given the sentence, the excerpts it cites, and the
problem found.

Return a corrected version of that sentence alone:
- State only what the cited excerpts support, keeping every citation marker like [c1].
- Drop a figure that is absent from the excerpts; never substitute a guess.
- Where attribution, certainty or magnitude was overstated, state the weaker supported form.
- Keep as much of the original meaning as the excerpts allow. Prefer weakening a claim to
  deleting it.
- If nothing in the sentence is supportable, return an empty string for `sentence`.
- Do not add new facts, do not comment on the correction, and do not return anything but the
  sentence.
Return JSON: {"sentence": string}
"""

REPAIR_SYSTEM = """You are correcting a Dow Jones / Factiva research answer that failed
verification. You are given the sources, the answer, and the specific problems found.

Rules:
- Remove or correct every flagged claim. Do not defend it.
- Drop a figure that is absent from the sources; never substitute a guess.
- Where attribution was overstated, restate only what the source says.
- Every remaining significant claim must carry a citation marker like [c1] that matches a
  real source label.
- Change ONLY what was flagged. Keep every unflagged claim, figure and attribution exactly
  as it was — this is a correction, not a rewrite, and quietly dropping sound material to
  play safe is itself a defect.
- Prefer correcting a claim to deleting it: if the sources support a weaker version, state
  the weaker version rather than removing the point.
- Keep the surviving analysis fluent and journalistic; do not leave stubs or fragments.
- Set abstain=true ONLY if nothing verifiable remains at all. If any supported claim
  survives, return it as the answer with abstain=false.
Return JSON with keys: answer (string), abstain (boolean), used_citations (list like ["c1"]).
"""


@dataclass
class Claim:
    index: int
    text: str
    labels: list[str]
    paragraph: int = 0
    # Journalistic style cites a claim group once, usually at its end, so a sentence with no
    # marker of its own is not necessarily unsourced.
    inherited: list[str] = field(default_factory=list)

    @property
    def effective_labels(self) -> list[str]:
        return self.labels or self.inherited


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
    """Every citation the sentence attempts, including prefix-less forms like "[9]"."""
    out: list[str] = []
    for match in _LOOSE_LABEL.finditer(text):
        for number in _LOOSE_NUMBER.findall(match.group(0)):
            label = f"c{int(number)}"
            if label not in out:
                out.append(label)
    return out


def strip_labels(text: str) -> str:
    """Remove citation markers so they cannot be mistaken for prose or figures."""
    return _LOOSE_LABEL.sub(" ", text)


def split_claims(answer: str) -> list[Claim]:
    """Split into sentence-level claims, keeping each sentence's citation labels."""
    claims: list[Claim] = []
    for para, block in enumerate(answer.split("\n")):
        block = block.strip()
        if not block:
            continue
        # Bullets and headings carry claims too, but a bare heading has no verb to check.
        block = re.sub(r"^[\-\*\u2022]\s*", "", block)
        for piece in _SENTENCE_SPLIT.split(block):
            piece = piece.strip()
            if len(piece) < 12:
                continue
            claims.append(
                Claim(
                    index=len(claims) + 1,
                    text=piece,
                    labels=_labels_in(piece),
                    paragraph=para,
                )
            )
    _assign_inherited(claims)
    return claims


def _assign_inherited(claims: list[Claim]) -> None:
    """Carry a claim group's citation to the sentences it covers.

    A marker at the end of a group is the normal way to source the sentences leading into
    it, so demanding one marker per sentence flags ordinary prose as unsourced. Inheritance
    only reaches within a paragraph, and figures are still checked against the inherited
    sources, so a sentence carrying a figure its neighbours do not support is still caught.
    """
    for i, c in enumerate(claims):
        if c.labels:
            continue
        for step in (1, -1):
            j = i + step
            while 0 <= j < len(claims) and claims[j].paragraph == c.paragraph:
                if claims[j].labels:
                    c.inherited = list(claims[j].labels)
                    break
                j += step
            if c.inherited:
                break


_HEDGES = (
    "the sources do not",
    "no information",
    "insufficient evidence",
    "not provide",
    "cannot answer",
)

# Reporting and movement verbs. Sentence length is a bad proxy for factual content:
# "Revenue hit $9.9bn." is short and dangerous, while "This is important context to
# consider." is longer and asserts nothing.
_CLAIM_VERBS = frozenset(
    """said says announced reported disclosed confirmed denied agreed plans planned
    expects expected forecast warned rose fell gained lost climbed dropped slid jumped
    surged tripled doubled halved cut raised lowered sold bought acquired divested
    launched exited hired fired appointed named resigned stepped filed sued settled
    posted earned generated grew shrank totalled totaled reached hit added removed
    approved rejected blocked banned fined invested committed explored valued
    say announce report disclose confirm deny agree plan expect warn rise fall gain
    lose climb drop slide jump surge triple double halve raise lower sell buy acquire
    divest launch exit hire fire appoint resign step file sue settle post earn generate
    grow shrink total reach add remove approve reject block ban fine invest commit
    explore value""".split()
)


# Forward-looking statements ("headcount will keep falling") are claims too, and among the
# riskiest, but the verb list only held finite past forms so they read as non-factual.
_MODALS = frozenset("will would could should expects expected plans planned forecast".split())


def _verb_like(token: str) -> bool:
    word = re.sub(r"\W", "", token.lower())
    if not word:
        return False
    if word in _CLAIM_VERBS or word in _MODALS:
        return True
    # Cheap inflection handling: falling -> fall, rising -> rise, cuts -> cut.
    for stem in (word[:-3], word[:-3] + "e", word[:-1], word[:-2]):
        if len(stem) >= 3 and stem in _CLAIM_VERBS:
            return True
    return False


def _has_factual_content(text: str) -> bool:
    """Does this sentence assert something about the world that needs a source?

    Biased towards saying yes: a missed hallucination costs the Accuracy gate, whereas an
    unnecessary citation request only costs a little brevity.
    """
    bare = strip_labels(text).strip()
    low = bare.lower()
    if len(bare) < 12 or any(h in low for h in _HEDGES):
        return False
    if re.search(r"\d", bare):
        return True
    tokens = bare.split()
    # A capitalised token past the opening word signals a named subject.
    if any(t[:1].isupper() for t in tokens[1:] if t[:1].isalpha()):
        return True
    return any(_verb_like(t) for t in tokens)


def _normalise_digits(text: str) -> str:
    return text.replace(",", "").replace("\u2009", "").replace("\u00a0", " ")


def figures(text: str) -> list[str]:
    return [_normalise_digits(m) for m in _NUMBER.findall(_normalise_digits(strip_labels(text)))]


def check_citations(claims: list[Claim], valid_labels: set[str]) -> list[Problem]:
    factual = [c for c in claims if _has_factual_content(c.text)]
    # An answer with no marker anywhere is a different failure from a few loose sentences,
    # and listing every sentence separately buries that. One answer in a 52-query run came
    # back entirely uncited and the repair pass, handed nine near-identical complaints, did
    # not fix any of them.
    if factual and not any(c.labels for c in claims):
        return [
            Problem(
                factual[0].index,
                "no_citations",
                f"the answer carries no citation marker at all across {len(factual)} factual "
                "sentences; every one needs the marker for the source that supports it",
            )
        ]

    problems: list[Problem] = []
    for c in claims:
        if not _has_factual_content(c.text):
            continue
        if not c.effective_labels:
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
        labels = c.effective_labels
        if not labels:
            # check_citations already reports this; re-reporting every figure in the
            # sentence buries the actual cause under duplicates.
            continue
        cited = _normalise_digits(
            " ".join(evidence_by_label.get(label, "") for label in labels)
        )
        for fig in figures(c.text):
            pattern = re.compile(rf"(?<!\d){re.escape(fig)}(?!\d)")
            if pattern.search(cited):
                continue
            if _YEAR.match(fig):
                problems.append(Problem(c.index, "unverified_year", f"year {fig} not in cited"))
            elif pattern.search(all_evidence):
                problems.append(
                    Problem(c.index, "miscited_figure", f"{fig} is in the sources but not {labels}")
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
    checkable = [c for c in claims if c.effective_labels and _has_factual_content(c.text)]
    if not checkable:
        return []

    blocks = []
    for c in checkable:
        cited = "\n".join(
            f"[{label}] {evidence_by_label.get(label, '(missing)')[:1200]}"
            for label in c.effective_labels
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
