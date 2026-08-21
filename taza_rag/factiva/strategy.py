from __future__ import annotations

import re

from taza_rag.models import SearchIntent
from taza_rag.retrieve.features import TOPIC_SYNONYMS, build_query_plan

# Common entity repairs seen in Factiva-style queries
ALIAS_FIXES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bdeutche\b", re.I), "Deutsche"),
    (re.compile(r"\bdeutsch\b(?!e)", re.I), "Deutsche"),
    (re.compile(r"\bjp\s?morgan\b", re.I), "JPMorgan"),
    (re.compile(r"\bjpm\b"), "JPMorgan"),
    (re.compile(r"\bgs\b"), "Goldman Sachs"),
    (re.compile(r"\bsoftbank\b", re.I), "SoftBank"),
    (re.compile(r"\bbytedance\b", re.I), "ByteDance"),
    (re.compile(r"\bblackrock\b", re.I), "BlackRock"),
]

RISK_CUES = ("risk", "compliance", "covenant", "regulatory", "sanction", "fraud", "probe")
GEO_CUES = (
    "climate", "country", "region", "southeast asia", "brazil", "nigeria", "india",
    "germany", "china", "europe", "middle east", "africa", "latin america",
)
PROFILING_CUES = ("ceo", "cfo", "chair", "letter to", "said", "says", "career", "board", "remarks")
TOPICAL_CUES = (
    "trends", "market", "future of", "how will", "what are", "ethics", "outlook",
    "landscape", "adoption", "impact of",
)


def detect_intent(query: str) -> SearchIntent:
    q = query.lower().strip()
    if any(x in q for x in RISK_CUES):
        return SearchIntent.RISK_COMPLIANCE
    if any(x in q for x in GEO_CUES):
        return SearchIntent.GEOGRAPHIC_ASSESSMENT
    if any(c in q for c in PROFILING_CUES) or _looks_like_person_query(query):
        return SearchIntent.EXECUTIVE_PROFILING
    if any(t in q for t in TOPICAL_CUES) or len(q.split()) >= 6:
        return SearchIntent.TOPICAL_EXPLORATION
    return SearchIntent.ENTITY_INVESTIGATION


def _looks_like_person_query(query: str) -> bool:
    tokens = [t for t in query.strip().split() if t.isalpha()]
    if 2 <= len(tokens) <= 3 and all(t[:1].isupper() for t in tokens):
        # Company-ish suffixes mean it is an organization, not a person
        lowered = " ".join(tokens).lower()
        if any(s in lowered for s in ("bank", "group", "inc", "corp", "holdings", "authority")):
            return False
        return True
    return False


def normalize_query(query: str) -> str:
    out = query.strip()
    for pattern, repl in ALIAS_FIXES:
        out = pattern.sub(repl, out)
    return out


def expand_queries(
    query: str,
    intent: SearchIntent | None = None,
    max_variants: int = 3,
) -> list[str]:
    """Heuristic multi-query expansion — no LLM required.

    Variants target recall from different angles: the literal ask, the entity on its
    own (so the reranker has entity-anchored candidates), and topic paraphrases that
    match how journalists write about the same event.
    """
    base = normalize_query(query)
    intent = intent or detect_intent(base)
    plan = build_query_plan(base, intent)

    variants: list[str] = [base]
    primary_entity = plan.entities[0] if plan.entities else ""

    if primary_entity and plan.topics:
        # Paraphrase groups matter more than a bare-entity call: the entity is already
        # covered by the literal query, whereas topic wording is what the API misses.
        for paraphrase in _topic_paraphrases(plan.topics):
            variants.append(f"{primary_entity} {paraphrase}")
        # Entity alone as a last-resort recall anchor.
        variants.append(primary_entity)

    if intent == SearchIntent.ENTITY_INVESTIGATION:
        variants.append(f"{base} strategy results")
    elif intent == SearchIntent.TOPICAL_EXPLORATION:
        variants.append(f"{base} outlook")
        variants.append(f"{base} risks")
    elif intent == SearchIntent.EXECUTIVE_PROFILING:
        variants.append(f"{base} comments strategy")
    elif intent == SearchIntent.GEOGRAPHIC_ASSESSMENT:
        variants.append(f"{base} economy investment")
    elif intent == SearchIntent.RISK_COMPLIANCE:
        variants.append(f"{base} regulator enforcement")
    else:
        variants.append(f"{base} analysis")

    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        v = v.strip()
        key = v.lower()
        if v and key not in seen:
            seen.add(key)
            out.append(v)
    return out[:max_variants]


def _topic_paraphrases(topics: list[str], groups: int = 2, per_group: int = 2) -> list[str]:
    """Synonym phrasings for the topic terms, e.g. restructuring -> "overhaul job cuts".

    Returns up to `groups` distinct phrasings so different vocabularies get their own
    Factiva call instead of competing inside one query string.
    """
    pools = [list(TOPIC_SYNONYMS.get(t, ())) for t in topics]
    out: list[str] = []
    for g in range(groups):
        terms: list[str] = []
        for pool in pools:
            window = pool[g * per_group : (g + 1) * per_group]
            terms.extend(window)
        phrase = " ".join(dict.fromkeys(terms))
        if phrase:
            out.append(phrase)
    return out


def default_days_range(intent: SearchIntent) -> str:
    if intent in {
        SearchIntent.ENTITY_INVESTIGATION,
        SearchIntent.EXECUTIVE_PROFILING,
        SearchIntent.EVENT_TRACKING,
    }:
        return "Last3Months"
    if intent == SearchIntent.KNOWN_ITEM:
        return "Last2Years"
    return "Last6Months"
