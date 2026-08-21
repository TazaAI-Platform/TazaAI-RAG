from __future__ import annotations

import re

from taza_rag.models import SearchIntent

# Common entity repairs seen in Factiva-style queries
ALIAS_FIXES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bdeutche\b", re.I), "Deutsche"),
    (re.compile(r"\bgs\b"), "Goldman Sachs"),
    (re.compile(r"\bjpm\b", re.I), "JPMorgan"),
]


def detect_intent(query: str) -> SearchIntent:
    q = query.lower().strip()
    if any(x in q for x in ("risk", "compliance", "covenant", "regulatory", "sanction")):
        return SearchIntent.RISK_COMPLIANCE
    if any(x in q for x in ("climate", "country", "region", "southeast asia", "brazil", "nigeria")):
        return SearchIntent.GEOGRAPHIC_ASSESSMENT
    profiling_cues = ("ceo", "cfo", "chair", "letter to", "said", "career", "board")
    # Person-like: two capitalized tokens without heavy topical words
    if any(c in q for c in profiling_cues) or _looks_like_person_query(query):
        return SearchIntent.EXECUTIVE_PROFILING
    topical = ("trends", "market", "future of", "how will", "what are", "ethics", "compliance")
    if any(t in q for t in topical) or len(q.split()) >= 5:
        return SearchIntent.TOPICAL_EXPLORATION
    return SearchIntent.ENTITY_INVESTIGATION


def _looks_like_person_query(query: str) -> bool:
    tokens = query.strip().split()
    if 2 <= len(tokens) <= 4 and all(t[:1].isupper() for t in tokens if t.isalpha()):
        return True
    return False


def normalize_query(query: str) -> str:
    out = query.strip()
    for pattern, repl in ALIAS_FIXES:
        out = pattern.sub(repl, out)
    return out


def expand_queries(query: str, intent: SearchIntent | None = None) -> list[str]:
    """Heuristic multi-query expansion — no LLM required.

    Returns de-duplicated variants ordered by priority (original first).
    """
    base = normalize_query(query)
    intent = intent or detect_intent(base)
    variants: list[str] = [base]

    if intent == SearchIntent.ENTITY_INVESTIGATION:
        variants.append(f"{base} latest news")
        variants.append(f"{base} strategy earnings")
    elif intent == SearchIntent.TOPICAL_EXPLORATION:
        variants.append(f"{base} market outlook")
        variants.append(f"{base} risks opportunities")
    elif intent == SearchIntent.EXECUTIVE_PROFILING:
        variants.append(f"{base} interview remarks")
        variants.append(f"{base} strategy comments")
    elif intent == SearchIntent.GEOGRAPHIC_ASSESSMENT:
        variants.append(f"{base} business climate investment")
        variants.append(f"{base} policy risk")
    elif intent == SearchIntent.RISK_COMPLIANCE:
        variants.append(f"{base} regulatory risk")
        variants.append(f"{base} compliance enforcement")
    else:
        variants.append(f"{base} analysis")

    # De-dupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
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
