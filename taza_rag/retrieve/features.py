"""Query/document features used by the reranker. No LLM, no network."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from taza_rag.models import RetrievedChunk, SearchIntent

STOPWORDS = {
    "a", "about", "after", "all", "and", "any", "are", "as", "at", "be", "been", "but", "by",
    "can", "did", "do", "does", "for", "from", "had", "has", "have", "how", "in", "into", "is",
    "it", "its", "latest", "me", "most", "my", "new", "news", "of", "on", "or", "our", "over",
    "recent", "should", "so", "some", "tell", "that", "the", "their", "there", "these", "they",
    "this", "to", "under", "up", "was", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "would", "you", "your",
}

# Corporate/legal suffixes that should not anchor an entity match on their own
ENTITY_SUFFIXES = {
    "inc", "inc.", "corp", "corp.", "co", "co.", "ltd", "ltd.", "plc", "ag", "sa", "s.a.",
    "nv", "n.v.", "llc", "lp", "group", "holdings", "holding", "company", "authority",
}

# Capitalized but generic — "letter to CEOs" must not make every leadership story
# look like it names the subject.
ROLE_WORDS = {
    "ceo", "ceos", "cfo", "cfos", "coo", "cto", "cio", "chair", "chairman", "chairwoman",
    "president", "board", "director", "directors", "executive", "executives", "management",
    "founder", "founders", "chief", "officer", "officers", "investor", "investors",
}

# Strong synonyms: specific enough that a body match is real topical evidence.
# Ordered most-distinct first — query expansion reads them in pairs, so adjacent
# entries should not be spelling variants of each other.
TOPIC_SYNONYMS: dict[str, tuple[str, ...]] = {
    "restructuring": (
        "overhaul", "job cuts", "cost cuts", "divest", "reorganization", "layoffs",
        "turnaround", "streamline", "restructure", "shake-up", "revamp",
        "reorganisation", "cost-cutting", "redundancies", "divestment",
    ),
    "earnings": ("results", "pretax profit", "guidance", "revenue", "quarterly"),
    "layoffs": ("job cuts", "headcount", "redundancies", "workforce reduction"),
    "merger": ("acquisition", "takeover", "m&a", "merge"),
    "regulation": ("regulator", "rulemaking", "regulatory", "supervisor"),
    "lawsuit": ("litigation", "settlement", "sued", "court filing"),
    "growth": ("expansion", "momentum", "expand"),
    "risk": ("exposure", "stress", "warning", "risks"),
    "trends": ("outlook", "shift", "landscape", "trend"),
    "strategy": ("strategic", "roadmap", "priorities"),
    "compliance": ("sanctions", "enforcement", "aml", "regulatory"),
    "inflation": ("prices", "cpi", "disinflation"),
    "deforestation": ("forest", "clearing", "logging", "amazon"),
    "recycling": ("recycled", "circular economy", "reuse", "scrap"),
}

# Weak synonyms: too generic for a body match (nearly every corporate story says
# "strategy"), but meaningful when a headline uses them.
TOPIC_WEAK_SYNONYMS: dict[str, tuple[str, ...]] = {
    "restructuring": ("strategy", "exit", "offload", "sell", "sale", "unit sale", "focus"),
    "earnings": ("beat", "miss", "outlook", "profit"),
    "merger": ("deal", "bid", "stake"),
    "growth": ("increase", "rise", "boom"),
    "risk": ("concern", "caution"),
    "trends": ("momentum", "growth"),
}


def topic_variants(topic: str, include_weak: bool = True) -> tuple[str, ...]:
    strong = (topic,) + TOPIC_SYNONYMS.get(topic, ())
    if include_weak:
        return strong + TOPIC_WEAK_SYNONYMS.get(topic, ())
    return strong

# Titles/sources that are aggregations rather than reported stories.
_DIGEST_TITLE = re.compile(
    r"(top\s+\w[\w\s&/]*headlines\s+at\s+\d{1,2}\s*(am|pm)\s*et"
    r"|^dow jones top\b"
    r"|market talk\b.*roundup"
    r"|news digest\b"
    r"|^press release summary"
    r"|briefing:\s|\bat a glance\b"
    r"|^top stories\b"
    r"|^factiva\b.*newsstand)",
    re.I,
)
_CONTINUATION = re.compile(r"\s-\d-\s|\s-\d-$")
# Newsletter round-ups read like articles but bundle unrelated items.
_NEWSLETTER_BODY = re.compile(
    r"(online version of .{0,60}newsletter"
    r"|sign up to receive it"
    r"|this week'?s? (?:top )?(?:stories|newsletter)"
    r"|the latest .{0,40}news from)",
    re.I,
)
_PROFILE_TITLE = re.compile(
    r"(-\s*(history|company profile|swot|key facts|financials)\s*$"
    r"|\bcompany profile\b"
    r"|\bswot analysis\b)",
    re.I,
)
_PROFILE_SOURCE = re.compile(
    r"(marketline|globaldata|company profiles|datamonitor|business monitor"
    r"|euromonitor|capital iq|refinitiv profile)",
    re.I,
)
_AGGREGATOR_SOURCE = re.compile(
    r"(zacks|simply wall st|investing\.com|benzinga|insider monkey|seeking alpha|tipranks)",
    re.I,
)

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9&'’\.\-]*")
_CAP_SPAN = re.compile(r"\b([A-Z][\w&’'\.\-]*(?:\s+(?:of|de|del|van|von)?\s*[A-Z][\w&’'\.\-]*)*)")

# Intents where digests/profiles are actively harmful (user wants reported news)
NEWS_INTENTS = {
    SearchIntent.ENTITY_INVESTIGATION,
    SearchIntent.EXECUTIVE_PROFILING,
    SearchIntent.EVENT_TRACKING,
    SearchIntent.COMPETITIVE_INTEL,
    SearchIntent.RISK_COMPLIANCE,
}

LEAD_CHARS = 400


def lead_text(hit: RetrievedChunk) -> str:
    """The article's opening, which only the first passage of a document has.

    Without this, splitting an article into passages would hand every paragraph its
    own "lead" and erase the distinction between a headline-level match and evidence
    buried deep in the story.
    """
    if hit.chunk.chunk_index != 0:
        return ""
    return (hit.chunk.text or "")[:LEAD_CHARS]


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


def content_terms(text: str) -> list[str]:
    return [w for w in words(text) if w not in STOPWORDS and len(w) > 1]


_SUFFIXES = ("'s", "’s", "ings", "ing", "ements", "ement", "ments", "ment", "ers", "es", "ed", "s")


def stem(word: str) -> str:
    """Tiny suffix stripper so "sells"/"sell" and "divestment"/"divest" unify.

    Deliberately conservative: real stemmers pull in a dependency and their extra
    aggression costs precision on entity names.
    """
    w = word.lower()
    for suffix in _SUFFIXES:
        # "cuts" -> "cut" needs a shorter floor than "divestment" -> "divest"
        floor = 3 if suffix == "s" else 4
        if w.endswith(suffix) and len(w) - len(suffix) >= floor:
            w = w[: -len(suffix)]
            break
    if len(w) >= 5 and w.endswith("e"):
        w = w[:-1]
    return w


def stems(text: str) -> list[str]:
    return [stem(w) for w in words(text)]


def contains_term(term: str, stem_list: list[str], stem_set: set[str]) -> bool:
    """Match a single word or a multi-word phrase against stemmed document tokens."""
    parts = words(term)
    if not parts:
        return False
    if len(parts) == 1:
        return stem(parts[0]) in stem_set
    target = [stem(p) for p in parts]
    n = len(target)
    return any(stem_list[i : i + n] == target for i in range(len(stem_list) - n + 1))


@dataclass
class QueryPlan:
    """Structured view of what the query is actually asking for."""

    raw: str
    intent: SearchIntent
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    expanded_topics: list[str] = field(default_factory=list)

    @property
    def entity_tokens(self) -> list[list[str]]:
        """Each entity as its list of significant tokens."""
        out: list[list[str]] = []
        for ent in self.entities:
            toks = [t for t in content_terms(ent) if t not in ENTITY_SUFFIXES]
            if toks:
                out.append(toks)
        return out

    @property
    def search_terms(self) -> list[str]:
        terms = list(self.expanded_topics)
        for ent in self.entities:
            terms.extend(content_terms(ent))
        return terms or content_terms(self.raw)


def _strip_possessive(span: str) -> str:
    return re.sub(r"[’']s\b", "", span).strip()


# Words that belong to the organisation name in front of them, so "Deutsche Bank" and
# "European Central Bank" survive the split below as single entities.
ORG_HEAD_WORDS = {
    "bank", "group", "holdings", "holding", "corp", "corporation", "company", "co",
    "capital", "partners", "ventures", "management", "advisors", "associates",
    "technologies", "technology", "systems", "solutions", "industries", "motors",
    "airlines", "airways", "energy", "pharma", "pharmaceuticals", "media", "networks",
    "exchange", "commission", "authority", "association", "institute", "university",
    "bancorp", "financial", "insurance", "securities", "asset", "investments",
}
_NAME_PARTICLES = {"jr", "sr", "ii", "iii", "iv"}


def _is_camel(token: str) -> bool:
    """BlackRock / JPMorgan / ByteDance — a self-contained brand, not a name fragment.

    Acronyms are excluded: "AI" in "Taza AI" belongs to the name before it.
    """
    return len(token) > 1 and not token.isupper() and any(c.isupper() for c in token[1:])


def split_entity_run(span: str) -> list[str]:
    """Break one run of capitalized words into the entities it actually names.

    A query like "Larry Fink BlackRock private markets" produces the single span
    "Larry Fink BlackRock", a phrase that appears in no document, which silently drove
    every entity signal to zero. Splitting recovers the person and the firm as separate
    entities so each can be matched on its own.
    """
    tokens = span.split()
    if len(tokens) < 3:
        return [span]

    groups: list[list[str]] = [[tokens[0]]]
    for token in tokens[1:]:
        bare = token.strip(".,").lower()
        current = groups[-1]
        starts_new = _is_camel(token) or (
            len(current) >= 2
            and bare not in ORG_HEAD_WORDS
            and bare not in ENTITY_SUFFIXES
            and bare not in _NAME_PARTICLES
        )
        if starts_new:
            groups.append([token])
        else:
            current.append(token)
    return [" ".join(g) for g in groups]


def extract_entities(query: str) -> list[str]:
    """Capitalized spans and quoted phrases are the entity candidates."""
    found: list[str] = []
    for quoted in re.findall(r'"([^"]{2,})"', query):
        found.append(quoted.strip())
    stripped = re.sub(r'"[^"]*"', " ", query)
    for run in _CAP_SPAN.findall(stripped):
        for span in split_entity_run(_strip_possessive(run)):
            if not span:
                continue
            toks = [t for t in content_terms(span) if t not in ENTITY_SUFFIXES]
            if not toks:
                continue
            # Generic role words ("CEOs", "Board") name a category, not a subject
            if all(t in ROLE_WORDS for t in toks):
                continue
            if len(span.split()) == 1 and span.lower() in STOPWORDS:
                continue
            found.append(span)
    # De-dupe, keep longest first so "Deutsche Bank" wins over "Deutsche"
    uniq: list[str] = []
    for f in sorted(set(found), key=lambda s: -len(s)):
        if not any(f.lower() in u.lower() for u in uniq):
            uniq.append(f)
    return uniq


def build_query_plan(query: str, intent: SearchIntent) -> QueryPlan:
    entities = extract_entities(query)
    # Compare stems so the possessive in "Larry Fink's letter" is not left behind
    # in the topic list as if it were something to search for.
    entity_stems = {stem(w) for e in entities for w in content_terms(e)}
    topics = [t for t in content_terms(query) if stem(t) not in entity_stems]

    expanded: list[str] = list(topics)
    for t in topics:
        expanded.extend(TOPIC_SYNONYMS.get(t, ()))
    return QueryPlan(
        raw=query,
        intent=intent,
        entities=entities,
        topics=topics,
        expanded_topics=list(dict.fromkeys(expanded)),
    )


# --- document-side features ---


def doc_kind(hit: RetrievedChunk) -> str:
    """One of: digest, profile, aggregator, article."""
    title = hit.chunk.title or ""
    source = f"{hit.chunk.source or ''} {(hit.chunk.metadata or {}).get('source_code') or ''}"
    if _DIGEST_TITLE.search(title) or _CONTINUATION.search(title):
        return "digest"
    # Many stacked headlines separated by pipes is a digest even without the marker
    if title.count("|") >= 2:
        return "digest"
    lead = f"{title}\n{(hit.chunk.text or '')[:LEAD_CHARS]}"
    if _NEWSLETTER_BODY.search(lead):
        return "digest"
    if _PROFILE_SOURCE.search(source) or _PROFILE_TITLE.search(title):
        return "profile"
    if _AGGREGATOR_SOURCE.search(source):
        return "aggregator"
    return "article"


def _phrase_hit(tokens: list[str], haystack: str) -> bool:
    """All entity tokens present, in order, within a short window (stem-insensitive)."""
    if not tokens:
        return False
    hay = stems(haystack)
    target = [stem(t) for t in tokens]
    if len(target) == 1:
        return target[0] in hay
    positions = [i for i, w in enumerate(hay) if w == target[0]]
    for start in positions:
        window = hay[start : start + len(target) + 3]
        if all(t in window for t in target):
            return True
    return False


def entity_signal(plan: QueryPlan, hit: RetrievedChunk) -> dict[str, float]:
    """Where the query's primary entity appears in the document.

    Only the primary (longest) entity gates: in "Larry Fink's annual letter to CEOs"
    a secondary span must never be enough to make an unrelated leadership story look
    like it names the subject. Secondary entities contribute a bonus instead.
    """
    groups = plan.entity_tokens
    if not groups:
        return {
            "entity_title": 0.0,
            "entity_lead": 0.0,
            "entity_body": 0.0,
            "entity_any": 1.0,
            "entity_extra": 0.0,
            "entity_all": 1.0,
        }

    title = hit.chunk.title or ""
    lead = lead_text(hit)
    body = hit.chunk.text or ""

    primary, rest = groups[0], groups[1:]
    in_title = 1.0 if _phrase_hit(primary, title) else 0.0
    in_lead = 1.0 if _phrase_hit(primary, lead) else 0.0
    in_body = 1.0 if _phrase_hit(primary, body) else 0.0
    # Secondary entities count wherever they appear, headline included: in
    # "Larry Fink BlackRock private markets" the firm is usually in the headline.
    searchable = f"{title}\n{body}"
    extra = (
        sum(1 for g in rest if _phrase_hit(g, searchable)) / len(rest) if rest else 0.0
    )
    return {
        "entity_title": in_title,
        "entity_lead": in_lead,
        "entity_body": in_body,
        "entity_any": max(in_title, in_lead, in_body),
        "entity_extra": extra,
        # Every entity the query named is present. "Andy Jassy AWS growth strategy" is
        # not answered by a Jassy story about Indian retail.
        "entity_all": 1.0 if max(in_title, in_lead, in_body) > 0.0 and extra >= 1.0 else 0.0,
    }


def topic_signal(plan: QueryPlan, hit: RetrievedChunk) -> dict[str, float]:
    """Does the document cover the non-entity part of the ask?

    Headlines may use weak paraphrases ("DB to Offload India Retail Unit"), but in the
    body only specific wording counts — otherwise boilerplate like "strategy" makes
    every corporate story look on-topic.
    """
    if not plan.topics:
        return {
            "topic_title": 1.0,
            "topic_lead": 1.0,
            "topic_body": 1.0,
            "topic_any": 1.0,
            "topic_strong": 1.0,
        }

    title_stems = stems(hit.chunk.title)
    lead_stems = stems(lead_text(hit))
    body_stems = stems(f"{hit.chunk.title}\n{hit.chunk.text}")
    sets = {
        "title": set(title_stems),
        "lead": set(lead_stems),
        "body": set(body_stems),
    }
    lists = {"title": title_stems, "lead": lead_stems, "body": body_stems}

    def covered(variants_for: dict[str, tuple[str, ...]], where: str) -> float:
        hits = 0
        for topic in plan.topics:
            if any(
                contains_term(v, lists[where], sets[where]) for v in variants_for[topic]
            ):
                hits += 1
        return hits / len(plan.topics)

    all_variants = {t: topic_variants(t, include_weak=True) for t in plan.topics}
    strong_variants = {t: topic_variants(t, include_weak=False) for t in plan.topics}

    t_title = covered(all_variants, "title")
    t_lead = covered(strong_variants, "lead")
    t_body = covered(all_variants, "body")
    t_body_strong = covered(strong_variants, "body")
    return {
        "topic_title": t_title,
        "topic_lead": t_lead,
        "topic_body": t_body,
        "topic_any": max(t_title, t_body),
        # Specific wording anywhere — weaker than headline/lead evidence.
        "topic_strong": max(t_title, t_body_strong),
    }


def kind_penalty(kind: str, intent: SearchIntent) -> float:
    """Subtractive penalty applied to the composite score.

    Digests and vendor profiles frequently match on keywords while carrying little
    reportable substance, so they are pushed below original reporting.
    """
    news = intent in NEWS_INTENTS
    if kind == "digest":
        return 1.10 if news else 0.70
    if kind == "profile":
        return 0.95 if news else 0.45
    if kind == "aggregator":
        return 0.20
    return 0.0


def title_shingles(text: str, n: int = 3) -> set[str]:
    toks = [t for t in content_terms(text)]
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def near_duplicate(
    a: RetrievedChunk,
    b: RetrievedChunk,
    threshold: float = 0.6,
    body_threshold: float = 0.3,
) -> bool:
    """Same story from two outlets — a matching headline *and* matching content.

    Requiring both means two passages of one article, which necessarily share a
    headline, are not mistaken for the same piece of evidence.
    """
    title_sim = jaccard(title_shingles(a.chunk.title), title_shingles(b.chunk.title))
    if title_sim < threshold:
        return False
    body_sim = jaccard(
        title_shingles(a.chunk.text[:600]), title_shingles(b.chunk.text[:600])
    )
    return body_sim >= body_threshold
