# TazaAI-RAG (trial)

Retrieval-first RAG prototype for the Taza engineering working period (**Option 1: contextual retrieval**).

Focus for this phase: **maximize retrieval quality** over Factiva / Dow Jones content. Answer generation is optional and secondary.

## Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| Factiva auth | Dow Jones OAuth2 service account (AuthN → AuthZ) |
| Corpus / retrieve | Factiva Retrieval API (`POST /content/gen-ai/retrieve`), parallel variants |
| Contextual retrieval | Passage splitting + situating context per passage (heuristic; LLM optional) |
| Ranking | Relevance tiering + RRF + BM25 + entity/topic signals + authority/freshness, doc-type penalties, near-duplicate collapse |
| Local index (ablation) | Dense embeddings + BM25 (`rank-bm25`, NumPy cosine) |
| Config / CLI | `pydantic-settings`, Typer, Rich |
| HTTP | `httpx` |
| Optional generation / judge | OpenAI chat + embeddings (not required for core retrieve path) |

## Architecture

```text
                    ┌─────────────────────────────────────┐
  query ──────────► │ normalize → intent → query plan     │
                    │ (entities vs topics + synonyms)     │
                    └─────────────────┬───────────────────┘
                                      │ N query variants
                                      ▼
                    ┌─────────────────────────────────────┐
                    │ Factiva Retrieval API (in parallel) │
                    └─────────────────┬───────────────────┘
                                      │ candidate articles
                                      ▼
                    ┌─────────────────────────────────────┐
                    │ contextual retrieval                │
                    │  article → passages (~180 tok)      │
                    │  + situating context per passage    │
                    │    (source, date, subject, position)│
                    └─────────────────┬───────────────────┘
                                      │ candidate passages
                                      ▼
                    ┌─────────────────────────────────────┐
                    │ relevance tiering                   │
                    │  t0 entity + topic in headline/lead │
                    │  t1 entity + topic deeper in body   │
                    │  t2 entity only                     │
                    │  t3 off-entity   → gated out        │
                    │ within tier: RRF + BM25 +           │
                    │ entity/topic placement + authority  │
                    │ + freshness − doc-type penalty      │
                    │ then best passage per document,     │
                    │ near-dup collapse + MMR             │
                    └─────────────────┬───────────────────┘
                                      ▼
                              evidence pack (top-k passages)

  optional ──► grounded answer + citations (LLM)
  eval     ──► aspect_coverage@5, entity_precision@5, noise_rate@5, salience@5
```

Design principle aligned with Taza: keep the **retrieval path measurable and auditable**; query planning and ranking can evolve independently.

## Retrieval quality pipeline

Implemented in `taza_rag/factiva/pipeline.py`, `taza_rag/retrieve/features.py`, `taza_rag/retrieve/quality.py`:

1. **Query plan** — normalize aliases/misspellings (`Deutche` → `Deutsche`), split the ask into **entities** (capitalized/quoted spans) and **topics** (everything else), expand topics with journalistic synonyms (`restructuring` → job cuts, overhaul, divest, …). Adjacent proper nouns are separated at real boundaries, so "Larry Fink BlackRock private markets" yields `Larry Fink` **and** `BlackRock` rather than one phrase that appears in no document, while `Deutsche Bank` and `European Central Bank` stay intact. When a query names two entities, evidence naming only one cannot reach the top tier — a Jassy story about Indian retail does not answer "Andy Jassy AWS growth strategy"
2. **Intent detection** — Factiva intents: entity, topical, executive profiling, geographic, risk/compliance, …
3. **Multi-query retrieve** — literal ask + entity anchor + topic paraphrase, issued **in parallel** with intent-aware `days_range`
4. **Contextual retrieval** (`taza_rag/factiva/contextual.py`) — the Retrieval API returns whole articles, so each one is split into ~180-token passages and each passage gets a situating prefix (publication, date, subject, position in the story) before it is indexed. Passages, not articles, are what get scored and cited. Passage ids are positional, which is what lets rank fusion recognise the same passage returned by two different query variants. The prefix is heuristic by default — no API key, 37 ms for ~70 passages; `--llm-context` writes it with an LLM instead
5. **Relevance tiering** — the decisive step, ordered by *where* the answer lives. A story whose headline covers the ask beats one that mentions it in paragraph nine, which beats one that names the entity but answers a different question (a buyback story for a *restructuring* query). Matching is stem-insensitive, so "sells Indian assets" registers as divestment news
6. **Within-tier scoring** — RRF across variants, BM25 over the contextualized text, entity placement (title > lead > body), topic coverage, source authority, freshness, minus a small penalty for passages deeper in the article. "Lead" is only granted to a document's opening passage, so evidence buried in paragraph nine cannot claim headline-level credit
7. **Document-type penalties** — headline digests (`Top ... Headlines at 12 AM ET`, multi-headline pipes, newsletter round-ups) and vendor profiles (MarketLine/GlobalData, `- History`, SWOT) are demoted; aggregators lightly penalized
8. **Entity gate** — for news-style intents, candidates that never name the entity are dropped when enough on-entity evidence exists
9. **Best passage per document, near-duplicate collapse + diversity** — each document contributes only its strongest passage, and the near-duplicate test requires a matching headline *and* matching content so two passages of one story are never mistaken for two sources. Per-source caps apply (2 for survey intents, 3 for entity intents), and **MMR** within each tier trades a little score for narrative breadth so one angle or region cannot fill the pack

Ablation: `taza-rag retrieve --raw` is the baseline (single Factiva call, API order, no scoring).

## Evaluation

### Live Factiva (`evals/gold/factiva_live_v1.jsonl`)

| Signal | Meaning |
|--------|---------|
| `term_hit@k` | Required terms present anywhere in top-k (a floor check — saturates at 1.0) |
| `term_hit_lead@3` | Stricter: required terms must reach the top-3 headlines/leads |
| `aspect_coverage@5` | Share of the gold set's substantive angles surfaced in the top-5 — the **Completeness** proxy and the most discriminative signal |
| `aspect_coverage@1200tok` | The same coverage measured from a fixed evidence budget, so a pack cannot win by returning more text |
| `evidence_tokens@5` | Size of the pack the customer pays to receive (lower is better) |
| `salience@5` | Query-token coverage in the top-5 pack |
| `entity_precision@5` | Share of top-5 that actually name the query entity in title/lead |
| `noise_rate@5` | Share of top-5 that are digests or vendor profiles (lower is better) |
| By-intent breakdown | Quality stratified by Factiva search intent |
| Markdown worksheet | Human Relevance / Completeness (1–3) on each pack |

`term_hit@k` is kept deliberately as a regression floor even though it saturates: the
Retrieval API almost always returns the subject, so it cannot separate a good pack from
a mediocre one. `aspect_coverage@5`, `entity_precision@5` and `noise_rate@5` are what
move when ranking changes.

`aspect_coverage@5` has a bias worth naming: it rewards returning **more text**, because
a long article covers more terms than a short passage simply by being longer — which is
exactly the over-retrieval behaviour a marketplace charges for. `aspect_coverage@1200tok`
compares packs at equal cost, and `evidence_tokens@5` reports the cost itself.

### Measured results

16 gold queries across all five Factiva intents, live API, `top_k=8`. Baseline is a
single Factiva call in API order; quality is the full stack.

| Metric | Baseline | Quality + contextual | Delta |
|--------|----------|----------------------|-------|
| `term_hit@k` | 1.000 | 1.000 | +0.000 |
| `term_hit_lead@3` | 1.000 | 1.000 | +0.000 |
| `aspect_coverage@5` | 0.552 | 0.635 | **+0.083** |
| `aspect_coverage@1200tok` | 0.688 | 0.750 | **+0.062** |
| `entity_precision@5` | 0.775 | 0.925 | **+0.150** |
| `noise_rate@5` (lower better) | 0.062 | 0.013 | **−0.050** |
| `salience@5` | 0.899 | 0.926 | +0.027 |
| `evidence_tokens@5` (lower better) | 2361 | 1127 | **−1234** |

More coverage from **half the text**: coverage per 1k tokens of evidence goes from 0.23 to
0.56. Median latency is ~4.3 s per query, of which contextualization is 37 ms.

Reproduce with `taza-rag eval-retrieve --top-k 8 --compare`. Numbers shift slightly run
to run because the underlying Factiva corpus is live.

#### Is contextual retrieval carrying its weight?

Isolating it against the same ranking stack over whole articles (`--no-contextual`):

| Metric | Articles | Passages | Delta |
|--------|----------|----------|-------|
| `aspect_coverage@5` | 0.688 | 0.635 | −0.052 |
| `aspect_coverage@1200tok` | 0.823 | 0.750 | −0.073 |
| `term_hit_lead@3` | 0.969 | 1.000 | +0.031 |
| `entity_precision@5` | 0.925 | 0.925 | +0.000 |
| `noise_rate@5` | 0.013 | 0.013 | +0.000 |
| `evidence_tokens@5` | 2280 | 1127 | **−1153** |
| coverage per 1k tokens | 0.30 | **0.56** | **+87%** |

Reported as measured rather than as a win. Passages cost a little raw coverage — a
contiguous passage carries narrower vocabulary than a whole article, and that holds even
at a fixed token budget. What they buy is half the text for that coverage, and a citation
that points at the specific paragraph supporting a claim instead of a whole story, which
is what the A1 **citation integrity** check actually asks for. Passage size barely matters
(320 tokens scores identically to 180), so the smaller window is kept.

Contextual retrieval is on by default because per-token efficiency and citation precision
are what a marketplace charges for; `--no-contextual` is the setting to prefer if raw
coverage is the only objective.

```bash
taza-rag eval-retrieve --gold evals/gold/factiva_live_v1.jsonl
taza-rag eval-retrieve --compare   # baseline vs quality stack, with deltas
```

Writes `evals/reports/factiva_retrieve_latest.json` + `.md` worksheet. `--compare` also runs
the single-call baseline per query and emits an ablation table.

### Local sample index (offline ablations)

Sample corpus + gold with known `doc_id`s for Recall@k / Precision@k. Useful when iterating ranking logic without Factiva quota.

Dow Jones **A1** dimensions (Accuracy gate, Relevance / Completeness / Clarity) are modeled in `taza_rag/eval/dj_a1.py` for later answer-stage scoring. Accuracy is not applied to retrieve-only packs.

## Setup

```bash
git clone https://github.com/TazaAI-Platform/TazaAI-RAG.git
cd TazaAI-RAG
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env   # fill Factiva credentials
```

### Environment

| Variable | Required | Purpose |
|----------|----------|---------|
| `FACTIVA_RAG_CLIENT_ID` | yes | RAG / Retrieval API client id |
| `FACTIVA_RAG_USERNAME` | yes | Service account email |
| `FACTIVA_RAG_PASSWORD` | yes | Service account API password |
| `FACTIVA_FEED_*` | optional | AI News Feed / Streams account |
| `FACTIVA_TOKEN_URL` | default | `https://accounts.dowjones.com/oauth2/v1/token` |
| `FACTIVA_API_BASE` | default | `https://api.dowjones.com` |
| `OPENAI_API_KEY` | optional | `answer`, local embeddings, LLM judge only |

`.env` is gitignored. Never commit credentials.

## CLI

```bash
# Auth
taza-rag factiva-auth --account rag

# Primary: quality retrieval (no OpenAI)
taza-rag retrieve "SoftBank Group"
taza-rag retrieve "Deutche Bank restructuring" --out /tmp/hits.json
taza-rag retrieve "private credit market trends" --raw   # baseline ablation

# Eval
taza-rag eval-retrieve --gold evals/gold/factiva_live_v1.jsonl
taza-rag eval-retrieve --limit 5              # smoke
taza-rag eval-retrieve --limit 5 --compare    # vs single-call baseline

# Optional generation (needs OPENAI_API_KEY)
taza-rag answer "EU AI Act compliance"

# Local hybrid index (needs embeddings provider)
taza-rag ingest --corpus data/sample_corpus/articles.jsonl
taza-rag eval-local --gold evals/gold/v1.jsonl --no-judge
```

Equivalent without installing the entrypoint:

```bash
python -m taza_rag.cli retrieve "SoftBank Group"
# or
./scripts/taza-rag retrieve "SoftBank Group"
```

### Ablations

Each ranking stage can be switched off to attribute the gain:

```bash
taza-rag retrieve "Deutsche Bank restructuring" --no-entity-gate
taza-rag retrieve "private credit market trends" --no-diversity
taza-rag retrieve "SoftBank Group" --variants 1     # single query, still reranked
taza-rag retrieve "SoftBank Group" --no-contextual  # whole articles, not passages
taza-rag retrieve "SoftBank Group" --raw            # no quality stack at all

# Contextual retrieval variants
taza-rag retrieve "SoftBank Group" --passage-tokens 320   # wider passages
taza-rag retrieve "SoftBank Group" --llm-context          # LLM-written context (needs key)
taza-rag retrieve "SoftBank Group" --semantic             # + embedding similarity (needs key)

# Same switches in eval, to attribute a metric change to one stage
taza-rag eval-retrieve --top-k 8 --no-contextual
```

## Tests

```bash
python scripts/run_tests.py     # stdlib runner, no pytest needed
pytest tests/                   # if dev extras are installed
```

39 offline tests cover entity extraction and splitting, multi-entity tiering,
document-type detection, stemming, MMR, near-duplicate collapse, and contextual passage
retrieval (splitting, id stability, lead-signal scoping, one-passage-per-document,
position penalty). None require network access or API credentials.

## What “good” looks like

- `aspect_coverage@5`, `entity_precision@5` and `noise_rate@5` beat the `--raw` baseline
- Coverage per 1k tokens of evidence improves, not just coverage
- Human Relevance/Completeness ≥ 2 on the generated worksheet
- Misspelled entities (`Deutche`) rank the same as correctly spelled ones
- Digests and vendor profiles stay out of the top-5 for news intents

## Repository layout

```text
taza_rag/
  factiva/          # OAuth, Retrieval API client, intent strategy, quality pipeline
  retrieve/         # query features, tiering / rerank / diversity (Factiva + local)
  index/            # local dense + BM25 store
  ingest/           # structure-aware chunking + contextual prefixes
  factiva/contextual.py  # passage splitting + situating context + optional embeddings
  generate/         # optional grounded answer
  eval/             # Factiva retrieval eval, local metrics, A1 judge schema
  cli.py
evals/
  gold/factiva_live_v1.jsonl
  gold/v1.jsonl                 # local sample gold
configs/
data/sample_corpus/
scripts/test_factiva_auth.sh
tests/
```

## Trial status / next

**Now:** Factiva retrieve quality loop, intent-aware ranking, contextual passage
retrieval, and offline-capable metrics including cost-normalized coverage.

**Known gaps, stated plainly:**
- **A1 Accuracy and Clarity are unmeasured.** The judge exists (`taza_rag/eval/dj_a1.py`)
  and covers all four dimensions, but it needs an OpenAI key and has only ever been wired
  to the local path. Everything measured here is retrieval-side.
- **Gold covers 5 of the 10 Factiva intents.** `industry_scan`, `event_tracking`,
  `known_item`, `competitive_intel` and `brand_perception` have no rows yet.
- **Abstention is implemented but unevaluated** — no gold query is deliberately
  unanswerable, so the refusal path has no measured precision.
- **Semantic scoring depends on the upstream API.** Local scoring is lexical by design
  (no key, ~150 ms); `--semantic` adds embedding similarity over contextualized passages
  when a key is present, but its contribution has not been measured.

**Next candidates:**
- Cross-encoder / vendor reranker on fused candidates
- News Feed / Streams → local PGVector index for owned corpus
- Value-before-access / entitlement-aware source selection (marketplace layer)

## License / access

Private trial repository for Taza. Factiva content subject to Dow Jones licensing and API usage terms.
