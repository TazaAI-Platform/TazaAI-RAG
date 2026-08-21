# TazaAI-RAG (trial)

Retrieval-first RAG prototype for the Taza engineering working period (**Option 1: contextual retrieval**).

Focus for this phase: **maximize retrieval quality** over Factiva / Dow Jones content. Answer generation is optional and secondary.

## Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| Factiva auth | Dow Jones OAuth2 service account (AuthN → AuthZ) |
| Corpus / retrieve | Factiva Retrieval API (`POST /content/gen-ai/retrieve`), parallel variants |
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
                                      │ candidate pool
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
                    │ then near-dup collapse + MMR        │
                    └─────────────────┬───────────────────┘
                                      ▼
                              evidence pack (top-k)

  optional ──► grounded answer + citations (LLM)
  eval     ──► aspect_coverage@5, entity_precision@5, noise_rate@5, salience@5
```

Design principle aligned with Taza: keep the **retrieval path measurable and auditable**; query planning and ranking can evolve independently.

## Retrieval quality pipeline

Implemented in `taza_rag/factiva/pipeline.py`, `taza_rag/retrieve/features.py`, `taza_rag/retrieve/quality.py`:

1. **Query plan** — normalize aliases/misspellings (`Deutche` → `Deutsche`), split the ask into **entities** (capitalized/quoted spans) and **topics** (everything else), expand topics with journalistic synonyms (`restructuring` → job cuts, overhaul, divest, …)
2. **Intent detection** — Factiva intents: entity, topical, executive profiling, geographic, risk/compliance, …
3. **Multi-query retrieve** — literal ask + entity anchor + topic paraphrase, issued **in parallel** with intent-aware `days_range`
4. **Relevance tiering** — the decisive step, ordered by *where* the answer lives. A story whose headline covers the ask beats one that mentions it in paragraph nine, which beats one that names the entity but answers a different question (a buyback story for a *restructuring* query). Matching is stem-insensitive, so "sells Indian assets" registers as divestment news
5. **Within-tier scoring** — RRF across variants, BM25 over the candidate pool, entity placement (title > lead > body), topic coverage, source authority, freshness
6. **Document-type penalties** — headline digests (`Top ... Headlines at 12 AM ET`, multi-headline pipes, newsletter round-ups) and vendor profiles (MarketLine/GlobalData, `- History`, SWOT) are demoted; aggregators lightly penalized
7. **Entity gate** — for news-style intents, candidates that never name the entity are dropped when enough on-entity evidence exists
8. **Near-duplicate collapse + diversity** — identical stories are merged, per-source caps apply (2 for survey intents, 3 for entity intents), and **MMR** within each tier trades a little score for narrative breadth so one angle or region cannot fill the pack

Ablation: `taza-rag retrieve --raw` is the baseline (single Factiva call, API order, no scoring).

## Evaluation

### Live Factiva (`evals/gold/factiva_live_v1.jsonl`)

| Signal | Meaning |
|--------|---------|
| `term_hit@k` | Required terms present anywhere in top-k (a floor check — saturates at 1.0) |
| `term_hit_lead@3` | Stricter: required terms must reach the top-3 headlines/leads |
| `aspect_coverage@5` | Share of the gold set's substantive angles surfaced in the top-5 — the **Completeness** proxy and the most discriminative signal |
| `salience@5` | Query-token coverage in the top-5 pack |
| `entity_precision@5` | Share of top-5 that actually name the query entity in title/lead |
| `noise_rate@5` | Share of top-5 that are digests or vendor profiles (lower is better) |
| By-intent breakdown | Quality stratified by Factiva search intent |
| Markdown worksheet | Human Relevance / Completeness (1–3) on each pack |

`term_hit@k` is kept deliberately as a regression floor even though it saturates: the
Retrieval API almost always returns the subject, so it cannot separate a good pack from
a mediocre one. `aspect_coverage@5`, `entity_precision@5` and `noise_rate@5` are what
move when ranking changes.

### Measured results

16 gold queries across all five Factiva intents, live API, `top_k=8`. Baseline is a
single Factiva call in API order; quality is the full stack.

| Metric | Baseline | Quality | Delta |
|--------|----------|---------|-------|
| `term_hit@k` | 1.000 | 1.000 | +0.000 |
| `term_hit_lead@3` | 1.000 | 1.000 | +0.000 |
| `aspect_coverage@5` | 0.552 | 0.688 | **+0.135** |
| `salience@5` | 0.899 | 0.954 | +0.055 |
| `entity_precision@5` | 0.700 | 0.856 | **+0.156** |
| `noise_rate@5` (lower better) | 0.062 | 0.013 | **−0.050** |

Reproduce with `taza-rag eval-retrieve --top-k 8 --compare`. Numbers shift slightly run
to run because the underlying Factiva corpus is live.

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
taza-rag retrieve "SoftBank Group" --raw            # no quality stack at all
```

## Tests

```bash
python scripts/run_tests.py     # stdlib runner, no pytest needed
pytest tests/                   # if dev extras are installed
```

24 offline tests cover entity extraction, document-type detection, tiering, stemming,
MMR, and near-duplicate collapse. None require network access or API credentials.

## What “good” looks like

- `aspect_coverage@5`, `entity_precision@5` and `noise_rate@5` beat the `--raw` baseline
- Human Relevance/Completeness ≥ 2 on the generated worksheet
- Misspelled entities (`Deutche`) rank the same as correctly spelled ones
- Digests and vendor profiles stay out of the top-5 for news intents

## Repository layout

```text
taza_rag/
  factiva/          # OAuth, Retrieval API client, intent strategy, quality pipeline
  retrieve/         # query features, tiering / rerank / diversity (Factiva + local)
  index/            # local dense + BM25 store
  ingest/           # chunking + contextual prefixes (local path)
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

**Now:** Factiva retrieve quality loop + intent-aware ranking + offline-capable metrics.

**Next candidates:**
- Cross-encoder / vendor reranker on fused candidates
- News Feed / Streams → local PGVector index for owned corpus
- Value-before-access / entitlement-aware source selection (marketplace layer)
- Full A1 scoring on generated answers once packs are stable

## License / access

Private trial repository for Taza. Factiva content subject to Dow Jones licensing and API usage terms.
