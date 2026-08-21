# TazaAI-RAG (trial)

Retrieval-first RAG prototype for the Taza engineering working period (**Option 1: contextual retrieval**).

Focus for this phase: **maximize retrieval quality** over Factiva / Dow Jones content. Answer generation is optional and secondary.

## Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| Factiva auth | Dow Jones OAuth2 service account (AuthN → AuthZ) |
| Corpus / retrieve | Factiva Retrieval API (`POST /content/gen-ai/retrieve`) |
| Ranking | Reciprocal Rank Fusion (RRF) + lexical / authority / freshness rerank |
| Local index (ablation) | Dense embeddings + BM25 (`rank-bm25`, NumPy cosine) |
| Config / CLI | `pydantic-settings`, Typer, Rich |
| HTTP | `httpx` |
| Optional generation / judge | OpenAI chat + embeddings (not required for core retrieve path) |

## Architecture

```text
                    ┌─────────────────────────────────────┐
  query ──────────► │ intent detect + multi-query expand  │
                    └─────────────────┬───────────────────┘
                                      │ N query variants
                                      ▼
                    ┌─────────────────────────────────────┐
                    │ Factiva Retrieval API (per variant) │
                    └─────────────────┬───────────────────┘
                                      │ ranked chunk lists
                                      ▼
                    ┌─────────────────────────────────────┐
                    │ RRF fuse → dedupe by doc_id          │
                    │ + lexical overlap                   │
                    │ + source authority prior            │
                    │ + freshness prior                   │
                    │ + per-source diversity cap          │
                    └─────────────────┬───────────────────┘
                                      ▼
                              evidence pack (top-k)

  optional ──► grounded answer + citations (LLM)
  eval     ──► term_hit@k, salience@5, intent breakdown, A1 worksheet
```

Design principle aligned with Taza: keep the **marketplace / retrieval path measurable and auditable**; intelligence around query expansion and ranking can evolve independently.

## Retrieval quality pipeline

Implemented in `taza_rag/factiva/pipeline.py` + `taza_rag/retrieve/quality.py`:

1. **Intent detection** (heuristic) — Factiva-style intents: entity, topical, executive profiling, geographic, risk/compliance, …
2. **Query expansion** — intent-specific variants (e.g. entity → `latest news`, misspelling fixes like `Deutche` → `Deutsche`)
3. **Multi-retrieve** — parallel Factiva calls with intent-aware `days_range`
4. **RRF fusion** — merge variant rankings
5. **Rerank** — lexical overlap + source authority + freshness
6. **Diversity** — cap docs per source code so one wire does not dominate top-k

Ablation: `taza-rag retrieve --raw` skips steps 1–2/4–6 (single Factiva call).

## Evaluation

### Live Factiva (`evals/gold/factiva_live_v1.jsonl`)

| Signal | Meaning |
|--------|---------|
| `term_hit@k` | Required terms present in top-k titles/text |
| `salience@5` | Query-token coverage in top-5 pack |
| By-intent breakdown | Quality stratified by Factiva search intent |
| Markdown worksheet | Human Relevance / Completeness (1–3) on the pack |

```bash
taza-rag eval-retrieve --gold evals/gold/factiva_live_v1.jsonl
```

Writes `evals/reports/factiva_retrieve_latest.json` + `.md` worksheet.

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
taza-rag eval-retrieve --limit 5   # smoke

# Optional generation (needs OPENAI_API_KEY)
taza-rag answer "EU AI Act compliance"

# Local hybrid index (needs embeddings provider)
taza-rag ingest --corpus data/sample_corpus/articles.jsonl
taza-rag eval-local --gold evals/gold/v1.jsonl --no-judge
```

<<<<<<< HEAD
## What “good” looks like

- Higher **term_hit@k** / **salience@5** on `factiva_live_v1` by intent
- Human Relevance/Completeness ≥ 2 on worksheet markdown
- Ablation: quality stack beats `--raw` on entity misspellings & topical queries
- Clear failure tags when packs are weak

## Layout
=======
Equivalent without entrypoint:
>>>>>>> b92ee8e (Rewrite README for trial: retrieval stack and eval focus)

```bash
python -m taza_rag.cli retrieve "SoftBank Group"
# or
./scripts/taza-rag retrieve "SoftBank Group"
```

## Repository layout

```text
taza_rag/
  factiva/          # OAuth, Retrieval API client, intent strategy, quality pipeline
  retrieve/         # RRF / rerank / diversity (Factiva + local)
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
