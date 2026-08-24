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
| Generation | OpenAI `gpt-4o-mini` (not required for core retrieve path) |
| A1 judge | `gpt-5` by default — deliberately not the generator, which inflated its own scores |
| Grounding verification | Deterministic figure + citation checks, one LLM entailment call, one repair pass |

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

#### Does the embedding signal help? (measured: no)

`--semantic` scores each contextualized passage by cosine similarity to the query using
`text-embedding-3-small`, fused into the composite score. Measured over the same 16 queries:

| Metric | Lexical | + semantic | Delta |
|--------|---------|-----------|-------|
| `term_hit_lead@3` | 1.000 | 1.000 | +0.000 |
| `aspect_coverage@5` | 0.635 | 0.667 | +0.031 |
| `aspect_coverage@1200tok` | 0.729 | 0.688 | −0.042 |
| `entity_precision@5` | 0.925 | 0.925 | +0.000 |
| `noise_rate@5` | 0.013 | 0.013 | +0.000 |
| `salience@5` | 0.926 | 0.926 | +0.000 |

No measurable gain, so it stays **off by default**. Two honest caveats: the lexical metrics
are near saturation (`entity_precision` 0.93, `noise_rate` 0.01), so there is little headroom
for a second signal to show up in them; and relevance tiering ranks before the composite
score, so an embedding can only reorder *within* a tier by design. The finding is "no gain on
this gold set", not "embeddings are useless" — a larger, harder gold set is the way to settle it.

Cost is real: embedding ~75 passages per query adds ~2 s. On a low-tier OpenAI account
(40,000 TPM) each call needs ~14K tokens, so throttling stretched this to ~40 s per query;
`taza_rag/llm.py` retries on `rate_limit_exceeded`, honouring the provider's suggested wait.

### A1 answer-level evaluation

Retrieval metrics score the evidence pack. Accuracy and Clarity are properties of the
**answer**, so they need generation — this is the only path where the full A1 rubric applies.

```bash
taza-rag eval-a1 --compare                                  # vs single-call baseline
taza-rag eval-a1 --gold evals/gold/factiva_abstain_v1.jsonl  # refusal behaviour
```

16 gold queries, `top_k=10`, generator `gpt-4o-mini`, judge `gpt-5`:

| A1 criterion | Verification off | Verification on | Delta |
|--------------|-----------------|-----------------|-------|
| **Accuracy** (hard gate, all 4 checks) | 0.438 | **0.688** | +0.250 |
| ├ factual correctness | 0.438 | 0.750 | +0.312 |
| ├ citation integrity | 0.438 | 0.688 | +0.250 |
| ├ no hallucinations | 0.688 | 0.750 | +0.062 |
| └ contextual integrity | 0.750 | 0.812 | +0.062 |
| Relevance (1–3) | 2.50 | 2.38 | −0.125 |
| Completeness (1–3) | 1.94 | 1.62 | −0.312 |
| Clarity (1–3) | 2.94 | 2.88 | −0.062 |

The verification-on column was re-measured after the claim-group fix described below; the
figures it replaced were produced by a verifier that mis-flagged correctly cited prose.

### Grounding verification

The generator is told to cite everything and invent nothing, and was then trusted. Under an
independent judge that trust does not survive: citation integrity failed on more than half of
answers. `taza_rag/factiva/verify.py` checks the answer instead.

| Check | Method | Catches |
|-------|--------|---------|
| Citation presence | deterministic | a factual sentence with no `[cN]` marker |
| Label validity | deterministic | `[c9]` when only 8 sources exist |
| Figure grounding | deterministic | a number appearing in no source, or cited to the wrong one |
| Claim support | one batched LLM call | attribution, certainty or magnitude the excerpt does not carry |

The figure and citation checks are deterministic, so they cost nothing and cannot themselves
hallucinate. Only paraphrase-level support needs a model. Anything flagged goes to a single
corrective pass, and the rewrite is then re-checked deterministically, since a rewrite can
introduce its own bad figures. Both reports are kept on the answer, so a residual problem is
visible rather than silently accepted.

Citation presence is checked per claim **group**, not per sentence. Journalistic prose sources
a group once, usually at its end, so requiring a marker on every sentence flagged ordinary
writing as unsourced — it fired on all 16 answers, forcing a needless rewrite each time, and
the repair pass could not "fix" answers that were already correct. A sentence with no marker
now inherits its neighbour's, within a paragraph only. Inheritance cannot launder a bad number:
figures are still checked against the inherited sources, so a fabricated figure in an
inheriting sentence is still caught. Fixing this cut spurious `uncited` flags from 31 to 7 and
took post-repair uncited claims to **zero**, which is a deterministic count rather than a
judge's opinion.

Figure grounding distinguishes three cases, because they are different defects: a number in
no source is invention, a real number attributed to the wrong chunk is miscitation, and a bare
year is only reported — "this year" is routinely paraphrased and should not trigger a rewrite.

**The cost is Completeness, and that is the interesting part.** Stripping unsupported claims
makes answers thinner: Completeness drops 0.25 and Relevance 0.125. Some of the earlier
Completeness score was being earned by content the sources did not support. That is the same
tension seen from the other side above — this judge rewards material that the Accuracy gate
should reject, and verification resolves it in favour of Accuracy.

Every Accuracy gate improves and `hallucination` falls out of the top failure tags. Treat the
size, not the direction, with caution: n=16 gives ±0.19 run-to-run, so +0.125 on any single
gate is within noise. Four gates moving up together is the part worth believing.

```bash
taza-rag answer "Abu Dhabi Investment Authority"    # verification on by default
taza-rag answer "..." --no-verify                   # ablate it
taza-rag eval-a1 --no-verify --judge-model gpt-5    # measure the difference
```

#### Why an earlier version of this file claimed Accuracy 1.000

It was wrong, for three separate reasons worth recording:

1. **The judge was grading its own homework.** `gpt-4o-mini` generating *and* scoring gave
   Accuracy 0.812; `gpt-5` scoring the identical answers gave 0.625. Per-query agreement
   between the two judges is only 0.250 — they disagree on 12 of 16 queries. The judge is
   now a separate, stronger model by default (`judge_model`, `--judge-model`).
2. **The judge was shown less evidence than the generator.** Excerpts were truncated to 900
   characters while the generator received the full passage, so supported claims were failed
   as unsupported. This alone accounted for most of the apparent collapse: `gpt-5` Accuracy
   was 0.125 before the fix and 0.625 after. `AnswerResult.context` now records the verbatim
   context and the judge scores against exactly that.
3. **n=16 is small and the corpus is live.** Three runs of the same configuration produced
   Accuracy 1.000, 0.875 and 0.812. One query is worth 6.25 points, so a single perfect run
   is consistent with a true pass rate near 0.80.
4. **Re-judging dropped the citations.** The re-judge path rebuilt answers without their
   citation records, so the judge failed citation integrity on answers that were properly
   cited. `rejudge-a1` now refuses a report that stores only citation doc ids.

5. **The verifier demanded a citation on every sentence.** It flagged all 16 answers, including
   correctly sourced prose that cites a claim group once at its end, so measured citation
   integrity was depressed by the checking tool rather than by the answers.

The residual failures are real, not judge pedantry. Spot-checking flagged claims against the
stored evidence by string search confirms them: one answer asserted "record profits" and
another "substantial market value gains", and neither phrase nor its support appears anywhere
in the retrieved text. **Citation integrity at 0.688 remains the open defect.** The shape of it
has changed, though: deterministic uncited claims are now zero after repair, so what is left is
paraphrase-level over-reach — the entailment check flags 31 claims across 16 answers for
attribution, certainty or magnitude the excerpt does not carry, and the single repair pass does
not resolve all of them.

Reproduce the judge comparison — the answers are held fixed so only the judge varies:

```bash
taza-rag eval-a1 --top-k 10                       # generate + judge, stores evidence
taza-rag rejudge-a1 --judge-model gpt-4o-mini     # re-score the same answers
```

**The Completeness ceiling is real and judge-independent.** Both judges score it 2.00 on the
same answers, and three independent fixes each moved it by exactly zero:

| Attempted fix | Completeness | Accuracy |
|---------------|--------------|----------|
| baseline of the six weak queries | 2.00 | 1.000 |
| prompt demands figures + dissent | 2.00 | 1.000 |
| whole articles instead of passages (max depth) | 2.00 | 1.000 |
| `top_k=20`, 5000-token budget, 12 distinct sources (max breadth) | 2.00 | 1.000 |

Reading what the judge asks for explains why: "broader market implications", "impact on
climate change", "potential risks associated with AI investments". These are requests for
interpretation that a licensed news corpus does not contain, and that the generation prompt
deliberately refuses to supply. Chasing them means speculating, which is precisely what the
Accuracy gate fails an answer for. Neither judge is a Dow Jones evaluator, so calibrating
these six against a human scorer is still the next step, and the `.md` worksheet exists for
exactly that.

Evidence for generation is budgeted by **tokens, not chunk count** — passages are about half
the size of an article, so a fixed chunk count silently handed the generator half the
evidence. Equalising the budget is what lifted Clarity to 3.00.

#### Abstention

Five deliberately unanswerable queries (`evals/gold/factiva_abstain_v1.jsonl`): a future
reporting period, a private unpublished act, personal data, a fictional company, and
confidential material.

**Abstention recall: 0.800** — 4 of 5 refused, including the invented company (no entity
hallucinated). The miss is `a005`, "confidential ECB minutes", which the system answered from
published reporting on ECB deliberations; that is arguably correct behaviour and a gold label
that is too strict, rather than a clean failure. Expected refusals are aggregated separately
from answer quality, because scoring a correct refusal against the answer rubric counts the
right behaviour as a failure.

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
| `OPENAI_API_KEY` | optional | `answer`, `eval-a1`, local embeddings, `--semantic`, `--llm-context` |

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

# Optional generation + A1 answer eval (needs OPENAI_API_KEY)
taza-rag answer "EU AI Act compliance"
taza-rag answer "EU AI Act compliance" --raw   # baseline retrieval, for comparison
taza-rag eval-a1 --compare                     # Accuracy gate + R/C/Clarity vs baseline
taza-rag eval-a1 --judge-model gpt-4.1         # swap the judge, not the generator
taza-rag rejudge-a1 --judge-model gpt-5        # re-score stored answers, judge isolated
taza-rag eval-a1 --gold evals/gold/factiva_abstain_v1.jsonl   # refusal behaviour

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

80 offline tests cover entity extraction and splitting, multi-entity tiering,
document-type detection, stemming, MMR, near-duplicate collapse, contextual passage
retrieval (splitting, id stability, lead-signal scoping, one-passage-per-document,
position penalty), claim verification (citation inheritance and its paragraph boundary,
figure grounding, short-claim detection), and provider-error handling (rate-limit retry,
quota as fatal, temperature fallback). None require network access or API credentials.

The provider-error and verification paths are the ones a live smoke test cannot reach, so
they are covered with a fake client rather than by hoping they work.

## What “good” looks like

- **A1 Accuracy under an independent judge** — the rubric's automatic-fail gate, currently
  0.688 and limited by citation integrity
- `aspect_coverage@5`, `entity_precision@5` and `noise_rate@5` beat the `--raw` baseline
- Coverage per 1k tokens of evidence improves, not just coverage
- Human Relevance/Completeness ≥ 2 on the generated worksheet
- Unanswerable queries are refused rather than answered
- Misspelled entities (`Deutche`) rank the same as correctly spelled ones
- Digests and vendor profiles stay out of the top-5 for news intents

## Repository layout

```text
taza_rag/
  factiva/          # OAuth, Retrieval API client, intent strategy, quality pipeline
  factiva/verify.py # claim-level grounding checks + repair pass
  retrieve/         # query features, tiering / rerank / diversity (Factiva + local)
  index/            # local dense + BM25 store
  ingest/           # structure-aware chunking + contextual prefixes
  factiva/contextual.py  # passage splitting + situating context + optional embeddings
  generate/         # optional grounded answer
  eval/             # Factiva retrieval eval, A1 answer eval, local metrics, A1 judge
  cli.py
evals/
  gold/factiva_live_v1.jsonl
  gold/factiva_abstain_v1.jsonl  # deliberately unanswerable queries
  gold/v1.jsonl                 # local sample gold
configs/
data/sample_corpus/
scripts/test_factiva_auth.sh
tests/
```

## Trial status / next

**Now:** Factiva retrieve quality loop, intent-aware ranking, contextual passage
retrieval, offline-capable retrieval metrics including cost-normalized coverage, and
answer-level A1 scoring with an independent judge, re-judgeable artifacts, and measured
abstention recall (0.800).

**Known gaps, stated plainly:**
- **Citation integrity is improved but still the weak gate (0.688).** Deterministic uncited
  claims are now zero after repair, so the remainder is paraphrase-level over-reach caught by
  the entailment check: 31 flagged claims across 16 answers, of which one repair pass clears
  most but not all. Looping repair until the checks are clean is the obvious next step.
- **Verification trades Completeness for Accuracy** (−0.31). Defensible for a Dow Jones
  product where Accuracy is the automatic-fail gate, but it is a real cost, not a free win.
- **Judges disagree with each other far more than expected.** `gpt-4o-mini` and `gpt-5` agree
  on only 0.250 of queries. Any single-judge number should be read as one noisy sample, and
  no A1 figure here has been calibrated against a human scorer.
- **n=16 with a live corpus is not a stable metric.** Repeat runs of one configuration moved
  Accuracy by 0.19. Expanding gold matters more than further ranking work.
- **Gold covers 5 of the 10 Factiva intents.** `industry_scan`, `event_tracking`,
  `known_item`, `competitive_intel` and `brand_perception` have no rows yet.
- **Abstention gold is 5 queries and one label is arguable** (`a005`, confidential ECB
  minutes, is answerable from published reporting). Too small to be a stable metric.
- **LLM-written passage context (`--llm-context`) is unmeasured.** It needs one chat call
  per passage (~1,100 per eval run), which is slow and rate-limited on a low-tier account;
  given that the cheaper embedding signal showed no gain, it was not prioritised.
- **The embedding signal shows no measurable gain** on this gold set and is off by default;
  see the caveats above on metric saturation and tier-ordered ranking.

**Next candidates:**
- Loop the repair pass until the deterministic checks are clean, rather than one attempt
- Human calibration of the six Completeness-limited queries against the A1 rubric
- Cross-encoder / vendor reranker on fused candidates
- Expand gold to the remaining five intents, and to a harder set with headroom
- News Feed / Streams → local PGVector index for owned corpus
- Value-before-access / entitlement-aware source selection (marketplace layer)

## License / access

Private trial repository for Taza. Factiva content subject to Dow Jones licensing and API usage terms.
