# Taza trial RAG — **retrieval quality first** (Factiva)

Built for the Taza working period (**Option 1: contextual retrieval**).

Priority order:
1. Retrieve the right Factiva evidence (saliency, authority, freshness)
2. Measure with Factiva intents + A1-oriented worksheets
3. Generate answers only after retrieval is strong (optional OpenAI)

## Quality retrieval stack (no OpenAI)

```
query
  → intent detect + heuristic multi-query expand
  → Factiva Retrieval API (N variants)
  → RRF fuse
  → lexical + authority + freshness rerank
  → source diversity cap
  → ranked evidence pack
```

## Setup

```bash
cd /home/Trustvs/Documents/TazaAI-RAG
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
# .env already has Factiva credentials (gitignored).
# OPENAI_API_KEY only needed for optional `answer` / embedding ingest.
```

## Primary commands

```bash
# OAuth
taza-rag factiva-auth --account rag

# Best evidence pack for a question (no OpenAI)
taza-rag retrieve "Jamie Dimon on the future of the economy"
taza-rag retrieve "Deutche Bank restructuring" --out /tmp/hits.json

# Ablation: single Factiva call vs quality stack
taza-rag retrieve "private credit market trends" --raw

# Live Factiva retrieval eval + human A1 worksheet (no OpenAI)
taza-rag eval-retrieve --gold evals/gold/factiva_live_v1.jsonl

# Optional generation AFTER retrieval is good
taza-rag answer "EU AI Act compliance"   # needs OPENAI_API_KEY
```

## What “good” looks like for the CEO

- Higher **term_hit@k** / **salience@5** on `factiva_live_v1` by intent
- Human Relevance/Completeness ≥ 2 on worksheet markdown
- Ablation: quality stack beats `--raw` on entity misspellings & topical queries
- Clear failure tags when packs are weak

## Layout

```
taza_rag/factiva/     # auth, retrieve, strategy, quality pipeline
taza_rag/retrieve/    # fuse / rerank helpers
taza_rag/eval/        # factiva retrieval eval (no LLM judge required)
evals/gold/factiva_live_v1.jsonl
```
