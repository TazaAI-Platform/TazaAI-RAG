# Appendix 2 — Retrieval: the tool boundary

Referenced from [AGENT.md](../../AGENT.md) § Architecture.

## The boundary

The agent reaches the corpus only through `SearchBackend` (`taza_rag/agent/gather.py`):

```python
class SearchBackend(Protocol):
    def search(self, query: str, *, top_k: int, intent: SearchIntent | None) -> list[RetrievedChunk]: ...
```

This mirrors Taza's own stated design principle — a clean boundary between an agent's
intelligence and the deterministic request path. The agent decides *what to ask and when to
stop*; the backend decides *how to rank*. Two consequences:

- The measured retrieval quality is reused rather than reimplemented: `aspect_coverage@5`
  0.615 against a 0.510 baseline, `noise_rate@5` 0.008, at 40% fewer evidence tokens. See the
  main [README](../../README.md) § Retrieval quality.
- The loop is testable offline. `FixtureSearch` supplies a fixed corpus, which is what makes
  the round-by-round behaviour verifiable without spending metered calls.

## Implementations

| Backend | Use |
|---|---|
| `FactivaSearch` | Live. Wraps `QualityRetriever`, one instance shared across threads so the OAuth exchange happens once. |
| `FixtureSearch` | Deterministic in-memory corpus for tests and offline regression. Whole-term overlap with a title bonus — monotonic in relevance, not an imitation of the ranker. |

## Parallelism and failure isolation

`gather()` issues one wave concurrently through a bounded `ThreadPoolExecutor`. Note there
are now **two levels** of parallelism: the agent fans out across sub-questions, and
`QualityRetriever` already fans out across query variants inside each call.

A failing step is recorded in `RoundRecord.failed_queries` and `ResearchResult.errors`, and
the run continues. The live corpus fails a few percent of calls even after retries, and a
four-step plan that aborts on the first 502 is useless. Guarded by
`test_one_failing_step_does_not_sink_the_run`.

## One pool, one label space

`EvidencePool` keeps one entry per passage, keyed on `chunk_id` (positional and stable, so the
same passage found by two steps collapses while two different passages of one article stay
distinct as genuinely different evidence).

**Labels `c1..cN` are assigned once for the whole run.** Per-section labelling was the first
approach and it collided: two sections each had a `c1` pointing at different documents, so a
citation in the final answer could not be resolved back to a source. Guarded by
`test_a_full_run_answers_from_the_corpus_with_resolvable_citations`.

Each entry records `found_by` — every sub-question that retrieved it. That is both an
explanation ("this passage answers two steps") and the cost signal below.

## Cost accounting

The priced unit in the brief is the chunk, so that is what is capped and reported.

| Field | Meaning |
|---|---|
| `chunks_returned` | Every passage handed back, including repeats. What you pay for. |
| `unique_chunks` | Distinct passages in the pool. What you actually learned from. |
| `reuse_rate` | `1 − unique/returned`. High reuse means overlapping sub-questions: a planner defect, paid for twice. |
| `retrieval_calls`, `llm_calls`, `evidence_tokens` | Latency and spend attribution. |

`reuse_rate` earns its place by being actionable: it points at the planner, not the ranker.

## Rejected

- **Whole-pool context for fact extraction.** Extraction runs per sub-question against that
  step's own evidence, capped at ~1,800 tokens. One large prompt over every passage was
  slower and diluted the extraction.
- **Semantic/embedding reranking inside the agent.** Measured on the retrieval gold with no
  gain, and off by default there; adding it here would only add latency.
