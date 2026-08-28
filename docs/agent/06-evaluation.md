# Appendix 6 — Evaluating answer quality

Referenced from [AGENT.md](../../AGENT.md) § Evaluation.

## Why the deterministic metrics lead

On the single-question path, re-scoring the **same 52 answers** with a different judge moved A1
Accuracy from 0.538 to 0.827 — 41 of 52 questions scored differently, not a byte of the answers
changed. A ruler with a 0.29 band cannot measure a 0.10 effect.

So this harness leads with string and rank computations that have no model in the loop, and
reports the judge second. `taza_rag/eval/research.py`, gold at
`evals/gold/research_v1.jsonl` (12 multi-hop questions, each needing 2–4 facets that no single
retrieval covers).

## The metrics

| Metric | n=12 | What it answers |
|---|---|---|
| **answer aspect coverage** | 0.688 | Does the finished answer deliver the gold aspects? Judged against the answer text, not the evidence pool — retrieving material and then failing to write it is a real and previously measured defect. |
| **plan disjointness** | 0.849 | Share of passages only one sub-question found. Near 1.0 means the steps looked at genuinely different evidence; near 0.0 means the plan was paraphrases and the run paid several times for one set of articles. |
| **calibration error** | −0.032 signed, 0.255 absolute | Self-assessed coverage at the stop minus delivered coverage. Signed, so over- and under-confidence are distinguishable. |
| **cost per covered aspect** | 10.0 | Unique passages bought per aspect delivered. In a marketplace that bills per chunk, an answer is not better for costing more. |
| **passage reuse rate** | 0.186 | Overlap between sub-questions. Points at the planner, not the ranker. |
| **stop-reason distribution** | plateau 6, target 5, cap 1 | How often runs converge vs. hit a cap. Caps and convergence mean opposite things about raising the budget. |
| **unsupported claims after repair** | 5 / 12 answers | Deterministic grounding residue from the verifier. |
| A1 Accuracy / Relevance / Completeness / Clarity | 0.250 / 2.00 / 1.50 / 2.00 | Reported, with the judge named, second. |

**Calibration error is the metric that matters most here**, because it is the only one that
says whether the stopping rule can be trusted — and an agent with no declared completion
criteria cannot produce it at all.

Read the signed and absolute versions together, or it flatters. Signed −0.032 suggests an
almost unbiased self-assessment; the 0.255 absolute error says the per-question estimate is
noisy in both directions. One question stopped at `target_coverage` after one round with a
66-word answer covering half the gold aspects; another stopped on `plateau` believing it had
0.25 while delivering 1.00.

## Matching rules

Aspect matching uses the same whole-term, stem-insensitive, majority-of-terms logic as the
agent's own coverage check ([Appendix 4](04-sufficiency.md)), including the structural
predicates.

Reusing the agent's matcher in its own eval is a real tension, and it is bounded honestly:
gold aspects are written independently of any run, so the matcher is being asked "does this
answer contain this pre-declared thing", not "did you succeed". It does mean a matcher bug
would flatter both numbers at once, which is why the earlier substring bug — `"AI"` matching
inside `"said"` — is called out in the main README and guarded by tests. Whole-term matching is
not a stylistic preference; it fixed metrics that were scoring gold terms for free.

## What is deliberately not scored

**Conflicts surfaced** is reported as an observed count, not graded. Grading it would require
knowing in advance that the live corpus contains a disagreement on a given question, which is
not knowable — the corpus moves. Reporting a count that cannot be graded is honest; inventing
a gold label for it would not be.

## Running it

```bash
taza-rag eval-research                                  # 12 questions, deterministic + A1
taza-rag eval-research --limit 4 --no-judge              # fast deterministic-only pass
taza-rag eval-research --max-rounds 1                    # ablate the refinement loop
taza-rag eval-research --judge-model gpt-4o-mini         # judge band, not system quality
python scripts/run_tests.py                              # 195 offline tests, network blocked
```

Ablating with `--max-rounds 1` is the cleanest way to ask whether the loop earns its cost: it
holds retrieval, synthesis and judge fixed and removes only the refinement rounds.

## Caveats

- **The corpus is live.** Two runs of the same question are not the same experiment. Paired
  A/B with the same gold and the same day is the only fair comparison.
- **n = 12.** One question is worth 0.083, so a single-question difference is not a result. Two
  near-identical configurations scored A1 Accuracy 0.417 and 0.250 on consecutive days, which
  is the whole argument for leading with the deterministic metrics.
- **A1 here is not comparable to the 0.712 on single questions.** Different, harder gold set,
  and answers run 218 words against 124. Citation integrity is a per-answer gate, so more
  claims mean more chances to fail it.
- **The judge is not calibrated against humans.** Same limitation as the retrieval work, and
  the same fix: score ~20 answers by hand against the A1 rubric before trusting a level.
