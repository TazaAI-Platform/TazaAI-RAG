# Taza RAG trial — decision document

A record of what was built, what it measures, what I got wrong, and what I would do next.
Every figure here comes from a stored report in `evals/reports/` and can be reproduced with
the commands at the end.

---

## 1. The choice

The brief offered several directions. I took **contextual retrieval over Factiva** because it
is closest to Taza's existing stack and because retrieval quality is the substrate everything
else rests on — metering, attribution and value-before-access are all meaningless if the
wrong chunk is returned.

One line in the brief shaped every design decision that followed: *every retrieved chunk has
an economic cost.* In a marketplace that pays per chunk, **precision beats coverage**, and a
system that answers well by retrieving twice as much is not a better system. So the headline
metric is not coverage but coverage per token of evidence.

## 2. What ships

```
query → intent detection → query expansion (3 variants)
      → parallel Factiva Retrieval API calls
      → contextual passage split (Anthropic-style prefixes)
      → RRF fusion → relevance tiering → MMR diversification → source capping
      → token-budgeted evidence pack
      → grounded answer with citations
      → claim-level verification + bounded repair loop
```

Ranking is deliberately non-LLM: entity and topic signals, source authority, freshness,
document-type penalties, light stemming, near-duplicate collapse. It is fast, explainable,
and every stage can be ablated from the CLI. An embedding signal was implemented and measured;
it showed no gain on this gold set and is off by default.

## 3. Retrieval quality — the core result

52 gold queries spanning all ten Factiva search intents, live API, `top_k=10`. Baseline is a
single Factiva call in API order. Quality is the full stack.

| Metric | Baseline | Quality | |
|---|---|---|---|
| `aspect_coverage@5` | 0.510 | **0.615** | +0.106 |
| `entity_precision@5` | 0.831 | **0.915** | +0.084 |
| `salience@5` | 0.894 | **0.939** | +0.045 |
| `noise_rate@5` (lower better) | 0.043 | **0.008** | 5× less noise |
| `evidence_tokens@5` (lower better) | 2,057 | **1,243** | **40% less evidence** |
| `aspect_coverage@1200tok` | 0.660 | 0.683 | +0.023 |
| `term_hit_lead@3` | **0.971** | 0.962 | −0.009 |

**The result that matters: more of the angles that matter, from 40% less text.** Coverage per
1,000 tokens of evidence doubles, from 0.25 to 0.50. For a per-chunk cost model that is the
whole argument.

`term_hit_lead@3` is the one metric where the baseline is marginally ahead. It is under half a
query out of 52 and is reported rather than explained away.

## 4. Answer quality — measured against the A1 rubric

Dow Jones A1 implemented as specified: Accuracy as a four-part automatic-fail gate, then
Relevance, Completeness and Clarity at 1–3. Generator `gpt-4o-mini`, judge `gpt-5` — a
different model from the generator, because a model scoring its own output inflates itself.

| | Score |
|---|---|
| **Accuracy** (gate: all four checks) | 0.538 |
| ├ factual correctness | 0.596 |
| ├ citation integrity | 0.577 |
| ├ no hallucinations | 0.654 |
| └ contextual integrity | 0.673 |
| Relevance (1–3) | 2.42 |
| Completeness (1–3) | 1.71 |
| Clarity (1–3) | 2.88 |
| Overall pass (gate + all dims ≥ 2) | 0.365 |
| Correct refusal of unanswerable queries | 0.900 |
| False refusal of answerable queries | 0.019 |

Median latency 19.2s per verified answer, p90 30.0s.

Accuracy varies more by intent than by anything I changed: `industry_scan` and
`event_tracking` reach 0.75, while `geographic_assessment` sits at 0.25 and `risk_compliance`
and `brand_perception` at 0.33. Top failure tag is `missing_narrative` on 41 of 52 answers —
**Completeness, not citation integrity, is the binding constraint.**

Claim verification does most of the mechanical work it promises: 118 unsupported claims
flagged across 52 answers, 15 surviving repair, 37 of 52 answers fully clean at exit, at 1.35
repair calls per answer.

## 5. What I got wrong

This is the section I would most want a reviewer to read, because the system is only as
trustworthy as the instruments measuring it — and mine were wrong repeatedly.

| Defect | Effect on reported numbers |
|---|---|
| Gold terms matched as substrings | `"AI"` matched "s**ai**d", `"EV"` "s**ev**en", `"AWS"` "l**aws**" — terms scored regardless of retrieval |
| Judge saw truncated evidence | Supported claims marked unsupported; Accuracy understated |
| Re-judge path dropped citations | Citation integrity failed on correctly cited answers |
| Citations required per sentence | Flagged normal claim-group prose; fired on all 16 answers and forced pointless rewrites |
| Repaired answers marked as refusals | Reported 25% abstention where the true rate was 1.9% |
| Two tests passed on swallowed network calls | Assertions never ran |
| Rich console ate `[c1]` markers | The demo path displayed correctly cited answers as uncited |
| One 429 aborted a whole eval run | 45 minutes of work discarded; exception caught but never imported |

The gold set was also expanded from 16 to 52 queries, and **the smaller set had been
flattering, not merely noisy**: Accuracy fell from 0.688 to 0.538 once the harder intents were
included. Reporting the lower number is the point of measuring.

These classes are now caught mechanically rather than by attention: the test runner blocks
network access, static checks reject an exception caught without being imported, a handler
that discards its error, or model text printed as markup, and a lint enforces the gold set's
intent floor and rejects degenerate terms. 126 offline tests, none able to reach the network.

## 6. What I tried that did not work

Reported because negative results are cheaper to inherit than to rediscover.

| Attempt | Outcome |
|---|---|
| Embedding/semantic signal in ranking | No measurable gain; off by default |
| Coverage-oriented answer prompt | Completeness +0.19, **Accuracy −0.096** → reverted |
| Per-sentence citation requirement | Recovered citation integrity, lost the Completeness → reverted |
| Whole articles instead of passages | No Completeness gain at higher token cost |
| `top_k=20`, 5,000-token budget | No Completeness gain at higher token cost |

The mechanism behind the two prompt failures constrains any future attempt: citation
integrity is a **per-answer binary gate**, so every additional claim is another chance to fail
it. Going from 96 to 131 median words took judge-observed citation failures from 22 to 27 of
52. Coverage is purchasable with prompt wording; passing the gate while carrying it is not.
Broader answers need per-claim verification strong enough to support them — a verification
problem, not a wording one.

## 7. What I would not claim

- **No A1 number here is calibrated against a human.** Two model judges scoring the identical
  answers agree on 31% of queries: `gpt-4o-mini` rates overall pass at 0.875 where `gpt-5`
  says 0.438. The judge is a larger source of variance than most changes I could make.
- **The corpus is live**, so retrieval figures move run to run, and roughly one query in
  twenty-five fails upstream even after retries.
- **There is no scale story in code.** No pgvector adapter, no MCP surface. Deliberate — the
  current path retrieves per query from Factiva, so a vector store would hold nothing until an
  owned corpus (News Feed / Streams) is ingested — but it is a gap, not a solved problem.
- **Answer quality is not production-ready** at 0.538 on a hard accuracy gate.

## 8. What I would do next, in order

1. **Human-calibrate the judge** on 20 answers against the A1 rubric. Until this exists, every
   answer-level number is soft, and no amount of tuning fixes that.
2. **Strengthen per-claim verification** so an answer can carry more claims without exposing
   the citation gate — the only route to Completeness that the measurements have not already
   closed off.
3. **Try a stronger generator.** `gpt-4o-mini` is the cheapest part of the pipeline and, on
   this evidence, the limiting one.
4. **Ingest News Feed / Streams into pgvector** — the point at which the owned-corpus and
   scale story becomes real rather than described.
5. **Expose retrieval as an MCP tool**, matching Taza's agent-facing boundary so retrieval
   stays a metered, callable service rather than something an agent reaches around.

## 9. Reproducing

```bash
taza-rag eval-retrieve --top-k 10 --compare            # retrieval vs raw baseline
taza-rag eval-a1 --verify --top-k 10                   # A1 gate + dimensions
taza-rag eval-a1 --gold evals/gold/factiva_abstain_v1.jsonl   # refusal behaviour
taza-rag rejudge-a1 --source <report> --judge-model gpt-4o-mini  # judge disagreement
python scripts/validate_gold.py                        # gold labels are satisfiable
python scripts/run_tests.py                            # 126 offline tests, no network
```

Stored reports: `evals/reports/retrieval_n52.json`, `a1_n52.json`, `abstain_n10.json`, plus
the two reverted prompt experiments (`a1_n52_coverage.json`, `a1_n52_v3.json`) so the
negative results can be checked rather than taken on trust.
