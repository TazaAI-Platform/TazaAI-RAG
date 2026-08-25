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
      → extract cited facts (drop any whose figures are not in the cited excerpt)
      → compose the answer from that list only
      → splice any grounded fact the writer dropped
      → claim-level verification + bounded repair loop
```

Writer is `gpt-4o` (`ANSWER_MODEL`), separate from `CHAT_MODEL` (`gpt-4o-mini`) and the
judge (`JUDGE_MODEL`, `gpt-5`). Mini left supported facts unused and mixed figures across
sources; a model scoring its own output inflates itself.

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
Relevance, Completeness and Clarity at 1–3. Writer `gpt-4o` (`ANSWER_MODEL`), judge `gpt-5`
— a different model from the generator, because a model scoring its own output inflates
itself. All 52 gold queries scored, none dropped.

The previous full-set run used a one-shot `gpt-4o-mini` writer (`a1_n52.json`). The shipping
path extracts facts, composes, splices unused grounded cards, then verifies
(`a1_gpt4o52.json`). Live corpus, so this is not a re-score of the same answers.

| | Previous (`mini`, one-shot) | **Shipping (`gpt-4o` + facts + splice)** |
|---|---|---|
| **Accuracy** (gate: all four checks) | 0.538 | **0.712** |
| ├ factual correctness | 0.596 | **0.846** |
| ├ citation integrity | 0.577 | **0.788** |
| ├ no hallucinations | 0.654 | **0.885** |
| └ contextual integrity | 0.673 | **0.846** |
| Relevance (1–3) | 2.42 | 2.44 |
| Completeness (1–3) | 1.71 | **1.79** |
| Clarity (1–3) | **2.88** | 2.71 |
| Overall pass (gate + all dims ≥ 2) | 0.365 | **0.519** |
| Completeness = 1 (worse) | 17 | **13** |
| Median words | 96 | 124 |
| Correct refusal of unanswerable queries | 0.900 | 0.900 |
| False refusal of answerable queries | 0.019 | 0.019 |

These are the **strict** readings. A second judge scores the same *previous* answers far
higher — see section 7 before quoting a level. The 0.29 judge band has not been re-measured
on the new answers.

Accuracy by intent under the shipping path: `industry_scan` and `competitive_intel` at 1.0,
`geographic_assessment` 0.75 (was 0.25), `risk_compliance` 0.67 (was 0.33). Still weak:
`known_item` 0.25, `brand_perception` 0.33. Top failure tag is `missing_narrative` on 45 of
52 — **Completeness, not citation integrity, is still the binding constraint.** Clarity
fell; splicing leftover facts buys coverage as a list, which the judge reads as structure
cost. Hallucination tags 16 → 6; uncited claims 18 → 8.

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
intent floor and rejects degenerate terms. 140 offline tests, none able to reach the network.

## 6. What I tried that did not work

Reported because negative results are cheaper to inherit than to rediscover.

| Attempt | Outcome |
|---|---|
| Embedding/semantic signal in ranking | No measurable gain; off by default |
| Coverage-oriented answer prompt | Completeness +0.19, **Accuracy −0.096** → reverted |
| Per-sentence citation requirement | Recovered citation integrity, lost the Completeness → reverted |
| Whole articles instead of passages | No Completeness gain at higher token cost |
| `top_k=20`, 5,000-token budget | No Completeness gain at higher token cost |

The mechanism behind the two prompt failures constrained the next attempt: citation
integrity is a **per-answer binary gate**, so asking the writer to cover more at once is the
wrong lever. The pipeline change that followed does the coverage work *before* writing.

**Extract cited facts, then compose from that list** (on by default; `--no-facts` ablates it).
A deterministic filter drops any extracted fact whose figures are not in the cited excerpt,
so the writer cannot invent a number extraction never produced. Measured on the first 16 gold
queries against the stored one-shot answers for the same ids (`evals/reports/a1_facts16.json`
vs the first 16 rows of `a1_n52.json`):

| | One-shot write | Extract then compose |
|---|---|---|
| **Accuracy** (gate) | 0.500 | **0.562** |
| ├ each of the four gates | 0.500–0.562 | **0.688** |
| Overall pass | 0.125 | **0.312** |
| Completeness | 1.56 | 1.50 |
| Relevance | 2.31 | 2.44 |
| Median words | 101 | 94 |

Accuracy moved the right way, Completeness did not. That is the opposite of the coverage
prompt, and it is why this path ships: it protects the automatic-fail gate. At n=16 the
Accuracy delta is one query, but all four gates moving together is the part worth believing.
This is not yet a 52-query number.

**Stronger writer + unused-fact splice** (`evals/reports/a1_gpt4o16.json`, same 16 ids,
same judge). Completeness failed because extracted facts never reached the page, not because
retrieval missed them. `gpt-4o` writes from the list; any card it still drops is appended
with its original citation — no new numbers, no new sources.

| | One-shot mini | Extract + mini | Extract + `gpt-4o` + splice |
|---|---|---|---|
| **Accuracy** (gate) | 0.500 | 0.562 | **0.688** |
| ├ factual / citation / no-halluc. | 0.500–0.562 | 0.688 | **0.875** |
| ├ contextual integrity | 0.500 | 0.688 | **0.812** |
| Overall pass | 0.125 | 0.312 | **0.500** |
| Completeness | 1.56 | 1.50 | **1.69** |
| Answer aspect coverage | — | 0.406 | **0.552** |
| Median words | 101 | 94 | 118 |
| `hallucination` tag | — | 5 | **1** |

This is the first time Completeness and Accuracy moved together. The coverage prompt bought
Completeness by asking the writer to invent more; here the extra sentences are pre-grounded
cards. Four queries newly passed the gate and two newly failed — net +2, which at n=16 is
still one-query noise on any single cell, but every Accuracy gate rose and Completeness
finally moved. Confirmed on all 52 (`a1_gpt4o52.json`): Accuracy **0.712**, Completeness
**1.79**, overall pass **0.519**. Completeness still sits below the overall-pass threshold
of 2; `missing_narrative` is on 45 of 52. Clarity is the one dimension that fell (2.88 →
2.71). At n=52 one query is 0.019, so the Accuracy lift versus the previous full set
(+0.173) is nine queries, not noise — with the live-corpus caveat in section 4.

## 7. What I would not claim

**The single most important caveat: the answer-level ruler is less precise than the
differences it is being used to measure.** The same 52 answers, re-scored changing only the
judge model:

| | judged by `gpt-5` | judged by `gpt-4o-mini` | band |
|---|---|---|---|
| **Accuracy** (gate) | 0.538 | 0.827 | **0.288** |
| ├ citation integrity | 0.577 | 0.923 | 0.346 |
| ├ factual correctness | 0.596 | 0.827 | 0.231 |
| ├ no hallucinations | 0.654 | 0.904 | 0.250 |
| └ contextual integrity | 0.673 | 0.827 | 0.154 |
| Relevance | 2.42 | 2.83 | 0.404 |
| Completeness | 1.71 | 2.00 | 0.288 |
| **Overall pass** | 0.365 | 0.827 | **0.462** |

The answers are byte-identical; only the grader changed. The two judges score 41 of 52
queries differently. That band was measured on the previous `mini` one-shot answers, not
on `a1_gpt4o52.json`. So **a level from a single judge should not be quoted as "the"
accuracy**, and the 0.29 gap is still three times the 0.096 effect a prompt change was
reverted over. Human raters are what would collapse it.

This does not invalidate the A/B comparisons in section 6, which hold the judge fixed across
both arms and so measure a delta rather than a level. It does mean **no single level in
section 4 should be quoted as "the" accuracy.** The conservative number is reported throughout
this document because a strict judge is the safer assumption for a Dow Jones product, not
because it is known to be the true one.

Everything in section 3 is free of this problem: retrieval metrics are deterministic string
and rank computations with no model in the loop, which is why I trust them and lead with them.

- **The corpus is live**, so retrieval figures move run to run, and roughly one query in
  twenty-five fails upstream even after retries.
- **There is no scale story in code.** No pgvector adapter, no MCP surface. Deliberate — the
  current path retrieves per query from Factiva, so a vector store would hold nothing until an
  owned corpus (News Feed / Streams) is ingested — but it is a gap, not a solved problem.
- **Answer quality is not production-ready.** Under the strict judge seven in ten answers
  clear the Accuracy gate and overall pass is still half, because Completeness averages 1.79
  against a threshold of 2. A Dow Jones audience still needs a human in the loop.

## 8. What I would do next, in order

1. **Human-calibrate the judge** on 20 answers against the A1 rubric. This is first by a wide
   margin. Section 7 shows the judge band (0.29 on Accuracy) is three times the size of the
   effects I was tuning against, so until a human anchors it, further answer-side work is
   fitting to a ruler whose graduations are wider than the differences being chased. Twenty
   scored answers would collapse that band and make every subsequent experiment meaningful.
2. **Ingest News Feed / Streams into pgvector** — the point at which the owned-corpus and
   scale story becomes real rather than described.
3. **Expose retrieval as an MCP tool**, matching Taza's agent-facing boundary so retrieval
   stays a metered, callable service rather than something an agent reaches around.

## 9. Reproducing

```bash
taza-rag eval-retrieve --top-k 10 --compare            # retrieval vs raw baseline
taza-rag eval-a1 --verify --top-k 10                   # A1 gate + dimensions (facts on)
taza-rag eval-a1 --verify --no-facts --top-k 10        # prior one-shot writer, for ablation
taza-rag eval-a1 --gold evals/gold/factiva_abstain_v1.jsonl   # refusal behaviour
taza-rag rejudge-a1 --source <report> --judge-model gpt-4o-mini  # judge disagreement
python scripts/validate_gold.py                        # gold labels are satisfiable
python scripts/run_tests.py                            # 140 offline tests, no network
```

Stored reports: `evals/reports/retrieval_n52.json`, `a1_n52.json` (previous writer),
`a1_gpt4o52.json` (shipping writer), `abstain_n10.json`, `a1_facts16.json` /
`a1_gpt4o16.json` (16-id ladder), plus the two reverted prompt experiments
(`a1_n52_coverage.json`, `a1_n52_v3.json`) so the negative results can be checked rather
than taken on trust.
