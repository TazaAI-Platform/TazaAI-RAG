# Appendix 7 — Limitations, rejected designs, and what I would do next

Referenced from [AGENT.md](../../AGENT.md) § What I would not claim.

## Limitations, stated plainly

- **Self-declared aspects are the weakest link.** The agent measures coverage against criteria
  its own planner wrote, so a plan that declares easy aspects can satisfy them and stop early.
  Observed directly: one question stopped at `target_coverage` after a single round with a
  66-word answer that covered half the gold aspects. Mean calibration error is −0.03, which
  flatters it; the per-question absolute error is 0.255.
- **`max_unique_chunks` stops the next round, it is not a hard ceiling.** The check happens
  between rounds, so a round already in flight can overshoot — one question finished on 47
  unique passages against a 40 budget. Capping mid-wave would mean discarding passages already
  paid for, which is worse.
- **`depends_on` orders steps but does not rewrite queries.** A dependent sub-question is
  issued with its original wording even after its prerequisite has been answered. Most plans
  declare no dependencies, so this has not been the binding constraint, but a question like
  "who runs the Vision Fund, and what have they said?" would benefit.
- **The judge is uncalibrated.** Levels from a single judge should not be quoted as *the*
  accuracy. Paired deltas with the judge held fixed remain valid.
- **n = 12 gold questions.** Enough to find defects — it found two — not enough to rank
  configurations.
- **The corpus is live**, so retrieval moves run to run and a few percent of calls fail
  upstream even after retries.
- **Conflict detection is conservative by design.** It will miss real disagreements that are
  phrased differently enough to fall below the subject-overlap threshold, or that are
  denominated differently. This is the chosen error direction, not an oversight.
- **Structural aspect predicates are keyword lists.** They will miss an attribution phrased
  unusually, and a contrary view expressed without any contrast marker.
- **No cost model in currency.** Chunks are counted, not priced. Real pricing needs the
  marketplace's rate card.
- **Latency is measured, not optimised.** Roughly 80–90 s per question at three rounds, and
  fact extraction dominates. It is serial by round, by construction.

## Rejected designs

| Attempt | Why it was rejected |
|---|---|
| Ask the model "do you have enough?" | Cannot know what it has not seen, agrees under pressure, unauditable. Replaced by measured coverage ([Appendix 4](04-sufficiency.md)). |
| Instruct the writer to cover more | Measured twice on 52 questions: Completeness +0.19, **Accuracy −0.096**. Citation integrity is a per-answer gate, so extra claims are extra chances to fail. |
| LLM conflict detection | Unauditable, and its false positives are the expensive kind. |
| Auto-resolve conflicts to the top-authority source | Cheap and wrong; the disagreement is often the most informative thing in the evidence. |
| Unit conversion across currencies | Needs FX at each article's publication date. Treating denominations as non-comparable costs only conflict recall. |
| LLM-rewritten refinement queries | Extra call, extra failure mode, no measured gain over `entity + aspect`. |
| Per-section labels `c1..cN` | Collided: two sections both had a `c1`, so citations could not be resolved. |
| Per-section sub-answers stitched together | Seams show, compose calls multiply. |
| Headings and bullets in the answer | Cheap apparent structure; list-shaped answers measurably cost Clarity. |
| Deep hierarchical plans | No gold question needed a second level; cost multiplies. |
| Embedding rerank inside the agent | Already measured as no-gain on the retrieval gold. |

## Defects found by measurement, not review

All four were found by running the thing, which is the argument for building the eval early.

1. **Unsatisfiable aspects.** The planner emitted "official comment" and "dissenting view".
   Those words never appear in a news story, so coverage was permanently understated at 0.444
   and the agent spent every remaining round re-asking for material it already had. Fixed two
   ways: structural predicates for abstract aspects, and explicit good/bad examples in the
   planner prompt.
2. **Plateau that only looked at fact counts.** A round returned nine new facts and moved
   coverage 0.00 — spending with no progress, which the original rule did not catch. Plateau
   now also compares coverage against the previous round.
3. **Nine fabricated disagreements on one question.** The Airbus/Boeing comparison produced
   facts like "Airbus delivered 60 aircraft in July" and "Boeing delivered 45 aircraft in
   July". Strip the digits and every remaining word matches, so the subject-overlap guard read
   two companies as one subject. Fixed with an actor guard: when both facts name entities and
   those names do not intersect, the numbers are not comparable. Total disagreements across the
   gold set fell from 11 to 2. Month names had to be excluded from the entity comparison, or
   "July" alone made every pair look like the same actor.
4. **A metric that measured my own vocabulary.** "Plan facet coverage" compared gold facet
   labels against sub-question wording and scored 0.194 on plans that were plainly correct —
   the agent asked "What financial results has SoftBank Group reported?" against a gold facet I
   had written as "quarterly earnings". The match is semantic and no lexical threshold recovers
   it, so the metric was deleted rather than kept and explained away. Plan quality is now read
   from **plan disjointness** (0.849: the steps do look at genuinely different evidence) and
   from what the answer delivers.

## What I would do next, in order

1. **Human-calibrate the judge** on ~20 answers. Highest value by a wide margin; without it
   every answer-side comparison is fitted to a ruler whose graduations are wider than the
   effects being measured.
2. **Expand the gold set** to ~40 multi-hop questions across all ten Factiva intents, matching
   what was done for retrieval — where expanding from 16 to 52 showed the small set had been
   *flattering*, not merely noisy.
3. **Ablate the refinement loop** (`--max-rounds 1` vs `3`) on the expanded gold. The loop's
   cost is known; its benefit is not yet quantified.
4. **Use dependency results to rewrite dependent queries**, closing the gap above.
5. **Price the budget in currency** rather than chunks, and let the stopping rule take an
   expected-value decision: buy the next passage only when it is likely to move an uncovered
   aspect. That is the "value before access" problem, and this loop's per-round record of
   spend against coverage gain is the dataset it would learn from.
6. **Expose the agent as an MCP tool** so retrieval and research are metered, callable services
   at Taza's agent-facing boundary.
