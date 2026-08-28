# Appendix 4 — Sufficiency: knowing when it has enough

Referenced from [AGENT.md](../../AGENT.md) § Architecture and § Stopping.

This is the part of the brief most easily skipped, so it is the part with the most design in
it.

## Why not ask the model

"Do you have enough information?" fails three ways: the model cannot know what it has not
seen, it agrees under pressure, and its verdict cannot be audited afterwards. So sufficiency
here is **measured**.

## Coverage

The plan declares aspects up front ([Appendix 1](01-planning.md)). Coverage is the share of
those aspects that some grounded fact satisfies, averaged over sub-questions.

Coverage is judged against **every fact in the run**, not only the facts retrieved for that
sub-question: the pool is shared, so a passage found while asking about borrowing can
legitimately answer an aspect of the earnings step. Attribution to a sub-question is still
kept, because it is what tells a refinement round *what* to go and ask for.

## Two kinds of aspect

`taza_rag/agent/aspects.py`. Most aspects are **lexical** — "record retail bond issuance" is
wording that appears in the copy, so whole-term stem-insensitive overlap settles it, with a
majority-of-terms threshold (a single common term like "profit" must not mark an aspect
covered).

Some are not, and a live probe made the cost obvious. The planner asked for an "official
comment" and a "dissenting view" — categories a complete answer genuinely needs, and which A1
explicitly rewards — but those words never appear in a news story. Judged lexically they could
never be satisfied, so **coverage sat at 0.444 while the agent burned every remaining round
re-asking for material it already had.**

So an aspect built only from generic vocabulary is treated as a **structural** requirement and
checked by a predicate over the facts instead:

| Category | Satisfied by |
|---|---|
| figure | any fact containing a number |
| attribution | `said / told / announced / according to / spokesman …` |
| contrary view | `however / but / critics / warned / risk / concern …` |
| timing | a year, a month, or a quarter |

An aspect carrying any distinctive term keeps the lexical path, because that is the stronger
signal when it is available: "OpenAI stake borrowing figures" must not be downgraded to "any
number will do". Guarded across both directions in `tests/test_agent_aspects.py`.

The planner prompt was also tightened with explicit good/bad aspect examples, so the common
case is fixed at the source and the structural path is the safety net.

## The five exits

Checked in this order, and the order is deliberate:

| # | Reason | Meaning |
|---|---|---|
| 1 | `target_coverage` | Coverage ≥ 0.8. The question is answered. |
| 2 | `round_cap` | Out of rounds. **Not** convergence. |
| 3 | `budget_chunks` | Out of passage budget. **Not** convergence. |
| 4 | `plateau` | Spending has stopped moving the answer. |
| 5 | `nothing_left_to_ask` / `no_new_query` | No uncovered aspect, or every follow-up was already issued. |

Caps are reported **before** plateau so a run that was cut off is never labelled as one that
converged. This matters: those two look identical in a final answer and mean opposite things
about whether raising the budget would help.

## Plateau, in detail

Two ways to be stuck, and the second was found by measurement:

- **No new grounded facts.** A round that bought passages and produced nothing new.
- **No coverage gain.** A round that produced plenty of facts which happen not to address
  anything still missing — coverage flat while the bill grows. The first live run did exactly
  this, so coverage is compared against the previous round's recorded value rather than
  trusting fact counts alone.

The comparison uses the previous round's coverage, not the current record's `coverage_delta`,
because the caller only fills that in after the decision is made. Getting this wrong would
have made plateau fire on every round.

Plateau is the empirical answer to "enough": the point where additional retrieval stops
changing the answer. It is also the cost discipline the marketplace demands — chunks that will
not change the answer should not be bought.

## Refinement

A continuing round does **not** re-ask the plan. It asks only for uncovered aspects, ordered
so the least-covered steps are served first, capped at three queries, anchored on the primary
entity so a bare aspect does not retrieve the whole market, and skipping any query already
issued. Guarded by `test_a_missing_aspect_triggers_a_second_round_that_asks_only_for_it` and
`test_a_query_already_issued_is_not_paid_for_twice`.

## Is the stopping rule any good?

The eval reports **calibration error**: self-assessed coverage at the stop minus the coverage
the answer actually delivers against gold. Signed, so over-confidence and under-confidence are
distinguishable. An agent without declared criteria cannot report this number at all.

Measured on 12 questions: **−0.032 signed, 0.255 absolute**, with exits distributed plateau 6,
`target_coverage` 5, `round_cap` 1. Read together those say the mechanism works and the
estimate is not yet reliable per question — near-unbiased on average, noisy in both directions
individually. The known cause is structural: **the agent measures itself against aspects its
own planner wrote**, so a plan that declares easy criteria can satisfy them and stop early. One
question stopped at `target_coverage` after a single round with a 66-word answer covering half
the gold aspects. See [Appendix 6](06-evaluation.md) and
[Appendix 7](07-limitations.md).

One more honest detail: `max_unique_chunks` is checked *between* rounds, so it stops the next
round rather than acting as a hard ceiling. One question finished on 47 unique passages against
a 40 budget. Capping mid-wave would mean discarding passages already paid for.
