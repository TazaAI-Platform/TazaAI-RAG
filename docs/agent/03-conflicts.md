# Appendix 3 — Conflicting and incomplete sources

Referenced from [AGENT.md](../../AGENT.md) § Architecture.

## The asymmetry that shapes the design

A research answer over a news archive will pull the same event from a wire, a national paper
and an aggregator. Their numbers rarely match exactly. Two reasons, opposite treatment:

- **Rounding** — one outlet writes an 18% fall, another 17.7%. Reporting that as a dispute is
  itself a factual error, and A1 fails an answer for exactly that distortion.
- **Disagreement** — materially different values. Silently picking one is the failure the
  brief asks about.

A false positive is worse than a miss, because it puts a fabricated dispute in front of a Dow
Jones reader. So `taza_rag/agent/conflict.py` is deliberately conservative, and every guard is
deterministic — a model asked "do these disagree?" can invent a conflict; a string comparison
cannot.

## The four guards

A pair of facts is only compared when all four hold:

1. **Different documents.** Two passages of one article restating a figure is not a conflict
   between sources.
2. **Same actor.** When both facts name entities and those names do not intersect, they are
   about different companies and their numbers are not comparable. This guard was added after
   measurement, not design: the Airbus/Boeing question produced **nine fabricated
   disagreements**, because "Airbus delivered 60 aircraft in July" and "Boeing delivered 45
   aircraft in July" are identical once the digits are stripped. Month names are excluded from
   the entity comparison, or "July" alone makes every pair look like the same actor. Total
   disagreements across the gold set fell from 11 to 2.
3. **Same subject** — ≥ 0.45 stemmed-term Jaccard overlap after stripping digits, so wording
   is compared rather than numbers.
4. **Same denomination.** "347.33 billion yen" and "$2.2 billion" are one fact twice, not a
   conflict. A percentage and an absolute amount are different measures of one event.

When either fact names no entity at all, the actor guard falls through to the overlap test
rather than guessing.

Then the numbers are compared, with **years excluded** — "this year" is routinely paraphrased,
so a year mismatch is weak evidence of a dispute, the same reasoning the verifier uses in
treating bare years as non-blocking.

## Rounding tolerance

Numbers within **2%** relative are a rounding restatement. Calibration:

| Pair | Relative difference | Verdict |
|---|---|---|
| 347.33 vs 347 | 0.09% | rounding |
| 18 vs 17.7 | 1.7% | rounding |
| 8.2 vs 8.5 | 3.7% | disagreement |

2% was chosen so that 18/17.7 is rounding while 8.2/8.5 is not. A wider tolerance starts
hiding real disputes; a narrower one starts manufacturing them out of house style.

## Resolution

`_prefer()` picks which side leads: **source authority first, publication date as tie-break** —
the same priors the ranker already uses, so the agent's preference is explainable in the same
terms as its retrieval.

The non-preferred side is **never dropped.** The composer is handed both values, both source
names, and which one to lead with, and is instructed to attribute each and never to average
two figures. Rounding restatements are filtered out of that brief entirely, since they are
noise to the reader.

## Incomplete sources

Handled as the mirror image. Any aspect the plan asked for that no grounded fact satisfies
becomes a `Gap` (see [Appendix 4](04-sufficiency.md)), and gaps are passed to the composer
under `DO NOT COVER` with an instruction to state plainly that the sources do not address
them, without speculating or padding.

Declaring a gap is the honest alternative to silent omission, and it is what the rubric's
intellectual-honesty dimension rewards. It also means a low-coverage run produces a visibly
partial answer rather than a confident-sounding thin one.

## Measured

Across the 12-question gold set: **2 disagreements surfaced, 1 rounding restatement filtered,
37 gaps declared**. The gap count being an order of magnitude larger than the conflict count is
the expected shape on a news archive — incompleteness is the common case, contradiction the
rare one.

## Rejected

- **LLM conflict detection.** Cannot be audited, and its false positives are precisely the
  expensive kind.
- **Auto-resolving to the highest-authority source.** Cheap and wrong: the disagreement is
  often the most informative thing in the evidence.
- **Unit conversion to compare across currencies.** Would need live FX rates at each article's
  publication date. Treating differing denominations as non-comparable is the conservative
  choice and costs only recall on conflict detection.
