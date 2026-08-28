# Multi-step research agent — Option 2

Takes a complex question, decomposes it into a research plan, searches Factiva in parallel,
decides whether what came back is enough, and writes an answer grounded in the sources.

Two pages. Detail lives in the [appendix](docs/agent/), one file per thing the brief said it
would judge. Retrieval-side work (ranking, contextual passages, the single-question A1 results)
is in the main [README](README.md) and [DECISION.md](DECISION.md).

```
question
  → plan            2–5 sub-questions, each with its own completion criteria     [A1]
  → gather          issued in parallel into one globally labelled evidence pool  [A2]
  → extract         grounded facts per step, in parallel; ungrounded figures dropped
  → assess          measure coverage; refine only what is missing, or stop and say why  [A4]
  → combine         compose from facts, attribute disagreements, declare gaps     [A3][A5]
  → verify          claim-level checks against cited excerpts, bounded repair
```

Retrieval sits behind a `SearchBackend` interface, so the agent's decisions stay separable
from the ranking work and the whole loop is testable offline against a fixed corpus. That
mirrors Taza's own principle of a clean boundary between agent intelligence and the
deterministic request path.

## Results

12 multi-hop gold questions, `top_k=6`, 3 rounds, 40-passage budget, writer `gpt-4o`, judge
`gpt-5`. Every question scored; none dropped upstream. Report:
`evals/reports/research_n12.json`.

| Deterministic | Value | |
|---|---|---|
| Answer aspect coverage | **0.688** | share of gold aspects the finished answer delivers |
| Plan disjointness | **0.849** | share of passages only one sub-question found |
| Calibration error (signed) | **−0.032** | self-assessed coverage minus delivered |
| Calibration error (absolute) | **0.255** | per-question, and the number that matters |
| Cost per covered aspect | **10.0** | unique passages bought per aspect delivered |
| Passage reuse rate | 0.186 | overlap between sub-questions |
| Rounds / unique passages | 1.83 / 26.7 | |
| Unsupported claims after repair | 5 across 12 answers | |
| Stop reasons | plateau 6, target 5, round cap 1 | |
| Median latency | 65 s | |

Judge, second and with the judge named: Accuracy **0.250**, Relevance 2.00, Completeness 1.50,
Clarity 2.00, overall pass 0.083. An earlier run of a near-identical config scored Accuracy
0.417 — at n=12 one question is worth 0.083, so **the honest statement is 0.25–0.42 and no
level should be quoted.** The deterministic numbers above are the ones I trust.

## The stopping rule

The brief singles out *how it knows when it has enough*, so that is where the design is.

Asking a model "are you done?" is not an answer — it cannot know what it has not seen, it
agrees under pressure, and its verdict cannot be audited. Instead the **plan declares its own
completion criteria up front** as aspects, and coverage is measured against them by whole-term
matching with no model in the loop.

Five labelled exits, checked in order: `target_coverage`, `round_cap`, `budget_chunks`,
`plateau`, `nothing_left_to_ask`. Caps are reported **before** plateau so a run that was cut
off is never mistaken for one that converged — they look identical in a finished answer and
mean opposite things about raising the budget.

`plateau` is the interesting one: a round that pays for passages and either returns no new
grounded fact, or returns facts that move no remaining aspect, is the point where retrieval has
stopped changing the answer. It fired on 6 of 12 questions. That is the empirical version of
"enough", and it is the cost discipline a metered corpus demands. Full detail in
[Appendix 4](docs/agent/04-sufficiency.md).

## What I would not claim

- **The agent grades itself against criteria it wrote itself.** Mean calibration error is
  −0.03, which looks excellent and hides the real number: 0.255 absolute. One question stopped
  at `target_coverage` after one round with a 66-word answer covering half the gold aspects;
  another stopped on `plateau` believing it had 0.25 while delivering 1.00. Self-declared
  sufficiency is the right mechanism and it is not yet trustworthy per question.
- **A1 on multi-hop questions is not comparable to the 0.712 on single questions.** Different,
  harder gold, and answers are 218 words against 124. Citation integrity is a per-answer gate,
  so more claims mean more chances to fail it — a mechanism already measured on the
  single-question path.
- **n = 12, live corpus.** Enough to find defects. Not enough to rank configurations.
- **The judge is uncalibrated against humans.** Same limitation, same fix, as the retrieval work.

Two defects were found by running it rather than reviewing it: aspects the planner phrased so
abstractly they could never be satisfied (coverage stuck at 0.444 while the agent re-asked for
material it already had), and conflict detection that reported **nine fabricated
disagreements** on one Airbus/Boeing question because two companies described in identical
words looked like one subject. Both are fixed and both now have tests.
[Appendix 7](docs/agent/07-limitations.md) has the rest, plus what I would do next.

## Agent-facing surface

The commercial boundary is MCP (`taza-rag mcp`), matching the product loop on
app.tazalabs.ai: **`query` is free**, **`transact` pays**, **`fetch_content` is
the only tool that returns bodies**. Packages are opaque handles labelled with a
closed tradeoff vocab (cheapest, densest, token_constrained, most_thorough,
balanced). Every result carries the same `usage` block — offered, bought,
refused, cited. No LLM sits on this path; ranking is the Factiva quality stack.
The research agent is a *client* of the loop (CLI / UI), not an MCP tool.

The query playground walks that loop: send a task, pick a package, fetch what
you bought. Ranking knobs stay under Advanced.

## Run it

```bash
taza-rag research "How exposed is SoftBank Group to its AI bets, and what do its numbers say?"
taza-rag research "..." --no-llm-plan          # heuristic plan, no planner call
taza-rag research "..." --max-rounds 1         # ablate the refinement loop
taza-rag eval-research                          # 12 questions, deterministic + A1
taza-rag eval-research --limit 4 --no-judge     # fast deterministic-only pass
taza-rag ui                                     # query playground (query → package → fetch)
taza-rag mcp                                    # stdio MCP: query / transact / fetch_content
python scripts/run_tests.py                     # offline tests, network blocked
```

Needs `OPENAI_API_KEY` and Factiva credentials in `.env` (see the main README). The
retrieval-only path still runs without a key.

## Appendix

| # | | Brief's question |
|---|---|---|
| A1 | [Planning](docs/agent/01-planning.md) | how it decides what work needs to happen |
| A2 | [Retrieval](docs/agent/02-retrieval.md) | how it retrieves information |
| A3 | [Conflicts and gaps](docs/agent/03-conflicts.md) | how it handles conflicting or incomplete sources |
| A4 | [Sufficiency](docs/agent/04-sufficiency.md) | how it knows when it has enough |
| A5 | [Synthesis](docs/agent/05-synthesis.md) | how it combines results |
| A6 | [Evaluation](docs/agent/06-evaluation.md) | how answer quality is evaluated |
| A7 | [Limitations](docs/agent/07-limitations.md) | what I would not claim, and what is next |
