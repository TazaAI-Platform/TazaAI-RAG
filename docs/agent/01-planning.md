# Appendix 1 — Planning: deciding what work needs to happen

Referenced from [AGENT.md](../../AGENT.md) § Architecture.

## What a plan is

`taza_rag/agent/plan.py` turns one question into 2–5 `SubQuestion`s. Each carries:

| Field | Purpose |
|---|---|
| `question` | Standalone archive query. Never contains "it" or "the company". |
| `intent` | Detected from the **sub-question's own** wording, not inherited. |
| `aspects` | 2–4 noun phrases naming what a complete answer must contain. |
| `depends_on` | Only when a step cannot be written until an earlier one is answered. |

## Why aspects exist

They are the plan's own completion criteria, declared before any retrieval happens. Coverage
is later measured against them deterministically, so the agent never has to ask a model
whether it feels finished. See [Appendix 4](04-sufficiency.md).

## Intent is per sub-question, not inherited

Intent drives the Factiva `days_range` window and the ranking priors. "What has Masayoshi Son
said about Arm?" is executive profiling even when the parent question is an entity
investigation, and inheriting the parent's intent would search the wrong window. Guarded by
`test_each_sub_question_gets_its_own_intent_not_the_parent_s`.

## Fewer steps is better, and the planner is not trusted on this

Every sub-question is a billed retrieval, so a plan holding six paraphrases of one ask pays
six times for one set of articles. The prompt says not to paraphrase; models do it anyway. So
duplicates are collapsed **after** planning, deterministically: any pair of sub-questions with
≥ 0.70 stemmed-term Jaccard overlap becomes one.

Two details matter when collapsing:

- **Aspects merge into the survivor.** Dropping a paraphrase must not quietly lower what the
  plan requires of the answer.
- **Ids are renumbered and `depends_on` is rewritten.** A dangling dependency on a removed
  step would otherwise stall the wave scheduler.

## Execution order

`execution_order()` groups steps into waves. Independent steps share a wave and are issued
concurrently. A step declaring a dependency waits.

A dependency **cycle** releases every remaining step as a final wave rather than deadlocking.
A late search beats a hung run.

## Heuristic fallback

With no `OPENAI_API_KEY`, or after a failed planner call, `heuristic_plan()` reuses the
retrieval stack's own query expansion: literal ask, entity anchor, topic paraphrases. It is a
genuinely weaker plan — variants are rephrasings rather than different angles, and they carry
no aspects — so the run records `method="heuristic"` and its coverage is not comparable to a
planned run. This is why every report prints the plan method.

## Rejected

- **LLM-rewritten refinement queries.** A second model call to rephrase a gap costs money and
  adds a failure mode. The deterministic `entity + aspect` form was good enough on the probe
  runs; revisit only with a measurement showing it is the binding constraint.
- **Deep hierarchical plans (sub-sub-questions).** No question in the gold set needed more
  than one level, and the cost multiplies.

## Known limitation

`depends_on` currently controls **ordering only**. A dependent step's query is not rewritten
using the prerequisite's findings. Most plans declare no dependencies, so this has not yet
been the binding constraint — but a question like "who runs the Vision Fund, and what have
they said?" would benefit, and the honest statement is that the second half is searched with
its original wording.
