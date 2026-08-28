# Appendix 5 — Combining results into one answer

Referenced from [AGENT.md](../../AGENT.md) § Architecture.

## Reused, not reinvented

The single-question path already settled what works, and `taza_rag/agent/synthesize.py` calls
straight into it:

1. **Extract** cited facts from the evidence.
2. **Filter** deterministically — any fact whose figures are absent from the excerpt it cites
   is dropped before the writer ever sees it. The writer cannot invent a number extraction
   never produced.
3. **Compose** only from what survived.
4. **Splice** back any grounded fact the writer dropped, with its own citation. No new
   numbers, no new sources.

On the 52-question gold set that path moved A1 Accuracy from 0.538 to 0.712 and was the first
change to move Completeness with it (1.71 → 1.79). Details in the main
[README](../../README.md) § Extract, then compose.

The earlier lesson behind it is worth restating because it constrains anything added here:
citation integrity is a **per-answer binary gate**, so every extra claim is another chance to
fail it. Two attempts to raise coverage by instructing the writer to cover more were measured
on all 52 questions and both cost more Accuracy than they returned. Coverage has to be bought
by pre-grounding material, not by asking for more prose.

## What the research level adds

**Extraction runs per sub-question, in parallel.** Each step's own evidence, capped at ~1,800
tokens. Keeps every extraction prompt small and on-topic; one prompt over the whole pool was
slower and diluted. A step whose extraction fails is recorded and skipped — one failure must
not lose the other steps' facts.

**Facts are grouped under the step that asked for them**, numbered globally. The grouping lets
the composer follow the plan's shape instead of writing one undifferentiated block; the global
numbering keeps every citation resolvable ([Appendix 2](02-retrieval.md)).

**Duplicate statements are collapsed** on `(label, normalised text)`. Overlapping
sub-questions extract the same sentence twice, and left alone the composer dutifully writes it
twice.

**The composer is handed structure the single-question path never had:**

| Section | Instruction |
|---|---|
| `PLAN` | Follow this order as short paragraphs. |
| `FACTS` | Use only these. Every factual sentence ends with its `[cN]`. |
| `DISAGREEMENTS` | Report both values, attribute each, lead with the preferred one, never average. |
| `DO NOT COVER` | State plainly that the sources do not address this. Do not speculate or pad. |

## Verification

The finished answer goes through the same claim-level check and bounded repair as the
single-question path (`_verify_and_repair`), against the whole pool's evidence:

- citation presence, per claim group rather than per sentence;
- label validity;
- figure grounding, distinguishing invention from miscitation from a bare year;
- one batched entailment call for paraphrase-level support.

Repair is sentence-level first and whole-answer only as a fallback, and every attempt is
scored so a rewrite can leave the answer unchanged but never degrade it.

## Rejected

- **Per-section sub-answers stitched together.** Reads as a report with the seams showing, and
  it multiplies compose calls. One composer over a grouped fact list keeps the lead sentence
  answering the actual question.
- **Headings and bullets.** Cheap apparent structure; A1 Clarity rewards journalistic prose,
  and the earlier splice work already showed that list-shaped answers cost Clarity.
- **Letting the composer resolve disagreements.** It picks one and moves on, which is the
  exact failure mode [Appendix 3](03-conflicts.md) exists to prevent.
