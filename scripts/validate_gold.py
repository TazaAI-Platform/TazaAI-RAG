#!/usr/bin/env python3
"""Check that gold labels are achievable, without tuning them to the ranking.

A required term that appears nowhere in the retrieved candidate pool makes its query
unscoreable: the metric then reports corpus coverage rather than retrieval quality, and no
ranking change can move it. This finds those labels.

The direction matters. It is only ever used to *relax or remove* a label that cannot be
satisfied, never to add a term because it happens to appear in the top-5 — that would tune
the gold set to the system under test and guarantee a good score.

Usage: python scripts/validate_gold.py [gold.jsonl]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from taza_rag.eval.factiva_retrieval import term_matches  # noqa: E402
from taza_rag.eval.retrieval import load_gold  # noqa: E402
from taza_rag.factiva.pipeline import QualityRetriever  # noqa: E402


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "evals/gold/factiva_live_v1.jsonl"
    gold = load_gold(path)
    retriever = QualityRetriever()
    unachievable: list[str] = []
    thin: list[str] = []

    for ex in gold:
        try:
            run = retriever.retrieve(ex.query, top_k=10, intent=ex.intent, contextual=False)
        except Exception as e:  # noqa: BLE001 - a dead query should not stop the audit
            print(f"{ex.id:6s} RETRIEVAL FAILED  {type(e).__name__}: {e}")
            continue

        pool = "\n".join(f"{h.chunk.title}\n{h.chunk.text}" for h in run.hits)
        missing = [t for t in ex.must_include_terms if not term_matches(t, pool)]
        aspects_missing = [t for t in ex.nice_to_have_terms if not term_matches(t, pool)]

        status = "ok"
        if missing:
            status = "UNACHIEVABLE"
            unachievable.append(f"{ex.id} ({ex.intent.value}): {missing}")
        elif len(aspects_missing) == len(ex.nice_to_have_terms) and ex.nice_to_have_terms:
            status = "no aspects in pool"
            thin.append(f"{ex.id}: {aspects_missing}")

        print(
            f"{ex.id:6s} {ex.intent.value:22s} pool={len(run.hits):3d} "
            f"must_missing={len(missing)} aspect_missing={len(aspects_missing)}  {status}"
        )

    print(f"\n{len(gold)} queries checked")
    if unachievable:
        print("\nRequired terms absent from the whole candidate pool (review the label):")
        for line in unachievable:
            print(f"  - {line}")
    if thin:
        print("\nNo aspect present anywhere in the pool (aspects may be mislabelled):")
        for line in thin:
            print(f"  - {line}")
    if not unachievable and not thin:
        print("Every label is satisfiable from the retrieved pool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
