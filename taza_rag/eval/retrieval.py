from __future__ import annotations

import json
from pathlib import Path

from taza_rag.models import GoldExample, RetrievedChunk, SearchIntent


def load_gold(path: Path) -> list[GoldExample]:
    rows: list[GoldExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(GoldExample.model_validate(json.loads(line)))
    return rows


def recall_at_k(
    hits: list[RetrievedChunk],
    must_include_doc_ids: list[str],
    k: int = 10,
) -> float:
    if not must_include_doc_ids:
        return 1.0
    top = {h.chunk.doc_id for h in hits[:k]}
    hit = sum(1 for d in must_include_doc_ids if d in top)
    return hit / len(must_include_doc_ids)


def precision_at_k(
    hits: list[RetrievedChunk],
    relevant_doc_ids: list[str],
    k: int = 5,
) -> float:
    if k == 0:
        return 0.0
    top = hits[:k]
    if not top:
        return 0.0
    rel = set(relevant_doc_ids)
    return sum(1 for h in top if h.chunk.doc_id in rel) / len(top)


def hard_negative_rate(
    hits: list[RetrievedChunk],
    hard_negatives: list[str],
    k: int = 5,
) -> float:
    if not hard_negatives:
        return 0.0
    top = {h.chunk.doc_id for h in hits[:k]}
    return sum(1 for d in hard_negatives if d in top) / len(hard_negatives)


def intent_counts(examples: list[GoldExample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ex in examples:
        key = ex.intent.value if isinstance(ex.intent, SearchIntent) else str(ex.intent)
        counts[key] = counts.get(key, 0) + 1
    return counts
