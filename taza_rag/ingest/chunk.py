from __future__ import annotations

import re

from taza_rag.models import Chunk, Document

_TOKEN_RE = re.compile(r"\S+")


def estimate_tokens(text: str) -> int:
    # Fast approx; good enough for chunk budgeting without requiring tiktoken always
    return max(1, len(_TOKEN_RE.findall(text)))


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_document(
    doc: Document,
    target_tokens: int = 450,
    overlap_tokens: int = 60,
) -> list[Chunk]:
    """Structure-aware chunking: pack paragraphs into ~target_tokens windows with overlap."""
    paragraphs = _split_paragraphs(doc.body)
    if not paragraphs:
        paragraphs = [doc.body.strip()]

    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        pt = estimate_tokens(para)
        if current and current_tokens + pt > target_tokens:
            windows.append("\n\n".join(current))
            # overlap: keep trailing paragraphs until overlap budget
            overlap: list[str] = []
            ot = 0
            for p in reversed(current):
                t = estimate_tokens(p)
                if ot + t > overlap_tokens:
                    break
                overlap.insert(0, p)
                ot += t
            current = overlap
            current_tokens = ot
        current.append(para)
        current_tokens += pt

    if current:
        windows.append("\n\n".join(current))

    chunks: list[Chunk] = []
    for i, text in enumerate(windows):
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::c{i:04d}",
                doc_id=doc.doc_id,
                text=text,
                title=doc.title,
                source=doc.source,
                source_tier=doc.source_tier,
                published_at=doc.published_at,
                url=doc.url,
                chunk_index=i,
                token_estimate=estimate_tokens(text),
                metadata={**doc.metadata, "entities": doc.entities},
            )
        )
    return chunks
