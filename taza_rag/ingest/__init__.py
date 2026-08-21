from __future__ import annotations

from taza_rag.config import settings
from taza_rag.ingest.chunk import chunk_document
from taza_rag.ingest.contextualize import contextualize_chunk
from taza_rag.models import Chunk, Document


def build_chunks(
    docs: list[Document],
    contextualize: bool = True,
    use_llm_context: bool = True,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        chunks = chunk_document(
            doc,
            target_tokens=settings.chunk_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        if contextualize:
            chunks = [contextualize_chunk(doc, c, use_llm=use_llm_context) for c in chunks]
        all_chunks.extend(chunks)
    return all_chunks
