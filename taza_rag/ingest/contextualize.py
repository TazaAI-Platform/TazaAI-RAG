from __future__ import annotations

from taza_rag.config import settings
from taza_rag.llm import chat_text
from taza_rag.models import Chunk, Document

CONTEXT_PROMPT = """You situate a chunk within its source document for retrieval indexing.
Write 2-3 sentences (50-100 tokens) that explain what this chunk is about in the context of
the full article. Include: document title, source, date if present, entities, and how the
chunk relates to the broader story. Do not quote the chunk at length. Output plain text only.
"""


def contextualize_chunk(doc: Document, chunk: Chunk, use_llm: bool = True) -> Chunk:
    """Anthropic-style contextual retrieval: prepend doc-aware context before indexing."""
    if not use_llm:
        header = (
            f"This chunk is from '{doc.title}' ({doc.source}"
            f"{', ' + doc.published_at if doc.published_at else ''}). "
            f"Entities: {', '.join(doc.entities) if doc.entities else 'n/a'}."
        )
        chunk.contextualized_text = f"{header}\n\n{chunk.text}"
        return chunk

    user = (
        f"<document>\nTitle: {doc.title}\nSource: {doc.source}\n"
        f"Published: {doc.published_at or 'unknown'}\n"
        f"Entities: {', '.join(doc.entities)}\n\n{doc.body}\n</document>\n\n"
        f"<chunk>\n{chunk.text}\n</chunk>"
    )
    context = chat_text(
        CONTEXT_PROMPT,
        user,
        model=settings.contextualize_model,
        temperature=0.0,
    )
    chunk.contextualized_text = f"{context.strip()}\n\n{chunk.text}"
    return chunk
