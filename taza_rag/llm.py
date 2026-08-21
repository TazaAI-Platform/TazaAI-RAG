from __future__ import annotations

import json
from typing import Any

from taza_rag.config import settings


class LLMError(RuntimeError):
    pass


def _client():
    if not settings.openai_api_key:
        raise LLMError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    try:
        from openai import OpenAI
    except ImportError as e:
        raise LLMError("Install openai: pip install 'taza-rag[openai]'") from e
    return OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    if not texts:
        return []
    client = _client()
    model = model or settings.embedding_model
    # OpenAI allows batching; keep batches modest
    out: list[list[float]] = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        ordered = sorted(resp.data, key=lambda x: x.index)
        out.extend([d.embedding for d in ordered])
    return out


def chat_json(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    client = _client()
    model = model or settings.chat_model
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content or "{}"
    return json.loads(content)


def chat_text(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> str:
    client = _client()
    model = model or settings.chat_model
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()
