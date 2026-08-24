from __future__ import annotations

import json
import random
import re
import time
from typing import Any, Callable, TypeVar

from taza_rag.config import settings

T = TypeVar("T")


class LLMError(RuntimeError):
    pass


class LLMQuotaError(LLMError):
    """The key is valid but the account cannot pay for the call."""


class LLMRateLimitError(LLMError):
    """Throttled, not broke. Safe to retry."""


_RETRY_AFTER = re.compile(r"try again in ([0-9.]+)(ms|s)")


def _retry_delay(msg: str, attempt: int) -> float:
    """Honour the provider's own suggested wait before falling back to backoff."""
    m = _RETRY_AFTER.search(msg)
    if m:
        value = float(m.group(1))
        seconds = value / 1000.0 if m.group(2) == "ms" else value
        return min(30.0, max(0.5, seconds * 1.5))
    return min(30.0, 1.5 * (2 ** (attempt - 1))) + random.uniform(0, 0.4)


def _call_with_retry(fn: Callable[[], T], max_retries: int = 5) -> T:
    """Retry throttling only.

    Low-tier accounts have small token-per-minute ceilings, so a whole evaluation
    run would otherwise die on one 429 partway through.
    """
    for attempt in range(1, max_retries + 2):
        try:
            return fn()
        except Exception as e:
            err = _wrap_openai_error(e)
            if not isinstance(err, LLMRateLimitError) or attempt > max_retries:
                raise err from e
            time.sleep(_retry_delay(str(e), attempt))
    raise LLMError("unreachable")


def _wrap_openai_error(e: Exception) -> LLMError:
    """Turn provider errors into something a reader can act on.

    A 429 for an exhausted balance is a billing problem, not a rate limit to back
    off from, and it should never be mistaken for a bug in the pipeline.
    """
    msg = str(e)
    if "insufficient_quota" in msg or "credit_balance_exhausted" in msg:
        return LLMQuotaError(
            "OpenAI rejected the call: the account has no credits remaining. "
            "Add credits at https://platform.openai.com/settings/organization/billing "
            "or run the key-free path (drop --semantic / --llm-context)."
        )
    if "rate_limit_exceeded" in msg or "Rate limit reached" in msg:
        return LLMRateLimitError(msg[:300])
    if "invalid_api_key" in msg or "Incorrect API key" in msg:
        return LLMError("OPENAI_API_KEY is not valid. Check .env for a stale or truncated key.")
    return LLMError(f"{type(e).__name__}: {msg[:300]}")


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
        resp = _call_with_retry(lambda: client.embeddings.create(model=model, input=batch))
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
    resp = _call_with_retry(
        lambda: client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
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
    resp = _call_with_retry(
        lambda: client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    )
    return (resp.choices[0].message.content or "").strip()
