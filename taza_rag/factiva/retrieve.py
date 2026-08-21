from __future__ import annotations

import hashlib
import random
import time
import uuid
from typing import Any

import httpx

from taza_rag.config import settings
from taza_rag.factiva.auth import FactivaAuth
from taza_rag.models import Citation, Chunk, RetrievedChunk


class FactivaRetrieveError(RuntimeError):
    pass


def _stable_user_id() -> str:
    raw = settings.factiva_metrics_user_id or settings.factiva_rag_username or "taza"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return digest[:32]


def _work_id() -> str:
    return uuid.uuid4().hex[:32]


def _extract_text(block: Any) -> str:
    if block is None:
        return ""
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        if "text" in block:
            return str(block["text"])
        content = block.get("content")
        if isinstance(content, list):
            return "\n".join(_extract_text(x) for x in content if x)
        if isinstance(content, dict):
            return _extract_text(content)
    if isinstance(block, list):
        return "\n".join(_extract_text(x) for x in block if x)
    return str(block)


def item_to_chunk(item: dict[str, Any], rank: int) -> RetrievedChunk:
    attrs = item.get("attributes") or {}
    meta = item.get("meta") or {}
    source = meta.get("source") or {}
    headline = attrs.get("headline") or {}
    main = headline.get("main") if isinstance(headline, dict) else {}
    title = _extract_text(main) or _extract_text(headline) or "Untitled"

    snippet = _extract_text((attrs.get("snippet") or {}).get("content"))
    body = _extract_text(attrs.get("content"))
    text = snippet or body
    if snippet and body and snippet not in body:
        text = f"{snippet}\n\n{body}"

    doc_id = (
        meta.get("original_doc_id")
        or item.get("id")
        or f"factiva-{rank}"
    )
    url = None
    links = item.get("links") or {}
    if isinstance(links, dict):
        url = links.get("self")

    published = attrs.get("publication_date") or attrs.get("load_date")
    source_name = source.get("name") or source.get("code") or "Factiva"
    source_code = (source.get("code") or "").lower()
    tier = "premium" if source_code in {"djdn", "j", "wsjo", "wsj"} else "standard"

    chunk = Chunk(
        chunk_id=f"{doc_id}::r{rank:04d}",
        doc_id=str(doc_id),
        text=text.strip(),
        title=title.strip(),
        source=str(source_name),
        source_tier=tier,
        published_at=str(published) if published else None,
        url=url,
        chunk_index=rank,
        token_estimate=max(1, len(text.split())),
        metadata={
            "factiva_id": item.get("id"),
            "language": (meta.get("language") or {}).get("code"),
            "source_code": source.get("code"),
            "attribution_code": source.get("attribution_code"),
        },
    )
    return RetrievedChunk(
        chunk=chunk,
        score=float(max(0.01, 1.0 - rank * 0.01)),
        rank=rank,
        method="factiva_retrieve",
        scores={"api_rank": float(rank)},
    )


RETRY_STATUS = {429, 500, 502, 503, 504}


class FactivaRetrievalClient:
    """Factiva Retrieval API — contextual news chunks for RAG."""

    def __init__(self, auth: FactivaAuth | None = None, max_retries: int = 2) -> None:
        self.auth = auth or FactivaAuth(account="rag")
        self.max_retries = max_retries

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 10,
        language: str = "en",
        days_range: str | None = "Last6Months",
        date_from: str | None = None,
        date_to: str | None = None,
        search_filters: list[dict[str, str]] | None = None,
    ) -> list[RetrievedChunk]:
        token = self.auth.get_access_token()
        filters = search_filters or [{"scope": "Language", "value": language}]

        date_obj: dict[str, Any] | None = None
        if date_from and date_to:
            date_obj = {"custom": {"from": date_from, "to": date_to}}
        elif days_range:
            date_obj = {"days_range": days_range}

        query_obj: dict[str, Any] = {
            "value": query,
            "search_filters": filters,
        }
        if date_obj:
            query_obj["date"] = date_obj

        payload = {
            "data": {
                "attributes": {
                    "response_limit": min(limit, 100),
                    "query": query_obj,
                    "metrics_data": {
                        "user_id": _stable_user_id(),
                        "work_id": _work_id(),
                        "application_id": settings.factiva_application_id[:32],
                    },
                },
                "id": "GenAIRetrieval",
                "type": "genai-content",
            }
        }

        url = f"{settings.factiva_api_base.rstrip('/')}/content/gen-ai/retrieve"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.dowjones.genai-content.v_1.0",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=90.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code == 401:
                token = self.auth.get_access_token(force=True)
                headers["Authorization"] = f"Bearer {token}"
                resp = client.post(url, headers=headers, json=payload)

            # The semantic service returns sporadic 5xx/429; retry with jittered backoff.
            for attempt in range(1, self.max_retries + 1):
                if resp.status_code not in RETRY_STATUS:
                    break
                time.sleep(min(4.0, 0.6 * (2 ** (attempt - 1))) + random.uniform(0, 0.3))
                resp = client.post(url, headers=headers, json=payload)

            if resp.status_code >= 400:
                raise FactivaRetrieveError(
                    f"Retrieve failed ({resp.status_code}) for query {query!r}: "
                    f"{resp.text[:400]}"
                )
            body = resp.json()

        items = body.get("data") or []
        if isinstance(items, dict):
            items = [items]
        return [item_to_chunk(item, i + 1) for i, item in enumerate(items)]


def hits_to_citations(hits: list[RetrievedChunk]) -> list[Citation]:
    out: list[Citation] = []
    for h in hits:
        c = h.chunk
        out.append(
            Citation(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                title=c.title,
                source=c.source,
                published_at=c.published_at,
                url=c.url,
                excerpt=c.text[:280],
            )
        )
    return out
