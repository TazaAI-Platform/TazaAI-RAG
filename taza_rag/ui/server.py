"""Local newsroom UI. Stdlib HTTP only — same process as `taza-rag retrieve`."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from taza_rag.config import settings
from taza_rag.factiva.answer import answer_with_factiva
from taza_rag.factiva.pipeline import QualityRetriever
from taza_rag.factiva.retrieve import FactivaRetrieveError
from taza_rag.llm import LLMError
from taza_rag.market import Market, MarketError
from taza_rag.ui.serialize import (
    SCORE_LEGEND,
    TIER_HELP,
    TIER_LABELS,
    answer_payload,
    health_payload,
    hit_from_payload,
    plan_payload,
    research_payload,
    run_payload,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_FILES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/app.css": "app.css",
    "/app.js": "app.js",
}
MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
}
MAX_BODY = 32_768
MAX_QUERY = 400


class UiHandler(BaseHTTPRequestHandler):
    server_version = "taza-rag-ui/1"

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the terminal readable; the CLI prints the bind address once.
        if self.path.startswith("/api/"):
            super().log_message(format, *args)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._json(
                200,
                health_payload(
                    factiva=bool(settings.factiva_rag_username and settings.factiva_rag_password),
                    openai=bool(settings.openai_api_key),
                ),
            )
            return
        if path == "/api/legend":
            self._json(
                200,
                {
                    "scores": SCORE_LEGEND,
                    "tiers": [
                        {"id": k, "label": TIER_LABELS[k], "help": TIER_HELP[k]}
                        for k in sorted(TIER_LABELS)
                    ],
                },
            )
            return
        name = STATIC_FILES.get(path)
        if not name:
            self._json(404, {"error": "not found"})
            return
        data = (STATIC_DIR / name).read_bytes()
        self._bytes(200, MIME[Path(name).suffix], data)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._read_json()
        except ValueError as e:
            self._json(400, {"error": str(e)})
            return
        try:
            if path == "/api/query":
                self._json(200, self._query(body))
                return
            if path == "/api/transact":
                self._json(200, self._market().transact(str(body.get("package_id") or "")))
                return
            if path == "/api/fetch":
                items = body.get("items")
                self._json(
                    200,
                    self._market().fetch_content(
                        str(body.get("grant_id") or ""),
                        items=[str(i) for i in items] if isinstance(items, list) else None,
                    ),
                )
                return
            query = str(body.get("query") or "").strip()
            if not query:
                self._json(400, {"error": "query is required"})
                return
            if len(query) > MAX_QUERY:
                self._json(400, {"error": f"query longer than {MAX_QUERY} characters"})
                return
            top_k = _clamp(body.get("top_k"), 1, 20, 10)
            raw = bool(body.get("raw"))
            if path == "/api/plan":
                self._json(200, plan_payload(query, max_variants=3))
                return
            if path == "/api/retrieve":
                self._json(200, self._retrieve(query, top_k=top_k, raw=raw))
                return
            if path == "/api/answer":
                self._json(
                    200,
                    self._answer(
                        query,
                        top_k=top_k,
                        raw=raw,
                        grant_id=str(body.get("grant_id") or "").strip(),
                    ),
                )
                return
            if path == "/api/research":
                self._json(200, self._research(query, body))
                return
        except MarketError as e:
            self._json(400, {"error": str(e)})
            return
        except FactivaRetrieveError as e:
            self._json(502, {"error": "Factiva retrieve failed", "detail": str(e)[:240]})
            return
        except LLMError as e:
            self._json(502, {"error": "Generator failed", "detail": str(e)[:240]})
            return
        self._json(404, {"error": "not found"})

    def _market(self) -> Market:
        existing = getattr(self.server, "market", None)
        if existing is not None:
            return existing
        market = Market()
        self.server.market = market  # type: ignore[attr-defined]
        return market

    def _query(self, body: dict[str, Any]) -> dict[str, Any]:
        query_fn = getattr(self.server, "query_fn", None)
        if query_fn:
            return query_fn(body)
        query = str(body.get("query") or body.get("text_query") or "").strip()
        if not query:
            raise MarketError("query is required")
        if len(query) > MAX_QUERY:
            raise MarketError(f"query longer than {MAX_QUERY} characters")
        budget = body.get("token_budget")
        return self._market().query(
            query,
            top_k=_clamp(body.get("top_k"), 1, 20, 10),
            token_budget=None if budget in (None, "") else _clamp(budget, 1, 50_000, 3000),
            max_packages_returned=_clamp(body.get("max_packages"), 1, 8, 5),
        )

    def _retrieve(self, query: str, *, top_k: int, raw: bool) -> dict[str, Any]:
        retrieve = getattr(self.server, "retrieve_fn", None)
        if retrieve:
            return retrieve(query, top_k=top_k, raw=raw)
        if raw:
            from taza_rag.factiva.retrieve import FactivaRetrievalClient
            from taza_rag.factiva.strategy import detect_intent

            hits = FactivaRetrievalClient().retrieve(query, limit=top_k, days_range="Last6Months")
            from taza_rag.factiva.pipeline import RetrievalRun

            run = RetrievalRun(
                query=query,
                intent=detect_intent(query),
                variants=[query],
                hits=hits,
                candidates=len(hits),
                passages=len(hits),
                config="factiva_raw",
            )
            return run_payload(run)
        run = QualityRetriever().retrieve(query, top_k=top_k)
        return run_payload(run)

    def _answer(
        self, query: str, *, top_k: int, raw: bool, grant_id: str = ""
    ) -> dict[str, Any]:
        answer = getattr(self.server, "answer_fn", None)
        if grant_id:
            from taza_rag.factiva.answer import answer_from_hits

            fetched = self._market().fetch_content(grant_id)
            hits = [hit_from_payload(item) for item in fetched.get("items") or []]
            write = getattr(self.server, "write_fn", None)
            if write:
                return write(query, hits)
            if not settings.openai_api_key:
                raise LLMError("OPENAI_API_KEY is not set. Retrieve still works without it.")
            result = answer_from_hits(query, hits, config_name="licensed_grant")
            payload = answer_payload(result)
            payload["usage"] = fetched.get("usage") or payload["usage"]
            payload["usage"]["cited"] = len(payload.get("citations") or [])
            payload["usage"]["llm_calls"] = 1
            return payload
        if answer:
            return answer(query, top_k=top_k, raw=raw)
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set. Retrieve still works without it.")
        result = answer_with_factiva(query, top_k=top_k, raw=raw)
        return answer_payload(result)

    def _research(self, query: str, body: dict[str, Any]) -> dict[str, Any]:
        research = getattr(self.server, "research_fn", None)
        if research:
            return research(query, body)
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set. The agent cannot plan or write without it.")

        from taza_rag.agent.gather import MarketBackend
        from taza_rag.agent.loop import research as run_research
        from taza_rag.agent.models import Budget

        budget = Budget(
            max_rounds=_clamp(body.get("max_rounds"), 1, 6, 3),
            max_unique_chunks=_clamp(body.get("max_chunks"), 4, 200, 40),
            max_sub_questions=_clamp(body.get("max_sub"), 1, 8, 5),
            top_k_per_query=_clamp(body.get("top_k"), 1, 20, 6),
            purchase_gate=bool(body.get("purchase_gate", True)),
        )
        result = run_research(
            query,
            backend=MarketBackend(market=self._market()),
            budget=budget,
            verify=bool(body.get("verify", True)),
            use_llm_plan=bool(body.get("llm_plan", True)),
        )
        return research_payload(result)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ValueError("request too large")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError("body must be JSON") from e
        if not isinstance(data, dict):
            raise ValueError("body must be an object")
        return data

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._bytes(status, "application/json; charset=utf-8", body)

    def _bytes(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _clamp(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def make_server(
    host: str,
    port: int,
    *,
    retrieve_fn: Callable[..., dict[str, Any]] | None = None,
    answer_fn: Callable[..., dict[str, Any]] | None = None,
    research_fn: Callable[..., dict[str, Any]] | None = None,
    query_fn: Callable[..., dict[str, Any]] | None = None,
    market: Market | None = None,
) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), UiHandler)
    httpd.retrieve_fn = retrieve_fn  # type: ignore[attr-defined]
    httpd.answer_fn = answer_fn  # type: ignore[attr-defined]
    httpd.research_fn = research_fn  # type: ignore[attr-defined]
    httpd.query_fn = query_fn  # type: ignore[attr-defined]
    httpd.market = market if market is not None else Market()  # type: ignore[attr-defined]
    return httpd


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    httpd = make_server(host, port)
    print(f"Taza RAG UI  http://{host}:{port}  (Ctrl-C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
