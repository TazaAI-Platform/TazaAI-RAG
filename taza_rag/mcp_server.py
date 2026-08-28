"""MCP surface: the product bid protocol, over this repo's ranking stack.

Taza's production MCP (app.tazalabs.ai) is a deterministic marketplace:

    query → packages → transact → fetch_content

No LLM sits on that path. This server speaks the same tool names and the same
cost split (query is free, transact pays, fetch reveals bodies). Factiva is the
first search backend, not the contract. The research agent is a client of these
tools, not one of them.

Transport is newline-delimited JSON-RPC 2.0 on stdin/stdout, stdlib only.
`handle()` is a pure function of a request plus the process-local Market.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable

from taza_rag.market import Market, MarketError

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "Taza Content Marketplace", "version": "0.1.0"}

METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

TOOLS: list[dict[str, Any]] = [
    {
        "name": "query",
        "description": (
            "Submit a bid for licensed content. Free: ranking runs, nothing is billed. "
            "Returns a bid_id plus priced packages, each an opaque handle labelled with "
            "the tradeoff it optimises for (cheapest, densest, token_constrained, "
            "most_thorough, balanced). Compare packages, then call transact to buy one. "
            "Zero packages is a valid response, not an error."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text_query": {"type": "string", "description": "Free-text search"},
                "query": {"type": "string", "description": "Alias of text_query"},
                "token_budget": {"type": "integer", "minimum": 1},
                "max_packages_returned": {"type": "integer", "minimum": 1, "maximum": 8, "default": 5},
                "purpose": {"type": "string", "default": "query_use"},
            },
            "required": ["text_query"],
        },
    },
    {
        "name": "transact",
        "description": (
            "Commit to one package from a bid. Atomic: a grant is issued and the rest "
            "of the bid becomes un-transactable, or nothing changes. This is the spend."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "package_id": {"type": "string", "description": "Opaque package handle from query"},
            },
            "required": ["package_id"],
        },
    },
    {
        "name": "fetch_content",
        "description": (
            "Retrieve document or chunk bodies under an active grant. Query and transact "
            "never return bodies; this is the only tool that does. Re-fetching the same "
            "item under the same grant is free."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "grant_id": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional subset of chunk ids; absent = the whole grant",
                },
            },
            "required": ["grant_id"],
        },
    },
    {
        "name": "reject_bid",
        "description": (
            "Walk away from a bid with a structured reason so the marketplace can learn "
            "what to offer next. Pure telemetry — holds nothing, bills nothing."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "bid_id": {"type": "string"},
                "reason": {"type": "string", "default": "other"},
            },
            "required": ["bid_id"],
        },
    },
]


_market = Market()


def reset_market(market: Market | None = None) -> Market:
    """Tests swap in a fixture-backed market so MCP stays offline."""
    global _market
    _market = market if market is not None else Market()
    return _market


def _clamp(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def tool_query(args: dict[str, Any]) -> dict[str, Any]:
    text = str(args.get("text_query") or args.get("query") or "").strip()
    if not text:
        raise ValueError("text_query is required")
    budget = args.get("token_budget")
    return _market.query(
        text,
        top_k=10,
        token_budget=None if budget in (None, "") else _clamp(budget, 1, 50_000, 3000),
        max_packages_returned=_clamp(args.get("max_packages_returned"), 1, 8, 5),
    )


def tool_transact(args: dict[str, Any]) -> dict[str, Any]:
    return _market.transact(str(args.get("package_id") or ""))


def tool_fetch(args: dict[str, Any]) -> dict[str, Any]:
    items = args.get("items")
    if items is not None and not isinstance(items, list):
        raise ValueError("items must be a list of content ids")
    return _market.fetch_content(
        str(args.get("grant_id") or ""),
        items=[str(i) for i in items] if items else None,
    )


def tool_reject(args: dict[str, Any]) -> dict[str, Any]:
    return _market.reject_bid(str(args.get("bid_id") or ""), reason=str(args.get("reason") or "other"))


HANDLERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "query": tool_query,
    "transact": tool_transact,
    "fetch_content": tool_fetch,
    "reject_bid": tool_reject,
}


def _text_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}],
        "isError": False,
    }


def _error_result(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}], "isError": True}


def call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    handler = HANDLERS.get(name)
    if handler is None:
        return _error_result(f"unknown tool {name!r}; available: {sorted(HANDLERS)}")
    try:
        return _text_result(handler(args or {}))
    except (ValueError, MarketError) as e:
        return _error_result(f"invalid arguments: {e}")
    except Exception as e:  # noqa: BLE001 - a tool failure must not kill the session
        return _error_result(f"{type(e).__name__}: {e}")


def handle(request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")
    if request_id is None:
        return None

    if method == "initialize":
        return _ok(request_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        })
    if method == "ping":
        return _ok(request_id, {})
    if method == "tools/list":
        return _ok(request_id, {"tools": TOOLS})
    if method == "tools/call":
        params = request.get("params") or {}
        name = str(params.get("name") or "")
        return _ok(request_id, call_tool(name, params.get("arguments") or {}))

    return _err(request_id, METHOD_NOT_FOUND, f"method not found: {method}")


def _ok(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _err(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def serve(stdin: Any = None, stdout: Any = None) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            _write(stdout, _err(None, INVALID_PARAMS, "malformed JSON"))
            continue
        if not isinstance(request, dict):
            _write(stdout, _err(None, INVALID_PARAMS, "request must be an object"))
            continue
        try:
            response = handle(request)
        except Exception as e:  # noqa: BLE001
            response = _err(request.get("id"), INTERNAL_ERROR, f"{type(e).__name__}: {e}")
        if response is not None:
            _write(stdout, response)


def _write(stdout: Any, payload: dict[str, Any]) -> None:
    stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    stdout.flush()
