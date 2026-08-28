"""MCP surface: retrieval and research as metered, callable tools.

Taza's architecture puts MCP at the agent-facing boundary, and the design principle behind it
is that the marketplace request path should be deterministic and auditable while the
intelligence above it evolves separately. That maps onto this repo exactly: the ranking stack
and the research loop are the intelligence, and what an external agent needs is a narrow,
priced interface to them.

Three tools, deliberately separated by what they cost:

- `taza_plan` — free. Decompose a question locally and see what it *would* search. An agent
  can inspect the plan before authorising any spend.
- `taza_retrieve` — one metered retrieval. Ranked passages with the score breakdown.
- `taza_research` — the full loop, bounded by an explicit passage budget the caller sets.

Every result carries the same `usage` block — offered, bought, refused, cited — so a caller
can reconcile spend without trusting the agent's narrative about it. The keys do not name a
corpus. Factiva is the first backend, not the contract.

Transport is newline-delimited JSON-RPC 2.0 on stdin/stdout, implemented against the stdlib so
the repo gains no dependency for it. `handle()` is a pure function of a request, which is what
makes the protocol testable offline.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "taza-rag", "version": "0.1.0"}

# JSON-RPC error codes we actually use.
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

TOOLS: list[dict[str, Any]] = [
    {
        "name": "taza_plan",
        "description": (
            "Decompose a complex question into a research plan without retrieving anything. "
            "Free: no corpus access, no passages billed. Use this to inspect what a research "
            "call would search before authorising spend."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The research question"},
                "max_sub_questions": {"type": "integer", "minimum": 1, "maximum": 8, "default": 5},
            },
            "required": ["question"],
        },
    },
    {
        "name": "taza_retrieve",
        "description": (
            "Retrieve ranked content for one query. The only required argument is the query; "
            "ranking knobs are optional and defaulted. Bills the returned pack. `usage` "
            "reports what was offered and bought. Each item includes a score breakdown so "
            "the caller can see why it ranked where it did."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
                "raw": {
                    "type": "boolean",
                    "default": False,
                    "description": "Single API call in provider order, skipping the quality stack",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "taza_research",
        "description": (
            "Run the multi-step research agent: decompose, search in parallel, judge coverage, "
            "refine what is missing, and answer with citations. Bounded by `max_chunks`, which "
            "is the passage budget the caller is willing to spend. Returns the answer, the "
            "coverage it achieved, why it stopped, declared gaps, source disagreements, and a "
            "full purchase ledger. `usage` is the consumption record: offered, bought, "
            "refused, cited."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "max_chunks": {
                    "type": "integer",
                    "minimum": 4,
                    "maximum": 200,
                    "default": 40,
                    "description": "Passage budget: the cap on what this call may buy",
                },
                "max_rounds": {"type": "integer", "minimum": 1, "maximum": 6, "default": 3},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 6},
                "purchase_gate": {"type": "boolean", "default": True},
            },
            "required": ["question"],
        },
    },
]


def _text_result(payload: dict[str, Any]) -> dict[str, Any]:
    """MCP tool result. JSON in a text block, which every client can read."""
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}],
        "isError": False,
    }


def _error_result(message: str) -> dict[str, Any]:
    """A failed tool call is a result, not a protocol error.

    Returning JSON-RPC errors for a Factiva outage would make the caller unable to tell a
    broken server from an unanswerable question.
    """
    return {"content": [{"type": "text", "text": message}], "isError": True}


def _clamp(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def tool_plan(args: dict[str, Any]) -> dict[str, Any]:
    from taza_rag.agent.plan import make_plan
    from taza_rag.ui.serialize import usage_payload

    question = str(args.get("question") or "").strip()
    if not question:
        raise ValueError("question is required")
    plan = make_plan(
        question, max_sub_questions=_clamp(args.get("max_sub_questions"), 1, 8, 5)
    )
    return {"plan": plan.payload(), "usage": usage_payload()}


def tool_retrieve(args: dict[str, Any]) -> dict[str, Any]:
    from taza_rag.factiva.pipeline import QualityRetriever
    from taza_rag.ui.serialize import run_payload

    query = str(args.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    top_k = _clamp(args.get("top_k"), 1, 20, 10)

    if args.get("raw"):
        from taza_rag.factiva.pipeline import RetrievalRun
        from taza_rag.factiva.retrieve import FactivaRetrievalClient
        from taza_rag.factiva.strategy import detect_intent

        hits = FactivaRetrievalClient().retrieve(query, limit=top_k, days_range="Last6Months")
        run = RetrievalRun(
            query=query,
            intent=detect_intent(query),
            variants=[query],
            hits=hits,
            candidates=len(hits),
            passages=len(hits),
            config="factiva_raw",
        )
    else:
        run = QualityRetriever().retrieve(query, top_k=top_k)

    return run_payload(run)


def tool_research(args: dict[str, Any]) -> dict[str, Any]:
    from taza_rag.agent.loop import research
    from taza_rag.agent.models import Budget
    from taza_rag.ui.serialize import research_payload

    question = str(args.get("question") or "").strip()
    if not question:
        raise ValueError("question is required")

    budget = Budget(
        max_rounds=_clamp(args.get("max_rounds"), 1, 6, 3),
        max_unique_chunks=_clamp(args.get("max_chunks"), 4, 200, 40),
        top_k_per_query=_clamp(args.get("top_k"), 1, 20, 6),
        purchase_gate=bool(args.get("purchase_gate", True)),
    )
    result = research(question, budget=budget)
    payload = research_payload(result)
    # The evidence bodies are large and the caller already has labels and citations; a tool
    # result is a decision record, not a content delivery channel.
    for item in payload.get("evidence", []):
        item.pop("text", None)
    return payload


HANDLERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "taza_plan": tool_plan,
    "taza_retrieve": tool_retrieve,
    "taza_research": tool_research,
}


def call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    handler = HANDLERS.get(name)
    if handler is None:
        return _error_result(f"unknown tool {name!r}; available: {sorted(HANDLERS)}")
    try:
        return _text_result(handler(args or {}))
    except ValueError as e:
        return _error_result(f"invalid arguments: {e}")
    except Exception as e:  # noqa: BLE001 - a tool failure must not kill the session
        # Deliberately broad: a Factiva outage, a missing key or a provider 500 are all
        # results the caller should see, and none of them should take the server down.
        return _error_result(f"{type(e).__name__}: {e}")


def handle(request: dict[str, Any]) -> dict[str, Any] | None:
    """Handle one JSON-RPC request. Returns None for notifications."""
    method = request.get("method")
    request_id = request.get("id")

    # Notifications carry no id and must not be answered.
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
    """Newline-delimited JSON-RPC on stdio.

    Nothing may be written to stdout except responses: a stray print corrupts the stream and
    the client sees a protocol error rather than the log line that caused it.
    """
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
        except Exception as e:  # noqa: BLE001 - never drop the session on one bad request
            response = _err(request.get("id"), INTERNAL_ERROR, f"{type(e).__name__}: {e}")
        if response is not None:
            _write(stdout, response)


def _write(stdout: Any, payload: dict[str, Any]) -> None:
    stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    stdout.flush()
