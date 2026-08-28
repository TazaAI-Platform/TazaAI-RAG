"""The MCP surface is a protocol, so the protocol is what gets tested.

A malformed frame, an unknown method or a failing tool must all leave the session alive: an
agent that loses its connection because one question was unanswerable is worse than one that
gets told the question was unanswerable.
"""

import io
import json

import taza_rag.mcp_server as mcp
from taza_rag.ui.serialize import USAGE_FIELDS


def _request(method, params=None, request_id=1):
    body = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        body["params"] = params
    return body


def test_initialize_announces_the_protocol_and_tool_capability():
    result = mcp.handle(_request("initialize"))["result"]
    assert result["protocolVersion"] == mcp.PROTOCOL_VERSION
    assert "tools" in result["capabilities"]
    assert result["serverInfo"]["name"] == "taza-rag"


def test_the_tool_list_declares_all_three_tools_with_schemas():
    tools = mcp.handle(_request("tools/list"))["result"]["tools"]
    assert {t["name"] for t in tools} == {"taza_plan", "taza_retrieve", "taza_research"}
    for tool in tools:
        assert tool["description"]
        schema = tool["inputSchema"]
        assert schema["type"] == "object"
        assert schema["required"], f"{tool['name']} declares no required argument"
        for field in schema["required"]:
            assert field in schema["properties"]


def test_the_retrieve_tool_does_not_require_ranking_knobs():
    """Playground contract: query in, options out. top_k is optional."""
    tools = {t["name"]: t for t in mcp.handle(_request("tools/list"))["result"]["tools"]}
    schema = tools["taza_retrieve"]["inputSchema"]
    assert schema["required"] == ["query"]
    assert "Factiva" not in tools["taza_retrieve"]["description"]


def test_the_research_tool_exposes_the_passage_budget_as_an_argument():
    """A caller must be able to cap what a single call may spend."""
    tools = {t["name"]: t for t in mcp.handle(_request("tools/list"))["result"]["tools"]}
    budget = tools["taza_research"]["inputSchema"]["properties"]["max_chunks"]
    assert budget["type"] == "integer"
    assert budget["maximum"] <= 200


def test_a_notification_is_never_answered():
    assert mcp.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_an_unknown_method_is_a_protocol_error():
    error = mcp.handle(_request("does/not/exist"))["error"]
    assert error["code"] == mcp.METHOD_NOT_FOUND


def test_an_unknown_tool_is_a_tool_error_not_a_protocol_error():
    """The session stays usable; the caller learns which tools exist."""
    result = mcp.handle(_request("tools/call", {"name": "nope", "arguments": {}}))["result"]
    assert result["isError"] is True
    assert "taza_research" in result["content"][0]["text"]


def test_a_missing_required_argument_is_reported_as_invalid_arguments():
    result = mcp.handle(_request("tools/call", {"name": "taza_plan", "arguments": {}}))["result"]
    assert result["isError"] is True
    assert "invalid arguments" in result["content"][0]["text"]


def test_a_failing_tool_returns_an_error_result_rather_than_killing_the_session():
    def boom(args):
        raise RuntimeError("Factiva is down")

    original = mcp.HANDLERS["taza_retrieve"]
    mcp.HANDLERS["taza_retrieve"] = boom
    try:
        result = mcp.handle(
            _request("tools/call", {"name": "taza_retrieve", "arguments": {"query": "x"}})
        )["result"]
    finally:
        mcp.HANDLERS["taza_retrieve"] = original
    assert result["isError"] is True
    assert "Factiva is down" in result["content"][0]["text"]


def test_planning_over_mcp_costs_nothing_and_returns_the_decomposition():
    """The free tool exists so an agent can look before it spends."""
    result = mcp.handle(
        _request("tools/call", {"name": "taza_plan", "arguments": {"question": "Deutche Bank restructuring"}})
    )["result"]
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert payload["usage"]["bought"] == 0
    assert payload["usage"]["offered"] == 0
    assert payload["usage"]["cited"] == 0
    assert tuple(payload["usage"]) == USAGE_FIELDS
    assert payload["plan"]["sub_questions"]
    # Normalisation still happens, or the entity signals go to zero downstream.
    assert any("Deutsche" in s["question"] for s in payload["plan"]["sub_questions"])


def test_the_stdio_loop_answers_requests_and_survives_a_malformed_frame():
    stdin = io.StringIO(
        json.dumps(_request("initialize", request_id=1))
        + "\n"
        + "{not json\n"
        + "\n"
        + json.dumps(_request("tools/list", request_id=2))
        + "\n"
    )
    stdout = io.StringIO()
    mcp.serve(stdin=stdin, stdout=stdout)

    responses = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    assert [r.get("id") for r in responses] == [1, None, 2]
    assert responses[1]["error"]["code"] == mcp.INVALID_PARAMS
    assert responses[2]["result"]["tools"]


def test_the_stdio_loop_writes_nothing_for_a_notification():
    stdin = io.StringIO(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
    stdout = io.StringIO()
    mcp.serve(stdin=stdin, stdout=stdout)
    assert stdout.getvalue() == ""
