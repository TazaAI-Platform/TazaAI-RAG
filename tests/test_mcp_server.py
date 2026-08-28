"""The MCP surface is a protocol, so the protocol is what gets tested.

A malformed frame, an unknown method or a failing tool must all leave the session alive.
The product loop (query → transact → fetch_content) is what gets exercised; LLMs stay off it.
"""

import io
import json

import taza_rag.mcp_server as mcp
from taza_rag.market import Market
from taza_rag.models import Chunk, RetrievedChunk
from taza_rag.ui.serialize import USAGE_FIELDS


def _request(method, params=None, request_id=1):
    body = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        body["params"] = params
    return body


def _hit():
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id="c-a",
            doc_id="d1",
            text="SECRET_BODY profit beat.",
            title="SoftBank profit",
            source="Dow Jones Newswires",
            published_at="2026-08-06",
            token_estimate=80,
        ),
        score=3.5,
        rank=1,
        scores={"tier": 0},
    )


def _offline_market() -> Market:
    return Market(search=lambda query, top_k: [_hit()])


def setup_function() -> None:
    mcp.reset_market(_offline_market())


# unittest-style: the runner has no pytest fixtures, so reset at import and in each test.
mcp.reset_market(_offline_market())


def test_initialize_announces_the_protocol_and_tool_capability():
    result = mcp.handle(_request("initialize"))["result"]
    assert result["protocolVersion"] == mcp.PROTOCOL_VERSION
    assert "tools" in result["capabilities"]
    assert result["serverInfo"]["name"] == "Taza Content Marketplace"


def test_the_tool_list_is_the_product_bid_protocol():
    tools = mcp.handle(_request("tools/list"))["result"]["tools"]
    assert {t["name"] for t in tools} == {"query", "transact", "fetch_content", "reject_bid"}
    for tool in tools:
        assert tool["description"]
        schema = tool["inputSchema"]
        assert schema["type"] == "object"
        assert schema["required"], f"{tool['name']} declares no required argument"
        for field in schema["required"]:
            assert field in schema["properties"]


def test_query_does_not_require_ranking_knobs():
    tools = {t["name"]: t for t in mcp.handle(_request("tools/list"))["result"]["tools"]}
    schema = tools["query"]["inputSchema"]
    assert schema["required"] == ["text_query"]
    assert "Factiva" not in tools["query"]["description"]


def test_transact_takes_only_the_package_handle():
    tools = {t["name"]: t for t in mcp.handle(_request("tools/list"))["result"]["tools"]}
    assert tools["transact"]["inputSchema"]["required"] == ["package_id"]


def test_a_notification_is_never_answered():
    assert mcp.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_an_unknown_method_is_a_protocol_error():
    error = mcp.handle(_request("does/not/exist"))["error"]
    assert error["code"] == mcp.METHOD_NOT_FOUND


def test_an_unknown_tool_is_a_tool_error_not_a_protocol_error():
    result = mcp.handle(_request("tools/call", {"name": "nope", "arguments": {}}))["result"]
    assert result["isError"] is True
    assert "fetch_content" in result["content"][0]["text"]


def test_a_missing_required_argument_is_reported_as_invalid_arguments():
    result = mcp.handle(_request("tools/call", {"name": "query", "arguments": {}}))["result"]
    assert result["isError"] is True
    assert "invalid arguments" in result["content"][0]["text"]


def test_a_failing_tool_returns_an_error_result_rather_than_killing_the_session():
    def boom(args):
        raise RuntimeError("Factiva is down")

    original = mcp.HANDLERS["query"]
    mcp.HANDLERS["query"] = boom
    try:
        result = mcp.handle(
            _request("tools/call", {"name": "query", "arguments": {"text_query": "x"}})
        )["result"]
    finally:
        mcp.HANDLERS["query"] = original
    assert result["isError"] is True
    assert "Factiva is down" in result["content"][0]["text"]


def test_query_over_mcp_costs_nothing_and_does_not_leak_bodies():
    mcp.reset_market(_offline_market())
    result = mcp.handle(
        _request("tools/call", {"name": "query", "arguments": {"text_query": "SoftBank profit"}})
    )["result"]
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert payload["usage"]["bought"] == 0
    assert tuple(payload["usage"]) == USAGE_FIELDS
    assert payload["packages"]
    assert "SECRET_BODY" not in result["content"][0]["text"]


def test_the_full_loop_over_mcp_is_query_transact_fetch():
    mcp.reset_market(_offline_market())
    bid = json.loads(
        mcp.handle(
            _request("tools/call", {"name": "query", "arguments": {"text_query": "SoftBank"}})
        )["result"]["content"][0]["text"]
    )
    grant = json.loads(
        mcp.handle(
            _request(
                "tools/call",
                {"name": "transact", "arguments": {"package_id": bid["packages"][0]["package_id"]}},
            )
        )["result"]["content"][0]["text"]
    )
    assert grant["usage"]["bought"] >= 1
    fetched = json.loads(
        mcp.handle(
            _request("tools/call", {"name": "fetch_content", "arguments": {"grant_id": grant["grant_id"]}})
        )["result"]["content"][0]["text"]
    )
    assert "SECRET_BODY" in fetched["items"][0]["text"]


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
