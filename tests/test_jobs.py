"""Hosted POSTs return a job id so a proxy cannot replace a long body with HTML."""

import time

from taza_rag.ui.jobs import JobBoard, JobFail
from taza_rag.ui.serialize import usage_payload
from taza_rag.ui.server import UiHandler, make_server


def test_job_board_records_success_and_typed_failure():
    board = JobBoard()
    done = board.submit(lambda: {"ok": True})
    snap = None
    for _ in range(50):
        snap = board.snapshot(done)
        if snap and snap["status"] != "running":
            break
        time.sleep(0.01)
    assert snap is not None
    assert snap["status"] == "done"
    assert snap["result"] == {"ok": True}

    def boom():
        raise JobFail(502, "Factiva retrieve failed", "upstream 429")

    failed = board.submit(boom)
    for _ in range(50):
        snap = board.snapshot(failed)
        if snap and snap["status"] != "running":
            break
        time.sleep(0.01)
    assert snap["status"] == "error"
    assert snap["error"]["http_status"] == 502
    assert "429" in snap["error"]["detail"]


def test_query_and_research_enqueue_then_complete():
    from tests.test_market import POOL
    from taza_rag.market import Market

    httpd = make_server(
        "127.0.0.1",
        0,
        market=Market(search=lambda query, top_k: POOL),
        research_fn=lambda query, body: {"answer": "ok", "usage": usage_payload()},
        ui_token="secret",
    )
    httpd.server_close()

    captured: list = []
    handler = UiHandler.__new__(UiHandler)
    handler.server = httpd
    handler._json = lambda status, payload: captured.append((status, payload))

    handler._enqueue("/api/query", {"query": "SoftBank profit", "top_k": 4})
    assert captured[0][0] == 202
    job_id = captured[0][1]["job_id"]
    snap = None
    for _ in range(50):
        snap = httpd.jobs.snapshot(job_id)
        if snap and snap["status"] != "running":
            break
        time.sleep(0.01)
    assert snap["status"] == "done"
    assert snap["result"]["packages"]
    assert snap["result"]["usage"]["bought"] == 0

    captured.clear()
    handler._enqueue("/api/research", {"query": "How exposed is SoftBank?"})
    job_id = captured[0][1]["job_id"]
    for _ in range(50):
        snap = httpd.jobs.snapshot(job_id)
        if snap and snap["status"] != "running":
            break
        time.sleep(0.01)
    assert snap["status"] == "done"
    assert snap["result"]["answer"] == "ok"

    handler.headers = {}
    handler.path = "/api/jobs/" + job_id
    handler.server.ui_token = "secret"
    assert handler._authed() is False
    handler.headers = {"X-UI-Token": "secret"}
    assert handler._authed() is True


def test_the_script_refuses_html_error_pages():
    from taza_rag.ui.server import STATIC_DIR

    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "isHtmlBody" in js
    assert "readJson" in js
    assert "postJob" in js
    assert "web page instead of a result" in js
    assert "<!doctype" in js
