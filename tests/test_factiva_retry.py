"""Factiva retry behaviour — offline, with a fake transport.

A single 429 aborted a 52-query evaluation five minutes in, so this path decides whether a
long run survives. It only executes when the upstream is failing, which is exactly when a
live smoke test is not watching.
"""

import time

import httpx

import taza_rag.factiva.retrieve as rmod
from taza_rag.factiva.retrieve import FactivaRetrievalClient, FactivaRetrieveError


class _FakeAuth:
    def get_access_token(self, force: bool = False) -> str:
        return "token"


def _response(status: int, headers: dict | None = None) -> httpx.Response:
    body = {"data": [] if status < 400 else None}
    if status >= 400:
        body = {"errors": [{"title": "Too Many Requests", "status": status}]}
    return httpx.Response(status, json=body, headers=headers or {})


class _FakeClient:
    """Replays a scripted sequence of responses and records the calls."""

    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = 0

    def post(self, url, headers=None, json=None):
        self.calls += 1
        status, hdrs = self.statuses.pop(0) if self.statuses else (200, {})
        return _response(status, hdrs)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _run(statuses, **kwargs):
    """Patch the transport and the clock so retries are exercised without waiting."""
    fake = _FakeClient(statuses)
    slept: list[float] = []
    real_client, real_sleep = httpx.Client, time.sleep
    httpx.Client = lambda *a, **k: fake
    rmod.time.sleep = slept.append
    try:
        client = FactivaRetrievalClient(auth=_FakeAuth(), **kwargs)
        try:
            hits = client.retrieve("test query", limit=3)
            error = None
        except FactivaRetrieveError as e:
            hits, error = None, e
    finally:
        httpx.Client = real_client
        rmod.time.sleep = real_sleep
    return hits, error, fake.calls, slept


def test_a_clean_response_makes_one_call():
    hits, error, calls, slept = _run([(200, {})])
    assert error is None and hits == []
    assert calls == 1 and slept == []


def test_a_transient_500_is_retried_and_succeeds():
    hits, error, calls, _ = _run([(500, {}), (200, {})])
    assert error is None and calls == 2


def test_a_rate_limit_gets_more_attempts_than_a_server_error():
    """A 429 clears with time, so it earns extra tries a 5xx does not."""
    _hits, error_5xx, calls_5xx, _ = _run([(500, {})] * 12)
    _hits, error_429, calls_429, _ = _run([(429, {})] * 12)
    assert error_5xx is not None and error_429 is not None
    assert calls_429 > calls_5xx, "429 should be retried more persistently than 5xx"


def test_a_rate_limit_that_clears_is_recovered():
    hits, error, calls, _ = _run([(429, {}), (429, {}), (200, {})])
    assert error is None, "a 429 that clears must not fail the run"
    assert calls == 3


def test_the_servers_retry_after_hint_is_honoured():
    _hits, _error, _calls, slept = _run([(429, {"retry-after": "7"}), (200, {})])
    assert slept and 7.0 <= slept[0] < 7.5, f"expected ~7s wait, got {slept}"


def test_an_http_date_retry_after_falls_back_to_backoff():
    """Some gateways send a date instead of seconds; that must not crash the retry."""
    _hits, error, calls, slept = _run(
        [(429, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}), (200, {})]
    )
    assert error is None and calls == 2
    assert slept and slept[0] < 45.0


def test_a_rate_limit_wait_can_exceed_the_5xx_ceiling():
    """The 4s ceiling that suits a transient 5xx cannot clear a per-minute quota."""
    _hits, _error, _calls, slept = _run([(429, {})] * 12)
    assert max(slept) > 4.0, f"429 backoff never exceeded the 5xx ceiling: {slept}"


def test_a_401_refreshes_the_token_once():
    hits, error, calls, _ = _run([(401, {}), (200, {})])
    assert error is None and calls == 2


def test_a_permanent_400_is_not_retried():
    _hits, error, calls, _ = _run([(400, {})])
    assert error is not None and calls == 1
    assert "400" in str(error)
