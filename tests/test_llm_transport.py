"""Provider-error handling — offline, with a fake client.

These paths only run when something goes wrong upstream, so they are exactly the paths a
live smoke test does not cover.
"""

import taza_rag.llm as llm
from taza_rag.llm import LLMError, LLMQuotaError, LLMRateLimitError, _retry_delay, _wrap_openai_error


class _Resp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]


class _FakeCompletions:
    def __init__(self, fail_on_temperature=False, fail_times=0, error=None):
        self.fail_on_temperature = fail_on_temperature
        self.fail_times = fail_times
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_on_temperature and "temperature" in kwargs:
            raise Exception(
                "Unsupported value: 'temperature' does not support 0.0 with this model. "
                "Only the default (1) value is supported."
            )
        if self.fail_times > 0:
            self.fail_times -= 1
            raise Exception(self.error)
        return _Resp('{"ok": true}')


class _FakeClient:
    def __init__(self, completions):
        self.chat = type("Chat", (), {"completions": completions})()


def _install(completions):
    llm._client = lambda: _FakeClient(completions)


def test_a_model_rejecting_temperature_is_retried_without_it():
    """Reasoning models accept only their default sampling; a judge run must not die on that."""
    fake = _FakeCompletions(fail_on_temperature=True)
    _install(fake)
    out = llm.chat_json("sys", "user", model="o3", temperature=0.0)
    assert out == {"ok": True}
    assert len(fake.calls) == 2
    assert "temperature" in fake.calls[0]
    assert "temperature" not in fake.calls[1]


def test_an_unrelated_error_is_not_retried_without_temperature():
    fake = _FakeCompletions(fail_times=1, error="Unsupported value: 'top_p' is not supported")
    _install(fake)
    try:
        llm.chat_json("sys", "user", model="gpt-4o-mini")
        raise AssertionError("should have raised")
    except LLMError:
        pass
    assert len(fake.calls) == 1


def test_a_rate_limit_is_retried_then_succeeds():
    fake = _FakeCompletions(
        fail_times=2,
        error="Rate limit reached for gpt-4o-mini ... rate_limit_exceeded. try again in 1ms",
    )
    _install(fake)
    assert llm.chat_json("sys", "user") == {"ok": True}
    assert len(fake.calls) == 3


def test_an_exhausted_balance_is_never_retried():
    """No amount of waiting fixes an unpaid account, and retrying hides the cause."""
    fake = _FakeCompletions(fail_times=5, error="insufficient_quota: no credits remaining")
    _install(fake)
    try:
        llm.chat_json("sys", "user")
        raise AssertionError("should have raised")
    except LLMQuotaError as e:
        assert "no credits" in str(e)
    assert len(fake.calls) == 1


def test_error_classification():
    def err(msg):
        return _wrap_openai_error(Exception(msg))

    assert isinstance(err("insufficient_quota"), LLMQuotaError)
    assert isinstance(err("Rate limit reached ... rate_limit_exceeded"), LLMRateLimitError)
    assert "not valid" in str(err("invalid_api_key"))
    assert isinstance(err("some other failure"), LLMError)


def test_retry_delay_prefers_the_provider_hint():
    assert _retry_delay("try again in 400ms", 1) < 1.0
    assert 2.0 < _retry_delay("try again in 2s", 1) < 4.0
    # Falls back to bounded exponential backoff, never unbounded
    assert _retry_delay("no hint", 1) <= 2.0
    assert _retry_delay("no hint", 10) <= 30.5
