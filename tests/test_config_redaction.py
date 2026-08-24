"""`show-config` must not print credentials.

The command exists to be pasted into an issue or shown on a screen while debugging, which
is exactly why a field leaking there is worse than one leaking to a log file. Client ids and
usernames are not passwords, but they identify the trial account and pair with one.
"""

from taza_rag.cli import _SECRET_KEYS
from taza_rag.config import Settings

# Anything matching these is a credential unless explicitly justified below.
CREDENTIAL_HINTS = ("password", "secret", "token_secret", "api_key", "client_id", "username")

# Endpoints and non-secret identifiers that are safe to display.
PUBLIC_ALLOWLIST = {"factiva_token_url", "factiva_authz_url", "factiva_retrieve_url"}


def test_every_credential_field_is_redacted():
    fields = set(Settings.model_fields)
    leaking = [
        name
        for name in sorted(fields)
        if name not in PUBLIC_ALLOWLIST
        and any(hint in name for hint in CREDENTIAL_HINTS)
        and name not in _SECRET_KEYS
    ]
    assert not leaking, f"show-config would print these in clear: {leaking}"


def test_redaction_actually_replaces_the_value():
    settings = Settings(openai_api_key="sk-live-should-never-appear")
    data = settings.model_dump(mode="json")
    for key in _SECRET_KEYS:
        if key in data and data[key]:
            data[key] = "***"
    assert data.get("openai_api_key") == "***"
    assert "sk-live-should-never-appear" not in str(data)


def test_the_secret_list_only_names_real_settings():
    """A typo in the list would silently stop redacting the field it was meant to cover."""
    unknown = sorted(_SECRET_KEYS - set(Settings.model_fields))
    assert not unknown, f"_SECRET_KEYS names non-existent settings: {unknown}"
