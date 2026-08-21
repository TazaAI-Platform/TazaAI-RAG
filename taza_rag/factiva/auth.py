from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Literal

import httpx

from taza_rag.config import settings

AccountKind = Literal["rag", "feed"]
MAX_TRANSPORT_RETRIES = 2


def _post_with_retry(
    client: httpx.Client, url: str, data: dict[str, str], headers: dict[str, str]
) -> httpx.Response:
    """The token endpoint occasionally resets the connection; retry before failing."""
    last: Exception | None = None
    for attempt in range(1, MAX_TRANSPORT_RETRIES + 2):
        try:
            return client.post(url, data=data, headers=headers)
        except httpx.TransportError as e:
            last = e
            if attempt <= MAX_TRANSPORT_RETRIES:
                time.sleep(min(4.0, 0.6 * (2 ** (attempt - 1))) + random.uniform(0, 0.3))
    raise FactivaAuthError(f"Token endpoint unreachable: {last}") from last


class FactivaAuthError(RuntimeError):
    pass


@dataclass
class TokenBundle:
    access_token: str
    expires_at: float
    account: AccountKind

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at - 60


class FactivaAuth:
    """Dow Jones service-account OAuth (AuthN → AuthZ)."""

    def __init__(self, account: AccountKind = "rag") -> None:
        self.account = account
        self._bundle: TokenBundle | None = None

    def _credentials(self) -> tuple[str, str, str]:
        if self.account == "rag":
            client_id = settings.factiva_rag_client_id
            username = settings.factiva_rag_username
            password = settings.factiva_rag_password
        else:
            client_id = settings.factiva_feed_client_id
            username = settings.factiva_feed_username
            password = settings.factiva_feed_password
        if not client_id or not username or not password:
            raise FactivaAuthError(f"Missing Factiva {self.account} credentials in .env")
        return client_id, username, password

    def get_access_token(self, force: bool = False) -> str:
        if self._bundle and not self._bundle.expired and not force:
            return self._bundle.access_token
        client_id, username, password = self._credentials()
        with httpx.Client(timeout=60.0) as client:
            authn = _post_with_retry(
                client,
                settings.factiva_token_url,
                {
                    "username": username,
                    "client_id": client_id,
                    "password": password,
                    "connection": "service-account",
                    "grant_type": "password",
                    "scope": "openid service_account_id",
                },
                {"Content-Type": "application/x-www-form-urlencoded"},
            )
            if authn.status_code >= 400:
                raise FactivaAuthError(
                    f"AuthN failed ({authn.status_code}): {authn.text[:500]}"
                )
            authn_body = authn.json()
            id_token = authn_body.get("id_token")
            access_token_n = authn_body.get("access_token")
            if not id_token:
                raise FactivaAuthError(f"AuthN missing id_token: {authn_body}")

            authz_data = {
                "assertion": id_token,
                "client_id": client_id,
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "scope": "openid pib",
            }
            # Essentials docs also send AuthN access_token; include when present.
            if access_token_n:
                authz_data["access_token"] = access_token_n

            authz = _post_with_retry(
                client,
                settings.factiva_token_url,
                authz_data,
                {"Content-Type": "application/x-www-form-urlencoded"},
            )
            if authz.status_code >= 400:
                raise FactivaAuthError(
                    f"AuthZ failed ({authz.status_code}): {authz.text[:500]}"
                )
            authz_body = authz.json()
            token = authz_body.get("access_token")
            if not token:
                raise FactivaAuthError(f"AuthZ missing access_token: {authz_body}")
            expires_in = int(authz_body.get("expires_in") or 3600)
            self._bundle = TokenBundle(
                access_token=token,
                expires_at=time.time() + expires_in,
                account=self.account,
            )
            return token
