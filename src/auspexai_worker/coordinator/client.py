"""HTTP client for the coordinator API.

The worker's network surface is small. M1 covers anonymous enrollment only;
heartbeat, assignment pull, and result submission follow in M2 through M4.
Requests are unsigned in M1 (enroll is anonymous-public per coordinator
§5.18). M2 introduces the RFC 9421 signer for worker-credential requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx


class CoordinatorError(Exception):
    """Base coordinator-client failure (network, HTTP, or schema)."""


class PubkeyAlreadyEnrolledError(CoordinatorError):
    """Coordinator returned 409 pubkey_already_enrolled.

    Usually means a previous enrollment succeeded but the worker lost its
    local state. Caller should treat as a recovery situation, not a fresh
    failure — but M1 surfaces it as an error and bails so the operator can
    investigate.
    """


class PubkeyAlreadyTenantError(CoordinatorError):
    """Coordinator returned 409 pubkey_already_tenant — the worker's keypair
    happens to collide with a registered tenant maintainer. Recovery: rotate
    the worker key (delete keystore + state, re-bootstrap)."""


@dataclass(frozen=True)
class EnrollmentResponse:
    worker_id: str
    trust_tier: int
    registered_at: datetime


class CoordinatorClient:
    """Synchronous httpx client for the coordinator API.

    Wraps `httpx.Client` rather than inheriting so tests can pass a custom
    transport (e.g. `httpx.MockTransport`) without monkey-patching.
    """

    def __init__(
        self,
        *,
        base_url: str,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
            headers={"User-Agent": "auspexai-worker/0.0.1"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> CoordinatorClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ---- /workers/enroll -------------------------------------------------

    def enroll(self, *, pubkey_hex: str, capabilities: dict[str, Any]) -> EnrollmentResponse:
        """POST /api/v0/workers/enroll. Anonymous; T0.

        Returns the assigned worker_id + trust_tier + registered_at. Raises
        PubkeyAlreadyEnrolledError / PubkeyAlreadyTenantError on 409, and
        CoordinatorError on other failures.
        """
        body = {"pubkey_hex": pubkey_hex, "capabilities": capabilities}
        try:
            response = self._client.post("/api/v0/workers/enroll", json=body)
        except httpx.HTTPError as exc:
            raise CoordinatorError(f"enroll: HTTP transport error: {exc}") from exc

        if response.status_code == 201:
            return _parse_enrollment(response.json())
        if response.status_code == 409:
            code = _error_code(response)
            if code == "pubkey_already_enrolled":
                raise PubkeyAlreadyEnrolledError(_error_message(response))
            if code == "pubkey_already_tenant":
                raise PubkeyAlreadyTenantError(_error_message(response))
            raise CoordinatorError(
                f"enroll: unexpected 409 conflict code {code!r}: {_error_message(response)}"
            )
        raise CoordinatorError(
            f"enroll: unexpected status {response.status_code}: {response.text[:500]}"
        )


def _parse_enrollment(payload: dict[str, Any]) -> EnrollmentResponse:
    try:
        worker_id = payload["worker_id"]
        trust_tier = payload["trust_tier"]
        registered_at_raw = payload["registered_at"]
    except KeyError as exc:
        raise CoordinatorError(f"enroll response missing field: {exc}") from exc
    if not isinstance(worker_id, str) or not isinstance(trust_tier, int):
        raise CoordinatorError(f"enroll response has wrong types: {payload!r}")
    registered_at = _parse_datetime(registered_at_raw)
    return EnrollmentResponse(
        worker_id=worker_id,
        trust_tier=trust_tier,
        registered_at=registered_at,
    )


def _parse_datetime(raw: object) -> datetime:
    if not isinstance(raw, str):
        raise CoordinatorError(f"registered_at must be a string, got {type(raw).__name__}")
    # FastAPI serializes datetimes in ISO 8601; tolerate trailing "Z".
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _error_code(response: httpx.Response) -> str | None:
    try:
        detail = response.json().get("detail", {})
    except ValueError:
        return None
    if not isinstance(detail, dict):
        return None
    err = detail.get("error")
    if not isinstance(err, dict):
        return None
    code = err.get("code")
    return code if isinstance(code, str) else None


def _error_message(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail", {})
    except ValueError:
        return response.text[:200]
    if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
        msg = detail["error"].get("message")
        if isinstance(msg, str):
            return msg
    return str(detail)[:500]
