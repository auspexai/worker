"""Direct RFC 8628 Device Authorization Flow against GitHub.

Q5 resolution (principles doc §9, ratified 2026-05-16): hand-rolled, no
OAuth library dependency. GitHub-only IdP at Phase 1; `read:user` scope.
Client ID is public-by-design per the Device Flow public-client model.

Flow shape:

  1. POST {client_id, scope} to GITHUB_DEVICE_CODE_URL → receive
     {device_code, user_code, verification_uri, expires_in, interval}.
  2. Show user_code + verification_uri to the volunteer (callback).
  3. Poll GITHUB_TOKEN_URL at `interval` seconds with the device_code.
     RFC 8628 §3.5 error codes:
       - authorization_pending  → keep polling at current interval
       - slow_down              → bump interval by 5s, continue polling
       - expired_token          → terminal (user didn't approve in time)
       - access_denied          → terminal (user explicitly declined)
  4. On success, the response carries `access_token`. Hand it to the
     coordinator's /accounts/oauth/exchange endpoint; never store it
     anywhere local. The worker discards the token as soon as the
     coordinator returns the binding token.

The access token never reaches `worker.db` or the keystore. The only thing
the worker persists across the upgrade is the resulting binding to the
account, in `worker_self.account_binding_json` (set by the M6b call).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

# Public-by-design Client ID for the auspexai-org GitHub OAuth App (Device
# Flow enabled, no client secret in use per memory `auspexai_github_oauth_app`).
GITHUB_CLIENT_ID = "Ov23lierutLLeF9skyHu"
GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_SCOPE = "read:user"

_DEVICE_FLOW_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"


class DeviceFlowError(Exception):
    """Base class for Device Flow failures."""


class ExpiredTokenError(DeviceFlowError):
    """The device_code expired before the user authorized (RFC 8628 §3.5)."""


class AccessDeniedError(DeviceFlowError):
    """The user explicitly declined authorization (RFC 8628 §3.5)."""


@dataclass(frozen=True)
class DeviceCode:
    """Result of the initial device-code request."""

    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int


def start_device_flow(
    *,
    client_id: str = GITHUB_CLIENT_ID,
    scope: str = GITHUB_SCOPE,
    transport: httpx.BaseTransport | None = None,
) -> DeviceCode:
    """Begin a Device Flow — returns the codes the user must enter."""
    with httpx.Client(transport=transport, timeout=10.0) as client:
        try:
            response = client.post(
                GITHUB_DEVICE_CODE_URL,
                data={"client_id": client_id, "scope": scope},
                headers={"Accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise DeviceFlowError(f"device-code request failed: {exc}") from exc

    if response.status_code != 200:
        raise DeviceFlowError(
            f"device-code request returned HTTP {response.status_code}: {response.text[:500]}"
        )
    payload = response.json()
    try:
        return DeviceCode(
            device_code=payload["device_code"],
            user_code=payload["user_code"],
            verification_uri=payload["verification_uri"],
            expires_in=int(payload["expires_in"]),
            interval=int(payload["interval"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DeviceFlowError(f"device-code response missing/invalid fields: {payload!r}") from exc


def _poll_once(
    *,
    client: httpx.Client,
    client_id: str,
    device_code: str,
) -> tuple[str | None, str | None]:
    """One poll cycle. Returns (access_token, error_code) — exactly one is None."""
    try:
        response = client.post(
            GITHUB_TOKEN_URL,
            data={
                "client_id": client_id,
                "device_code": device_code,
                "grant_type": _DEVICE_FLOW_GRANT_TYPE,
            },
            headers={"Accept": "application/json"},
        )
    except httpx.HTTPError as exc:
        raise DeviceFlowError(f"token poll request failed: {exc}") from exc

    if response.status_code not in (200, 400):
        raise DeviceFlowError(
            f"token poll returned unexpected HTTP {response.status_code}: {response.text[:500]}"
        )
    payload = response.json()
    if "access_token" in payload:
        return str(payload["access_token"]), None
    return None, str(payload.get("error", "unknown_error"))


def run_device_flow(
    *,
    on_code: Callable[[DeviceCode], None],
    client_id: str = GITHUB_CLIENT_ID,
    scope: str = GITHUB_SCOPE,
    transport: httpx.BaseTransport | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Run the full Device Flow end-to-end. Returns the access_token.

    `on_code` is invoked once with the DeviceCode so the caller can display
    `user_code` + `verification_uri` to the volunteer. `clock` and `sleep`
    are injectable for tests.
    """
    code = start_device_flow(client_id=client_id, scope=scope, transport=transport)
    on_code(code)

    deadline = clock() + code.expires_in
    interval = code.interval
    with httpx.Client(transport=transport, timeout=10.0) as client:
        while clock() < deadline:
            sleep(interval)
            token, error = _poll_once(
                client=client, client_id=client_id, device_code=code.device_code
            )
            if token is not None:
                return token
            if error == "authorization_pending":
                continue
            if error == "slow_down":
                interval += 5
                continue
            if error == "expired_token":
                raise ExpiredTokenError(
                    "device code expired before authorization; run `auspexai-worker login` again"
                )
            if error == "access_denied":
                raise AccessDeniedError("user declined the authorization request")
            raise DeviceFlowError(f"token poll returned unexpected error code {error!r}")
    raise ExpiredTokenError("device code expired before authorization completed")
