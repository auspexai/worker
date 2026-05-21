"""HTTP client for the coordinator API.

M1: anonymous enroll. M2: RFC 9421-signed heartbeat. M3+: assignment pull
and result submission follow with the same signed-request pattern. The
client accepts an optional `Rfc9421Signer` — when provided, every
authenticated request is signed via `_signed_request()`; `enroll()` stays
unsigned (anonymous-public per coordinator §5.18).
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from auspexai_worker.signing import Rfc9421Signer


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


class WorkerIdMismatchError(CoordinatorError):
    """403 worker_id_mismatch — signer's worker doesn't match URL worker_id."""


class WorkerNotFoundError(CoordinatorError):
    """404 worker_not_found — the worker has been retired or never existed."""


class AssignmentNotFoundError(CoordinatorError):
    """404 assignment_not_found — no assignment exists for this (unit, worker)."""


class AssignmentAlreadyResolvedError(CoordinatorError):
    """409 assignment_already_resolved — already has a result, or already refused."""


class UnauthorizedError(CoordinatorError):
    """401/403 from a signed endpoint. Usually means the signature was bad,
    the keyid resolved to nothing, or the credential class wasn't allowed."""


@dataclass(frozen=True)
class EnrollmentResponse:
    worker_id: str
    trust_tier: int
    registered_at: datetime


@dataclass(frozen=True)
class WorkerStatusResponse:
    """Subset of WorkerResponse the coordinator returns to a worker calling
    its own endpoints. operator_only fields (pubkey_hex, account_id,
    capabilities) are exposure-filtered out before the worker sees them."""

    worker_id: str
    trust_tier: int
    registered_at: datetime | None
    last_heartbeat_at: datetime | None
    retired_at: datetime | None


@dataclass(frozen=True)
class WorkUnitEnvelope:
    """Wire shape of the coordinator's WorkUnitEnvelopeOut. Mirrors the SDK's
    workunit_v0_1 schema. Payload is opaque; tenant code interprets it."""

    schema_version: str
    unit_id: str
    tenant_id: str
    experiment_id: str  # tenant's experiment_label (NOT the coordinator's exp-... id)
    manifest_sha256: str
    created_at: datetime
    payload: dict[str, Any]


@dataclass(frozen=True)
class RefuseResponse:
    """Coordinator's acknowledgment of a refuse call."""

    assignment_id: str
    unit_id: str
    refused_at: datetime | None
    refused_kind: str | None


@dataclass(frozen=True)
class AssignmentResponse:
    """Coordinator's response to GET /workers/{id}/assignments.

    When no work is available, all four fields are None — `work_unit is None`
    is the canonical check.
    """

    assignment_id: str | None
    assigned_at: datetime | None
    coordinator_experiment_id: str | None  # coordinator's exp-... id (stable)
    work_unit: WorkUnitEnvelope | None


class CoordinatorClient:
    """Synchronous httpx client for the coordinator API."""

    def __init__(
        self,
        *,
        base_url: str,
        signer: Rfc9421Signer | None = None,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
            headers={"User-Agent": "auspexai-worker/0.0.1"},
        )
        self._signer = signer

    @property
    def signer(self) -> Rfc9421Signer | None:
        return self._signer

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> CoordinatorClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ---- /workers/enroll (anonymous) ------------------------------------

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

    # ---- /workers/{id}/heartbeat (signed) -------------------------------

    def heartbeat(
        self,
        *,
        worker_id: str,
        capabilities: dict[str, Any] | None = None,
    ) -> WorkerStatusResponse:
        """POST /api/v0/workers/{worker_id}/heartbeat. Worker-credentialed.

        When `capabilities` is None, the coordinator advances last_heartbeat_at
        without rewriting the stored capabilities. Pass a dict on every tick
        if you want capabilities to stay current.
        """
        if self._signer is None:
            raise CoordinatorError(
                "heartbeat requires a signer; CoordinatorClient was constructed without one"
            )
        body: dict[str, Any] = {}
        if capabilities is not None:
            body["capabilities"] = capabilities
        response = self._signed_request(
            method="POST",
            path=f"/api/v0/workers/{worker_id}/heartbeat",
            json_body=body,
        )
        if response.status_code == 200:
            return _parse_worker_status(response.json())
        if response.status_code == 403:
            code = _error_code(response)
            if code == "worker_id_mismatch":
                raise WorkerIdMismatchError(_error_message(response))
            raise UnauthorizedError(_error_message(response))
        if response.status_code == 401:
            raise UnauthorizedError(_error_message(response))
        if response.status_code == 404:
            raise WorkerNotFoundError(_error_message(response))
        raise CoordinatorError(
            f"heartbeat: unexpected status {response.status_code}: {response.text[:500]}"
        )

    # ---- /workers/{id}/assignments (signed) -----------------------------

    def get_assignment(self, *, worker_id: str) -> AssignmentResponse:
        """GET /api/v0/workers/{worker_id}/assignments. Worker-credentialed.

        Returns an AssignmentResponse. When no work is available the response
        has `work_unit is None`; otherwise the envelope carries the assigned
        unit. Coordinator's first-fit scheduler creates the assignment row
        as a side effect of this call (per the platform's M6d design); the
        worker should treat receiving a non-null work_unit as a commitment
        and submit a result (or let the assignment lapse).
        """
        if self._signer is None:
            raise CoordinatorError(
                "get_assignment requires a signer; CoordinatorClient was constructed without one"
            )
        response = self._signed_request(
            method="GET",
            path=f"/api/v0/workers/{worker_id}/assignments",
            json_body=None,
        )
        if response.status_code == 200:
            return _parse_assignment(response.json())
        if response.status_code == 403:
            code = _error_code(response)
            if code == "worker_id_mismatch":
                raise WorkerIdMismatchError(_error_message(response))
            raise UnauthorizedError(_error_message(response))
        if response.status_code == 401:
            raise UnauthorizedError(_error_message(response))
        if response.status_code == 404:
            raise WorkerNotFoundError(_error_message(response))
        raise CoordinatorError(
            f"get_assignment: unexpected status {response.status_code}: {response.text[:500]}"
        )

    # ---- /workers/{id}/assignments/{unit_id}/refuse (signed) -----------

    def refuse_assignment(
        self,
        *,
        worker_id: str,
        unit_id: str,
        kind: str,
        reason: str,
    ) -> RefuseResponse:
        """POST .../refuse. Worker-credentialed.

        Tells the coordinator the worker is declining a previously-pulled
        assignment so the coordinator can free the replication slot and
        operators can see the refusal reason. The local audit log is the
        primary record; this call is the network half of the same fact.

        Raises:
            AssignmentNotFoundError: 404, no such (unit, worker) assignment.
            AssignmentAlreadyResolvedError: 409, already has result or
                already refused.
            WorkerIdMismatchError: 403, signer's worker doesn't match URL.
            UnauthorizedError: 401/403 on signature/credential failure.
        """
        if self._signer is None:
            raise CoordinatorError(
                "refuse_assignment requires a signer; CoordinatorClient was constructed without one"
            )
        response = self._signed_request(
            method="POST",
            path=f"/api/v0/workers/{worker_id}/assignments/{unit_id}/refuse",
            json_body={"kind": kind, "reason": reason},
        )
        if response.status_code == 200:
            return _parse_refuse(response.json())
        if response.status_code == 403:
            code = _error_code(response)
            if code == "worker_id_mismatch":
                raise WorkerIdMismatchError(_error_message(response))
            raise UnauthorizedError(_error_message(response))
        if response.status_code == 401:
            raise UnauthorizedError(_error_message(response))
        if response.status_code == 404:
            raise AssignmentNotFoundError(_error_message(response))
        if response.status_code == 409:
            raise AssignmentAlreadyResolvedError(_error_message(response))
        raise CoordinatorError(
            f"refuse_assignment: unexpected status {response.status_code}: {response.text[:500]}"
        )

    # ---- internals ------------------------------------------------------

    def _signed_request(
        self,
        *,
        method: str,
        path: str,
        json_body: dict[str, Any] | None,
    ) -> httpx.Response:
        """Build, sign, and send a worker-credentialed request."""
        assert self._signer is not None
        body_bytes: bytes
        headers: dict[str, str] = {}
        if json_body is None:
            body_bytes = b""
        else:
            body_bytes = _json.dumps(json_body, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"

        # Resolve authority the same way coordinator computes it
        # (Starlette's request.url.netloc == httpx.URL.netloc for the same
        # bind). Build the absolute URL via the client to inherit base_url.
        full_url = self._client.build_request(method, path).url
        authority = full_url.netloc.decode("ascii")

        sig_headers = self._signer.sign(
            method=method,
            path=full_url.raw_path.decode("ascii") if full_url.raw_path else full_url.path,
            authority=authority,
            body=body_bytes,
        )
        headers.update(sig_headers)
        try:
            return self._client.request(
                method,
                path,
                content=body_bytes,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            raise CoordinatorError(f"{method} {path}: HTTP transport error: {exc}") from exc


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


def _parse_assignment(payload: dict[str, Any]) -> AssignmentResponse:
    work_unit_raw = payload.get("work_unit")
    work_unit: WorkUnitEnvelope | None = None
    if work_unit_raw is not None:
        try:
            work_unit = WorkUnitEnvelope(
                schema_version=str(work_unit_raw.get("schema_version", "0.1")),
                unit_id=str(work_unit_raw["unit_id"]),
                tenant_id=str(work_unit_raw["tenant_id"]),
                experiment_id=str(work_unit_raw["experiment_id"]),
                manifest_sha256=str(work_unit_raw["manifest_sha256"]).lower(),
                created_at=_parse_datetime(work_unit_raw["created_at"]),
                payload=dict(work_unit_raw.get("payload") or {}),
            )
        except KeyError as exc:
            raise CoordinatorError(f"assignment work_unit missing field: {exc}") from exc
    return AssignmentResponse(
        assignment_id=_opt_str(payload.get("assignment_id")),
        assigned_at=_parse_optional_datetime(payload.get("assigned_at")),
        coordinator_experiment_id=_opt_str(payload.get("experiment_id")),
        work_unit=work_unit,
    )


def _opt_str(raw: object) -> str | None:
    return None if raw is None else str(raw)


def _parse_refuse(payload: dict[str, Any]) -> RefuseResponse:
    try:
        return RefuseResponse(
            assignment_id=str(payload["assignment_id"]),
            unit_id=str(payload["unit_id"]),
            refused_at=_parse_optional_datetime(payload.get("refused_at")),
            refused_kind=_opt_str(payload.get("refused_kind")),
        )
    except KeyError as exc:
        raise CoordinatorError(f"refuse response missing field: {exc}") from exc


def _parse_worker_status(payload: dict[str, Any]) -> WorkerStatusResponse:
    try:
        worker_id = payload["worker_id"]
        trust_tier = payload["trust_tier"]
    except KeyError as exc:
        raise CoordinatorError(f"worker status response missing field: {exc}") from exc
    return WorkerStatusResponse(
        worker_id=str(worker_id),
        trust_tier=int(trust_tier),
        registered_at=_parse_optional_datetime(payload.get("registered_at")),
        last_heartbeat_at=_parse_optional_datetime(payload.get("last_heartbeat_at")),
        retired_at=_parse_optional_datetime(payload.get("retired_at")),
    )


def _parse_datetime(raw: object) -> datetime:
    if not isinstance(raw, str):
        raise CoordinatorError(f"datetime must be a string, got {type(raw).__name__}")
    # FastAPI serializes datetimes in ISO 8601; tolerate trailing "Z".
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _parse_optional_datetime(raw: object) -> datetime | None:
    if raw is None:
        return None
    return _parse_datetime(raw)


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
