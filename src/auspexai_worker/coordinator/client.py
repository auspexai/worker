"""HTTP client for the coordinator API.

M1: anonymous enroll. M2: RFC 9421-signed heartbeat. M3+: assignment pull
and result submission follow with the same signed-request pattern. The
client accepts an optional `Rfc9421Signer` — when provided, every
authenticated request is signed via `_signed_request()`; `enroll()` stays
unsigned (anonymous-public per coordinator §5.18).
"""

from __future__ import annotations

import base64
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


class WorkerQuarantinedError(CoordinatorError):
    """423 worker_quarantined — the maintainer has paused this worker.

    Reversible: the maintainer can unquarantine, after which assignment
    fetches resume normally. Carries the quarantined_at timestamp and the
    maintainer's quarantine *reason* from the error details, so the worker
    can log/show both. The reason is intentionally surfaced to the worker:
    a volunteer running this machine is entitled to know why it was paused
    (trust-boundary transparency, ratified 2026-05-30). It is not operator-
    only — the coordinator only sends it to the worker itself and the
    worker's own-account researcher, never to third parties."""

    def __init__(
        self,
        message: str,
        *,
        quarantined_at: str | None = None,
        quarantine_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.quarantined_at = quarantined_at
        self.quarantine_reason = quarantine_reason


class WorkerPausedError(CoordinatorError):
    """423 worker_paused — the operator paused this worker (§2.1 #11). Like
    quarantine it's reversible + carries the operator's reason for the volunteer,
    but it's a NO-FAULT operational hold (distinct code + `no_fault` flag) — not a
    trust/fault signal. The worker surfaces it as an operational pause."""

    def __init__(
        self,
        message: str,
        *,
        paused_at: str | None = None,
        pause_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.paused_at = paused_at
        self.pause_reason = pause_reason


class AssignmentNotFoundError(CoordinatorError):
    """404 assignment_not_found — no assignment exists for this (unit, worker)."""


class AssignmentAlreadyResolvedError(CoordinatorError):
    """409 assignment_already_resolved — already has a result, or already refused."""


class ResultAlreadySubmittedError(CoordinatorError):
    """409 result_already_submitted — this assignment already has a result.

    The coordinator includes `existing_assignment_id` and `existing_result_id`
    in the 409 details so a worker retrying after a transient submit failure
    can reconcile its local state instead of losing the submission record.
    """

    def __init__(
        self,
        message: str,
        *,
        existing_assignment_id: str | None = None,
        existing_result_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.existing_assignment_id = existing_assignment_id
        self.existing_result_id = existing_result_id


class UnitIdMismatchError(CoordinatorError):
    """422 unit_id_mismatch — body.unit_id ≠ URL unit_id."""


class WorkerPubkeyMismatchError(CoordinatorError):
    """403 worker_pubkey_mismatch — Result.worker_pubkey ≠ signing credential."""


class UnauthorizedError(CoordinatorError):
    """401/403 from a signed endpoint. Usually means the signature was bad,
    the keyid resolved to nothing, or the credential class wasn't allowed."""


class UnsupportedIdpError(CoordinatorError):
    """400 unsupported_idp — coordinator doesn't accept this identity provider."""


class InvalidAccessTokenError(CoordinatorError):
    """401 invalid_access_token — IdP rejected the access token at verify time."""


class BindingTokenNotFoundError(CoordinatorError):
    """404 binding_token_not_found — the upgrade call's binding token is unknown."""


class BindingTokenExpiredError(CoordinatorError):
    """400 binding_token_expired — the binding token has aged out (5-min TTL)."""


class BindingTokenConsumedError(CoordinatorError):
    """409 binding_token_consumed — binding tokens are one-shot; already used."""


@dataclass(frozen=True)
class EnrollmentResponse:
    worker_id: str
    trust_tier: int
    registered_at: datetime


@dataclass(frozen=True)
class LatestRelease:
    """§9 #46: the coordinator's release announcement, relayed in the
    heartbeat response. Informational only — upgrading is the volunteer's
    election; the worker never acts on this beyond surfacing it."""

    version: str
    notes: str | None
    url: str | None


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
    latest_release: LatestRelease | None = None


@dataclass(frozen=True)
class OAuthExchangeResponse:
    """Coordinator's reply to POST /accounts/oauth/exchange. The binding_token
    is a one-shot 5-minute token consumed by POST /workers/{id}/upgrade.
    account_id is informational for the caller; the worker doesn't otherwise
    need it (the worker stores the binding the coordinator records, not the
    account_id itself)."""

    account_id: str
    binding_token: str
    expires_at: datetime
    is_new_account: bool


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
class ResultSubmissionResponse:
    """Coordinator's acknowledgment of a submitted Result.

    `unit_status_after` is "in_progress" until completions_so_far reaches
    replication_target, at which point it transitions to "completed".
    """

    result_id: str
    unit_id: str
    unit_status_after: str
    completions_so_far: int
    replication_target: int


@dataclass(frozen=True)
class RefuseResponse:
    """Coordinator's acknowledgment of a refuse call."""

    assignment_id: str
    unit_id: str
    refused_at: datetime | None
    refused_kind: str | None


@dataclass(frozen=True)
class CanonicalReceiptResponse:
    """Coordinator's response to GET /workers/{id}/results/{result_id}/canonical-receipt.

    The `cose_signed_blob` is the canonical COSE-Sign1 bytes the worker
    stores in `submitted_results.canonical_blob`. The other fields are
    metadata the worker logs but doesn't otherwise persist.
    """

    receipt_id: str
    experiment_id: str
    cose_signed_blob: bytes
    signing_key_pubkey_hex: str


@dataclass(frozen=True)
class PrestageDirective:
    """One model the conductor wants this worker to pre-stage (M3b)."""

    model_id: str
    hf_repo: str
    hf_filename: str


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
        if response.status_code == 423:
            # An operator hold — quarantine (fault) or pause (§2.1 #11, no-fault).
            # Both carry the operator's reason in the error details so the worker
            # can surface it (the reason is worker-visible by design).
            code = _error_code(response)
            try:
                details = response.json().get("detail", {}).get("error", {}).get("details", {})
            except (ValueError, AttributeError):
                details = {}
            if code == "worker_paused":
                raise WorkerPausedError(
                    _error_message(response),
                    paused_at=details.get("paused_at"),
                    pause_reason=details.get("pause_reason"),
                )
            raise WorkerQuarantinedError(
                _error_message(response),
                quarantined_at=details.get("quarantined_at"),
                quarantine_reason=details.get("quarantine_reason"),
            )
        raise CoordinatorError(
            f"get_assignment: unexpected status {response.status_code}: {response.text[:500]}"
        )

    # ---- /workers/{id}/prestage (signed) — M3b eager conductor --------

    def get_prestage(self, *, worker_id: str) -> list[PrestageDirective]:
        """GET /api/v0/workers/{worker_id}/prestage. Worker-credentialed. Returns
        the models the conductor wants this worker to pre-stage (pull ahead of
        assignment); empty when nothing is queued. The worker fulfils each via its
        M3 auto-acquire path."""
        if self._signer is None:
            raise CoordinatorError(
                "get_prestage requires a signer; CoordinatorClient was constructed without one"
            )
        response = self._signed_request(
            method="GET",
            path=f"/api/v0/workers/{worker_id}/prestage",
            json_body=None,
        )
        if response.status_code == 200:
            items = response.json().get("prestage") or []
            return [
                PrestageDirective(
                    model_id=i["model_id"], hf_repo=i["hf_repo"], hf_filename=i["hf_filename"]
                )
                for i in items
            ]
        if response.status_code in (401, 403):
            raise UnauthorizedError(_error_message(response))
        if response.status_code == 404:
            raise WorkerNotFoundError(_error_message(response))
        raise CoordinatorError(
            f"get_prestage: unexpected status {response.status_code}: {response.text[:500]}"
        )

    # ---- /workers/{id}/assignments/{unit_id}/result (signed) ----------

    def submit_result(
        self,
        *,
        worker_id: str,
        unit_id: str,
        worker_pubkey: str,
        completed_at: str,
        exit_code: int,
        payload: dict[str, Any],
        worker_signature: str,
    ) -> ResultSubmissionResponse:
        """POST .../result. Worker-credentialed.

        Per coordinator M6d the body's `worker_signature` is stored but
        not verified at submit time — M7 will re-verify when issuing
        receipts. The worker still produces a valid signature (see
        `signing.sign_result`) so M7 verification just works.
        """
        if self._signer is None:
            raise CoordinatorError(
                "submit_result requires a signer; CoordinatorClient was constructed without one"
            )
        body: dict[str, Any] = {
            "unit_id": unit_id,
            "worker_pubkey": worker_pubkey.lower(),
            "completed_at": completed_at,
            "exit_code": int(exit_code),
            "payload": payload,
            "worker_signature": worker_signature,
        }
        response = self._signed_request(
            method="POST",
            path=f"/api/v0/workers/{worker_id}/assignments/{unit_id}/result",
            json_body=body,
        )
        if response.status_code == 201:
            return _parse_result_submission(response.json())
        if response.status_code == 422:
            code = _error_code(response)
            if code == "unit_id_mismatch":
                raise UnitIdMismatchError(_error_message(response))
            raise CoordinatorError(f"submit_result: 422 {code!r}: {_error_message(response)}")
        if response.status_code == 403:
            code = _error_code(response)
            if code == "worker_pubkey_mismatch":
                raise WorkerPubkeyMismatchError(_error_message(response))
            if code == "worker_id_mismatch":
                raise WorkerIdMismatchError(_error_message(response))
            raise UnauthorizedError(_error_message(response))
        if response.status_code == 401:
            raise UnauthorizedError(_error_message(response))
        if response.status_code == 404:
            raise AssignmentNotFoundError(_error_message(response))
        if response.status_code == 409:
            details = _error_details(response)
            raise ResultAlreadySubmittedError(
                _error_message(response),
                existing_assignment_id=(
                    details.get("assignment_id")
                    if isinstance(details.get("assignment_id"), str)
                    else None
                ),
                existing_result_id=(
                    details.get("existing_result_id")
                    if isinstance(details.get("existing_result_id"), str)
                    else None
                ),
            )
        raise CoordinatorError(
            f"submit_result: unexpected status {response.status_code}: {response.text[:500]}"
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

    # ---- /accounts/oauth/exchange (anonymous-public) -------------------

    def oauth_exchange(self, *, idp: str, access_token: str) -> OAuthExchangeResponse:
        """POST /api/v0/accounts/oauth/exchange. Anonymous-public.

        Hand the IdP's access token to the coordinator; the coordinator
        verifies it with the IdP and mints a one-shot binding token to be
        consumed by `upgrade_worker`. The access_token is never persisted on
        the worker — pass it straight from `oauth.run_device_flow()` into
        this method.

        Raises:
            UnsupportedIdpError: 400 — IdP not enabled.
            InvalidAccessTokenError: 401 — IdP rejected the token.
        """
        try:
            response = self._client.post(
                "/api/v0/accounts/oauth/exchange",
                json={"idp": idp, "access_token": access_token},
            )
        except httpx.HTTPError as exc:
            raise CoordinatorError(f"oauth_exchange: HTTP transport error: {exc}") from exc

        if response.status_code == 200:
            return _parse_oauth_exchange(response.json())
        code = _error_code(response)
        if response.status_code == 400 and code == "unsupported_idp":
            raise UnsupportedIdpError(_error_message(response))
        if response.status_code == 401 and code == "invalid_access_token":
            raise InvalidAccessTokenError(_error_message(response))
        raise CoordinatorError(
            f"oauth_exchange: unexpected status {response.status_code}: {response.text[:500]}"
        )

    # ---- /workers/{id}/upgrade (signed) --------------------------------

    def upgrade_worker(self, *, worker_id: str, binding_token: str) -> WorkerStatusResponse:
        """POST /api/v0/workers/{worker_id}/upgrade. Worker-credentialed.

        Consumes a one-shot binding_token from `oauth_exchange` and promotes
        the worker from T0 to T1. The returned status's trust_tier should be 1.

        Raises:
            BindingTokenNotFoundError: 404, token unknown.
            BindingTokenExpiredError: 400, token aged out (>5min).
            BindingTokenConsumedError: 409, token already used.
            WorkerIdMismatchError: 403, signer's worker doesn't match URL.
            UnauthorizedError: 401/403 on signature/credential failure.
            WorkerNotFoundError: 404, worker has been retired.
        """
        if self._signer is None:
            raise CoordinatorError(
                "upgrade_worker requires a signer; CoordinatorClient was constructed without one"
            )
        response = self._signed_request(
            method="POST",
            path=f"/api/v0/workers/{worker_id}/upgrade",
            json_body={"binding_token": binding_token},
        )
        if response.status_code == 200:
            return _parse_worker_status(response.json())
        code = _error_code(response)
        if response.status_code == 400 and code == "binding_token_expired":
            raise BindingTokenExpiredError(_error_message(response))
        if response.status_code == 404 and code == "binding_token_not_found":
            raise BindingTokenNotFoundError(_error_message(response))
        if response.status_code == 409 and code == "binding_token_consumed":
            raise BindingTokenConsumedError(_error_message(response))
        if response.status_code == 403:
            if code == "worker_id_mismatch":
                raise WorkerIdMismatchError(_error_message(response))
            raise UnauthorizedError(_error_message(response))
        if response.status_code == 401:
            raise UnauthorizedError(_error_message(response))
        if response.status_code == 404:
            raise WorkerNotFoundError(_error_message(response))
        raise CoordinatorError(
            f"upgrade_worker: unexpected status {response.status_code}: {response.text[:500]}"
        )

    # ---- /workers/{id}/actions/retire (signed) -------------------------

    def retire_worker(self, *, worker_id: str) -> WorkerStatusResponse:
        """POST /api/v0/workers/{worker_id}/actions/retire. Worker-credentialed.

        The withdrawal call. Coordinator marks the worker retired and the
        scheduler stops handing it work. The coordinator-side `retired_keys`
        registry that prevents re-binding the same key (per §5.15) lands in
        coordinator M7; until then, retire is effective for scheduling but
        re-enroll with the same key isn't yet forbidden — the worker still
        purges its local state regardless of that property.

        Raises:
            WorkerNotFoundError: 404, worker doesn't exist (or already retired).
            WorkerIdMismatchError: 403, signer's worker doesn't match URL.
            UnauthorizedError: 401/403 on signature/credential failure.
        """
        if self._signer is None:
            raise CoordinatorError(
                "retire_worker requires a signer; CoordinatorClient was constructed without one"
            )
        response = self._signed_request(
            method="POST",
            path=f"/api/v0/workers/{worker_id}/actions/retire",
            json_body=None,
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
            f"retire_worker: unexpected status {response.status_code}: {response.text[:500]}"
        )

    def get_canonical_receipt(
        self, *, worker_id: str, result_id: str
    ) -> CanonicalReceiptResponse | None:
        """GET /api/v0/workers/{worker_id}/results/{result_id}/canonical-receipt.

        Worker-credentialed. Used by the M7-tail background fetch loop to
        populate this worker's local `submitted_results.canonical_blob`
        cache.

        Returns None on 404 (`receipt_not_issued` — most often because the
        unit's quorum disagreed; the worker should leave the row as
        placeholder and stop retrying after a reasonable bound). Raises on
        unexpected statuses.
        """
        if self._signer is None:
            raise CoordinatorError(
                "get_canonical_receipt requires a signer; CoordinatorClient "
                "was constructed without one"
            )
        response = self._signed_request(
            method="GET",
            path=f"/api/v0/workers/{worker_id}/results/{result_id}/canonical-receipt",
            json_body=None,
        )
        if response.status_code == 200:
            payload = response.json()
            cose_b64 = payload.get("cose_signed_blob_b64")
            if not isinstance(cose_b64, str):
                raise CoordinatorError(
                    f"get_canonical_receipt: 200 response missing "
                    f"cose_signed_blob_b64 ({list(payload.keys())})"
                )
            return CanonicalReceiptResponse(
                receipt_id=str(payload.get("receipt_id", "")),
                experiment_id=str(payload.get("experiment_id", "")),
                cose_signed_blob=base64.b64decode(cose_b64),
                signing_key_pubkey_hex=str(payload.get("signing_key_pubkey_hex", "")),
            )
        if response.status_code == 404:
            return None
        if response.status_code == 403:
            raise UnauthorizedError(_error_message(response))
        if response.status_code == 401:
            raise UnauthorizedError(_error_message(response))
        raise CoordinatorError(
            f"get_canonical_receipt: unexpected status {response.status_code}: "
            f"{response.text[:500]}"
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


def _parse_result_submission(payload: dict[str, Any]) -> ResultSubmissionResponse:
    try:
        return ResultSubmissionResponse(
            result_id=str(payload["result_id"]),
            unit_id=str(payload["unit_id"]),
            unit_status_after=str(payload["unit_status_after"]),
            completions_so_far=int(payload["completions_so_far"]),
            replication_target=int(payload["replication_target"]),
        )
    except KeyError as exc:
        raise CoordinatorError(f"result-submission response missing field: {exc}") from exc


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
        latest_release=_parse_latest_release(payload.get("latest_release")),
    )


def _parse_latest_release(raw: object) -> LatestRelease | None:
    """Tolerant parse of the §9 #46 announcement block. The announcement is
    coordinator-supplied display data — a malformed block must NEVER fail the
    heartbeat, so anything unexpected collapses to None. Notes are truncated
    at this wire boundary; the dashboard additionally HTML-escapes."""
    if not isinstance(raw, dict):
        return None
    version = raw.get("version")
    if not isinstance(version, str) or not version:
        return None
    notes = raw.get("headline")
    url = raw.get("release_url")
    return LatestRelease(
        version=version,
        notes=notes[:500] if isinstance(notes, str) else None,
        url=url if isinstance(url, str) else None,
    )


def _parse_oauth_exchange(payload: dict[str, Any]) -> OAuthExchangeResponse:
    try:
        account_id = payload["account_id"]
        binding_token = payload["binding_token"]
        expires_at = _parse_datetime(payload["expires_at"])
        is_new_account = payload["is_new_account"]
    except KeyError as exc:
        raise CoordinatorError(f"oauth_exchange response missing field: {exc}") from exc
    return OAuthExchangeResponse(
        account_id=str(account_id),
        binding_token=str(binding_token),
        expires_at=expires_at,
        is_new_account=bool(is_new_account),
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


def _error_details(response: httpx.Response) -> dict[str, Any]:
    """Extract the `error.details` dict from a FastAPI error response."""
    try:
        detail = response.json().get("detail", {})
    except ValueError:
        return {}
    if not isinstance(detail, dict):
        return {}
    err = detail.get("error")
    if not isinstance(err, dict):
        return {}
    details = err.get("details")
    return details if isinstance(details, dict) else {}


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
