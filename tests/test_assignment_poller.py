"""Tests for the assignment poller (M3 daemon loop)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from auspexai_worker.coordinator import CoordinatorClient
from auspexai_worker.daemon import AssignmentPoller
from auspexai_worker.signing import Rfc9421Signer
from auspexai_worker.state import (
    AcceptedSensitiveRepository,
    AssignmentAuditRepository,
    Database,
    ManifestPinRepository,
    MigrationRunner,
    TenantListRepository,
)


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "worker.db")
    MigrationRunner(d).apply_all()
    return d


def _make_signer() -> Rfc9421Signer:
    pk = Ed25519PrivateKey.generate()
    pub = pk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    return Rfc9421Signer(pk, pub)


def _no_work_response() -> dict[str, object]:
    return {
        "assignment_id": None,
        "assigned_at": None,
        "experiment_id": None,
        "work_unit": None,
    }


def _refuse_ok_response(*, assignment_id: str = "asg-1", unit_id: str = "u-1") -> dict:
    return {
        "assignment_id": assignment_id,
        "unit_id": unit_id,
        "refused_at": "2026-05-20T12:00:01+00:00",
        "refused_kind": "manual",
    }


def _route(assignment_responses: list[dict], refuse_response: dict | None = None):
    """Build a httpx.MockTransport handler that dispatches by URL.

    GET /api/v0/workers/{id}/assignments → pops from `assignment_responses`.
    POST .../refuse → returns `refuse_response` (default: 200 OK ack).
    """
    refuse_payload = refuse_response or _refuse_ok_response()

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and req.url.path.endswith("/assignments"):
            if not assignment_responses:
                return httpx.Response(200, json=_no_work_response())
            return httpx.Response(200, json=assignment_responses.pop(0))
        if req.method == "POST" and "/refuse" in req.url.path:
            return httpx.Response(200, json=refuse_payload)
        return httpx.Response(404, json={"detail": {"error": {"code": "unhandled"}}})

    return handler


def _work_response(
    *,
    assignment_id: str = "asg-1",
    coordinator_experiment_id: str = "exp-coord-1",
    unit_id: str = "u-1",
    tenant_id: str = "t-1",
    manifest_sha256: str = "a" * 64,
    payload: dict | None = None,
) -> dict[str, object]:
    return {
        "assignment_id": assignment_id,
        "assigned_at": "2026-05-20T12:00:00+00:00",
        "experiment_id": coordinator_experiment_id,
        "work_unit": {
            "schema_version": "0.1",
            "unit_id": unit_id,
            "tenant_id": tenant_id,
            "experiment_id": "tenant-label",
            "manifest_sha256": manifest_sha256,
            "created_at": "2026-05-20T11:50:00+00:00",
            "payload": payload or {},
        },
    }


def _make_poller(
    db: Database,
    handler,
    *,
    interval: float = 0.0,
) -> tuple[AssignmentPoller, CoordinatorClient]:
    client = CoordinatorClient(
        base_url="http://test-coord.invalid",
        signer=_make_signer(),
        transport=httpx.MockTransport(handler),
    )
    poller = AssignmentPoller(
        coordinator=client,
        worker_id="wkr-test",
        manifest_pins=ManifestPinRepository(db),
        accepted_sensitive=AcceptedSensitiveRepository(db),
        tenant_lists=TenantListRepository(db),
        audit=AssignmentAuditRepository(db),
        interval_seconds=interval,
    )
    return poller, client


class TestPollerNoWork:
    def test_no_work_increments_no_work_polls(self, db: Database) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_no_work_response())

        poller, client = _make_poller(db, handler)
        try:
            stats = poller.run(max_polls=2)
        finally:
            client.close()
        assert stats.polls_attempted == 2
        assert stats.no_work_polls == 2
        assert stats.units_accepted == 0
        # No audit rows written for no-work polls.
        assert AssignmentAuditRepository(db).recent() == []


class TestPollerAccept:
    def test_acceptable_assignment_audited_as_accepted(self, db: Database) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_work_response())

        poller, client = _make_poller(db, handler)
        try:
            stats = poller.run(max_polls=1)
        finally:
            client.close()
        assert stats.units_accepted == 1
        rows = AssignmentAuditRepository(db).recent()
        assert len(rows) == 1
        assert rows[0].action == "accepted"
        assert rows[0].unit_id == "u-1"
        assert rows[0].manifest_sha256 == "a" * 64


class TestPollerRefusalPaths:
    def test_tenant_deny_refuses_and_audits_and_calls_refuse(self, db: Database) -> None:
        TenantListRepository(db).deny_add("t-1")
        handler = _route([_work_response(tenant_id="t-1")])
        poller, client = _make_poller(db, handler)
        try:
            stats = poller.run(max_polls=1)
        finally:
            client.close()
        assert stats.units_refused == 1
        assert stats.refuse_calls_succeeded == 1
        assert stats.refuse_calls_failed == 0
        rows = AssignmentAuditRepository(db).recent()
        assert rows[0].action == "refused_tenant_deny"

    def test_manifest_swap_refuses_after_pin(self, db: Database) -> None:
        handler = _route(
            [
                _work_response(unit_id="u-1", manifest_sha256="a" * 64),
                _work_response(unit_id="u-2", manifest_sha256="b" * 64),
            ]
        )
        poller, client = _make_poller(db, handler)
        try:
            stats = poller.run(max_polls=2)
        finally:
            client.close()
        assert stats.units_accepted == 1
        assert stats.units_refused == 1
        assert stats.refuse_calls_succeeded == 1
        rows = AssignmentAuditRepository(db).recent()
        assert rows[0].action == "refused_manifest_swap"
        assert rows[1].action == "accepted"

    def test_sensitive_default_decline_audits(self, db: Database) -> None:
        handler = _route([_work_response(payload={"sensitive_content_flags": ["dual-use"]})])
        poller, client = _make_poller(db, handler)
        try:
            stats = poller.run(max_polls=1)
        finally:
            client.close()
        assert stats.units_refused == 1
        assert stats.refuse_calls_succeeded == 1
        rows = AssignmentAuditRepository(db).recent()
        assert rows[0].action == "refused_sensitive"

    def test_refuse_call_failure_does_not_kill_loop(self, db: Database) -> None:
        TenantListRepository(db).deny_add("t-1")

        def handler(req: httpx.Request) -> httpx.Response:
            if req.method == "GET" and req.url.path.endswith("/assignments"):
                return httpx.Response(200, json=_work_response(tenant_id="t-1"))
            if "/refuse" in req.url.path:
                return httpx.Response(500, text="coord blew up on refuse")
            return httpx.Response(404)

        poller, client = _make_poller(db, handler)
        try:
            stats = poller.run(max_polls=1)
        finally:
            client.close()
        # Local audit still recorded; only the refuse network call failed.
        assert stats.units_refused == 1
        assert stats.refuse_calls_failed == 1
        assert stats.refuse_calls_succeeded == 0
        rows = AssignmentAuditRepository(db).recent()
        assert rows[0].action == "refused_tenant_deny"

    def test_sensitive_after_accept_passes(self, db: Database) -> None:
        AcceptedSensitiveRepository(db).accept("exp-coord-1")

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=_work_response(
                    payload={"sensitive_content_flags": ["dual-use"]},
                ),
            )

        poller, client = _make_poller(db, handler)
        try:
            stats = poller.run(max_polls=1)
        finally:
            client.close()
        assert stats.units_accepted == 1


class TestPollerErrorResilience:
    def test_coordinator_error_does_not_kill_loop(self, db: Database) -> None:
        responses = [
            httpx.Response(500, text="server blew up"),
            httpx.Response(200, json=_no_work_response()),
        ]

        def handler(req: httpx.Request) -> httpx.Response:
            return responses.pop(0)

        poller, client = _make_poller(db, handler)
        try:
            stats = poller.run(max_polls=2)
        finally:
            client.close()
        assert stats.polls_attempted == 2
        assert stats.polls_failed == 1
        assert stats.polls_succeeded == 1
