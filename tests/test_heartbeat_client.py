"""Tests for CoordinatorClient.heartbeat() — the M2 signed endpoint."""

from __future__ import annotations

import json

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from auspexai_worker.coordinator import (
    CoordinatorClient,
    CoordinatorError,
    UnauthorizedError,
    WorkerIdMismatchError,
    WorkerNotFoundError,
)
from auspexai_worker.signing import Rfc9421Signer
from tests._signature_oracle import verify_request as oracle_verify


def _make_signer() -> tuple[Rfc9421Signer, str]:
    privkey = Ed25519PrivateKey.generate()
    pub = privkey.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    return Rfc9421Signer(privkey, pub), pub


def _make_client(handler, signer: Rfc9421Signer | None) -> CoordinatorClient:
    return CoordinatorClient(
        base_url="http://test-coordinator.invalid",
        signer=signer,
        transport=httpx.MockTransport(handler),
    )


class TestHeartbeatHappyPath:
    def test_signed_request_arrives_with_capabilities(self) -> None:
        signer, pub = _make_signer()
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["method"] = request.method
            captured["path"] = request.url.path
            captured["host"] = request.headers.get("host")
            captured["headers"] = dict(request.headers)
            captured["body"] = bytes(request.content)
            # Oracle-verify so we exercise the wire format.
            oracle_verify(
                method=request.method,
                path=request.url.raw_path.decode("ascii"),
                authority=request.headers["host"],
                body=bytes(request.content),
                headers={
                    "Signature-Input": request.headers["Signature-Input"],
                    "Signature": request.headers["Signature"],
                    **(
                        {"Content-Digest": request.headers["Content-Digest"]}
                        if "Content-Digest" in request.headers
                        else {}
                    ),
                },
                expected_keyid_hex=pub,
            )
            return httpx.Response(
                200,
                json={
                    "worker_id": "wkr-a",
                    "trust_tier": 0,
                    "registered_at": "2026-05-20T12:00:00+00:00",
                    "last_heartbeat_at": "2026-05-20T12:05:00+00:00",
                },
            )

        with _make_client(handler, signer) as client:
            response = client.heartbeat(worker_id="wkr-a", capabilities={"os": "linux"})

        assert captured["method"] == "POST"
        assert captured["path"] == "/api/v0/workers/wkr-a/heartbeat"
        assert json.loads(captured["body"]) == {"capabilities": {"os": "linux"}}  # type: ignore[arg-type]
        assert response.worker_id == "wkr-a"
        assert response.trust_tier == 0
        assert response.last_heartbeat_at is not None

    def test_no_capabilities_sends_empty_body_object(self) -> None:
        signer, _pub = _make_signer()

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body == {}
            return httpx.Response(
                200,
                json={
                    "worker_id": "wkr-a",
                    "trust_tier": 0,
                },
            )

        with _make_client(handler, signer) as client:
            response = client.heartbeat(worker_id="wkr-a")
        assert response.worker_id == "wkr-a"


class TestHeartbeatErrors:
    def test_403_worker_id_mismatch_maps_to_typed_error(self) -> None:
        signer, _pub = _make_signer()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                403,
                json={
                    "detail": {
                        "error": {
                            "code": "worker_id_mismatch",
                            "message": "credential worker_id does not match URL worker_id",
                        }
                    }
                },
            )

        with _make_client(handler, signer) as client:
            with pytest.raises(WorkerIdMismatchError):
                client.heartbeat(worker_id="wkr-b", capabilities={})

    def test_403_other_codes_map_to_unauthorized(self) -> None:
        signer, _pub = _make_signer()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                403,
                json={"detail": {"error": {"code": "forbidden", "message": "no"}}},
            )

        with _make_client(handler, signer) as client:
            with pytest.raises(UnauthorizedError):
                client.heartbeat(worker_id="wkr-a", capabilities={})

    def test_401_maps_to_unauthorized(self) -> None:
        signer, _pub = _make_signer()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={"detail": {"error": {"code": "bad_signature", "message": "no"}}},
            )

        with _make_client(handler, signer) as client:
            with pytest.raises(UnauthorizedError):
                client.heartbeat(worker_id="wkr-a", capabilities={})

    def test_404_maps_to_worker_not_found(self) -> None:
        signer, _pub = _make_signer()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                json={"detail": {"error": {"code": "worker_not_found", "message": "no"}}},
            )

        with _make_client(handler, signer) as client:
            with pytest.raises(WorkerNotFoundError):
                client.heartbeat(worker_id="wkr-a", capabilities={})

    def test_heartbeat_without_signer_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        with _make_client(handler, signer=None) as client:
            with pytest.raises(CoordinatorError):
                client.heartbeat(worker_id="wkr-a", capabilities={})


# ---- /assignments/{unit_id}/refuse ---------------------------------------


class TestRefuseAssignment:
    def test_200_returns_parsed_response(self) -> None:
        signer, _pub = _make_signer()
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "assignment_id": "asg-1",
                    "unit_id": "u-1",
                    "refused_at": "2026-05-20T12:00:00+00:00",
                    "refused_kind": "manifest_swap",
                },
            )

        with _make_client(handler, signer) as client:
            response = client.refuse_assignment(
                worker_id="wkr-a",
                unit_id="u-1",
                kind="manifest_swap",
                reason="hash diverged",
            )
        assert captured["path"] == "/api/v0/workers/wkr-a/assignments/u-1/refuse"
        assert captured["body"] == {"kind": "manifest_swap", "reason": "hash diverged"}
        assert response.assignment_id == "asg-1"
        assert response.refused_kind == "manifest_swap"

    def test_404_maps_to_assignment_not_found(self) -> None:
        from auspexai_worker.coordinator import AssignmentNotFoundError

        signer, _pub = _make_signer()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                json={"detail": {"error": {"code": "assignment_not_found", "message": "no"}}},
            )

        with _make_client(handler, signer) as client:
            with pytest.raises(AssignmentNotFoundError):
                client.refuse_assignment(worker_id="wkr-a", unit_id="u-1", kind="x", reason="y")

    def test_409_maps_to_already_resolved(self) -> None:
        from auspexai_worker.coordinator import AssignmentAlreadyResolvedError

        signer, _pub = _make_signer()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                409,
                json={
                    "detail": {"error": {"code": "assignment_already_resolved", "message": "no"}}
                },
            )

        with _make_client(handler, signer) as client:
            with pytest.raises(AssignmentAlreadyResolvedError):
                client.refuse_assignment(worker_id="wkr-a", unit_id="u-1", kind="x", reason="y")

    def test_403_worker_id_mismatch_maps_to_typed_error(self) -> None:
        signer, _pub = _make_signer()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                403,
                json={"detail": {"error": {"code": "worker_id_mismatch", "message": "no"}}},
            )

        with _make_client(handler, signer) as client:
            with pytest.raises(WorkerIdMismatchError):
                client.refuse_assignment(worker_id="wkr-b", unit_id="u-1", kind="x", reason="y")
