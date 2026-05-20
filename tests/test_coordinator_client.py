"""Tests for the coordinator HTTP client.

We use httpx.MockTransport rather than spinning up a real coordinator, so
these tests stay hermetic.
"""

from __future__ import annotations

import json
from datetime import datetime

import httpx
import pytest

from auspexai_worker.coordinator import (
    CoordinatorClient,
    CoordinatorError,
    PubkeyAlreadyEnrolledError,
    PubkeyAlreadyTenantError,
)


def _make_client(handler) -> CoordinatorClient:
    return CoordinatorClient(
        base_url="http://test-coordinator.invalid",
        transport=httpx.MockTransport(handler),
    )


class TestEnrollHappyPath:
    def test_201_returns_parsed_response(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "POST"
            assert request.url.path == "/api/v0/workers/enroll"
            body = json.loads(request.content)
            assert body["pubkey_hex"] == "a" * 64
            assert body["capabilities"] == {"os": "linux"}
            return httpx.Response(
                201,
                json={
                    "worker_id": "wkr-abc123",
                    "trust_tier": 0,
                    "registered_at": "2026-05-20T12:00:00+00:00",
                },
            )

        with _make_client(handler) as client:
            response = client.enroll(pubkey_hex="a" * 64, capabilities={"os": "linux"})
        assert response.worker_id == "wkr-abc123"
        assert response.trust_tier == 0
        assert response.registered_at == datetime.fromisoformat("2026-05-20T12:00:00+00:00")

    def test_accepts_trailing_z_in_registered_at(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                201,
                json={
                    "worker_id": "wkr-abc",
                    "trust_tier": 0,
                    "registered_at": "2026-05-20T12:00:00Z",
                },
            )

        with _make_client(handler) as client:
            response = client.enroll(pubkey_hex="a" * 64, capabilities={})
        assert response.registered_at.tzinfo is not None


class TestEnrollErrors:
    def test_409_pubkey_already_enrolled_maps_to_typed_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                409,
                json={
                    "detail": {
                        "error": {
                            "code": "pubkey_already_enrolled",
                            "message": "this pubkey is already registered as a worker",
                        }
                    }
                },
            )

        with _make_client(handler) as client:
            with pytest.raises(PubkeyAlreadyEnrolledError):
                client.enroll(pubkey_hex="a" * 64, capabilities={})

    def test_409_pubkey_already_tenant_maps_to_typed_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                409,
                json={
                    "detail": {
                        "error": {
                            "code": "pubkey_already_tenant",
                            "message": "this pubkey is registered as a tenant maintainer",
                        }
                    }
                },
            )

        with _make_client(handler) as client:
            with pytest.raises(PubkeyAlreadyTenantError):
                client.enroll(pubkey_hex="a" * 64, capabilities={})

    def test_unexpected_5xx_raises_coordinator_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="internal blowup")

        with _make_client(handler) as client:
            with pytest.raises(CoordinatorError):
                client.enroll(pubkey_hex="a" * 64, capabilities={})

    def test_malformed_201_body_raises_coordinator_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(201, json={"missing": "fields"})

        with _make_client(handler) as client:
            with pytest.raises(CoordinatorError):
                client.enroll(pubkey_hex="a" * 64, capabilities={})

    def test_transport_error_raises_coordinator_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        with _make_client(handler) as client:
            with pytest.raises(CoordinatorError):
                client.enroll(pubkey_hex="a" * 64, capabilities={})
