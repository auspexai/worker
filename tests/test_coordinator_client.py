"""Tests for the coordinator HTTP client.

We use httpx.MockTransport rather than spinning up a real coordinator, so
these tests stay hermetic.
"""

from __future__ import annotations

import json
from datetime import datetime

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from auspexai_worker.coordinator import (
    CoordinatorClient,
    CoordinatorError,
    PubkeyAlreadyEnrolledError,
    PubkeyAlreadyTenantError,
    UnauthorizedError,
)
from auspexai_worker.signing import Rfc9421Signer


def _make_client(handler) -> CoordinatorClient:
    return CoordinatorClient(
        base_url="http://test-coordinator.invalid",
        transport=httpx.MockTransport(handler),
    )


def _make_signer() -> Rfc9421Signer:
    privkey = Ed25519PrivateKey.generate()
    pub = privkey.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    return Rfc9421Signer(privkey, pub)


def _make_client_signed(handler, signer: Rfc9421Signer) -> CoordinatorClient:
    return CoordinatorClient(
        base_url="http://test-coordinator.invalid",
        signer=signer,
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


class TestGetSupportedModels:
    """GET /api/v0/models/supported — the coordinator-owned provisionable catalog
    that `model recommend` now consults (worker-credentialed, signed)."""

    def test_200_parses_models(self) -> None:
        signer = _make_signer()

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "GET"
            assert request.url.path == "/api/v0/models/supported"
            # signed request carries the RFC 9421 headers
            assert "Signature" in request.headers
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "model_id": "qwen2.5-0.5b-instruct-q4_k_m",
                            "display_name": "Qwen2.5 0.5B Instruct",
                            "family": "qwen",
                            "param_b": 0.5,
                            "quant": "Q4_K_M",
                            "approx_ram_gb": 1.5,
                            "on_worker_count": 2,
                            "fits_worker_count": 3,
                            "ram_known_workers": 3,
                            "status": "available",
                            "in_catalog": True,
                            "hf_repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                        },
                        {
                            "model_id": "held-but-uncurated",
                            "display_name": "held-but-uncurated",
                            "family": "",
                            "param_b": None,
                            "quant": "",
                            "approx_ram_gb": None,
                            "on_worker_count": 1,
                            "status": "available",
                            "in_catalog": False,
                            "hf_repo": None,
                        },
                    ],
                    "total_active_workers": 3,
                    "fleet_can_auto_acquire": True,
                    "catalog_source": "curated",
                    "catalog_fetched_at": None,
                },
            )

        with _make_client_signed(handler, signer) as client:
            models = client.get_supported_models()

        assert [m.model_id for m in models] == [
            "qwen2.5-0.5b-instruct-q4_k_m",
            "held-but-uncurated",
        ]
        assert models[0].approx_ram_gb == 1.5
        assert models[0].param_b == 0.5
        assert models[0].hf_repo == "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
        assert models[0].status == "available"
        # uncurated entry: sizing metadata is null-safe
        assert models[1].approx_ram_gb is None
        assert models[1].param_b is None
        assert models[1].hf_repo is None

    def test_requires_signer(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            return httpx.Response(200, json={"models": []})

        with _make_client(handler) as client:  # no signer
            with pytest.raises(CoordinatorError):
                client.get_supported_models()

    def test_403_raises_unauthorized(self) -> None:
        signer = _make_signer()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                403,
                json={
                    "detail": {
                        "error": {"code": "forbidden", "message": "worker credential required"}
                    }
                },
            )

        with _make_client_signed(handler, signer) as client:
            with pytest.raises(UnauthorizedError):
                client.get_supported_models()

    def test_unexpected_status_raises_coordinator_error(self) -> None:
        signer = _make_signer()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        with _make_client_signed(handler, signer) as client:
            with pytest.raises(CoordinatorError):
                client.get_supported_models()
