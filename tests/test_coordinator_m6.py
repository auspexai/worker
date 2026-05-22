"""Tests for the M6 coordinator-client methods: oauth_exchange, upgrade_worker,
retire_worker. MockTransport stands in for the coordinator; signed endpoints
are exercised through the same shared signer / oracle pattern the heartbeat
tests use.
"""

from __future__ import annotations

import json

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from auspexai_worker.coordinator import (
    BindingTokenConsumedError,
    BindingTokenExpiredError,
    BindingTokenNotFoundError,
    CoordinatorClient,
    CoordinatorError,
    InvalidAccessTokenError,
    UnauthorizedError,
    UnsupportedIdpError,
    WorkerIdMismatchError,
    WorkerNotFoundError,
)
from auspexai_worker.signing import Rfc9421Signer


def _make_signer() -> tuple[Rfc9421Signer, str]:
    privkey = Ed25519PrivateKey.generate()
    pub = privkey.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    return Rfc9421Signer(privkey, pub), pub


def _make_client(handler, signer: Rfc9421Signer | None = None) -> CoordinatorClient:
    return CoordinatorClient(
        base_url="http://test-coordinator.invalid",
        signer=signer,
        transport=httpx.MockTransport(handler),
    )


def _error_response(status: int, code: str, message: str = "test error") -> httpx.Response:
    # FastAPI HTTPException serializes detail under a top-level "detail" key.
    return httpx.Response(status, json={"detail": {"error": {"code": code, "message": message}}})


# ---- /accounts/oauth/exchange (anonymous-public) ---------------------------


class TestOAuthExchange:
    def test_200_returns_parsed_response(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["method"] = request.method
            captured["path"] = request.url.path
            body = json.loads(request.content)
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "account_id": "acct-abc",
                    "binding_token": "bnd-xyz",
                    "expires_at": "2026-05-22T10:05:00+00:00",
                    "is_new_account": True,
                },
            )

        with _make_client(handler) as client:
            result = client.oauth_exchange(idp="github", access_token="gho_test")

        assert captured["method"] == "POST"
        assert captured["path"] == "/api/v0/accounts/oauth/exchange"
        assert captured["body"] == {"idp": "github", "access_token": "gho_test"}
        assert result.account_id == "acct-abc"
        assert result.binding_token == "bnd-xyz"
        assert result.is_new_account is True

    def test_400_unsupported_idp_raises(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return _error_response(400, "unsupported_idp")

        with _make_client(handler) as client:
            with pytest.raises(UnsupportedIdpError):
                client.oauth_exchange(idp="github", access_token="gho_test")

    def test_401_invalid_access_token_raises(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return _error_response(401, "invalid_access_token")

        with _make_client(handler) as client:
            with pytest.raises(InvalidAccessTokenError):
                client.oauth_exchange(idp="github", access_token="bad")

    def test_unexpected_status_raises_generic(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        with _make_client(handler) as client:
            with pytest.raises(CoordinatorError, match="unexpected status 500"):
                client.oauth_exchange(idp="github", access_token="gho_test")


# ---- /workers/{id}/upgrade (signed) ----------------------------------------


class TestUpgradeWorker:
    def test_200_returns_status_with_T1(self) -> None:
        signer, _pub = _make_signer()
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            body = json.loads(request.content)
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "worker_id": "wkr-1",
                    "trust_tier": 1,
                    "registered_at": "2026-05-20T10:00:00+00:00",
                    "last_heartbeat_at": "2026-05-22T10:00:00+00:00",
                    "retired_at": None,
                },
            )

        with _make_client(handler, signer=signer) as client:
            result = client.upgrade_worker(worker_id="wkr-1", binding_token="bnd-xyz")

        assert captured["path"] == "/api/v0/workers/wkr-1/upgrade"
        assert captured["body"] == {"binding_token": "bnd-xyz"}
        assert result.trust_tier == 1

    def test_400_binding_token_expired_raises(self) -> None:
        signer, _pub = _make_signer()

        def handler(_request: httpx.Request) -> httpx.Response:
            return _error_response(400, "binding_token_expired")

        with _make_client(handler, signer=signer) as client:
            with pytest.raises(BindingTokenExpiredError):
                client.upgrade_worker(worker_id="wkr-1", binding_token="bnd-xyz")

    def test_404_binding_token_not_found_raises(self) -> None:
        signer, _pub = _make_signer()

        def handler(_request: httpx.Request) -> httpx.Response:
            return _error_response(404, "binding_token_not_found")

        with _make_client(handler, signer=signer) as client:
            with pytest.raises(BindingTokenNotFoundError):
                client.upgrade_worker(worker_id="wkr-1", binding_token="bnd-xyz")

    def test_409_binding_token_consumed_raises(self) -> None:
        signer, _pub = _make_signer()

        def handler(_request: httpx.Request) -> httpx.Response:
            return _error_response(409, "binding_token_consumed")

        with _make_client(handler, signer=signer) as client:
            with pytest.raises(BindingTokenConsumedError):
                client.upgrade_worker(worker_id="wkr-1", binding_token="bnd-xyz")

    def test_403_worker_id_mismatch_raises(self) -> None:
        signer, _pub = _make_signer()

        def handler(_request: httpx.Request) -> httpx.Response:
            return _error_response(403, "worker_id_mismatch")

        with _make_client(handler, signer=signer) as client:
            with pytest.raises(WorkerIdMismatchError):
                client.upgrade_worker(worker_id="wkr-1", binding_token="bnd-xyz")

    def test_403_other_raises_unauthorized(self) -> None:
        signer, _pub = _make_signer()

        def handler(_request: httpx.Request) -> httpx.Response:
            return _error_response(403, "credential_class_required")

        with _make_client(handler, signer=signer) as client:
            with pytest.raises(UnauthorizedError):
                client.upgrade_worker(worker_id="wkr-1", binding_token="bnd-xyz")

    def test_404_worker_not_found_raises(self) -> None:
        signer, _pub = _make_signer()

        def handler(_request: httpx.Request) -> httpx.Response:
            return _error_response(404, "worker_not_found")

        with _make_client(handler, signer=signer) as client:
            with pytest.raises(WorkerNotFoundError):
                client.upgrade_worker(worker_id="wkr-missing", binding_token="bnd-xyz")

    def test_without_signer_raises(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        with _make_client(handler, signer=None) as client:
            with pytest.raises(CoordinatorError, match="requires a signer"):
                client.upgrade_worker(worker_id="wkr-1", binding_token="bnd-xyz")


# ---- /workers/{id}/actions/retire (signed) ---------------------------------


class TestRetireWorker:
    def test_200_returns_status_with_retired_at(self) -> None:
        signer, _pub = _make_signer()
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["method"] = request.method
            return httpx.Response(
                200,
                json={
                    "worker_id": "wkr-1",
                    "trust_tier": 1,
                    "registered_at": "2026-05-20T10:00:00+00:00",
                    "last_heartbeat_at": "2026-05-22T10:00:00+00:00",
                    "retired_at": "2026-05-22T11:00:00+00:00",
                },
            )

        with _make_client(handler, signer=signer) as client:
            result = client.retire_worker(worker_id="wkr-1")

        assert captured["method"] == "POST"
        assert captured["path"] == "/api/v0/workers/wkr-1/actions/retire"
        assert result.retired_at is not None

    def test_404_worker_not_found_raises(self) -> None:
        signer, _pub = _make_signer()

        def handler(_request: httpx.Request) -> httpx.Response:
            return _error_response(404, "worker_not_found")

        with _make_client(handler, signer=signer) as client:
            with pytest.raises(WorkerNotFoundError):
                client.retire_worker(worker_id="wkr-missing")

    def test_403_worker_id_mismatch_raises(self) -> None:
        signer, _pub = _make_signer()

        def handler(_request: httpx.Request) -> httpx.Response:
            return _error_response(403, "worker_id_mismatch")

        with _make_client(handler, signer=signer) as client:
            with pytest.raises(WorkerIdMismatchError):
                client.retire_worker(worker_id="wkr-other")

    def test_without_signer_raises(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        with _make_client(handler, signer=None) as client:
            with pytest.raises(CoordinatorError, match="requires a signer"):
                client.retire_worker(worker_id="wkr-1")
