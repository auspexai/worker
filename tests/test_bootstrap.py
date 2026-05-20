"""Tests for the bootstrap orchestration."""

from __future__ import annotations

import httpx
import pytest

from auspexai_worker.bootstrap import bootstrap, collect_capabilities, initialize_state
from auspexai_worker.config import WorkerConfig
from auspexai_worker.coordinator import CoordinatorClient
from auspexai_worker.keystore import InMemoryKeystore


def _make_coordinator(handler) -> CoordinatorClient:
    return CoordinatorClient(
        base_url="http://test-coordinator.invalid",
        transport=httpx.MockTransport(handler),
    )


class TestBootstrap:
    def test_fresh_run_generates_key_enrolls_and_persists(self, tmp_config: WorkerConfig) -> None:
        keystore = InMemoryKeystore()
        calls: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json

            calls.append(_json.loads(request.content))
            return httpx.Response(
                201,
                json={
                    "worker_id": "wkr-new-001",
                    "trust_tier": 0,
                    "registered_at": "2026-05-20T12:00:00+00:00",
                },
            )

        with _make_coordinator(handler) as client:
            result = bootstrap(
                tmp_config, keystore=keystore, coordinator=client, capabilities={"os": "linux"}
            )
        assert result.fresh_enrollment is True
        assert result.worker_self.worker_id == "wkr-new-001"
        assert result.worker_self.trust_tier == 0
        # Pubkey persisted matches the keystore's actual public key.
        assert keystore.has_key()
        assert len(calls) == 1
        assert calls[0]["pubkey_hex"] == result.worker_self.pubkey_hex
        assert calls[0]["capabilities"] == {"os": "linux"}

    def test_idempotent_re_run_does_not_call_coordinator(self, tmp_config: WorkerConfig) -> None:
        keystore = InMemoryKeystore()
        call_count = [0]

        def handler(request: httpx.Request) -> httpx.Response:
            call_count[0] += 1
            return httpx.Response(
                201,
                json={
                    "worker_id": "wkr-idem-001",
                    "trust_tier": 0,
                    "registered_at": "2026-05-20T12:00:00+00:00",
                },
            )

        with _make_coordinator(handler) as client:
            first = bootstrap(tmp_config, keystore=keystore, coordinator=client)
            # Second run — no coordinator call, same identity.
            second = bootstrap(tmp_config, keystore=keystore, coordinator=client)
        assert call_count[0] == 1
        assert first.fresh_enrollment is True
        assert second.fresh_enrollment is False
        assert first.worker_self.worker_id == second.worker_self.worker_id

    def test_coordinator_409_propagates_typed_error(self, tmp_config: WorkerConfig) -> None:
        from auspexai_worker.coordinator import PubkeyAlreadyEnrolledError

        keystore = InMemoryKeystore()

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

        with _make_coordinator(handler) as client:
            with pytest.raises(PubkeyAlreadyEnrolledError):
                bootstrap(tmp_config, keystore=keystore, coordinator=client)
        # Failed enrollment must not have persisted a worker_self row.
        _, repo = initialize_state(tmp_config)
        assert repo.get() is None


class TestCollectCapabilities:
    def test_returns_lowercase_os_arch_and_python_version(self) -> None:
        caps = collect_capabilities()
        assert caps["os"] == caps["os"].lower()
        assert caps["arch"] == caps["arch"].lower()
        assert isinstance(caps["python_version"], str)
        assert "." in caps["python_version"]
