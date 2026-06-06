"""Tests for the heartbeat daemon loop."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from auspexai_worker.capabilities import (
    Capabilities,
    DeclaredCaps,
    GpuDeclaration,
    GpuObservation,
)
from auspexai_worker.coordinator import CoordinatorClient
from auspexai_worker.daemon import HeartbeatLoop
from auspexai_worker.signing import Rfc9421Signer
from auspexai_worker.state import Database, MigrationRunner, WorkerSelfRepository


def _make_signer() -> Rfc9421Signer:
    pk = Ed25519PrivateKey.generate()
    pub = pk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    return Rfc9421Signer(pk, pub)


def _make_fake_capabilities() -> Capabilities:
    return Capabilities(
        os="linux",
        arch="x86_64",
        python_version="3.12.0",
        ram_total_gb=16.0,
        cpu_count=8,
        gpus_observed=GpuObservation(nvidia=0, amd=False),
        gpus_declared=GpuDeclaration(),
        declared_caps=DeclaredCaps(),
    )


@pytest.fixture
def repo(tmp_path: Path) -> WorkerSelfRepository:
    db = Database(tmp_path / "worker.db")
    MigrationRunner(db).apply_all()
    r = WorkerSelfRepository(db)
    r.insert(
        worker_id="wkr-loop-001",
        trust_tier=0,
        pubkey_hex="a" * 64,
        enrolled_at=datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC),
    )
    return r


class TestLoopTicks:
    def test_max_ticks_bounds_execution(self, repo: WorkerSelfRepository) -> None:
        count = [0]

        def handler(request: httpx.Request) -> httpx.Response:
            count[0] += 1
            return httpx.Response(
                200,
                json={
                    "worker_id": "wkr-loop-001",
                    "trust_tier": 0,
                    "last_heartbeat_at": "2026-05-20T12:05:00+00:00",
                },
            )

        signer = _make_signer()
        with CoordinatorClient(
            base_url="http://test-coord.invalid",
            signer=signer,
            transport=httpx.MockTransport(handler),
        ) as client:
            loop = HeartbeatLoop(
                coordinator=client,
                repo=repo,
                worker_id="wkr-loop-001",
                capability_collector=_make_fake_capabilities,
                interval_seconds=0.0,  # no sleep between ticks
            )
            stats = loop.run(max_ticks=3)

        assert count[0] == 3
        assert stats.ticks_attempted == 3
        assert stats.ticks_succeeded == 3
        assert stats.ticks_failed == 0

    def test_records_heartbeat_in_local_db(self, repo: WorkerSelfRepository) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"worker_id": "wkr-loop-001", "trust_tier": 0})

        with CoordinatorClient(
            base_url="http://test-coord.invalid",
            signer=_make_signer(),
            transport=httpx.MockTransport(handler),
        ) as client:
            loop = HeartbeatLoop(
                coordinator=client,
                repo=repo,
                worker_id="wkr-loop-001",
                capability_collector=_make_fake_capabilities,
                interval_seconds=0.0,
            )
            loop.run(max_ticks=1)

        worker = repo.get()
        assert worker is not None
        assert worker.last_heartbeat_at is not None

    def test_heartbeat_refreshes_local_trust_tier(self, repo: WorkerSelfRepository) -> None:
        # The coordinator's heartbeat response carries the worker's live tier;
        # the loop must persist it so `status`/the dashboard don't show a stale
        # enrollment-time tier after a coord-side promotion/demotion.
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "worker_id": "wkr-loop-001",
                    "trust_tier": 2,  # promoted coord-side; was T0 at enrollment
                    "last_heartbeat_at": "2026-05-20T12:05:00+00:00",
                },
            )

        assert repo.get().trust_tier == 0  # enrollment tier
        with CoordinatorClient(
            base_url="http://test-coord.invalid",
            signer=_make_signer(),
            transport=httpx.MockTransport(handler),
        ) as client:
            loop = HeartbeatLoop(
                coordinator=client,
                repo=repo,
                worker_id="wkr-loop-001",
                capability_collector=_make_fake_capabilities,
                interval_seconds=0.0,
            )
            loop.run(max_ticks=1)

        assert repo.get().trust_tier == 2  # refreshed from the heartbeat response


class TestLoopStopEvent:
    def test_stop_event_aborts_loop(self, repo: WorkerSelfRepository) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"worker_id": "wkr-loop-001", "trust_tier": 0})

        stop_event = threading.Event()
        # Pre-set the event so the loop exits immediately after first tick.
        stop_event.set()

        with CoordinatorClient(
            base_url="http://test-coord.invalid",
            signer=_make_signer(),
            transport=httpx.MockTransport(handler),
        ) as client:
            loop = HeartbeatLoop(
                coordinator=client,
                repo=repo,
                worker_id="wkr-loop-001",
                capability_collector=_make_fake_capabilities,
                interval_seconds=60.0,
                stop_event=stop_event,
            )
            stats = loop.run()

        # stop_event is checked at loop top; since pre-set, no ticks run.
        assert stats.ticks_attempted == 0

    def test_stop_called_after_tick(self, repo: WorkerSelfRepository) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"worker_id": "wkr-loop-001", "trust_tier": 0})

        with CoordinatorClient(
            base_url="http://test-coord.invalid",
            signer=_make_signer(),
            transport=httpx.MockTransport(handler),
        ) as client:
            loop = HeartbeatLoop(
                coordinator=client,
                repo=repo,
                worker_id="wkr-loop-001",
                capability_collector=_make_fake_capabilities,
                interval_seconds=10.0,
            )

            # Stop after the first tick by scheduling stop in another thread.
            def stop_soon() -> None:
                loop.stop()

            timer = threading.Timer(0.1, stop_soon)
            timer.start()
            stats = loop.run(max_ticks=100)
            timer.cancel()

        # At least one tick attempted, but bounded well below max_ticks.
        assert stats.ticks_attempted >= 1
        assert stats.ticks_attempted < 100


class TestLoopErrorResilience:
    def test_coordinator_error_does_not_kill_loop(self, repo: WorkerSelfRepository) -> None:
        responses = [
            httpx.Response(500, text="server blew up"),
            httpx.Response(200, json={"worker_id": "wkr-loop-001", "trust_tier": 0}),
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            return responses.pop(0)

        with CoordinatorClient(
            base_url="http://test-coord.invalid",
            signer=_make_signer(),
            transport=httpx.MockTransport(handler),
        ) as client:
            loop = HeartbeatLoop(
                coordinator=client,
                repo=repo,
                worker_id="wkr-loop-001",
                capability_collector=_make_fake_capabilities,
                interval_seconds=0.0,
            )
            stats = loop.run(max_ticks=2)

        assert stats.ticks_attempted == 2
        assert stats.ticks_succeeded == 1
        assert stats.ticks_failed == 1
        assert stats.last_error is not None

    def test_capability_collector_error_does_not_kill_loop(
        self, repo: WorkerSelfRepository
    ) -> None:
        # Regression: a thermal sysfs read raised TypeError inside capability
        # collection, escaped _tick_once (which only caught CoordinatorError),
        # and killed the heartbeat thread → fleet went silently offline.
        calls = [0]

        def flaky_collector() -> Capabilities:
            calls[0] += 1
            if calls[0] == 1:
                raise TypeError("can't concat NoneType to bytes")  # the Jetson crash
            return _make_fake_capabilities()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"worker_id": "wkr-loop-001", "trust_tier": 0})

        with CoordinatorClient(
            base_url="http://test-coord.invalid",
            signer=_make_signer(),
            transport=httpx.MockTransport(handler),
        ) as client:
            loop = HeartbeatLoop(
                coordinator=client,
                repo=repo,
                worker_id="wkr-loop-001",
                capability_collector=flaky_collector,
                interval_seconds=0.0,
            )
            stats = loop.run(max_ticks=2)

        # The loop survived the collector error and kept going.
        assert stats.ticks_attempted == 2
        assert stats.ticks_failed == 1
        assert stats.ticks_succeeded == 1
        assert stats.last_error is not None
