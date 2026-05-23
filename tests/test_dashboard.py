"""Tests for the worker dashboard FastAPI app (v0.1.4 §5.14 Layer B)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from auspexai_worker.config import WorkerConfig
from auspexai_worker.dashboard import build_app
from auspexai_worker.state import (
    AssignmentAuditRepository,
    Database,
    MigrationRunner,
    WorkerSelfRepository,
)


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "worker.db")
    MigrationRunner(d).apply_all()
    return d


@pytest.fixture
def config(tmp_path: Path) -> WorkerConfig:
    return WorkerConfig.load(
        config_path=tmp_path / "no-such-config.toml",
        env={
            "AUSPEXAI_WORKER_STATE_DIR": str(tmp_path / "state"),
            "AUSPEXAI_WORKER_DATA_DIR": str(tmp_path / "data"),
        },
    )


@pytest.fixture
def client(db: Database, config: WorkerConfig) -> TestClient:
    return TestClient(build_app(db=db, config=config))


def _enroll(db: Database) -> None:
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key().public_bytes_raw().hex()
    WorkerSelfRepository(db).insert(
        worker_id="wkr-test",
        trust_tier=0,
        pubkey_hex=pub,
        enrolled_at=datetime.now(UTC),
    )


class TestOverview:
    def test_root_shows_not_enrolled_banner_when_no_worker(self, client: TestClient) -> None:
        r = client.get("/")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        assert "not yet enrolled" in r.text
        assert "auspexai-worker bootstrap" in r.text

    def test_root_shows_identity_when_enrolled(self, client: TestClient, db: Database) -> None:
        _enroll(db)
        r = client.get("/")
        assert r.status_code == 200
        assert "wkr-test" in r.text
        assert "T0 anonymous" in r.text
        assert "coord.auspexai.network" in r.text or "Coordinator" in r.text


class TestActivity:
    def test_activity_empty_state(self, client: TestClient, db: Database) -> None:
        r = client.get("/activity")
        assert r.status_code == 200
        assert "No assignment activity yet" in r.text

    def test_activity_renders_recent_rows(self, client: TestClient, db: Database) -> None:
        audit = AssignmentAuditRepository(db)
        audit.append(
            assignment_id="asg-1",
            coordinator_experiment_id="exp-coord-1",
            tenant_id="t-1",
            unit_id="u-1",
            manifest_sha256="a" * 64,
            action="assignment.accepted",
            reason=None,
        )
        r = client.get("/activity")
        assert r.status_code == 200
        assert "u-1" in r.text
        assert "assignment.accepted" in r.text
        assert "t-1" in r.text


class TestReceipts:
    def test_receipts_empty_state(self, client: TestClient) -> None:
        r = client.get("/receipts")
        assert r.status_code == 200
        assert "No receipts yet" in r.text


class TestConfig:
    def test_config_page_renders_loaded_values(
        self, client: TestClient, config: WorkerConfig
    ) -> None:
        r = client.get("/config")
        assert r.status_code == 200
        assert "coordinator_url" in r.text
        assert config.coordinator_url in r.text
        assert "heartbeat interval" in r.text
        assert "60s" in r.text


class TestAPI:
    def test_api_stats_returns_json(self, client: TestClient) -> None:
        r = client.get("/api/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["worker_id"] is None  # not enrolled
        assert body["receipts_count"] == 0
        assert body["audit_count"] == 0
        assert "coordinator_url" in body

    def test_api_stats_after_enroll(self, client: TestClient, db: Database) -> None:
        _enroll(db)
        r = client.get("/api/stats")
        body = r.json()
        assert body["worker_id"] == "wkr-test"
        assert body["trust_tier"] == 0


class TestNoExternalSurface:
    """Sanity-check that the dashboard is read-only: no PUT/POST/DELETE
    surface on any route."""

    def test_no_post_routes_on_html_pages(self, client: TestClient) -> None:
        for path in ("/", "/activity", "/receipts", "/config", "/api/stats"):
            r = client.post(path, json={})
            # 405 (method not allowed) is the FastAPI default for an
            # unmatched method on a defined route. We don't actually
            # care about 405 vs 404 — what matters is there's no
            # 2xx success.
            assert r.status_code >= 400, f"{path} accepted POST: {r.status_code}"
