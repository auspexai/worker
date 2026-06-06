"""Tests for the worker dashboard FastAPI app (v0.1.4 §5.14 Layer B)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

import auspexai_worker.dashboard.app as dash_app
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
def client(db: Database, config: WorkerConfig, tmp_path: Path) -> TestClient:
    # M9 leg 4: point the executor setter at a tmp worker.toml so tests never
    # touch the real XDG config.
    return TestClient(build_app(db=db, config=config, config_path=tmp_path / "worker.toml"))


@pytest.fixture
def toml_path(tmp_path: Path) -> Path:
    return tmp_path / "worker.toml"


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
    def test_overview_surfaces_health_and_execution(self, client: TestClient, db: Database) -> None:
        _enroll(db)
        r = client.get("/")
        assert r.status_code == 200
        # the new transparency block + executor-policy badge (default synthetic)
        assert "Health" in r.text and "execution" in r.text
        assert "synthetic only" in r.text
        assert "models in store" in r.text

    def test_models_page_renders_store_and_accelerator(self, client: TestClient) -> None:
        r = client.get("/models")
        assert r.status_code == 200
        assert "Local model store" in r.text
        assert "No models in the store yet." in r.text  # empty store
        # the dashboard shows the detected accelerator + points to the CLI for
        # live HF suggestions (it does not query HF on render)
        assert "This host can run" in r.text
        assert "accelerator" in r.text
        assert "model recommend" in r.text

    def test_config_page_shows_new_blocks(self, client: TestClient) -> None:
        r = client.get("/config")
        assert "execute tenant code" in r.text
        assert "model store dir" in r.text
        assert "thermal thresholds" in r.text

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


class TestSelfPause:
    """§2.1 #11 + follow-on: the dashboard self-pause form carries an optional
    reason (parsed from the urlencoded body, no python-multipart), stored as a
    local note and surfaced back on the overview — mirroring `pause --reason`."""

    def test_self_pause_form_has_reason_input(self, client: TestClient, db: Database) -> None:
        _enroll(db)
        r = client.get("/")
        assert 'action="/self-pause"' in r.text
        assert 'name="reason"' in r.text  # the operator can now enter a reason

    def test_self_pause_with_reason_is_stored_and_surfaced(
        self, client: TestClient, db: Database
    ) -> None:
        _enroll(db)
        r = client.post(
            "/self-pause",
            data={"reason": "rebooting the box"},
            follow_redirects=False,
        )
        assert r.status_code == 303
        self_ = WorkerSelfRepository(db).get()
        assert self_ is not None and self_.self_paused is True
        assert self_.self_pause_reason == "rebooting the box"
        # and it surfaces on the overview's self-paused notice
        overview = client.get("/")
        assert "rebooting the box" in overview.text

    def test_self_pause_without_reason_stores_none(self, client: TestClient, db: Database) -> None:
        _enroll(db)
        r = client.post("/self-pause", data={"reason": "   "}, follow_redirects=False)
        assert r.status_code == 303
        self_ = WorkerSelfRepository(db).get()
        assert self_ is not None and self_.self_paused is True
        # blank ⇒ None (no synthetic "paused from dashboard" placeholder)
        assert self_.self_pause_reason is None

    def test_self_unpause_clears(self, client: TestClient, db: Database) -> None:
        _enroll(db)
        WorkerSelfRepository(db).set_self_pause(True, reason="x")
        r = client.post("/self-unpause", follow_redirects=False)
        assert r.status_code == 303
        self_ = WorkerSelfRepository(db).get()
        assert self_ is not None and self_.self_paused is False


class TestExecutorSetter:
    """M9 leg 4: the dashboard executor-policy setter — one-click downgrades, a
    confirm step before enabling provisioned, writing the worker.toml the daemon
    reads on its next restart."""

    def test_config_page_shows_setter_with_current_mode(self, client: TestClient) -> None:
        r = client.get("/config")
        assert "Code-execution policy" in r.text
        assert 'action="/executor"' in r.text
        # default config is synthetic → the page offers off + enable-provisioned
        assert "enable provisioned" in r.text

    def test_set_off_writes_toml_and_auto_restarts(
        self, client: TestClient, toml_path: Path, monkeypatch
    ) -> None:
        # auto-restart on a UI config change (systemd present → scheduled)
        monkeypatch.setattr(dash_app, "_schedule_detached_restart", lambda: True)
        r = client.post("/executor", data={"policy": "off"})
        assert r.status_code == 200
        assert 'execute_tenant_code = "off"' in toml_path.read_text()
        assert "restarting now" in r.text  # UI says it's restarting to apply

    def test_set_without_systemd_shows_manual_restart(
        self, client: TestClient, toml_path: Path, monkeypatch
    ) -> None:
        # no systemd managing the worker → fall back to a manual-restart instruction
        monkeypatch.setattr(dash_app, "_schedule_detached_restart", lambda: False)
        r = client.post("/executor", data={"policy": "off"})
        assert r.status_code == 200
        assert 'execute_tenant_code = "off"' in toml_path.read_text()
        assert "systemctl --user restart auspexai-worker" in r.text

    def test_enable_provisioned_without_confirm_redirects_to_confirm(
        self, client: TestClient, toml_path: Path
    ) -> None:
        r = client.post("/executor", data={"policy": "provisioned"}, follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/executor/confirm"
        # nothing was written — the deliberate-act gate held
        assert not toml_path.exists() or 'execute_tenant_code = "provisioned"' not in (
            toml_path.read_text()
        )

    def test_enable_provisioned_with_confirm_writes_and_restarts(
        self, client: TestClient, toml_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(dash_app, "_schedule_detached_restart", lambda: True)
        r = client.post("/executor", data={"policy": "provisioned", "confirm": "yes"})
        assert r.status_code == 200
        assert 'execute_tenant_code = "provisioned"' in toml_path.read_text()
        assert "restarting now" in r.text

    def test_confirm_page_warns_and_mentions_restart(self, client: TestClient) -> None:
        r = client.get("/executor/confirm")
        assert r.status_code == 200
        assert "third-party tenant code" in r.text
        assert "restarts the worker" in r.text  # the action's restart is disclosed
        assert "enable provisioned + restart" in r.text

    def test_pending_restart_surfaced_when_file_differs_from_running(
        self, client: TestClient, db: Database, toml_path: Path
    ) -> None:
        """The dashboard reflects the running daemon's snapshot (synthetic here);
        once worker.toml is changed to provisioned, /config shows a pending-restart
        banner and drives the buttons off the configured (on-disk) value — the
        mayhem0 confusion (file=provisioned, daemon still synthetic)."""
        _enroll(db)  # the overview health block (with the badge) renders when enrolled
        toml_path.write_text('[executor]\nexecute_tenant_code = "provisioned"\n')
        r = client.get("/config")
        assert "Pending restart" in r.text
        assert "running" in r.text  # shows the still-running mode
        # buttons reflect configured=provisioned → offers downgrades, NOT "enable provisioned"
        assert "set synthetic (echo only)" in r.text
        assert "enable provisioned" not in r.text
        # overview surfaces it too
        ov = client.get("/")
        assert "pending restart" in ov.text

    def test_invalid_policy_is_ignored(
        self, client: TestClient, toml_path: Path, monkeypatch
    ) -> None:
        # never restarts on a rejected/invalid policy
        monkeypatch.setattr(dash_app, "_schedule_detached_restart", lambda: True)
        r = client.post("/executor", data={"policy": "bogus"}, follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/config"
        assert not toml_path.exists()


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
