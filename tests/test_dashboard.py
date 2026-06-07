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

    def test_enrolled_overview_is_live(self, client: TestClient, db: Database) -> None:
        """M6 #3: the overview carries the baseline-poll updater — a "● live"
        indicator, the /api/stats poll script, and data-live markers on EVERY
        metric the poll refreshes (heartbeat, activity counts, progress, thermal)
        — not just one field."""
        _enroll(db)
        r = client.get("/")
        assert r.status_code == 200
        assert 'id="live-ind"' in r.text  # header indicator
        assert "/api/stats" in r.text  # the poll target is wired into the script
        for marker in (
            'data-live="last_heartbeat_at"',
            'data-live="receipts_count"',
            'data-live="pending_submissions"',
            'data-live="audit_count"',
            'data-live="completed_units"',
            'data-live="distinct_experiments"',
            'data-live="thermal"',
            'data-live="worker_state"',
        ):
            assert marker in r.text, marker

    def test_api_stats_includes_thermal_progress_and_state(
        self, client: TestClient, db: Database
    ) -> None:
        """The poll source carries the telemetry + derived state the overview shows."""
        _enroll(db)
        d = client.get("/api/stats").json()
        for key in (
            "thermal_enabled",
            "completed_units",
            "distinct_experiments",
            "worker_state",
            "state_label",
            "state_tone",
        ):
            assert key in d, key

    def test_overview_status_badge_and_fault_tone(
        self, client: TestClient, db: Database
    ) -> None:
        """§2.1 #11: the overview shows a single worker-state badge, and a
        quarantine (the one fault signal) renders the fault-toned notice while a
        no-fault operator pause does not."""
        from auspexai_worker.state import WorkerSelfRepository

        _enroll(db)
        repo = WorkerSelfRepository(db)
        # Fresh heartbeat → active; no fault notice; the volunteer pause control is offered.
        repo.record_heartbeat(datetime.now(UTC), trust_tier=0)
        r = client.get("/")
        assert "active" in r.text
        assert "notice fault" not in r.text
        assert "pause this worker" in r.text

        # Quarantine = fault → fault-toned notice; the pause control is withdrawn
        # (operator-controlled, the volunteer can't lift it).
        repo.record_operator_hold("quarantine", reason="manipulation suspected")
        r = client.get("/")
        assert "quarantined" in r.text
        assert "notice fault" in r.text
        assert "pause this worker" not in r.text

        # No-fault operator pause → neutral notice (no fault tone).
        repo.record_operator_hold("pause", reason="rolling upgrade")
        r = client.get("/")
        assert "paused by operator" in r.text
        assert "notice fault" not in r.text

    def test_static_pages_are_not_live(self, client: TestClient, db: Database) -> None:
        """Static log/config pages don't carry the poll script (no live data)."""
        _enroll(db)
        assert 'id="live-ind"' not in client.get("/activity").text
        assert 'id="live-ind"' not in client.get("/config").text


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
        # execute_tenant_code now lives in its own live "Code-execution policy"
        # section (not the read-only table, which would show a stale snapshot).
        assert "Code-execution policy" in r.text
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
    confirm step before enabling provisioned, written to worker.toml and HOT-RELOADED
    by the daemon (no restart). The dashboard reflects the live on-disk value."""

    def test_config_page_shows_setter_with_current_mode(self, client: TestClient) -> None:
        r = client.get("/config")
        assert "Code-execution policy" in r.text
        assert 'action="/executor"' in r.text
        # default config is synthetic → the page offers off + enable-provisioned
        assert "enable provisioned" in r.text
        assert "no restart needed" in r.text  # hot-reload disclosed

    def test_set_off_writes_toml_and_redirects(self, client: TestClient, toml_path: Path) -> None:
        r = client.post("/executor", data={"policy": "off"}, follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/config"
        assert 'execute_tenant_code = "off"' in toml_path.read_text()

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

    def test_enable_provisioned_with_confirm_writes_toml(
        self, client: TestClient, toml_path: Path
    ) -> None:
        r = client.post(
            "/executor", data={"policy": "provisioned", "confirm": "yes"}, follow_redirects=False
        )
        assert r.status_code == 303
        assert r.headers["location"] == "/config"
        assert 'execute_tenant_code = "provisioned"' in toml_path.read_text()

    def test_confirm_page_discloses_no_restart(self, client: TestClient) -> None:
        r = client.get("/executor/confirm")
        assert r.status_code == 200
        assert "third-party tenant code" in r.text
        assert "no restart needed" in r.text  # hot-reload, not a restart
        assert "Yes, enable provisioned execution" in r.text

    def test_dashboard_reflects_live_on_disk_policy(
        self, client: TestClient, db: Database, toml_path: Path
    ) -> None:
        """No stale snapshot / pending-restart split — the dashboard reads the live
        on-disk policy (which the daemon hot-reloads). Writing provisioned → /config
        shows it as current immediately + offers downgrades, no banner."""
        _enroll(db)  # overview health block (with the badge) renders when enrolled
        toml_path.write_text('[executor]\nexecute_tenant_code = "provisioned"\n')
        r = client.get("/config")
        assert "Pending restart" not in r.text  # no stale-snapshot banner anymore
        assert "set synthetic (echo only)" in r.text  # offers downgrades from provisioned
        assert "enable provisioned" not in r.text
        assert "runs provisioned tenant code" in r.text
        # the mayhem1 bug: NO stale "synthetic only" badge from a snapshot read
        # while the live policy is provisioned (the read-only kv row was removed).
        assert "synthetic only" not in r.text
        # overview shows the live mode too
        ov = client.get("/")
        assert "runs provisioned tenant code" in ov.text

    def test_invalid_policy_is_ignored(self, client: TestClient, toml_path: Path) -> None:
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
