"""Tests for the worker dashboard FastAPI app (v0.1.4 §5.14 Layer B)."""

from __future__ import annotations

import json
import time
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
        assert "wkr-test" in r.text  # worker_id rides the heart header now
        assert "T0 anonymous" in r.text
        # the coordinator URL rides the heart's status tooltip now (the heartbeat
        # IS the contact — no separate vital); it's still carried in /api/stats
        assert "coord.auspexai.network" in client.get("/api/stats").json()["coordinator_url"]

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
            # completed_units / distinct_experiments render via renderHeart;
            # receipts + pending are folded into the heart's metrics row (data-live).
            'data-live="receipts_count"',
            'data-live="pending_submissions"',
            'data-live="state_banner"',  # the state is the live banner now
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
            # the activity-heart line (plain headline/detail, unwrapped from the banner)
            "activity_headline",
            "activity_detail",
        ):
            assert key in d, key

    def test_overview_renders_activity_heart(self, client: TestClient, db: Database) -> None:
        """The volunteer's heart monitor — its skeleton (filled by the immediate
        live tick) is on the overview, and the poll script renders into it."""
        _enroll(db)
        r = client.get("/")
        assert r.status_code == 200
        for marker in (
            'id="wkr-heart"',
            'id="heart-strip"',
            'id="heart-dot"',
            'id="heart-narration"',
            "renderHeart",  # the poll fills the heart
        ):
            assert marker in r.text, marker

    def test_overview_card_layout_heart_id_and_no_store(
        self, client: TestClient, db: Database
    ) -> None:
        """The sections render as cards (.grid/.field, researcher-dashboard style);
        worker_id + version ride the heart header; the page is no-store so a reload
        after a daemon roll fetches fresh HTML."""
        _enroll(db)
        r = client.get("/")
        assert 'class="grid"' in r.text and 'class="field"' in r.text
        assert 'class="heart-id"' in r.text  # worker_id · version in the heart
        assert "<dt>worker_id</dt>" not in r.text  # moved OUT of the Identity dl
        assert r.headers.get("cache-control") == "no-store"

    def test_overview_status_badge_and_fault_tone(self, client: TestClient, db: Database) -> None:
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
        assert 'action="/self-pause"' in r.text  # the pause toggle lives in the heart

        # Quarantine = fault → fault-toned notice; the pause control is withdrawn
        # (operator-controlled, the volunteer can't lift it).
        repo.record_operator_hold("quarantine", reason="manipulation suspected")
        r = client.get("/")
        assert "quarantined" in r.text
        assert "notice fault" in r.text
        assert 'action="/self-pause"' not in r.text  # withdrawn under an operator hold

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
    def test_overview_identity_section_is_pure_identity(
        self, client: TestClient, db: Database
    ) -> None:
        """The heart owns the worker's operational state — accelerator, execution,
        thermal, inference are its vitals (client-rendered from /api/stats); the
        remaining server section is pure IDENTITY: public key, enrolled, flavor."""
        _enroll(db)
        r = client.get("/")
        assert r.status_code == 200
        assert "<h2>Identity</h2>" in r.text
        assert "public key" in r.text
        assert "enrolled" in r.text
        # no leftover section headers from the earlier passes:
        assert "<h2>Details</h2>" not in r.text
        assert "<h2>Capabilities</h2>" not in r.text
        assert "<h2>Contribution ledger</h2>" not in r.text
        # accelerator + execution are heart VITALS now — in /api/stats, not a card:
        d = client.get("/api/stats").json()
        assert d.get("accelerator")
        assert "execution" in d
        # The heart leads; then the Identity section.
        assert r.text.index('id="wkr-heart"') < r.text.index("<h2>Identity</h2>")

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
        assert "model store dir" in r.text  # under the grouped Storage section
        assert "<h3>Thermal</h3>" in r.text  # read-only settings are grouped now
        assert "thresholds" in r.text

    def test_config_page_renders_loaded_values(
        self, client: TestClient, config: WorkerConfig
    ) -> None:
        r = client.get("/config")
        assert r.status_code == 200
        assert "coordinator url" in r.text
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
    """§2.1 #11: the dashboard self-pause is a single no-argument button — no
    reason is collected (pausing your own box needs no justification)."""

    def test_self_pause_form_has_no_reason_input(self, client: TestClient, db: Database) -> None:
        _enroll(db)
        r = client.get("/")
        assert 'action="/self-pause"' in r.text
        assert 'name="reason"' not in r.text  # the reason field is gone

    def test_self_pause_toggles_state(self, client: TestClient, db: Database) -> None:
        _enroll(db)
        r = client.post("/self-pause", follow_redirects=False)
        assert r.status_code == 303
        self_ = WorkerSelfRepository(db).get()
        assert self_ is not None and self_.self_paused is True
        assert self_.self_pause_reason is None  # column stays dormant

    def test_self_unpause_clears(self, client: TestClient, db: Database) -> None:
        _enroll(db)
        WorkerSelfRepository(db).set_self_pause(True)
        r = client.post("/self-unpause", follow_redirects=False)
        assert r.status_code == 303
        self_ = WorkerSelfRepository(db).get()
        assert self_ is not None and self_.self_paused is False


class _NoopUnbindClient:
    """Stand-in for CoordinatorClient: a context manager whose `unbind_worker`
    is a no-op (the local dashboard logout route never touches the network)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __enter__(self) -> _NoopUnbindClient:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def unbind_worker(self, *, worker_id: str) -> None:
        return None


def _enroll_bound(db: Database) -> None:
    """Seed a worker that is enrolled AND bound to an account (T1)."""
    _enroll(db)
    WorkerSelfRepository(db).update_after_upgrade(
        new_tier=1,
        account_binding_json='{"account_id": "acct-test"}',
    )


class TestLogout:
    """The local dashboard's `log out` action mirrors the CLI `logout`: unbind at
    the coordinator, then clear the local binding + revert to T0 (worker keeps
    running). Hermetic — the coordinator client + keystore are stubbed."""

    def _patch_deps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The route lazy-imports these from their home modules; patch there.
        monkeypatch.setattr(
            "auspexai_worker.coordinator.client.CoordinatorClient",
            _NoopUnbindClient,
        )
        monkeypatch.setattr("auspexai_worker.bootstrap.open_keystore", lambda config: object())
        monkeypatch.setattr("auspexai_worker.bootstrap.build_signer", lambda keystore: object())

    def test_logout_form_shown_only_when_bound(self, client: TestClient, db: Database) -> None:
        _enroll(db)  # T0 anonymous: no logout control
        assert 'action="/logout"' not in client.get("/").text
        WorkerSelfRepository(db).update_after_upgrade(
            new_tier=1, account_binding_json='{"account_id": "acct-test"}'
        )
        assert 'action="/logout"' in client.get("/").text

    def test_logout_unbinds_and_reverts_to_t0(
        self, client: TestClient, db: Database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enroll_bound(db)
        self._patch_deps(monkeypatch)
        r = client.post("/logout", follow_redirects=False)
        assert r.status_code == 303
        self_ = WorkerSelfRepository(db).get()
        assert self_ is not None
        assert self_.trust_tier == 0
        assert self_.account_binding_json is None

    def test_logout_when_anonymous_is_noop(self, client: TestClient, db: Database) -> None:
        _enroll(db)  # already T0 anonymous
        r = client.post("/logout", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/?logout=anon"
        self_ = WorkerSelfRepository(db).get()
        assert self_ is not None and self_.account_binding_json is None


class _FakeDeviceCode:
    """Minimal stand-in for oauth.DeviceCode (only the fields the route reads)."""

    user_code = "WDJB-MJHT"
    verification_uri = "https://github.com/login/device"
    expires_in = 900
    interval = 5


class _FakeExchange:
    """Stand-in for OAuthExchangeResponse."""

    account_id = "acct-login-test"
    binding_token = "btok-test"
    expires_at = datetime(2026, 6, 20, tzinfo=UTC)
    is_new_account = True


class _FakeStatusAfter:
    trust_tier = 1


class _UpgradeClient:
    """Stand-in for CoordinatorClient used by the dashboard login flow: a context
    manager whose oauth_exchange / upgrade_worker return canned objects (the local
    dashboard test never touches the network)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __enter__(self) -> _UpgradeClient:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def oauth_exchange(self, *, idp: str, access_token: str) -> _FakeExchange:
        return _FakeExchange()

    def upgrade_worker(self, *, worker_id: str, binding_token: str) -> _FakeStatusAfter:
        return _FakeStatusAfter()


class TestLogin:
    """The local dashboard's `log in` action mirrors the CLI `login`: drive GitHub's
    Device Flow in a background thread, exchange + upgrade at the coordinator, then
    persist the binding. Hermetic — run_device_flow, the coordinator client, and the
    keystore are stubbed so nothing touches network/disk. Binds ANONYMOUS (no
    citation opt-in prompt) by design."""

    def _patch_deps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # run_device_flow runs synchronously in the test: invoke on_code (so the
        # user_code is captured) then return a fake token. The route imports it
        # from auspexai_worker.oauth; patch there.
        def _fake_run_device_flow(*, on_code, **kwargs):  # type: ignore[no-untyped-def]
            on_code(_FakeDeviceCode())
            return "fake-access-token"

        monkeypatch.setattr("auspexai_worker.oauth.run_device_flow", _fake_run_device_flow)
        monkeypatch.setattr(
            "auspexai_worker.coordinator.client.CoordinatorClient",
            _UpgradeClient,
        )
        monkeypatch.setattr("auspexai_worker.bootstrap.open_keystore", lambda config: object())
        monkeypatch.setattr("auspexai_worker.bootstrap.build_signer", lambda keystore: object())

    def test_login_form_shown_only_when_anonymous(self, client: TestClient, db: Database) -> None:
        _enroll(db)  # T0 anonymous: the login control shows
        r = client.get("/")
        assert 'action="/login"' in r.text
        # ...and the logout control does NOT (mutually exclusive)
        assert 'action="/logout"' not in r.text
        WorkerSelfRepository(db).update_after_upgrade(
            new_tier=1, account_binding_json='{"account_id": "acct-test"}'
        )
        r2 = client.get("/")
        assert 'action="/login"' not in r2.text  # now bound → no login control
        assert 'action="/logout"' in r2.text

    def test_login_completes_and_binds_anonymous(
        self, client: TestClient, db: Database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enroll(db)  # T0 anonymous
        self._patch_deps(monkeypatch)

        r = client.post("/login", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/login"

        # The flow runs in a daemon thread; poll the worker row for the bound state.
        repo = WorkerSelfRepository(db)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            self_ = repo.get()
            if self_ is not None and self_.account_binding_json is not None:
                break
            time.sleep(0.02)

        self_ = repo.get()
        assert self_ is not None
        assert self_.trust_tier == 1
        assert self_.account_binding_json is not None
        binding = json.loads(self_.account_binding_json)
        assert binding["account_id"] == "acct-login-test"
        assert binding["idp"] == "github"

        # After binding, the dashboard GATES on the one-time citation prompt — even
        # navigating straight to "/" bounces there until the choice is made (no sidestep).
        assert client.get("/", follow_redirects=False).headers["location"] == "/login/citation"
        # The prompt has NO free-text name — credit is the verified GitHub identity.
        assert 'name="name"' not in client.get("/login/citation").text
        # Resolve it (default anonymous; consumes the one-time prompt).
        client.post("/login/citation", data={"choice": "anonymous"})
        # Now the overview offers logout (not login) — mutually exclusive.
        text = client.get("/").text
        assert 'action="/logout"' in text
        assert 'action="/login"' not in text

    def test_login_when_already_bound_is_noop(self, client: TestClient, db: Database) -> None:
        _enroll_bound(db)
        r = client.post("/login", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/"


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
        assert "Runs provisioned tenant code" in r.text  # the calm indicator's tooltip
        # the mayhem1 bug: NO stale "synthetic only" badge from a snapshot read
        # while the live policy is provisioned (the read-only kv row was removed).
        assert "synthetic only" not in r.text
        # the live mode is also a heart vital on the overview — client-rendered from
        # /api/stats (not a server-side card), so assert it on the poll source.
        assert client.get("/api/stats").json()["execution"] == "provisioned"

    def test_invalid_policy_is_ignored(self, client: TestClient, toml_path: Path) -> None:
        r = client.post("/executor", data={"policy": "bogus"}, follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/config"
        assert not toml_path.exists()


class TestNoExternalSurface:
    """The DISPLAY pages take no POST — they're read-only views. (The control
    routes — /self-pause, /executor, /login, /logout — are separate POST
    endpoints with their own tests; this guards the view pages.)"""

    def test_no_post_routes_on_html_pages(self, client: TestClient) -> None:
        for path in ("/", "/activity", "/receipts", "/config", "/api/stats"):
            r = client.post(path, json={})
            # 405 (method not allowed) is the FastAPI default for an
            # unmatched method on a defined route. We don't actually
            # care about 405 vs 404 — what matters is there's no
            # 2xx success.
            assert r.status_code >= 400, f"{path} accepted POST: {r.status_code}"


class TestI4Overview:
    """I4 (ui_triage_first_ia_redesign.md §5): state-banner-first + inference."""

    def test_active_idle_worker_shows_no_banner_and_honest_activity(
        self, client: TestClient, db: Database
    ) -> None:
        """Option B: an idle worker shows NO state banner (the heart owns active/
        idle) — the container stays present-but-empty so the live poll can flip a
        hold in, and the heart's activity source honestly says 'Idle', never
        'Receiving work'."""
        from auspexai_worker.state import WorkerSelfRepository

        _enroll(db)
        WorkerSelfRepository(db).record_heartbeat(datetime.now(UTC), trust_tier=0)
        r = client.get("/")
        start = r.text.index('data-live="state_banner"')
        banner = r.text[start : r.text.index("</div>", start)]
        assert "receiving work" not in banner.lower()  # no overclaim
        # the honest activity lives in the heart's source now
        d = client.get("/api/stats").json()
        assert d["activity_headline"] == "Idle"

    def test_active_worker_with_recent_work_says_receiving(
        self, client: TestClient, db: Database
    ) -> None:
        """A worker that submitted a unit recently accurately reports 'Receiving
        work' — via the heart's activity source (the banner is hold-only now)."""
        from auspexai_worker.state import SubmittedResultRepository, WorkerSelfRepository

        _enroll(db)
        WorkerSelfRepository(db).record_heartbeat(datetime.now(UTC), trust_tier=0)
        SubmittedResultRepository(db).record(
            unit_id="u-1",
            assignment_id="asg-1",
            result_id="res-1",
            exit_code=0,
            completed_at=datetime.now(UTC).isoformat(),
            coord_unit_status_after="in_progress",
            coord_completions_so_far=1,
            coord_replication_target=3,
            payload_json="{}",
        )
        d = client.get("/api/stats").json()
        assert d["activity_headline"] == "Receiving work"

    def test_api_stats_banner_empty_for_active_activity_in_headline(
        self, client: TestClient, db: Database
    ) -> None:
        """The banner is a HOLD alert (option B): EMPTY for an active/idle worker
        (the container collapses); the live activity is carried in
        activity_headline, which the heart renders."""
        from auspexai_worker.state import WorkerSelfRepository

        _enroll(db)
        WorkerSelfRepository(db).record_heartbeat(datetime.now(UTC), trust_tier=0)
        d = client.get("/api/stats").json()
        assert d["state_banner_html"] == "" and d["state_banner_class"] == ""
        assert d["activity_headline"] == "Idle"

    def test_inference_absent_when_backend_none(self, client: TestClient, db: Database) -> None:
        """Not an inference host → the heart's inference vital is null (no card)."""
        _enroll(db)
        assert client.get("/api/stats").json()["inference"] is None

    def test_inference_vital_present_when_ollama(
        self, db: Database, config: WorkerConfig, tmp_path: Path
    ) -> None:
        """An inference host carries live backend reachability in /api/stats, which
        the heart renders as a vital (dot) — not a Capabilities card."""
        import dataclasses

        cfg = dataclasses.replace(config, inference_backend="ollama")
        c = TestClient(build_app(db=db, config=cfg, config_path=tmp_path / "worker.toml"))
        _enroll(db)
        inf = c.get("/api/stats").json()["inference"]
        assert inf["backend"] == "ollama"
        # No Ollama running in the test → the honest unreachable probe result.
        assert inf["reachable"] is False


class TestUpdateNotice:
    """§9 #46: the update-available notice (server-built, escaped, election-only)."""

    def test_no_notice_when_nothing_announced(self, client: TestClient, db: Database) -> None:
        _enroll(db)
        r = client.get("/")
        assert "Update available" not in r.text
        stats = client.get("/api/stats").json()
        assert stats["update_available"] is False
        assert stats["update_notice_html"] == ""

    def test_notice_renders_when_newer_announced(self, client: TestClient, db: Database) -> None:
        _enroll(db)
        WorkerSelfRepository(db).record_latest_release(
            version="99.0.0",
            notes="Worker flavors + official Ollama",
            url="https://github.com/auspexai/worker/releases/tag/v99.0.0",
            at=datetime.now(UTC),
        )
        r = client.get("/")
        assert "Update available: v99.0.0" in r.text
        assert "Worker flavors + official Ollama" in r.text
        assert "getworker.auspexai.network" in r.text
        assert "never automatic" in r.text
        # one-click copy affordance: command in data-cmd, copied via the
        # volunteer's clipboard — still PRINTED/copied, never executed.
        assert 'class="copy-cmd"' in r.text
        assert "data-cmd=" in r.text
        stats = client.get("/api/stats").json()
        assert stats["update_available"] is True
        assert "v99.0.0" in stats["update_notice_html"]

    def test_notice_hidden_when_current(self, client: TestClient, db: Database) -> None:
        # Announcing an OLDER version than the running build shows nothing —
        # display-time comparison, no clearing logic needed.
        _enroll(db)
        WorkerSelfRepository(db).record_latest_release(
            version="0.0.1", notes="ancient", url=None, at=datetime.now(UTC)
        )
        assert "Update available" not in client.get("/").text
        assert client.get("/api/stats").json()["update_available"] is False

    def test_notes_are_escaped(self, client: TestClient, db: Database) -> None:
        # The headline is coordinator-supplied text — treat as untrusted input.
        _enroll(db)
        WorkerSelfRepository(db).record_latest_release(
            version="99.0.0",
            notes='<script>alert("xss")</script>',
            url="javascript:alert(1)",  # non-https → not linked
            at=datetime.now(UTC),
        )
        text = client.get("/").text
        assert "<script>alert" not in text
        assert "&lt;script&gt;" in text
        assert 'href="javascript:' not in text

    def test_flavor_shown_in_identity_and_stats(self, db: Database, tmp_path: Path) -> None:
        _enroll(db)
        cfg = WorkerConfig.load(
            config_path=tmp_path / "no-such-config.toml",
            env={
                "AUSPEXAI_WORKER_STATE_DIR": str(tmp_path / "state"),
                "AUSPEXAI_WORKER_DATA_DIR": str(tmp_path / "data"),
                "AUSPEXAI_WORKER_FLAVOR": "inference",
            },
        )
        c = TestClient(build_app(db=db, config=cfg, config_path=tmp_path / "worker.toml"))
        assert ">inference<" in c.get("/").text.replace("<code>inference</code>", ">inference<")
        assert c.get("/api/stats").json()["flavor"] == "inference"


class TestLoginCitation:
    """The one-time post-login public-citation prompt (System B): shown only right
    after a fresh bind, defaulting to anonymous, re-asked at every login. NOT a
    standing dashboard control. (The bound + fresh-login render is validated live —
    it depends on the in-process login-session state.)"""

    def test_no_persistent_citation_tab(self, client: TestClient) -> None:
        # the standing settings page + nav tab are gone — it lives in the login flow now
        assert 'href="/citation"' not in client.get("/").text

    def test_login_citation_not_fresh_goes_to_dashboard(self, client: TestClient) -> None:
        # not freshly bound (no worker / idle session) ⇒ no prompt, straight to "/"
        r = client.get("/login/citation", follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/"

    def test_login_citation_post_goes_to_dashboard(self, client: TestClient) -> None:
        r = client.post("/login/citation", data={"choice": "anonymous"}, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/"
