"""§2.1 #11 CLI: volunteer self-pause + the executor policy setter."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from click.testing import CliRunner

from auspexai_worker.cli import cli
from auspexai_worker.config import WorkerConfig
from auspexai_worker.state import Database, MigrationRunner, WorkerSelfRepository


def _cfg(tmp_path: Path) -> Path:
    cfg = tmp_path / "worker.toml"
    cfg.write_text('[coordinator]\nurl = "http://t.invalid"\n')
    return cfg


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "AUSPEXAI_WORKER_STATE_DIR": str(tmp_path / "state"),
        "AUSPEXAI_WORKER_DATA_DIR": str(tmp_path / "data"),
    }


def _enroll(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    db = Database(state / "worker.db")
    MigrationRunner(db).apply_all()
    WorkerSelfRepository(db).insert(
        worker_id="wkr-x", trust_tier=0, pubkey_hex="a" * 64, enrolled_at=datetime.now(UTC)
    )
    db.close()


def _self_paused(tmp_path: Path) -> bool:
    db = Database(tmp_path / "state" / "worker.db")
    try:
        return bool(WorkerSelfRepository(db).get().self_paused)
    finally:
        db.close()


def test_pause_then_unpause_toggles_state(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _enroll(tmp_path)
    r = CliRunner().invoke(cli, ["--config", str(cfg), "pause"], env=_env(tmp_path))
    assert r.exit_code == 0, r.output
    assert _self_paused(tmp_path) is True
    r2 = CliRunner().invoke(cli, ["--config", str(cfg), "unpause"], env=_env(tmp_path))
    assert r2.exit_code == 0, r2.output
    assert _self_paused(tmp_path) is False


def test_pause_requires_enrollment(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)  # no worker enrolled
    r = CliRunner().invoke(cli, ["--config", str(cfg), "pause"], env=_env(tmp_path))
    assert r.exit_code == 1
    assert "not enrolled" in r.output


def test_executor_set_writes_toml(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    r = CliRunner().invoke(
        cli,
        ["--config", str(cfg), "executor", "set", "provisioned", "--auto-acquire"],
        env=_env(tmp_path),
    )
    assert r.exit_code == 0, r.output
    loaded = WorkerConfig.load(config_path=cfg)
    assert loaded.execute_tenant_code == "provisioned"
    assert loaded.auto_acquire is True
    # The pre-existing [coordinator] section is preserved (targeted upsert).
    assert loaded.coordinator_url == "http://t.invalid"
    # Hot-reload: no restart — the daemon picks it up within a heartbeat.
    assert "no restart needed" in r.output


def test_executor_auto_acquire_toggles_only_the_flag(tmp_path: Path) -> None:
    """The onramp/dashboard auto-acquire toggle: writes ONLY [executor]
    auto_acquire, leaving execute_tenant_code (and other sections) untouched."""
    cfg = _cfg(tmp_path)
    # Establish a non-default execution policy, so preservation is observable.
    CliRunner().invoke(
        cli, ["--config", str(cfg), "executor", "set", "provisioned"], env=_env(tmp_path)
    )
    r = CliRunner().invoke(
        cli, ["--config", str(cfg), "executor", "auto-acquire", "on"], env=_env(tmp_path)
    )
    assert r.exit_code == 0, r.output
    loaded = WorkerConfig.load(config_path=cfg)
    assert loaded.auto_acquire is True
    assert loaded.execute_tenant_code == "provisioned"  # policy untouched
    assert loaded.coordinator_url == "http://t.invalid"  # other section preserved
    assert "no restart needed" in r.output
    # Toggling off flips only the flag.
    r2 = CliRunner().invoke(
        cli, ["--config", str(cfg), "executor", "auto-acquire", "off"], env=_env(tmp_path)
    )
    assert r2.exit_code == 0, r2.output
    loaded2 = WorkerConfig.load(config_path=cfg)
    assert loaded2.auto_acquire is False
    assert loaded2.execute_tenant_code == "provisioned"
