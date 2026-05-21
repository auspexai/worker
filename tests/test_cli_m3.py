"""Tests for the M3 CLI verbs: queue, peek, accept, refuse, tenant."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from auspexai_worker.cli import cli
from auspexai_worker.state import (
    AcceptedSensitiveRepository,
    AssignmentAuditRepository,
    Database,
    MigrationRunner,
)


def _write_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "worker.toml"
    cfg.write_text(
        '[coordinator]\nurl = "http://m3-test.invalid"\n'
        '[identity]\nkeystore_backend = "encrypted_file"\n'
    )
    return cfg


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "AUSPEXAI_WORKER_STATE_DIR": str(tmp_path / "state"),
        "AUSPEXAI_WORKER_DATA_DIR": str(tmp_path / "data"),
    }


def _bootstrap_state(tmp_path: Path) -> Database:
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    db = Database(state_dir / "worker.db")
    MigrationRunner(db).apply_all()
    return db


class TestQueueCommand:
    def test_empty_queue(self, tmp_path: Path) -> None:
        _bootstrap_state(tmp_path)
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(cfg), "queue"], env=_env(tmp_path))
        assert result.exit_code == 0
        assert "no assignment activity" in result.output

    def test_renders_recent_rows(self, tmp_path: Path) -> None:
        db = _bootstrap_state(tmp_path)
        AssignmentAuditRepository(db).append(action="accepted", unit_id="u-1", tenant_id="t-1")
        db.close()
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(cfg), "queue"], env=_env(tmp_path))
        assert result.exit_code == 0
        assert "accepted" in result.output
        assert "u-1" in result.output


class TestPeekCommand:
    def test_missing_unit(self, tmp_path: Path) -> None:
        _bootstrap_state(tmp_path)
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(cfg), "peek", "u-missing"], env=_env(tmp_path))
        assert result.exit_code == 0
        assert "no local record" in result.output

    def test_shows_audit_for_known_unit(self, tmp_path: Path) -> None:
        db = _bootstrap_state(tmp_path)
        AssignmentAuditRepository(db).append(
            action="accepted",
            unit_id="u-1",
            tenant_id="t-1",
            coordinator_experiment_id="exp-1",
            manifest_sha256="a" * 64,
        )
        db.close()
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(cfg), "peek", "u-1"], env=_env(tmp_path))
        assert result.exit_code == 0
        assert "accepted" in result.output
        assert "a" * 64 in result.output


class TestAcceptCommand:
    def test_inserts_into_accepted_table(self, tmp_path: Path) -> None:
        _bootstrap_state(tmp_path)
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(cfg), "accept", "exp-1"], env=_env(tmp_path))
        assert result.exit_code == 0
        assert "accepted: exp-1" in result.output
        # Open a fresh connection to confirm persistence.
        db2 = Database(tmp_path / "state" / "worker.db")
        assert AcceptedSensitiveRepository(db2).contains("exp-1")


class TestTenantCommand:
    def test_allow_then_list(self, tmp_path: Path) -> None:
        _bootstrap_state(tmp_path)
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        runner.invoke(cli, ["--config", str(cfg), "tenant", "allow", "t-1"], env=_env(tmp_path))
        result = runner.invoke(cli, ["--config", str(cfg), "tenant", "list"], env=_env(tmp_path))
        assert result.exit_code == 0
        assert "t-1" in result.output

    def test_deny_then_list(self, tmp_path: Path) -> None:
        _bootstrap_state(tmp_path)
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        runner.invoke(cli, ["--config", str(cfg), "tenant", "deny", "t-bad"], env=_env(tmp_path))
        result = runner.invoke(cli, ["--config", str(cfg), "tenant", "list"], env=_env(tmp_path))
        assert result.exit_code == 0
        assert "t-bad" in result.output


class TestRefuseCommand:
    def test_records_audit_row(self, tmp_path: Path) -> None:
        _bootstrap_state(tmp_path)
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--config", str(cfg), "refuse", "u-1", "--reason", "testing"],
            env=_env(tmp_path),
        )
        assert result.exit_code == 0
        db = Database(tmp_path / "state" / "worker.db")
        rows = AssignmentAuditRepository(db).recent()
        assert len(rows) == 1
        assert rows[0].action == "refused_manual"
        assert rows[0].unit_id == "u-1"
        assert rows[0].reason == "testing"
