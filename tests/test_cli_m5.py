"""Tests for the M5 CLI verbs: `receipts list`, `receipts show`, `log`.

Mirrors the test_cli_m3.py pattern (CliRunner against the click group with
env-var-controlled state-dir).
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from auspexai_worker.cli import cli
from auspexai_worker.state import (
    AssignmentAuditRepository,
    Database,
    MigrationRunner,
    SubmittedResultRepository,
)


def _write_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "worker.toml"
    cfg.write_text(
        '[coordinator]\nurl = "http://m5-test.invalid"\n'
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


def _seed_receipt(
    db: Database,
    *,
    unit_id: str,
    result_id: str,
    payload: str = '{"answer": 42}',
    tenant_id: str | None = None,
) -> None:
    SubmittedResultRepository(db).record(
        unit_id=unit_id,
        assignment_id=f"asg-{unit_id}",
        result_id=result_id,
        exit_code=0,
        completed_at="2026-05-22T10:00:00",
        coord_unit_status_after="completed",
        coord_completions_so_far=3,
        coord_replication_target=3,
        payload_json=payload,
    )
    if tenant_id is not None:
        AssignmentAuditRepository(db).append(
            action="assignment.accept", unit_id=unit_id, tenant_id=tenant_id
        )


class TestReceiptsList:
    def test_empty(self, tmp_path: Path) -> None:
        _bootstrap_state(tmp_path)
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(cfg), "receipts", "list"], env=_env(tmp_path))
        assert result.exit_code == 0, result.output
        assert "no receipts yet" in result.output

    def test_lists_recent_receipts(self, tmp_path: Path) -> None:
        db = _bootstrap_state(tmp_path)
        _seed_receipt(db, unit_id="u-1", result_id="r-1")
        _seed_receipt(db, unit_id="u-2", result_id="r-2")
        db.close()
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(cfg), "receipts", "list"], env=_env(tmp_path))
        assert result.exit_code == 0, result.output
        assert "unit=u-1" in result.output
        assert "unit=u-2" in result.output
        assert "status=placeholder" in result.output

    def test_since_filter(self, tmp_path: Path) -> None:
        db = _bootstrap_state(tmp_path)
        _seed_receipt(db, unit_id="u-1", result_id="r-1")
        db.close()
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--config", str(cfg), "receipts", "list", "--since", "2099-01-01"],
            env=_env(tmp_path),
        )
        assert result.exit_code == 0, result.output
        assert "no receipts match the given filters" in result.output

    def test_tenant_filter(self, tmp_path: Path) -> None:
        db = _bootstrap_state(tmp_path)
        _seed_receipt(db, unit_id="u-1", result_id="r-1", tenant_id="tenant-a")
        _seed_receipt(db, unit_id="u-2", result_id="r-2", tenant_id="tenant-b")
        db.close()
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--config", str(cfg), "receipts", "list", "--tenant", "tenant-a"],
            env=_env(tmp_path),
        )
        assert result.exit_code == 0, result.output
        assert "unit=u-1" in result.output
        assert "unit=u-2" not in result.output


class TestReceiptsShow:
    def test_show_by_result_id(self, tmp_path: Path) -> None:
        db = _bootstrap_state(tmp_path)
        _seed_receipt(db, unit_id="u-1", result_id="r-1", payload='{"answer": 42}')
        db.close()
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--config", str(cfg), "receipts", "show", "r-1"],
            env=_env(tmp_path),
        )
        assert result.exit_code == 0, result.output
        assert "unit_id:" in result.output
        assert "u-1" in result.output
        assert "result_id:" in result.output
        assert "r-1" in result.output
        assert "receipt_status:" in result.output
        assert "placeholder" in result.output
        assert '"answer": 42' in result.output

    def test_show_by_unit_id_falls_back(self, tmp_path: Path) -> None:
        db = _bootstrap_state(tmp_path)
        _seed_receipt(db, unit_id="u-1", result_id="r-1")
        db.close()
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--config", str(cfg), "receipts", "show", "u-1"],
            env=_env(tmp_path),
        )
        assert result.exit_code == 0, result.output
        assert "u-1" in result.output
        assert "r-1" in result.output

    def test_show_not_found(self, tmp_path: Path) -> None:
        _bootstrap_state(tmp_path)
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--config", str(cfg), "receipts", "show", "does-not-exist"],
            env=_env(tmp_path),
        )
        assert result.exit_code == 1
        assert "no receipt found" in result.output


class TestLogCommand:
    def test_empty(self, tmp_path: Path) -> None:
        _bootstrap_state(tmp_path)
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(cfg), "log"], env=_env(tmp_path))
        assert result.exit_code == 0, result.output
        assert "no audit rows match" in result.output

    def test_lists_audit_rows(self, tmp_path: Path) -> None:
        db = _bootstrap_state(tmp_path)
        repo = AssignmentAuditRepository(db)
        repo.append(action="assignment.accept", unit_id="u-1", tenant_id="t-1")
        repo.append(
            action="assignment.refuse", unit_id="u-2", tenant_id="t-2", reason="tenant_deny"
        )
        db.close()
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(cfg), "log"], env=_env(tmp_path))
        assert result.exit_code == 0, result.output
        assert "assignment.accept" in result.output
        assert "assignment.refuse" in result.output
        assert "u-1" in result.output
        assert "u-2" in result.output
        assert "tenant_deny" in result.output

    def test_unit_filter(self, tmp_path: Path) -> None:
        db = _bootstrap_state(tmp_path)
        repo = AssignmentAuditRepository(db)
        repo.append(action="assignment.accept", unit_id="u-1")
        repo.append(action="assignment.accept", unit_id="u-2")
        db.close()
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--config", str(cfg), "log", "--unit", "u-1"], env=_env(tmp_path)
        )
        assert result.exit_code == 0, result.output
        assert "u-1" in result.output
        assert "u-2" not in result.output

    def test_action_filter(self, tmp_path: Path) -> None:
        db = _bootstrap_state(tmp_path)
        repo = AssignmentAuditRepository(db)
        repo.append(action="assignment.accept", unit_id="u-1")
        repo.append(action="assignment.refuse", unit_id="u-2")
        db.close()
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--config", str(cfg), "log", "--action", "assignment.refuse"],
            env=_env(tmp_path),
        )
        assert result.exit_code == 0, result.output
        assert "assignment.refuse" in result.output
        assert "u-2" in result.output
        assert "u-1" not in result.output

    def test_since_filter_future(self, tmp_path: Path) -> None:
        db = _bootstrap_state(tmp_path)
        AssignmentAuditRepository(db).append(action="assignment.accept", unit_id="u-1")
        db.close()
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--config", str(cfg), "log", "--since", "2099-01-01"],
            env=_env(tmp_path),
        )
        assert result.exit_code == 0, result.output
        assert "no audit rows match" in result.output
