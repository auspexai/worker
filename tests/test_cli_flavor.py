"""§9 #46 CLI surfaces: `flavor`, `inference set-backend`, and the `status`
update-available block."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from click.testing import CliRunner

from auspexai_worker.cli import cli
from auspexai_worker.state import Database, MigrationRunner, WorkerSelfRepository


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "AUSPEXAI_WORKER_STATE_DIR": str(tmp_path / "state"),
        "AUSPEXAI_WORKER_DATA_DIR": str(tmp_path / "data"),
    }


def _config(tmp_path: Path, body: str = "") -> Path:
    p = tmp_path / "worker.toml"
    p.write_text(body, encoding="utf-8")
    return p


class TestFlavorCommands:
    def test_show_default_lean(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        r = CliRunner().invoke(cli, ["--config", str(cfg), "flavor", "show"], env=_env(tmp_path))
        assert r.exit_code == 0, r.output
        assert "lean (default" in r.output

    def test_set_then_show(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        runner = CliRunner()
        r = runner.invoke(
            cli, ["--config", str(cfg), "flavor", "set", "inference"], env=_env(tmp_path)
        )
        assert r.exit_code == 0, r.output
        assert 'flavor = "inference"' in cfg.read_text(encoding="utf-8")
        r = runner.invoke(
            cli, ["--config", str(cfg), "flavor", "show", "--raw"], env=_env(tmp_path)
        )
        assert r.output.strip() == "inference"

    def test_set_rejects_bad_shape(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        r = CliRunner().invoke(
            cli, ["--config", str(cfg), "flavor", "set", "Not Valid!"], env=_env(tmp_path)
        )
        assert r.exit_code == 1
        assert "ERROR" in r.output


class TestInferenceCommands:
    def test_set_backend_writes_toml_and_warns_restart(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        r = CliRunner().invoke(
            cli,
            ["--config", str(cfg), "inference", "set-backend", "ollama"],
            env=_env(tmp_path),
        )
        assert r.exit_code == 0, r.output
        assert 'backend = "ollama"' in cfg.read_text(encoding="utf-8")
        assert "restart the daemon" in r.output

    def test_show(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path, '[inference]\nbackend = "ollama"\n')
        r = CliRunner().invoke(cli, ["--config", str(cfg), "inference", "show"], env=_env(tmp_path))
        assert r.exit_code == 0, r.output
        assert "backend:    ollama" in r.output


class TestStatusUpdateBlock:
    def _enroll_with_announcement(self, tmp_path: Path, *, version: str) -> None:
        state = tmp_path / "state"
        state.mkdir(parents=True, exist_ok=True)
        db = Database(state / "worker.db")
        MigrationRunner(db).apply_all()
        repo = WorkerSelfRepository(db)
        repo.insert(
            worker_id="wkr-cli-test",
            trust_tier=0,
            pubkey_hex="a" * 64,
            enrolled_at=datetime(2026, 6, 1, tzinfo=UTC),
        )
        repo.record_latest_release(
            version=version,
            notes="Worker flavors + official Ollama",
            url="https://github.com/auspexai/worker/releases/tag/v99.0.0",
            at=datetime.now(UTC),
        )
        db.close()

    def test_status_shows_update_block_when_newer(self, tmp_path: Path) -> None:
        self._enroll_with_announcement(tmp_path, version="99.0.0")
        cfg = _config(tmp_path, '[worker]\nflavor = "inference"\n')
        r = CliRunner().invoke(cli, ["--config", str(cfg), "status"], env=_env(tmp_path))
        assert r.exit_code == 0, r.output
        assert "update available: v99.0.0" in r.output
        assert "Worker flavors + official Ollama" in r.output
        assert "--flavor inference" in r.output  # the printed (never run) command
        assert "never automatic" in r.output
        assert "flavor:      inference" in r.output

    def test_status_silent_when_current(self, tmp_path: Path) -> None:
        self._enroll_with_announcement(tmp_path, version="0.0.1")
        cfg = _config(tmp_path)
        r = CliRunner().invoke(cli, ["--config", str(cfg), "status"], env=_env(tmp_path))
        assert r.exit_code == 0, r.output
        assert "update available" not in r.output
