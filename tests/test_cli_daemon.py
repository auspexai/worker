"""Tests for the daemon CLI command (M2)."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from auspexai_worker.cli import cli


def _write_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "worker.toml"
    cfg.write_text(
        "[coordinator]\n"
        'url = "http://daemon-test.invalid"\n'
        "[identity]\n"
        'keystore_backend = "encrypted_file"\n'
    )
    return cfg


class TestDaemonRefusesWhenNotEnrolled:
    def test_exits_with_actionable_message(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--config", str(cfg), "daemon"],
            env={
                "AUSPEXAI_WORKER_STATE_DIR": str(tmp_path / "state"),
                "AUSPEXAI_WORKER_DATA_DIR": str(tmp_path / "data"),
            },
        )
        assert result.exit_code == 2
        assert "not enrolled" in result.output
        assert "auspexai-worker bootstrap" in result.output
