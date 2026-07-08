"""Tests for the daemon CLI command (M2)."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from auspexai_worker.cli import cli
from auspexai_worker.keystore.base import pubkey_hex as _pubkey_hex
from auspexai_worker.keystore.encrypted_file import EncryptedFileKeystore
from auspexai_worker.state import Database, MigrationRunner, WorkerSelfRepository


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


class TestDaemonRefusesForgedStrict:
    """AUD-25 (A9 audit): on Linux, policy=strict + use_bubblewrap=false would run
    every unit uncontained yet sign ran_under="strict" — a forged containment
    attestation. The daemon must refuse to START rather than serve forged strict."""

    def test_linux_strict_without_bubblewrap_refuses_to_start(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        state_dir = tmp_path / "state"
        data_dir = tmp_path / "data"
        state_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)

        # Real keystore so the daemon's keystore↔enrolled pubkey check passes and
        # execution reaches the sandbox gate.
        ks = EncryptedFileKeystore(data_dir / "keystore.enc")
        priv = ks.generate_and_store()
        pubkey = _pubkey_hex(priv)

        db = Database(state_dir / "worker.db")
        MigrationRunner(db).apply_all()
        WorkerSelfRepository(db).insert(
            worker_id="wkr-strict",
            trust_tier=0,
            pubkey_hex=pubkey,
            enrolled_at=datetime(2026, 7, 8, 10, 0, 0, tzinfo=UTC),
        )
        db.close()

        cfg = tmp_path / "worker.toml"
        cfg.write_text(
            "[coordinator]\n"
            'url = "http://daemon-test.invalid"\n'
            "[identity]\n"
            'keystore_backend = "encrypted_file"\n'
            "[sandbox]\n"
            'policy = "strict"\n'
            "use_bubblewrap = false\n"
        )

        result = CliRunner().invoke(
            cli,
            ["--config", str(cfg), "daemon", "--max-ticks", "1"],
            env={
                "AUSPEXAI_WORKER_STATE_DIR": str(state_dir),
                "AUSPEXAI_WORKER_DATA_DIR": str(data_dir),
            },
        )
        assert result.exit_code == 2
        assert "policy=strict requires use_bubblewrap" in result.output
