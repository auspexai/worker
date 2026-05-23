"""Tests for the click CLI surface."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import httpx
from click.testing import CliRunner

from auspexai_worker.cli import cli
from auspexai_worker.config import WorkerConfig


def _write_config(tmp_path: Path, coordinator_url: str) -> Path:
    state_dir = tmp_path / "state"
    data_dir = tmp_path / "data"
    state_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "worker.toml"
    config_path.write_text(
        "[coordinator]\n"
        f'url = "{coordinator_url}"\n'
        "[identity]\n"
        'keystore_backend = "encrypted_file"\n'
    )
    # State + data dirs come via env vars (TOML config doesn't expose them in M1).
    return config_path


class TestStatusCommand:
    def test_status_when_not_enrolled(self, tmp_path: Path) -> None:
        config_path = _write_config(tmp_path, "http://example.invalid")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--config", str(config_path), "status"],
            env={
                "AUSPEXAI_WORKER_STATE_DIR": str(tmp_path / "state"),
                "AUSPEXAI_WORKER_DATA_DIR": str(tmp_path / "data"),
            },
        )
        assert result.exit_code == 0, result.output
        assert "not enrolled" in result.output
        assert "http://example.invalid" in result.output

    def test_status_after_enrollment(self, tmp_path: Path) -> None:
        # Pre-populate state by running bootstrap with an injected coordinator.
        from auspexai_worker.bootstrap import bootstrap
        from auspexai_worker.coordinator import CoordinatorClient
        from auspexai_worker.keystore import InMemoryKeystore

        config = WorkerConfig(
            coordinator_url="http://example.invalid",
            heartbeat_interval_seconds=60,
            assignment_poll_interval_seconds=30,
            state_dir=tmp_path / "state",
            data_dir=tmp_path / "data",
            keystore_backend="encrypted_file",
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                201,
                json={
                    "worker_id": "wkr-status-001",
                    "trust_tier": 0,
                    "registered_at": "2026-05-20T12:00:00+00:00",
                },
            )

        keystore = InMemoryKeystore()
        with CoordinatorClient(
            base_url="http://example.invalid",
            transport=httpx.MockTransport(handler),
        ) as client:
            bootstrap(config, keystore=keystore, coordinator=client)

        config_path = _write_config(tmp_path, "http://example.invalid")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--config", str(config_path), "status"],
            env={
                "AUSPEXAI_WORKER_STATE_DIR": str(tmp_path / "state"),
                "AUSPEXAI_WORKER_DATA_DIR": str(tmp_path / "data"),
            },
        )
        assert result.exit_code == 0, result.output
        assert "wkr-status-001" in result.output
        assert "T0" in result.output


class TestBootstrapCommand:
    def test_bootstrap_drives_first_run_enrollment(self, tmp_path: Path) -> None:
        config_path = _write_config(tmp_path, "http://bootstrap-test.invalid")

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/v0/workers/enroll"
            body = json.loads(request.content)
            assert "pubkey_hex" in body
            assert "capabilities" in body
            return httpx.Response(
                201,
                json={
                    "worker_id": "wkr-cli-001",
                    "trust_tier": 0,
                    "registered_at": "2026-05-20T12:00:00+00:00",
                },
            )

        runner = CliRunner()
        with patch("auspexai_worker.bootstrap.CoordinatorClient") as mock_client_cls:
            mock = mock_client_cls.return_value.__enter__.return_value
            mock.enroll.return_value = _make_enrollment_response()
            mock.close.return_value = None
            result = runner.invoke(
                cli,
                ["--config", str(config_path), "bootstrap"],
                env={
                    "AUSPEXAI_WORKER_STATE_DIR": str(tmp_path / "state"),
                    "AUSPEXAI_WORKER_DATA_DIR": str(tmp_path / "data"),
                },
            )
        assert result.exit_code == 0, result.output
        assert "enrolled: wkr-cli-001" in result.output


def _make_enrollment_response():
    from datetime import UTC, datetime

    from auspexai_worker.coordinator import EnrollmentResponse

    return EnrollmentResponse(
        worker_id="wkr-cli-001",
        trust_tier=0,
        registered_at=datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC),
    )


class TestSandboxProbeCommand:
    """`auspexai-worker sandbox probe` exposes the daemon's bwrap check
    as a standalone subcommand so the .deb postinst can invoke it."""

    def test_probe_returns_ok_when_bwrap_works(self) -> None:
        from auspexai_worker.sandbox import BubblewrapProbeResult

        runner = CliRunner()
        with patch(
            "auspexai_worker.cli.probe_bubblewrap",
            return_value=BubblewrapProbeResult(ok=True),
        ):
            result = runner.invoke(cli, ["sandbox", "probe"])
        assert result.exit_code == 0, result.output
        assert "OK" in result.output

    def test_probe_exits_nonzero_when_bwrap_fails(self) -> None:
        from auspexai_worker.sandbox import BubblewrapProbeResult

        runner = CliRunner()
        with patch(
            "auspexai_worker.cli.probe_bubblewrap",
            return_value=BubblewrapProbeResult(
                ok=False,
                reason="bwrap probe exit=1: setting up uid map: Permission denied",
            ),
        ):
            result = runner.invoke(cli, ["sandbox", "probe"])
        assert result.exit_code == 1
        assert "FAILED" in result.output
        assert "Permission denied" in result.output
