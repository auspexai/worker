"""Shared fixtures for worker tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from auspexai_worker.config import WorkerConfig


@pytest.fixture
def tmp_config(tmp_path: Path) -> WorkerConfig:
    """A WorkerConfig that uses isolated tmp_path for state + data dirs.

    The coordinator URL points at a placeholder; tests that need network
    behavior inject their own CoordinatorClient.
    """
    return WorkerConfig(
        coordinator_url="http://test-coordinator.invalid",
        heartbeat_interval_seconds=60,
        assignment_poll_interval_seconds=30,
        state_dir=tmp_path / "state",
        data_dir=tmp_path / "data",
        keystore_backend="encrypted_file",
    )
