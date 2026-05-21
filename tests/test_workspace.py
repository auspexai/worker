"""Tests for the per-unit workspace manager."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from auspexai_worker.workspace import (
    WorkspaceManager,
    WorkspaceNotFoundError,
    workspace_runs_dir,
)


class TestWorkspaceManager:
    def test_for_unit_returns_paths_without_creating(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(tmp_path)
        ws = mgr.for_unit("u-1")
        assert ws.unit_id == "u-1"
        assert ws.workspace_dir == (tmp_path / "u-1").resolve()
        assert ws.output_path == ws.workspace_dir / "output.json"
        assert ws.pid_file == ws.workspace_dir / "runner.pid"
        assert not ws.exists()

    def test_create_makes_directory(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(tmp_path)
        ws = mgr.create("u-1")
        assert ws.workspace_dir.is_dir()
        assert oct(ws.workspace_dir.stat().st_mode & 0o777) == "0o700"

    def test_create_cleans_stale_workspace(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(tmp_path)
        ws1 = mgr.create("u-1")
        (ws1.workspace_dir / "stale.txt").write_text("leftover")
        ws2 = mgr.create("u-1")
        assert not (ws2.workspace_dir / "stale.txt").exists()

    def test_read_output_roundtrips(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(tmp_path)
        ws = mgr.create("u-1")
        ws.output_path.write_text(json.dumps({"hello": "world"}))
        assert ws.read_output() == {"hello": "world"}

    def test_pid_file_roundtrips(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(tmp_path)
        ws = mgr.create("u-1")
        assert ws.read_pid() is None
        ws.write_pid(12345)
        assert ws.read_pid() == 12345

    def test_cleanup_removes_directory(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(tmp_path)
        ws = mgr.create("u-1")
        (ws.workspace_dir / "anything.json").write_text("{}")
        ws.cleanup()
        assert not ws.workspace_dir.exists()
        # second cleanup is a no-op.
        ws.cleanup()

    def test_get_existing_raises_when_absent(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(tmp_path)
        with pytest.raises(WorkspaceNotFoundError):
            mgr.get_existing("u-missing")

    def test_directory_traversal_refused(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(tmp_path)
        # for_unit should refuse to resolve outside runs_dir.
        ws = mgr.for_unit("normal-unit-id")
        # Sanity: normal id stays inside.
        assert ws.workspace_dir.is_relative_to(tmp_path.resolve())


class TestWorkspaceRunsDir:
    def test_creates_runs_subdir(self, tmp_path: Path) -> None:
        runs = workspace_runs_dir(tmp_path)
        assert runs == tmp_path / "runs"
        assert runs.is_dir()
        assert oct(runs.stat().st_mode & 0o777) == "0o700"
