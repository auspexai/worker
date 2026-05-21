"""Workspace path management."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class WorkspaceNotFoundError(Exception):
    """Raised when a workspace is expected but absent (e.g. abort target)."""


@dataclass(frozen=True)
class RunnerWorkspace:
    """All paths the daemon + runner + abort CLI need for one unit."""

    unit_id: str
    workspace_dir: Path
    output_path: Path
    pid_file: Path

    def exists(self) -> bool:
        return self.workspace_dir.is_dir()

    def read_output(self) -> dict[str, Any]:
        """Read + parse the runner's output. Raises FileNotFoundError if
        the runner never wrote it, JSONDecodeError on malformed content."""
        with self.output_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def write_pid(self, pid: int) -> None:
        self.pid_file.write_text(f"{pid}\n", encoding="ascii")

    def read_pid(self) -> int | None:
        try:
            raw = self.pid_file.read_text(encoding="ascii").strip()
        except FileNotFoundError:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def cleanup(self) -> None:
        """Remove the workspace directory and everything in it. No-op if
        already gone."""
        shutil.rmtree(self.workspace_dir, ignore_errors=True)


class WorkspaceManager:
    """Resolves workspace paths under a runs-dir root and creates them
    on demand."""

    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir

    @property
    def runs_dir(self) -> Path:
        return self._runs_dir

    def for_unit(self, unit_id: str) -> RunnerWorkspace:
        """Return the workspace path object for a unit. Does NOT create
        the directory."""
        # unit_id is supposed to be a tenant-controlled string; defend
        # against directory traversal by replacing any slash / parent
        # segments. Worst case the workspace lands in an odd subdirectory;
        # tenant can't escape the runs root.
        safe = unit_id.replace("/", "_").replace("..", "_")
        ws_dir = (self._runs_dir / safe).resolve()
        # If sanitization produced something outside runs_dir, refuse.
        try:
            ws_dir.relative_to(self._runs_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"unit_id {unit_id!r} resolves outside runs_dir; refusing") from exc
        return RunnerWorkspace(
            unit_id=unit_id,
            workspace_dir=ws_dir,
            output_path=ws_dir / "output.json",
            pid_file=ws_dir / "runner.pid",
        )

    def create(self, unit_id: str) -> RunnerWorkspace:
        """Create the workspace dir (0o700) and return its path object.

        Cleans up any pre-existing workspace for the same unit_id first
        — a stale workspace usually means a previous run died abnormally,
        and we don't want runner output from a prior attempt to leak into
        this one's result.
        """
        ws = self.for_unit(unit_id)
        if ws.workspace_dir.exists():
            ws.cleanup()
        ws.workspace_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
        return ws

    def get_existing(self, unit_id: str) -> RunnerWorkspace:
        """Return the workspace for a unit only if it exists on disk.

        Raises WorkspaceNotFoundError when the directory is absent — used
        by `abort` so it can give a clear "no such unit" message rather
        than a confusing chmod error from sending signals to a nonexistent
        PID.
        """
        ws = self.for_unit(unit_id)
        if not ws.exists():
            raise WorkspaceNotFoundError(f"no workspace for unit {unit_id!r}")
        return ws


def workspace_runs_dir(state_dir: Path) -> Path:
    """Standard runs/ subdirectory under the worker state dir.

    Lives at `$XDG_STATE_HOME/auspexai-worker/runs/`. Created on first
    use; survives daemon restart so abort+observe paths can still find
    in-progress runs.
    """
    runs_dir = state_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    return runs_dir
