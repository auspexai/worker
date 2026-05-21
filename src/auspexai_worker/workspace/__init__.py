"""Per-unit workspaces — the directory the runner subprocess sees.

One workspace per (unit_id) at `$XDG_STATE_HOME/auspexai-worker/runs/<unit_id>/`.
The daemon creates it before spawning the runner, the runner writes its
result body to `output.json` inside it, the daemon reads + cleans up on
completion. The daemon writes `runner.pid` so `auspexai-worker abort
<unit-id>` can find the runner process.

A separate top-level module (not under `daemon/`) so the `abort` CLI can
read the PID file without instantiating any daemon-only types.
"""

from __future__ import annotations

from .paths import (
    RunnerWorkspace,
    WorkspaceManager,
    WorkspaceNotFoundError,
    workspace_runs_dir,
)

__all__ = [
    "RunnerWorkspace",
    "WorkspaceManager",
    "WorkspaceNotFoundError",
    "workspace_runs_dir",
]
