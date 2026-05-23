"""Read-only local dashboard for the worker daemon.

§5.14 "Layer B" volunteer-transparency surface. HTTP server bound to
localhost:7799 by default (configurable). Read-only — withdrawal,
tier upgrades, etc. remain CLI-only.

Wired into the daemon as a third thread alongside HeartbeatLoop +
AssignmentPoller. `[dashboard] enabled = false` in worker.toml disables.
"""

from __future__ import annotations

from .app import build_app
from .server import DashboardServer

__all__ = ["DashboardServer", "build_app"]
