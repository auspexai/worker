"""Worker daemon — background loops that run continuously.

M2: heartbeat loop. M3: assignment poll. M4: runner dispatch. Each loop is a
small class with `run()` / `stop()` so a single daemon process can compose
several together without prescribing an event loop framework.
"""

from __future__ import annotations

from .loop import HeartbeatLoop, HeartbeatStats

__all__ = ["HeartbeatLoop", "HeartbeatStats"]
