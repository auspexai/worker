"""Sandbox wrapper — Phase 1 permissive, Phase 2 strict (same code path).

Per §5.17 the worker is a two-tier process model: a trusted daemon and an
untrusted, sandboxed runner subprocess. M4 ships the wrapper that
constructs the bubblewrap (Linux) invocation so Phase 2 can flip
permissive → strict by changing config rather than rewriting the daemon.

For tests + headless CI hosts without bubblewrap available, a
`passthrough` mode runs the runner directly (no sandbox at all). The
config flag `[sandbox] use_bubblewrap = false` selects passthrough.
"""

from __future__ import annotations

from .resources import ResourceLimits, UnitCgroup
from .wrapper import (
    BubblewrapProbeResult,
    SandboxConfig,
    SandboxNotAvailableError,
    SandboxPolicy,
    SeatbeltProbeResult,
    build_argv,
    check_bubblewrap_available,
    enforced_policy,
    probe_bubblewrap,
    probe_seatbelt,
)

__all__ = [
    "BubblewrapProbeResult",
    "ResourceLimits",
    "SandboxConfig",
    "SandboxNotAvailableError",
    "SandboxPolicy",
    "SeatbeltProbeResult",
    "UnitCgroup",
    "build_argv",
    "check_bubblewrap_available",
    "enforced_policy",
    "probe_bubblewrap",
    "probe_seatbelt",
]
