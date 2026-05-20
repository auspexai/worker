"""Worker capability detection.

The capabilities payload is opaque to the coordinator at M6b but the scheduler
will eventually use it for capability-matched scheduling (§5.8). M2 ships a
deliberately conservative set that's stable across Linux distros: OS, arch,
RAM total, CPU core count, GPU presence flags. Declared resource caps (from
the `[resources]` config block) ride alongside as `declared_caps` so the
volunteer's chosen ceilings are visible to the operator and the future
scheduler.

Detection is stdlib-only (no psutil) — we read /proc and check device-file
existence. All probes accept an optional `sysroot` argument so tests can
substitute a fake filesystem.
"""

from __future__ import annotations

from .detect import (
    Capabilities,
    DeclaredCaps,
    GpuDeclaration,
    GpuObservation,
    GpuProbe,
    collect,
    detect_gpus,
    detect_ram_total_gb,
)

__all__ = [
    "Capabilities",
    "DeclaredCaps",
    "GpuDeclaration",
    "GpuObservation",
    "GpuProbe",
    "collect",
    "detect_gpus",
    "detect_ram_total_gb",
]
