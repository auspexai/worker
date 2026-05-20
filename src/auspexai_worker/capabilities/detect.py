"""Stdlib-only capability detection for the worker.

Returns the JSON-serializable payload the worker sends to the coordinator on
every heartbeat. Detection is intentionally cheap so calling on every tick
(default 60 s) is fine.
"""

from __future__ import annotations

import glob
import os
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GpuInventory:
    """What the worker thinks it has for accelerators."""

    nvidia: int  # count of /dev/nvidia<N> device files
    amd: bool  # /dev/kfd present (ROCm-capable)

    def has_any(self) -> bool:
        return self.nvidia > 0 or self.amd


@dataclass(frozen=True)
class DeclaredCaps:
    """Volunteer-declared resource caps from `[resources]`. Coordinator
    scheduler may use these for capability-matched scheduling; the worker
    enforces them locally in M4 (sandbox)."""

    max_ram_gb: float | None = None
    max_vram_gb: float | None = None
    max_cpu_cores: int | None = None
    network_quota_mb_per_hour: int | None = None


@dataclass(frozen=True)
class Capabilities:
    """Full payload sent in heartbeat.capabilities."""

    os: str
    arch: str
    python_version: str
    ram_total_gb: float | None
    cpu_count: int | None
    gpus: GpuInventory
    declared_caps: DeclaredCaps = field(default_factory=DeclaredCaps)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable shape. Drops None-valued declared_caps fields so
        the wire payload stays compact when nothing is configured."""
        d = asdict(self)
        d["gpus"] = asdict(self.gpus)
        declared = {k: v for k, v in asdict(self.declared_caps).items() if v is not None}
        if declared:
            d["declared_caps"] = declared
        else:
            d.pop("declared_caps", None)
        return d


# ---- probes ----------------------------------------------------------------


def detect_ram_total_gb(*, meminfo_path: Path | None = None) -> float | None:
    """Parse `/proc/meminfo` for `MemTotal`. Returns GB (binary, /1024^2 KiB).

    Returns None on non-Linux or unreadable meminfo.
    """
    path = meminfo_path or Path("/proc/meminfo")
    try:
        content = path.read_text(encoding="ascii", errors="replace")
    except (FileNotFoundError, PermissionError):
        return None
    for line in content.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            # Standard form: "MemTotal:   12345678 kB"
            if len(parts) >= 2 and parts[1].isdigit():
                kb = int(parts[1])
                return round(kb / (1024 * 1024), 2)
    return None


def detect_cpu_count() -> int | None:
    return os.cpu_count()


def detect_gpus(*, sysroot: Path | None = None) -> GpuInventory:
    """Probe device files for GPU presence.

    Args:
        sysroot: Override for testing. Probes look for files under
            `sysroot / "dev" / "nvidiaN"` etc. instead of the real `/dev/`.

    Detection is presence-based, not driver-state — a stale device file
    without a working driver will overreport. The scheduler should pair this
    with declared local model availability before assigning real GPU work
    (M3 / M4).
    """
    if sysroot is None:
        nvidia_glob = "/dev/nvidia[0-9]*"
        kfd_path = Path("/dev/kfd")
    else:
        nvidia_glob = str(sysroot / "dev" / "nvidia[0-9]*")
        kfd_path = sysroot / "dev" / "kfd"
    nvidia_devices = [p for p in glob.glob(nvidia_glob) if Path(p).name != "nvidiactl"]
    return GpuInventory(
        nvidia=len(nvidia_devices),
        amd=kfd_path.exists(),
    )


# ---- top-level collect -----------------------------------------------------


def collect(
    *,
    declared: DeclaredCaps | None = None,
    sysroot: Path | None = None,
    meminfo_path: Path | None = None,
) -> Capabilities:
    """Top-level capability snapshot. Cheap; safe to call on every heartbeat."""
    return Capabilities(
        os=platform.system().lower(),
        arch=platform.machine().lower(),
        python_version=platform.python_version(),
        ram_total_gb=detect_ram_total_gb(meminfo_path=meminfo_path),
        cpu_count=detect_cpu_count(),
        gpus=detect_gpus(sysroot=sysroot),
        declared_caps=declared or DeclaredCaps(),
    )
