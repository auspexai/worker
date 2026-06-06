"""Stdlib-only capability detection for the worker.

Returns the JSON-serializable payload the worker sends to the coordinator on
every heartbeat. Detection is intentionally cheap so calling on every tick
(default 60 s) is fine.

**GPU probe robustness (Q-W2 resolution):** the worker treats device-file
*existence* as a hint, not as authority. A device file that doesn't respond
to `open()` — stale node from a partial driver uninstall, container
bind-mount without a working CUDA runtime, race during boot, kernel module
unloaded at runtime — is excluded from the observed count. The volunteer's
declared GPU hardware in `[capabilities.gpus]` is the routing-relevant
signal (per §5.8 BYOM); the observed-probe travels alongside as
corroboration / mismatch diagnostic, not as the authoritative inventory.
"""

from __future__ import annotations

import glob
import os
import platform
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GpuObservation:
    """What the worker *observed* by probing device files. Not authoritative
    — only the volunteer (via `[capabilities.gpus]` config / `GpuDeclaration`)
    is authoritative about what hardware is actually usable."""

    nvidia: int  # /dev/nvidia[0-9]* device files that responded to open()
    amd: bool  # /dev/kfd exists and responded to open()

    def has_any(self) -> bool:
        return self.nvidia > 0 or self.amd


@dataclass(frozen=True)
class GpuDeclaration:
    """Volunteer-declared GPU hardware from `[capabilities.gpus]` config.

    All fields are optional — a worker that wants the coordinator's
    scheduler to consider it for GPU work must declare at least the count
    and VRAM total of its accelerators. Authoritative VRAM/model
    auto-detection (via `nvidia-smi` shell-out or NVML) is deferred to a
    later milestone.
    """

    nvidia: int | None = None
    nvidia_model: str | None = None
    vram_total_gb: float | None = None
    amd: bool | None = None
    amd_model: str | None = None

    def is_empty(self) -> bool:
        return all(
            v is None
            for v in (
                self.nvidia,
                self.nvidia_model,
                self.vram_total_gb,
                self.amd,
                self.amd_model,
            )
        )


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
    gpus_observed: GpuObservation
    gpus_declared: GpuDeclaration = field(default_factory=GpuDeclaration)
    declared_caps: DeclaredCaps = field(default_factory=DeclaredCaps)
    # Running worker code version (hatch-vcs). Always reported so the operator
    # can tell which workers support which features across a mixed-version
    # fleet (who has §9 #37 executor dispatch / W-M / W-H). Part of the
    # version-surfacing epic; rides the opaque capabilities channel.
    worker_version: str | None = None
    # Locally-available model ids (the BYOM store inventory, W-M). The §5.8
    # capability the scheduler will route on (#30). Omitted from the wire when
    # empty. The coordinator stores capabilities as an opaque dict, so this is
    # forward-compatible — it consumes `models` only once #30 lands.
    models: list[str] = field(default_factory=list)
    # Current thermal/health snapshot (W-H), or None where no sensor exists.
    # Lets the coordinator route work away from a degraded/overheating worker
    # (forward-compatible; opaque until consumed).
    thermal: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable shape.

        - `gpus_observed` is always included (operators rely on its presence
          for fleet-view diagnostics).
        - `gpus_declared` is omitted entirely when the volunteer hasn't
          declared anything, to keep the wire payload compact.
        - `declared_caps` is omitted when no caps are set.
        """
        d = asdict(self)
        d["gpus_observed"] = asdict(self.gpus_observed)
        if self.gpus_declared.is_empty():
            d.pop("gpus_declared", None)
        else:
            d["gpus_declared"] = {
                k: v for k, v in asdict(self.gpus_declared).items() if v is not None
            }
        declared = {k: v for k, v in asdict(self.declared_caps).items() if v is not None}
        if declared:
            d["declared_caps"] = declared
        else:
            d.pop("declared_caps", None)
        if not self.models:
            d.pop("models", None)  # compact wire when the store is empty
        if self.thermal is None:
            d.pop("thermal", None)  # omit where no sensor / health disabled
        if self.worker_version is None:
            d.pop("worker_version", None)
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


def _device_responsive(path: Path) -> bool:
    """Open the device file non-blocking; close immediately on success.

    Stale `/dev/nvidiaN` nodes (driver unloaded, partial uninstall,
    container bind-mount with no runtime, boot race) return ENXIO / ENODEV
    on open. `EACCES` (permission denied) is also treated as "not
    available" — if this worker user can't open the device, it can't
    use the GPU even if one exists.

    `O_NONBLOCK` is used so this can't hang on character devices with
    weird semantics.
    """
    try:
        fd = os.open(str(path), os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return False
    try:
        os.close(fd)
    except OSError:
        pass
    return True


GpuProbe = Callable[[Path], bool]


def detect_gpus(
    *,
    sysroot: Path | None = None,
    probe: GpuProbe = _device_responsive,
) -> GpuObservation:
    """Probe device files for GPU presence, with `open()` validation.

    Args:
        sysroot: Override for testing. Probes look for files under
            `sysroot / "dev" / "nvidiaN"` etc. instead of the real `/dev/`.
        probe: Injectable for tests. Returns True iff the device file is
            actually openable. Default validates via `os.open()`.
    """
    if sysroot is None:
        nvidia_glob = "/dev/nvidia[0-9]*"
        kfd_path = Path("/dev/kfd")
    else:
        nvidia_glob = str(sysroot / "dev" / "nvidia[0-9]*")
        kfd_path = sysroot / "dev" / "kfd"
    candidates = [Path(p) for p in glob.glob(nvidia_glob) if Path(p).name != "nvidiactl"]
    nvidia_responsive = sum(1 for path in candidates if probe(path))
    amd = kfd_path.exists() and probe(kfd_path)
    return GpuObservation(nvidia=nvidia_responsive, amd=amd)


# ---- top-level collect -----------------------------------------------------


def collect(
    *,
    declared_caps: DeclaredCaps | None = None,
    declared_gpus: GpuDeclaration | None = None,
    sysroot: Path | None = None,
    meminfo_path: Path | None = None,
    probe: GpuProbe = _device_responsive,
    # Locally-available model ids (BYOM store inventory). Caller-supplied — the
    # collector doesn't read the store itself (keeps detection store-agnostic).
    models: list[str] | None = None,
    # Current thermal snapshot (W-H), caller-supplied (the daemon owns the
    # stateful monitor so hysteresis is shared with the dispatch gate).
    thermal: dict[str, Any] | None = None,
    # Back-compat alias kept for callers that still pass `declared=...`.
    declared: DeclaredCaps | None = None,
) -> Capabilities:
    """Top-level capability snapshot. Cheap; safe to call on every heartbeat."""
    resolved_caps = declared_caps if declared_caps is not None else (declared or DeclaredCaps())
    # Lazy import avoids any package-load cycle (detect is imported early).
    from auspexai_worker import __version__ as worker_version

    return Capabilities(
        os=platform.system().lower(),
        arch=platform.machine().lower(),
        python_version=platform.python_version(),
        ram_total_gb=detect_ram_total_gb(meminfo_path=meminfo_path),
        cpu_count=detect_cpu_count(),
        gpus_observed=detect_gpus(sysroot=sysroot, probe=probe),
        gpus_declared=declared_gpus or GpuDeclaration(),
        declared_caps=resolved_caps,
        models=models or [],
        thermal=thermal,
        worker_version=worker_version,
    )
