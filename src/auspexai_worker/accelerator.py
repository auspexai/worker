"""General accelerator detection — what compute + how much memory the worker
has for model inference (W-M / v0.1.11).

Replaces the old recommend path that read only *declared* VRAM and so labelled
every host "no GPU — CPU (slow)". This detects the accelerator class across
platforms and computes one **effective accelerator-memory budget** the fit logic
sizes against:

  - **NVIDIA discrete** — `nvidia-smi` reports total VRAM (a separate pool).
  - **AMD discrete** — `/dev/kfd` + sysfs `mem_info_vram_total`.
  - **Apple Silicon** — Metal on **unified memory**: the accelerator budget *is*
    system RAM.
  - **Jetson / tegra** — also **unified memory** (device-tree says Jetson/Orin/
    Tegra; the GPU shares the system RAM). This is the mayhems.
  - **CPU-only** — no accelerator; budget = RAM (the "slow" note is then real).

"Unified" is a general property (Apple + Jetson share it), not a one-off. All
probes are injectable so detection is testable without the hardware.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from auspexai_worker.capabilities import detect_ram_total_gb


class AcceleratorKind(StrEnum):
    NVIDIA = "nvidia"  # discrete
    AMD = "amd"  # discrete
    APPLE = "apple"  # unified (Metal)
    JETSON = "jetson"  # unified (tegra)
    CPU = "cpu"  # no accelerator


@dataclass(frozen=True)
class Accelerator:
    kind: AcceleratorKind
    memory_budget_gb: float | None  # effective accelerator memory ceiling
    unified: bool  # budget shared with system RAM (Apple/Jetson)
    label: str  # human-readable

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "memory_budget_gb": (
                round(self.memory_budget_gb, 1) if self.memory_budget_gb is not None else None
            ),
            "unified": self.unified,
            "label": self.label,
        }


def _read_device_tree_model() -> str:
    try:
        return Path("/proc/device-tree/model").read_text(errors="replace").strip("\x00 \n")
    except OSError:
        return ""


def _nvidia_smi_vram_gb() -> float | None:
    """Total VRAM (GB) of the first NVIDIA GPU via nvidia-smi, or None."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    first = out.stdout.strip().splitlines()
    if not first or not first[0].strip().isdigit():
        return None
    return int(first[0].strip()) / 1024.0  # MiB -> GiB


def _amd_vram_gb() -> float | None:
    for p in sorted(Path("/sys/class/drm").glob("card*/device/mem_info_vram_total")):
        try:
            return int(p.read_text().strip()) / 1e9
        except (OSError, ValueError):
            continue
    return None


def detect_accelerator(
    *,
    system: str | None = None,
    machine: str | None = None,
    ram_total_gb: float | None = None,
    device_tree_model: str | None = None,
    nvidia_smi: Callable[[], float | None] = _nvidia_smi_vram_gb,
    amd_vram: Callable[[], float | None] = _amd_vram_gb,
    kfd_present: bool | None = None,
) -> Accelerator:
    """Detect the accelerator + its effective memory budget. All inputs are
    injectable for testing; real defaults probe the host."""
    system = system if system is not None else platform.system()
    machine = (machine if machine is not None else platform.machine()).lower()
    ram = ram_total_gb if ram_total_gb is not None else detect_ram_total_gb()
    dt_model = device_tree_model if device_tree_model is not None else _read_device_tree_model()
    kfd = kfd_present if kfd_present is not None else Path("/dev/kfd").exists()

    dt = dt_model.lower()
    # Jetson FIRST: tegra has NVIDIA silicon but is unified, so it must not fall
    # into the discrete-NVIDIA branch.
    if any(tag in dt for tag in ("jetson", "tegra", "orin", "xavier")):
        return Accelerator(
            AcceleratorKind.JETSON, ram, unified=True, label=f"Jetson unified ({_g(ram)})"
        )
    # Apple Silicon: Metal on unified memory.
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return Accelerator(
            AcceleratorKind.APPLE,
            ram,
            unified=True,
            label=f"Apple Silicon unified ({_g(ram)})",
        )
    # Discrete NVIDIA.
    vram = nvidia_smi()
    if vram is not None:
        return Accelerator(
            AcceleratorKind.NVIDIA, vram, unified=False, label=f"NVIDIA discrete ({_g(vram)})"
        )
    # Discrete AMD.
    if kfd:
        amd = amd_vram()
        return Accelerator(
            AcceleratorKind.AMD, amd, unified=False, label=f"AMD discrete ({_g(amd)})"
        )
    return Accelerator(AcceleratorKind.CPU, ram, unified=False, label=f"CPU only ({_g(ram)})")


def _g(gb: float | None) -> str:
    return f"{gb:.0f} GB" if gb is not None else "unknown"
