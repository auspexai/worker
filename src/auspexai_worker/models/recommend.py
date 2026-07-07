"""Resource survey + selection parsing (W-M).

The network's provisionable-model catalog now lives on the coordinator
(`GET /api/v0/models/supported`); `model recommend` intersects that catalog with
this host's resources. What remains here are the local, catalog-free utilities:
the resource survey and the interactive-selection parser.

Resource survey uses stdlib + the worker's existing capability detection (no
psutil): disk via `shutil.disk_usage`, RAM via `detect_ram_total_gb`, VRAM from
the volunteer's GPU declaration (the volunteer is the source of truth, §capabilities).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from auspexai_worker.capabilities import detect_ram_total_gb


@dataclass(frozen=True)
class WorkerResources:
    disk_free_bytes: int
    ram_gb: float | None
    vram_gb: float | None


def _nearest_existing(path: Path) -> Path:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def survey_resources(store_root: Path, *, declared_vram_gb: float | None = None) -> WorkerResources:
    """Probe the resources relevant to model fit. VRAM comes from a declaration
    if set, else from general accelerator detection (so even this fallback path
    recognizes the GPU / unified memory instead of defaulting to 'no GPU')."""
    usage = shutil.disk_usage(_nearest_existing(store_root))
    vram = declared_vram_gb
    if vram is None:
        from auspexai_worker.accelerator import AcceleratorKind, detect_accelerator

        acc = detect_accelerator()
        if acc.kind is not AcceleratorKind.CPU:
            vram = acc.memory_budget_gb
    return WorkerResources(
        disk_free_bytes=usage.free,
        ram_gb=detect_ram_total_gb(),
        vram_gb=vram,
    )


def parse_selection(raw: str, count: int) -> list[int]:
    """Parse a multi-select reply into 0-based indices into a `count`-long list.

    Accepts `all`/`*`, `none`/empty/`q`, or comma/space-separated 1-based
    numbers. Out-of-range and non-numeric tokens are ignored (forgiving prompt).
    Returns sorted unique indices.
    """
    raw = raw.strip().lower()
    if raw in ("", "none", "n", "q"):
        return []
    if raw in ("all", "a", "*"):
        return list(range(count))
    idxs: set[int] = set()
    for tok in raw.replace(",", " ").split():
        if tok.isdigit():
            i = int(tok) - 1
            if 0 <= i < count:
                idxs.add(i)
    return sorted(idxs)
