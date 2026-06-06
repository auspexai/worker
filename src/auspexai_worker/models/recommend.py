"""Resource survey + model recommendation (W-M).

`recommend` intersects the catalog (what the network wants) with the worker's
resources (what it can run) — the supply-side half of closing #32's empty-pool
problem: "these in-demand models would actually run on your hardware."

Resource survey uses stdlib + the worker's existing capability detection (no
psutil): disk via `shutil.disk_usage`, RAM via `detect_ram_total_gb`, VRAM from
the volunteer's GPU declaration (the volunteer is the source of truth, §capabilities).
VRAM is a soft signal — a model that wants a GPU can still run (slowly) on CPU, so
insufficient/absent VRAM is reported as a note, not a hard fail; disk + RAM are
hard gates.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from auspexai_worker.capabilities import detect_ram_total_gb
from auspexai_worker.models.catalog import ModelCatalog, ModelCatalogEntry
from auspexai_worker.models.store import ModelStore


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


@dataclass(frozen=True)
class Recommendation:
    entry: ModelCatalogEntry
    fits: bool  # hard gates (disk, RAM) satisfied
    installed: bool  # already in the store
    blockers: list[str] = field(default_factory=list)  # why it doesn't fit
    notes: list[str] = field(default_factory=list)  # soft signals (e.g. VRAM/CPU)
    in_demand: bool = False


def _gb(b: float) -> str:
    return f"{b / 1e9:.1f} GB"


def recommend(
    catalog: ModelCatalog,
    store: ModelStore,
    resources: WorkerResources,
    *,
    demand: tuple[str, ...] = (),
) -> list[Recommendation]:
    out: list[Recommendation] = []
    for entry in catalog:
        blockers: list[str] = []
        notes: list[str] = []
        if entry.disk_bytes and entry.disk_bytes > resources.disk_free_bytes:
            blockers.append(
                f"needs {_gb(entry.disk_bytes)} disk, only {_gb(resources.disk_free_bytes)} free"
            )
        if (
            entry.min_ram_gb
            and resources.ram_gb is not None
            and entry.min_ram_gb > resources.ram_gb
        ):
            blockers.append(
                f"needs {entry.min_ram_gb:.0f} GB RAM, host has {resources.ram_gb:.1f} GB"
            )
        if entry.vram_load_gb:
            if resources.vram_gb is None:
                notes.append("no GPU declared; would run on CPU (slow)")
            elif entry.vram_load_gb > resources.vram_gb:
                notes.append(
                    f"wants ~{entry.vram_load_gb:.0f} GB VRAM, host declares "
                    f"{resources.vram_gb:.1f} GB; would offload/CPU"
                )
        out.append(
            Recommendation(
                entry=entry,
                fits=not blockers,
                installed=store.has(entry.id),
                blockers=blockers,
                notes=notes,
                in_demand=entry.id in demand,
            )
        )
    # in-demand first, then fitting, then smaller download.
    return sorted(out, key=lambda r: (not r.in_demand, not r.fits, r.entry.disk_bytes))


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
