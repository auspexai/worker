"""Model catalog — what models the network wants + their resource footprints (W-M).

The catalog drives the BYOM onramp: `model recommend` reads it to suggest models
that fit the volunteer's hardware, and `model pull <id>` reads it to know where
to fetch from + how to verify. It is deliberately a **pluggable source** behind
`CatalogSource`:

  - Phase-1 seed: a small JSON bundled with the worker (`seed_catalog.json`) —
    works offline, no infra.
  - Near-term: a URL source (e.g. `auspexai.network/models.json`).
  - The honest target (per `worker_model_management_design.md`): a coordinator
    demand-board aggregating researcher model-requests, since "what the network
    wants" is inherently dynamic. The supply CLI works against the seed first;
    only the source swaps.

`model id` matches the manifest `models[].id` the §9 #37 store resolves on, so a
pulled model lands where an executor's `--models` will find it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelCatalogEntry:
    """One catalogued model + its (approximate) resource footprint."""

    id: str  # canonical model id; matches manifest models[].id + store dir name
    version: str
    hf_repo: str  # HuggingFace repo, e.g. "openai-community/gpt2"
    note: str = ""  # one-line "what / why" shown to the volunteer
    disk_bytes: int = 0  # approx total download size
    vram_load_gb: float | None = None  # approx VRAM to load (None = CPU-friendly)
    min_ram_gb: float | None = None
    files: list[str] = field(default_factory=list)  # [] = whole-repo snapshot
    # filename -> sha256 for post-download integrity (the supply-chain analog of
    # manifest_sha256). Optional in the seed; required once the catalog is trusted.
    sha256: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelCatalogEntry:
        return cls(
            id=d["id"],
            version=str(d.get("version", "")),
            hf_repo=d["hf_repo"],
            note=d.get("note", ""),
            disk_bytes=int(d.get("disk_bytes", 0)),
            vram_load_gb=d.get("vram_load_gb"),
            min_ram_gb=d.get("min_ram_gb"),
            files=list(d.get("files", [])),
            sha256=dict(d.get("sha256", {})),
        )


class CatalogSource(Protocol):
    """Where the catalog comes from. Returns the raw list of entry dicts."""

    def load(self) -> list[dict[str, Any]]: ...


class BundledCatalogSource:
    """The seed catalog shipped inside the worker package (offline default)."""

    def load(self) -> list[dict[str, Any]]:
        raw = resources.files("auspexai_worker.models").joinpath("seed_catalog.json")
        return json.loads(raw.read_text(encoding="utf-8"))


class FileCatalogSource:
    """A catalog read from a local JSON file (tests / operator override)."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> list[dict[str, Any]]:
        return json.loads(self._path.read_text(encoding="utf-8"))


class ModelCatalog:
    """An immutable view over a catalog source."""

    def __init__(self, entries: list[ModelCatalogEntry]) -> None:
        self._by_id = {e.id: e for e in entries}

    @classmethod
    def load(cls, source: CatalogSource | None = None) -> ModelCatalog:
        source = source or BundledCatalogSource()
        return cls([ModelCatalogEntry.from_dict(d) for d in source.load()])

    def __iter__(self):
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    def get(self, model_id: str) -> ModelCatalogEntry | None:
        return self._by_id.get(model_id)
