"""The worker-local BYOM model store (W-M) — operations over the §9 #37 store.

Layout (shared with the executor dispatch): `<root>/<model_id>/...`. `list()` is
also the worker's **model inventory** — the declaration #30 capability-matching
will route on. The store holds weights only; the platform never distributes them
(§5.8), so it's filled by the volunteer (`model pull`).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


@dataclass(frozen=True)
class StoredModel:
    id: str
    path: Path
    size_bytes: int


class ModelStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, model_id: str) -> Path:
        return self._root / model_id

    def has(self, model_id: str) -> bool:
        """True if the model is present and non-empty (a bare/partial dir
        doesn't count — a half-finished pull must not read as installed)."""
        d = self.path_for(model_id)
        return d.is_dir() and any(f.is_file() for f in d.rglob("*"))

    def list(self) -> list[StoredModel]:
        if not self._root.is_dir():
            return []
        out: list[StoredModel] = []
        for d in sorted(self._root.iterdir()):
            if d.is_dir() and any(f.is_file() for f in d.rglob("*")):
                out.append(StoredModel(id=d.name, path=d, size_bytes=_dir_size_bytes(d)))
        return out

    def inventory(self) -> list[str]:
        """The model ids present locally — the #30 capability declaration."""
        return [m.id for m in self.list()]

    def remove(self, model_id: str) -> bool:
        """Delete a model from the store. Returns False if it wasn't present."""
        d = self.path_for(model_id)
        if not d.is_dir():
            return False
        shutil.rmtree(d)
        return True
