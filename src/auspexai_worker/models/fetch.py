"""Model fetch — pull weights from HuggingFace into the store, verified (W-M).

The fetch backend is pluggable (`ModelFetcher`) so the testable logic (disk
pre-check, atomic staging, hash verification, idempotence) is exercised with a
fake, and the real `huggingface_hub` dependency is optional + lazily imported.

`huggingface_hub` is an optional extra (`auspexai-worker[models]`): the core
worker stays lean, and a host that only *runs* provisioned models (weights staged
by other means) never needs it. Only `model pull` does.

Integrity: weights are a supply-chain surface (a malicious file). When the
catalog entry carries `sha256`, the pull verifies it after download — the
acquisition-side analog of the §9 #37 `manifest_sha256` check.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Protocol

from auspexai_worker.models.catalog import ModelCatalogEntry
from auspexai_worker.models.recommend import WorkerResources
from auspexai_worker.models.store import ModelStore


class ModelFetchError(Exception):
    """Pull failed (disk, network, integrity, or missing backend)."""


class ModelFetcher(Protocol):
    def fetch(self, entry: ModelCatalogEntry, dest_dir: Path) -> None:
        """Download the model's files into dest_dir (created by the caller)."""
        ...


class HfHubFetcher:
    """Fetch a model snapshot from HuggingFace. Lazily imports huggingface_hub."""

    def fetch(self, entry: ModelCatalogEntry, dest_dir: Path) -> None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ModelFetchError(
                "huggingface_hub is not installed. Install the models extra: "
                "`pip install 'auspexai-worker[models]'` (or `pip install huggingface_hub`)."
            ) from exc
        snapshot_download(
            repo_id=entry.hf_repo,
            local_dir=str(dest_dir),
            allow_patterns=entry.files or None,
        )


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(dest_dir: Path, expected: dict[str, str]) -> None:
    for fname, sha in expected.items():
        target = dest_dir / fname
        if not target.is_file():
            raise ModelFetchError(f"integrity: expected file {fname!r} missing after download")
        actual = _sha256_of(target)
        if actual != sha.lower():
            raise ModelFetchError(
                f"integrity: {fname} sha256 {actual} != catalog {sha.lower()} (refusing)"
            )


def pull_model(
    entry: ModelCatalogEntry,
    store: ModelStore,
    fetcher: ModelFetcher,
    *,
    resources: WorkerResources | None = None,
) -> Path:
    """Pull a model into the store. Idempotent (already-present → no-op).

    Fetches into a `.partial` dir and renames on success so an interrupted pull
    never reads as installed. Verifies catalog sha256s before the rename.
    """
    dest = store.path_for(entry.id)
    if store.has(entry.id):
        return dest  # already installed

    if resources is not None and entry.disk_bytes and entry.disk_bytes > resources.disk_free_bytes:
        raise ModelFetchError(
            f"insufficient disk: {entry.id} needs ~{entry.disk_bytes / 1e9:.1f} GB, "
            f"{resources.disk_free_bytes / 1e9:.1f} GB free"
        )

    staging = dest.parent / f"{entry.id}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        fetcher.fetch(entry, staging)
        if entry.sha256:
            _verify(staging, entry.sha256)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if dest.exists():
        shutil.rmtree(dest)
    staging.rename(dest)
    return dest
