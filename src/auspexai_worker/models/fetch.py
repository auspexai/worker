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
import os
import shutil
import threading
from pathlib import Path
from typing import Protocol

from auspexai_worker.models.catalog import ModelCatalogEntry
from auspexai_worker.models.download_progress import DownloadProgressPoller
from auspexai_worker.models.recommend import WorkerResources
from auspexai_worker.models.store import ModelStore

# M3 fetch-hardening (mayhem0, 2026-06-06): a `model setup` download wedged for
# hours on a half-closed HuggingFace connection because the default **Xet**
# transfer path has no read timeout — the process blocked in a socket read that
# never returned. Force the classic resumable HTTP downloader (it honors
# HF_HUB_DOWNLOAD_TIMEOUT as a per-read timeout AND resumes the `.incomplete`
# shard) and bound that read. Set BEFORE huggingface_hub is imported anywhere;
# `setdefault` so an operator can still override via the environment.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")


class ModelFetchError(Exception):
    """Pull failed (disk, network, integrity, or missing backend)."""


class ModelFetcher(Protocol):
    def fetch(self, entry: ModelCatalogEntry, dest_dir: Path) -> None:
        """Download the model's files into dest_dir (created by the caller)."""
        ...


class HfHubFetcher:
    """Fetch from HuggingFace. Lazily imports huggingface_hub."""

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

    def fetch_file(self, repo: str, filename: str, dest_dir: Path) -> None:
        """Download a single file (e.g. one GGUF quant) — the HF-direct path."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ModelFetchError(
                "huggingface_hub is not installed. Install the models extra: "
                "`pip install 'auspexai-worker[models]'`."
            ) from exc
        hf_hub_download(repo_id=repo, filename=filename, local_dir=str(dest_dir))


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


def pull_quant(quant, store: ModelStore, fetcher, *, disk_free_bytes: int | None = None) -> Path:
    """Pull a single HF GGUF quant (`hf_browse.ModelQuant`) into the store under
    its model_id. Idempotent; atomic `.partial` staging; disk pre-check.

    `fetcher` must expose `fetch_file(repo, filename, dest_dir)` (HfHubFetcher)."""
    dest = store.path_for(quant.model_id)
    if store.has(quant.model_id):
        return dest
    if disk_free_bytes is not None and quant.size_bytes > disk_free_bytes:
        raise ModelFetchError(
            f"insufficient disk: {quant.model_id} needs ~{quant.size_gb:.1f} GB, "
            f"{disk_free_bytes / 1e9:.1f} GB free"
        )
    staging = dest.parent / f"{quant.model_id}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        with DownloadProgressPoller(quant.model_id, staging, getattr(quant, "size_bytes", None)):
            fetcher.fetch_file(quant.repo, quant.filename, staging)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if dest.exists():
        shutil.rmtree(dest)
    staging.rename(dest)
    return dest


def _hf_total_bytes(hf_repo: str, hf_filename: str) -> int | None:
    """Best-effort total size of an HF file, for a download % (D12 5c). None on
    any failure — the poller still reports bytes-downloaded; only the % is lost.
    Never raises into the pull path."""
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url

        meta = get_hf_file_metadata(hf_hub_url(repo_id=hf_repo, filename=hf_filename))
        return int(meta.size) if getattr(meta, "size", None) else None
    except Exception:
        return None


def _free_bytes_for(path: Path) -> int:
    """Free bytes on the filesystem holding `path` (or its nearest existing
    ancestor — the store root may not exist yet on a fresh worker)."""
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


# A pull stages into `<model>.partial` and rmtree's it at the start; the prestage
# loop and the dispatch-time auto-acquire can both pull the same model at once, so
# without a guard one's rmtree wipes the other's in-flight HuggingFace download
# (Errno 2 on the `.incomplete` file, and every retry re-races → the model never
# lands). One lock per model_id serializes them; the loser re-checks the store and
# returns the winner's result.
_PULL_LOCKS: dict[str, threading.Lock] = {}
_PULL_LOCKS_GUARD = threading.Lock()


def _pull_lock_for(model_id: str) -> threading.Lock:
    with _PULL_LOCKS_GUARD:
        lock = _PULL_LOCKS.get(model_id)
        if lock is None:
            lock = threading.Lock()
            _PULL_LOCKS[model_id] = lock
        return lock


class StoreModelAcquirer:
    """Concrete `provisioning.ModelAcquirer` (M3 lazy auto-acquire): pull a pinned
    file into the worker store, guarded by a coarse free-disk headroom check.
    Wired into the dispatcher only when `[executor] auto_acquire` is on."""

    def __init__(
        self,
        store: ModelStore,
        fetcher: ModelFetcher | None = None,
        *,
        min_headroom_bytes: int = 2_000_000_000,
    ) -> None:
        self._store = store
        self._fetcher = fetcher or HfHubFetcher()
        self._min_headroom_bytes = min_headroom_bytes

    def acquire(self, *, model_id: str, hf_repo: str, hf_filename: str) -> Path:
        return pull_from_coords(
            model_id=model_id,
            hf_repo=hf_repo,
            hf_filename=hf_filename,
            store=self._store,
            fetcher=self._fetcher,
            disk_free_bytes=_free_bytes_for(self._store.root),
            min_headroom_bytes=self._min_headroom_bytes,
        )


def pull_from_coords(
    *,
    model_id: str,
    hf_repo: str,
    hf_filename: str,
    store: ModelStore,
    fetcher,
    disk_free_bytes: int | None = None,
    min_headroom_bytes: int = 2_000_000_000,
) -> Path:
    """Pull a single pinned file (`hf_repo`/`hf_filename`) into the store under
    `model_id` — the M3 lazy-auto-acquire path. Mirrors `pull_quant`'s atomic
    `.partial` staging + idempotence, but takes the explicit acquisition coords
    the manifest carries (we don't know the size up front, so the disk guard is
    a coarse free-headroom check rather than an exact size pre-check).

    `fetcher` must expose `fetch_file(repo, filename, dest_dir)` (HfHubFetcher)."""
    dest = store.path_for(model_id)
    if store.has(model_id):
        return dest
    # Serialize concurrent acquires of the SAME model (prestage loop vs. dispatch):
    # both stage into `<model>.partial`, and one's rmtree-at-start would wipe the
    # other's in-flight download. The loser re-checks the store and returns.
    with _pull_lock_for(model_id):
        if store.has(model_id):  # the winner finished while we waited on the lock
            return dest
        if disk_free_bytes is not None and disk_free_bytes < min_headroom_bytes:
            raise ModelFetchError(
                f"insufficient disk headroom to acquire {model_id!r}: "
                f"{disk_free_bytes / 1e9:.1f} GB free, need >{min_headroom_bytes / 1e9:.1f} GB"
            )
        staging = dest.parent / f"{model_id}.partial"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            with DownloadProgressPoller(model_id, staging, _hf_total_bytes(hf_repo, hf_filename)):
                fetcher.fetch_file(hf_repo, hf_filename, staging)
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise ModelFetchError(
                f"auto-acquire of {model_id!r} from {hf_repo}/{hf_filename} failed: {exc}"
            ) from exc
        if dest.exists():
            shutil.rmtree(dest)
        staging.rename(dest)
    return dest
