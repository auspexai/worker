"""Model fetch — pull weights from HuggingFace into the store, verified (W-M).

The testable logic (disk pre-check, atomic staging, idempotence) is exercised
with a fake fetcher exposing `fetch_file`, and the real `huggingface_hub`
dependency is optional + lazily imported.

`huggingface_hub` is an optional extra (`auspexai-worker[models]`): the core
worker stays lean, and a host that only *runs* provisioned models (weights staged
by other means) never needs it. Only `model pull` / auto-acquire does.
"""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path

from auspexai_worker.models.download_progress import DownloadProgressPoller
from auspexai_worker.models.hf_browse import memory_fits
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


class HfHubFetcher:
    """Fetch from HuggingFace. Lazily imports huggingface_hub."""

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
        fetcher: HfHubFetcher | None = None,
        *,
        min_headroom_bytes: int = 2_000_000_000,
        usable_memory_gb: float | None = None,
    ) -> None:
        self._store = store
        self._fetcher = fetcher or HfHubFetcher()
        self._min_headroom_bytes = min_headroom_bytes
        # RAM guard: this worker won't auto-acquire a model it can't load/serve.
        self._usable_memory_gb = usable_memory_gb

    def acquire(self, *, model_id: str, hf_repo: str, hf_filename: str) -> Path:
        return pull_from_coords(
            model_id=model_id,
            hf_repo=hf_repo,
            hf_filename=hf_filename,
            store=self._store,
            fetcher=self._fetcher,
            disk_free_bytes=_free_bytes_for(self._store.root),
            min_headroom_bytes=self._min_headroom_bytes,
            usable_memory_gb=self._usable_memory_gb,
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
    usable_memory_gb: float | None = None,
) -> Path:
    """Pull a single pinned file (`hf_repo`/`hf_filename`) into the store under
    `model_id` — the M3 lazy-auto-acquire path. Mirrors `pull_quant`'s atomic
    `.partial` staging + idempotence, but takes the explicit acquisition coords
    the manifest carries.

    `usable_memory_gb` (when known) is the RAM GUARD: a worker must not download a
    model it can't serve — presence-on-disk is worthless if it never loads (the
    stranded-156GB-on-a-7GB-mayhem class). We size the file from HF and REFUSE
    up-front when its load footprint exceeds this worker's usable memory; a failed
    size query falls through (best-effort, disk guard still applies).

    `fetcher` must expose `fetch_file(repo, filename, dest_dir)` (HfHubFetcher)."""
    dest = store.path_for(model_id)
    if store.has(model_id):
        return dest
    # RAM guard (AUD/fleet-fit): refuse to acquire a model this worker can't serve.
    if usable_memory_gb is not None:
        size = _hf_total_bytes(hf_repo, hf_filename)
        if size is not None and not memory_fits(size, usable_memory_gb):
            raise ModelFetchError(
                f"refusing to auto-acquire {model_id!r}: ~{size / 1e9:.1f} GB "
                f"(load footprint) exceeds this worker's usable memory "
                f"(~{usable_memory_gb:.1f} GB) — it could never serve"
            )
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
