"""In-flight model-download progress (D12 Inc 5c).

A tiny thread-safe registry the pull path writes to and the heartbeat loop reads,
so the coordinator — and the watching researcher — can see a model download
advance (bytes / %) instead of a binary "downloading".

Robust by construction: a background poller samples the size of the `.partial`
staging directory the pull already owns (never hf's internal cache layout), and
the total is best-effort — a missing total degrades to bytes-only, it never
blocks or fails a pull.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
# model_id -> (bytes_downloaded, total_bytes | None)
_PROGRESS: dict[str, tuple[int, int | None]] = {}


def _set(model_id: str, downloaded: int, total: int | None) -> None:
    with _LOCK:
        _PROGRESS[model_id] = (downloaded, total)


def clear(model_id: str) -> None:
    with _LOCK:
        _PROGRESS.pop(model_id, None)


def snapshot() -> dict[str, dict[str, int | None]]:
    """Current in-flight downloads, shaped for the heartbeat payload."""
    with _LOCK:
        return {
            mid: {"bytes_downloaded": dl, "total_bytes": tot}
            for mid, (dl, tot) in _PROGRESS.items()
        }


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


class DownloadProgressPoller:
    """Background sampler for one pull: while the (blocking) fetch runs, publish
    the staging dir's growing size under `model_id` on an interval. Use as a
    context manager around the fetch; cleans up the registry entry on exit.
    Never raises into the pull path."""

    def __init__(
        self,
        model_id: str,
        staging: Path,
        total_bytes: int | None,
        *,
        interval: float = 2.0,
    ) -> None:
        self._model_id = model_id
        self._staging = staging
        self._total = total_bytes
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                _set(self._model_id, _dir_size(self._staging), self._total)
            except Exception:  # defense-in-depth: progress must never break a pull
                logger.debug(
                    "download-progress sample failed for %s", self._model_id, exc_info=True
                )

    def __enter__(self) -> DownloadProgressPoller:
        _set(self._model_id, 0, self._total)
        self._thread = threading.Thread(
            target=self._run, name=f"dlprogress-{self._model_id}", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        clear(self._model_id)
