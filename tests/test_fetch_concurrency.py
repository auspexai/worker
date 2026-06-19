"""Regression: concurrent auto-acquire of the SAME model must not race.

The prestage loop and the dispatch-time auto-acquire both pull a model into the
shared `<model>.partial` staging dir, and `pull_from_coords` rmtree's that dir at
the start of each attempt. Without a guard, one call's rmtree wiped the other's
in-flight HuggingFace download — `[Errno 2]` on the `.incomplete` file — and every
retry re-raced, so the model never landed (seen live on mayhem1, 2026-06-19). The
per-model lock serializes them; only the winner downloads.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from auspexai_worker.models.fetch import pull_from_coords
from auspexai_worker.models.store import ModelStore


class _SlowFetcher:
    """fetch_file writes into staging after a beat, recording peak concurrency."""

    def __init__(self) -> None:
        self.calls = 0
        self.max_concurrent = 0
        self._active = 0
        self._lk = threading.Lock()

    def fetch_file(self, repo: str, filename: str, dest_dir: Path) -> None:
        with self._lk:
            self.calls += 1
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
        time.sleep(0.05)
        (Path(dest_dir) / filename).write_bytes(b"gguf-bytes")
        with self._lk:
            self._active -= 1


def test_pull_from_coords_serializes_concurrent_same_model_acquires(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    fetcher = _SlowFetcher()
    errors: list[Exception] = []

    def go() -> None:
        try:
            pull_from_coords(
                model_id="gemma-x",
                hf_repo="org/repo",
                hf_filename="m.gguf",
                store=store,
                fetcher=fetcher,
                disk_free_bytes=None,
            )
        except Exception as e:  # the test records any failure
            errors.append(e)

    threads = [threading.Thread(target=go) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert store.has("gemma-x")  # the model actually landed
    assert fetcher.max_concurrent == 1  # never two downloads into the same .partial
    assert fetcher.calls == 1  # idempotent re-check → only the winner pulled
