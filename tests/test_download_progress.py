"""DownloadProgressPoller + registry (D12 Inc 5c)."""

from __future__ import annotations

import time
from pathlib import Path

from auspexai_worker.models import download_progress as dp


def test_poller_publishes_growing_size_then_clears(tmp_path: Path) -> None:
    staging = tmp_path / "m1.partial"
    staging.mkdir()
    (staging / "shard").write_bytes(b"x" * 128)
    assert "m1" not in dp.snapshot()
    with dp.DownloadProgressPoller("m1", staging, total_bytes=1000, interval=0.01):
        # __enter__ publishes immediately with the known total.
        assert dp.snapshot()["m1"]["total_bytes"] == 1000
        # the poller samples the staging dir's growing size.
        time.sleep(0.05)
        assert dp.snapshot()["m1"]["bytes_downloaded"] >= 128
    # cleared on exit
    assert "m1" not in dp.snapshot()


def test_poller_total_none_degrades_to_bytes_only(tmp_path: Path) -> None:
    staging = tmp_path / "m2.partial"
    staging.mkdir()
    with dp.DownloadProgressPoller("m2", staging, total_bytes=None, interval=0.01):
        snap = dp.snapshot()
        assert snap["m2"]["total_bytes"] is None
        assert snap["m2"]["bytes_downloaded"] == 0
    assert "m2" not in dp.snapshot()
