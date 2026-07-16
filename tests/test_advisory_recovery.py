"""Advisory auto-recovery — the card clears once the volunteer's fix takes effect."""

from __future__ import annotations

from types import SimpleNamespace

from auspexai_worker.daemon.advisory_recovery import advisory_recovered, run_recovery_check


def _row(kind: str, *, available_at_raise_gb=None):
    return SimpleNamespace(kind=kind, model_id="m", available_at_raise_gb=available_at_raise_gb)


class _Backend:
    def __init__(self, version: str | None):
        self._v = version

    def version(self) -> str | None:
        return self._v


class _Repo:
    """Minimal serve-advisory repo: holds one row, clear() drops it."""

    def __init__(self, row):
        self._row = row
        self.cleared = False

    def get(self):
        return self._row

    def clear(self):
        self._row = None
        self.cleared = True


def test_stale_backend_recovers_when_ollama_updated():
    # Below the floor (0.17.7) → not recovered; at/above (0.32.0) → recovered.
    assert not advisory_recovered(
        _row("stale_backend"), backend_version="0.17.7", available_memory_gb=None
    )
    assert advisory_recovered(
        _row("stale_backend"), backend_version="0.32.0", available_memory_gb=None
    )


def test_gpu_oom_recovers_when_free_memory_rises_above_baseline():
    row = _row("gpu_oom", available_at_raise_gb=1.0)  # was 1.0 GB free when it OOM'd
    # Still low → not recovered; freed (drop_caches) to well above baseline → recovered.
    assert not advisory_recovered(row, backend_version=None, available_memory_gb=1.2)
    assert advisory_recovered(row, backend_version=None, available_memory_gb=3.5)


def test_gpu_oom_no_baseline_recovers_on_healthy_free_memory():
    # A baseline-less (legacy / orphaned) card can't do the relative comparison, so
    # it clears once free memory is comfortably healthy in absolute terms — the next
    # dispatch re-raises WITH a baseline if the serve still fails. Starved → stays.
    row = _row("gpu_oom", available_at_raise_gb=None)
    assert advisory_recovered(row, backend_version=None, available_memory_gb=8.0)
    assert not advisory_recovered(row, backend_version=None, available_memory_gb=1.5)


def test_gpu_oom_no_baseline_stays_without_a_memory_reading():
    # No memory reading at all → can't judge; leave the card for the next dispatch.
    row = _row("gpu_oom", available_at_raise_gb=None)
    assert not advisory_recovered(row, backend_version=None, available_memory_gb=None)


def test_generic_serve_error_has_no_cheap_recovery():
    assert not advisory_recovered(
        _row("serve_error"), backend_version="0.32.0", available_memory_gb=8.0
    )


def test_run_recovery_clears_a_recovered_stale_backend_card():
    repo = _Repo(_row("stale_backend"))
    cleared = run_recovery_check(repo, backend=_Backend("0.32.0"), available_memory_gb=None)
    assert cleared is True
    assert repo.cleared is True
    assert repo.get() is None


def test_run_recovery_leaves_an_unrecovered_card():
    repo = _Repo(_row("stale_backend"))
    cleared = run_recovery_check(repo, backend=_Backend("0.17.7"), available_memory_gb=None)
    assert cleared is False
    assert repo.cleared is False


def test_run_recovery_no_advisory_is_a_noop():
    repo = _Repo(None)
    assert run_recovery_check(repo, backend=_Backend("0.32.0"), available_memory_gb=None) is False
