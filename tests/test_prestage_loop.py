"""PrestageLoop (M3b worker side) — polls directives + pulls via the acquirer."""

from __future__ import annotations

from pathlib import Path

from auspexai_worker.coordinator import CoordinatorError
from auspexai_worker.coordinator.client import PrestageDirective
from auspexai_worker.daemon import PrestageLoop


class _FakeCoord:
    def __init__(self, directives, *, raises=None):
        self._directives = directives
        self._raises = raises
        self.calls = 0

    def get_prestage(self, *, worker_id):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._directives


class _FakeAcquirer:
    def __init__(self, *, fail_for=()):
        self.acquired: list[str] = []
        self._fail = set(fail_for)

    def acquire(self, *, model_id, hf_repo, hf_filename):
        if model_id in self._fail:
            raise RuntimeError("pull failed")
        self.acquired.append(model_id)
        return Path("/store") / model_id


def _loop(coord, acq):
    return PrestageLoop(coordinator=coord, worker_id="wkr-a", acquirer=acq, interval_seconds=0.01)


def test_prestage_pulls_each_directive():
    coord = _FakeCoord(
        [
            PrestageDirective("m-x", "Org/X-GGUF", "X.gguf"),
            PrestageDirective("m-y", "Org/Y-GGUF", "Y.gguf"),
        ]
    )
    acq = _FakeAcquirer()
    stats = _loop(coord, acq).run(max_ticks=1)
    assert acq.acquired == ["m-x", "m-y"]
    assert stats.pulls_succeeded == 2
    assert stats.pulls_failed == 0


def test_prestage_continues_past_a_failed_pull():
    coord = _FakeCoord(
        [
            PrestageDirective("m-x", "Org/X-GGUF", "X.gguf"),
            PrestageDirective("m-y", "Org/Y-GGUF", "Y.gguf"),
        ]
    )
    acq = _FakeAcquirer(fail_for={"m-x"})
    stats = _loop(coord, acq).run(max_ticks=1)
    assert acq.acquired == ["m-y"]  # failure on m-x didn't stop m-y
    assert stats.pulls_failed == 1
    assert stats.pulls_succeeded == 1


def test_prestage_survives_coordinator_error():
    coord = _FakeCoord([], raises=CoordinatorError("coordinator unreachable"))
    acq = _FakeAcquirer()
    stats = _loop(coord, acq).run(max_ticks=1)
    assert stats.pulls_succeeded == 0
    assert stats.last_error is not None  # logged, loop didn't crash
