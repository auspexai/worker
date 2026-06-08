"""Unit tests for the volunteer-facing worker-state deriver (§2.1 #11).

The precedence is the load-bearing behavior: quarantine (the one fault signal)
must outrank everything; an operator hold of either kind must outrank the
volunteer's own self-pause + overheating; offline only when nothing else holds.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from auspexai_worker.state import WorkerSelf
from auspexai_worker.worker_state import SelfState, derive_self_state

NOW = datetime(2026, 6, 7, 12, 0, 0, tzinfo=UTC)


def _w(**kw: object) -> WorkerSelf:
    base: dict[str, object] = {
        "worker_id": "wkr-x",
        "trust_tier": 1,
        "pubkey_hex": "ab" * 32,
        "enrolled_at": datetime(2026, 1, 1, tzinfo=UTC),
        "last_heartbeat_at": NOW - timedelta(seconds=10),  # fresh
        "account_binding_json": None,
    }
    base.update(kw)
    return WorkerSelf(**base)  # type: ignore[arg-type]


def test_active_when_fresh_and_unheld() -> None:
    s = derive_self_state(_w(), now=NOW)
    assert s.state is SelfState.ACTIVE
    assert s.tone == "ok"
    assert s.fault is False


def test_quarantine_is_the_only_fault() -> None:
    s = derive_self_state(_w(operator_hold_kind="quarantine", operator_hold_reason="r"), now=NOW)
    assert s.state is SelfState.QUARANTINED
    assert s.fault is True
    assert s.tone == "error"


def test_operator_pause_is_no_fault() -> None:
    s = derive_self_state(_w(operator_hold_kind="pause", operator_hold_reason="maint"), now=NOW)
    assert s.state is SelfState.OPERATOR_PAUSED
    assert s.fault is False


def test_quarantine_outranks_self_pause() -> None:
    s = derive_self_state(_w(operator_hold_kind="quarantine", self_paused=True), now=NOW)
    assert s.state is SelfState.QUARANTINED


def test_operator_pause_outranks_self_pause_and_overheating() -> None:
    s = derive_self_state(
        _w(operator_hold_kind="pause", self_paused=True), thermal_critical=True, now=NOW
    )
    assert s.state is SelfState.OPERATOR_PAUSED


def test_self_pause_outranks_overheating() -> None:
    s = derive_self_state(_w(self_paused=True), thermal_critical=True, now=NOW)
    assert s.state is SelfState.SELF_PAUSED


def test_overheating_when_only_thermal_critical() -> None:
    s = derive_self_state(_w(), thermal_critical=True, now=NOW)
    assert s.state is SelfState.OVERHEATING
    assert s.tone == "warn"


def test_offline_when_heartbeat_stale() -> None:
    s = derive_self_state(_w(last_heartbeat_at=NOW - timedelta(minutes=10)), now=NOW)
    assert s.state is SelfState.OFFLINE


def test_offline_when_never_heartbeated() -> None:
    s = derive_self_state(_w(last_heartbeat_at=None), now=NOW)
    assert s.state is SelfState.OFFLINE


def test_naive_heartbeat_is_treated_as_utc() -> None:
    # A naive (tz-less) recent heartbeat must read as fresh, not crash/offline.
    s = derive_self_state(
        _w(last_heartbeat_at=(NOW - timedelta(seconds=5)).replace(tzinfo=None)), now=NOW
    )
    assert s.state is SelfState.ACTIVE
