"""Derived volunteer-facing worker state (§2.1 #11 — the volunteer surface).

The volunteer needs one at-a-glance answer to "what is my worker doing?" The
worker carries several independent local signals — the operator hold
(`operator_hold_kind` = pause | quarantine, cached from assignment polls), the
volunteer's own `self_paused`, thermal-critical, and heartbeat freshness. This
collapses them into a single effective state with a consistent label, one-line
explanation, and tone, so the CLI `status` and the local dashboard agree
(persona coherence is the whole point of the #11-residual surface).

Precedence (first match wins):

    quarantined (fault) > operator-paused (no-fault) > self-paused
    > overheating > offline (stale heartbeat) > active

A quarantine outranks an operator-pause (it's the actionable fault signal); both
operator holds outrank the volunteer's own self-pause; a hold of any kind
outranks overheating (the hold is the operative reason work isn't flowing).
**Quarantine is the only fault/trust signal** — every other non-active state is
no-fault, and the surface must not make a volunteer feel flagged for, say, a
routine maintenance pause.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auspexai_worker.state import WorkerSelf

# Heartbeat freshness window — mirrors coordinator
# worker_status.STALE_HEARTBEAT_MINUTES so the volunteer's "offline" and the
# operator's "offline" mean the same thing.
STALE_HEARTBEAT_MINUTES = 3


class SelfState(StrEnum):
    QUARANTINED = "quarantined"
    OPERATOR_PAUSED = "operator-paused"
    SELF_PAUSED = "self-paused"
    OVERHEATING = "overheating"
    OFFLINE = "offline"
    ACTIVE = "active"


@dataclass(frozen=True)
class StatePresentation:
    """Display-ready worker state. `tone` maps to the dashboard badge classes
    (ok / warn / error / "" = neutral) and renders as a plain word in the CLI.
    `fault` is True only for quarantine — the one fault/trust signal."""

    state: SelfState
    label: str
    detail: str
    tone: str
    fault: bool


def derive_self_state(
    worker: WorkerSelf,
    *,
    thermal_critical: bool = False,
    now: datetime,
) -> StatePresentation:
    """Collapse the worker's local signals into one volunteer-facing state.

    `now` is passed in (not read here) so callers control the clock and the
    function stays pure/testable; `thermal_critical` is the host's CRITICAL
    thermal reading (WARM is advisory and does not surface as a headline state).
    """
    if worker.operator_hold_kind == "quarantine":
        return StatePresentation(
            SelfState.QUARANTINED,
            "quarantined",
            "Flagged by the operator — receiving no work. This is a fault/trust "
            f"signal. Reason: {worker.operator_hold_reason or '—'}",
            "error",
            fault=True,
        )
    if worker.operator_hold_kind == "pause":
        return StatePresentation(
            SelfState.OPERATOR_PAUSED,
            "paused by operator",
            "No-fault operational hold — temporarily out of rotation (e.g. host "
            "maintenance or a rolling upgrade). Nothing is wrong with your worker. "
            f"Reason: {worker.operator_hold_reason or '—'}",
            "",
            fault=False,
        )
    if worker.self_paused:
        sp = f" Reason: {worker.self_pause_reason}" if worker.self_pause_reason else ""
        return StatePresentation(
            SelfState.SELF_PAUSED,
            "self-paused",
            "You paused this worker — it stays enrolled (tier preserved) but "
            f"receives no work until you resume.{sp}",
            "",
            fault=False,
        )
    if thermal_critical:
        return StatePresentation(
            SelfState.OVERHEATING,
            "overheating",
            "Thermal-critical — auto-refusing work until the host cools. No action "
            "needed; it resumes on its own.",
            "warn",
            fault=False,
        )
    hb = worker.last_heartbeat_at
    if hb is not None and hb.tzinfo is None:
        hb = hb.replace(tzinfo=UTC)
    if hb is None or hb < now - timedelta(minutes=STALE_HEARTBEAT_MINUTES):
        return StatePresentation(
            SelfState.OFFLINE,
            "offline",
            "Not heartbeating — the worker daemon may be stopped. Start it with "
            "`auspexai-worker run`.",
            "warn",
            fault=False,
        )
    return StatePresentation(
        SelfState.ACTIVE,
        "active",
        "Enrolled and heartbeating — available for work.",
        "ok",
        fault=False,
    )
