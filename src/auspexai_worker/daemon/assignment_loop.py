"""Assignment poller — pulls assignments from the coordinator and applies
the M3 gate pipeline. Mirrors `HeartbeatLoop` in shape so the two can run
side-by-side as separate threads in the worker daemon.

Phase 1 / M3 behavior: every accepted assignment is logged to the local
audit table and then DROPPED (no runner subprocess yet — that lands in
M4). The coordinator will re-schedule the unit when the assignment row
exceeds its timeout. Q-W4 resolution per the design doc default: no
explicit refuse endpoint, drop semantics carry the M3 cost.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from auspexai_worker.assignment import (
    AssignmentDecision,
    DecisionKind,
    GateContext,
    apply_gates,
)
from auspexai_worker.coordinator import (
    AssignmentAlreadyResolvedError,
    AssignmentNotFoundError,
    AssignmentResponse,
    CoordinatorClient,
    CoordinatorError,
)
from auspexai_worker.state import (
    AcceptedSensitiveRepository,
    AssignmentAuditRepository,
    ManifestPinRepository,
    TenantListRepository,
)

logger = logging.getLogger(__name__)


@dataclass
class AssignmentStats:
    polls_attempted: int = 0
    polls_succeeded: int = 0
    polls_failed: int = 0
    units_accepted: int = 0
    units_refused: int = 0
    no_work_polls: int = 0
    refuse_calls_succeeded: int = 0
    refuse_calls_failed: int = 0
    last_error: str | None = None
    errors: list[str] = field(default_factory=list)


class AssignmentPoller:
    """Periodic GET /workers/{id}/assignments + gate-pipeline + audit-log writer."""

    def __init__(
        self,
        *,
        coordinator: CoordinatorClient,
        worker_id: str,
        manifest_pins: ManifestPinRepository,
        accepted_sensitive: AcceptedSensitiveRepository,
        tenant_lists: TenantListRepository,
        audit: AssignmentAuditRepository,
        interval_seconds: float,
        stop_event: threading.Event | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._worker_id = worker_id
        self._manifest_pins = manifest_pins
        self._accepted_sensitive = accepted_sensitive
        self._tenant_lists = tenant_lists
        self._audit = audit
        self._interval = float(interval_seconds)
        self._stop_event = stop_event or threading.Event()
        self._stats = AssignmentStats()

    @property
    def stats(self) -> AssignmentStats:
        return self._stats

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def stop(self) -> None:
        self._stop_event.set()

    def run(self, *, max_polls: int | None = None) -> AssignmentStats:
        """Loop until `stop()` is called or `max_polls` is reached."""
        logger.info(
            "assignment poller starting (worker_id=%s, interval=%.0fs)",
            self._worker_id,
            self._interval,
        )
        while not self._stop_event.is_set():
            if max_polls is not None and self._stats.polls_attempted >= max_polls:
                break
            self._poll_once()
            self._stop_event.wait(timeout=self._interval)
        logger.info(
            "assignment poller stopped (polls=%d accepted=%d refused=%d failed=%d)",
            self._stats.polls_attempted,
            self._stats.units_accepted,
            self._stats.units_refused,
            self._stats.polls_failed,
        )
        return self._stats

    # ---- internals ------------------------------------------------------

    def _poll_once(self) -> None:
        self._stats.polls_attempted += 1
        try:
            response = self._coordinator.get_assignment(worker_id=self._worker_id)
            self._stats.polls_succeeded += 1
        except CoordinatorError as exc:
            self._stats.polls_failed += 1
            self._stats.last_error = str(exc)
            self._stats.errors.append(str(exc))
            logger.warning(
                "assignment poll failed (poll=%d): %s",
                self._stats.polls_attempted,
                exc,
            )
            return

        self._handle_assignment(response)

    def _handle_assignment(self, response: AssignmentResponse) -> None:
        if response.work_unit is None or response.coordinator_experiment_id is None:
            self._stats.no_work_polls += 1
            logger.debug("no work available (poll=%d)", self._stats.polls_attempted)
            return

        ctx = GateContext(
            coordinator_experiment_id=response.coordinator_experiment_id,
            manifest_pins=self._manifest_pins,
            accepted_sensitive=self._accepted_sensitive,
            tenant_lists=self._tenant_lists,
        )
        decision = apply_gates(response.work_unit, ctx)

        self._audit.append(
            action=decision.kind.value,
            assignment_id=response.assignment_id,
            coordinator_experiment_id=response.coordinator_experiment_id,
            tenant_id=response.work_unit.tenant_id,
            unit_id=response.work_unit.unit_id,
            manifest_sha256=response.work_unit.manifest_sha256,
            reason=decision.reason,
        )

        if decision.accepted:
            self._stats.units_accepted += 1
            logger.info(
                "assignment accepted (unit=%s experiment=%s tenant=%s) — M3 drops "
                "(runner subprocess arrives in M4)",
                response.work_unit.unit_id,
                response.coordinator_experiment_id,
                response.work_unit.tenant_id,
            )
        else:
            self._stats.units_refused += 1
            self._on_refused(decision, response)

    def _on_refused(
        self,
        decision: AssignmentDecision,
        response: AssignmentResponse,
    ) -> None:
        level = (
            logging.WARNING if decision.kind == DecisionKind.REFUSED_MANIFEST_SWAP else logging.INFO
        )
        unit_id = response.work_unit.unit_id if response.work_unit else None
        logger.log(
            level,
            "assignment refused (unit=%s reason=%s): %s",
            unit_id or "?",
            decision.kind.value,
            decision.reason,
        )
        # Tell the coordinator about the refusal so the operator console can
        # see the reason and the unit can be re-offered to another worker
        # (per Q-W4 / Option A). Network errors here log a warning but don't
        # halt the poller — the local audit row is the primary record.
        if unit_id is None:
            return
        try:
            self._coordinator.refuse_assignment(
                worker_id=self._worker_id,
                unit_id=unit_id,
                kind=decision.kind.value,
                reason=decision.reason or "",
            )
            self._stats.refuse_calls_succeeded += 1
        except (AssignmentNotFoundError, AssignmentAlreadyResolvedError) as exc:
            # Coordinator state diverged from worker state — common cause:
            # worker restarted between GET and the refuse follow-up, and the
            # coordinator already has a refuse row from a previous run. Not
            # a hard error.
            self._stats.refuse_calls_failed += 1
            logger.info("refuse no-op (unit=%s): %s", unit_id, exc)
        except CoordinatorError as exc:
            self._stats.refuse_calls_failed += 1
            logger.warning(
                "refuse coordinator call failed (unit=%s): %s — local audit row "
                "still recorded; coordinator-side assignment stays in_progress",
                unit_id,
                exc,
            )
