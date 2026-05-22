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
from auspexai_worker.daemon.dispatch import (
    DispatchOutcome,
    DispatchOutcomeKind,
    RunnerDispatcher,
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
    units_submitted: int = 0  # M4: dispatched + run + submitted ok
    units_dispatch_failed: int = 0  # M4: gate-accepted but runner / submit failed
    no_work_polls: int = 0
    refuse_calls_succeeded: int = 0
    refuse_calls_failed: int = 0
    last_error: str | None = None
    errors: list[str] = field(default_factory=list)


class AssignmentPoller:
    """Periodic GET /workers/{id}/assignments + gate-pipeline + audit-log writer.

    M4: when the gate decision is accepted AND a `dispatcher` is configured,
    the poller calls `dispatcher.run_unit(...)` synchronously to execute
    the unit and submit the result before polling again. Without a
    dispatcher (M3 behavior, kept for backward compatibility in tests),
    accepted units are dropped after the audit row is written.
    """

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
        dispatcher: RunnerDispatcher | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._worker_id = worker_id
        self._manifest_pins = manifest_pins
        self._accepted_sensitive = accepted_sensitive
        self._tenant_lists = tenant_lists
        self._audit = audit
        self._interval = float(interval_seconds)
        self._dispatcher = dispatcher
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
        # M6-tail: drain any queued pending submissions before pulling new
        # work. Submits that failed transiently on a previous tick get
        # another shot here. Bounded by max_per_tick so a long backlog
        # doesn't starve the assignment poll.
        if self._dispatcher is not None:
            try:
                retry_outcomes = self._dispatcher.retry_pending()
            except Exception:
                logger.exception("retry_pending raised; continuing with new-work poll")
            else:
                for outcome in retry_outcomes:
                    if outcome.kind == DispatchOutcomeKind.SUBMITTED:
                        self._stats.units_submitted += 1
                    elif outcome.kind == DispatchOutcomeKind.SUBMIT_FAILED_TERMINAL:
                        self._stats.units_dispatch_failed += 1
                    # SUBMIT_FAILED_TRANSIENT outcomes leave the row pending
                    # for the next tick — no stat change.

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
            if self._dispatcher is not None:
                self._dispatch_accepted(response)
            else:
                logger.info(
                    "assignment accepted (unit=%s experiment=%s tenant=%s) — no "
                    "dispatcher configured; dropping (M3-mode)",
                    response.work_unit.unit_id,
                    response.coordinator_experiment_id,
                    response.work_unit.tenant_id,
                )
        else:
            self._stats.units_refused += 1
            self._on_refused(decision, response)

    def _dispatch_accepted(self, response: AssignmentResponse) -> None:
        """M4: hand an accepted assignment to the runner dispatcher,
        record the outcome, and translate failures into a coordinator-side
        refuse + audit so dangling in_progress rows don't accumulate."""
        assert self._dispatcher is not None and response.work_unit is not None
        unit = response.work_unit
        try:
            outcome = self._dispatcher.run_unit(response)
        except Exception as exc:
            logger.exception("dispatcher raised for unit %s", unit.unit_id)
            outcome = DispatchOutcome(
                kind=DispatchOutcomeKind.RUNNER_CRASH,
                reason=f"dispatcher exception: {type(exc).__name__}: {exc}",
            )

        if outcome.kind == DispatchOutcomeKind.SUBMITTED:
            self._stats.units_submitted += 1
            self._audit.append(
                action="submitted",
                assignment_id=response.assignment_id,
                coordinator_experiment_id=response.coordinator_experiment_id,
                tenant_id=unit.tenant_id,
                unit_id=unit.unit_id,
                manifest_sha256=unit.manifest_sha256,
                reason=(
                    f"result_id={outcome.result_response.result_id} "
                    f"status={outcome.result_response.unit_status_after} "
                    f"completions={outcome.result_response.completions_so_far}/"
                    f"{outcome.result_response.replication_target}"
                    if outcome.result_response
                    else None
                ),
            )
            return

        # Anything else is a failed dispatch — record locally and tell the
        # coordinator we refused so the assignment row doesn't dangle.
        self._stats.units_dispatch_failed += 1
        submit_failed_kinds = (
            DispatchOutcomeKind.SUBMIT_FAILED_TRANSIENT,
            DispatchOutcomeKind.SUBMIT_FAILED_TERMINAL,
        )
        if outcome.kind == DispatchOutcomeKind.SUBMIT_FAILED_TRANSIENT:
            action_tag = "submit_failed_transient"
        elif outcome.kind == DispatchOutcomeKind.SUBMIT_FAILED_TERMINAL:
            action_tag = "submit_failed_terminal"
        else:
            action_tag = outcome.kind  # runner_failed / sandbox_unavailable
        self._audit.append(
            action=action_tag,
            assignment_id=response.assignment_id,
            coordinator_experiment_id=response.coordinator_experiment_id,
            tenant_id=unit.tenant_id,
            unit_id=unit.unit_id,
            manifest_sha256=unit.manifest_sha256,
            reason=outcome.reason,
        )
        if outcome.kind in submit_failed_kinds:
            # Per M6-tail: a SUBMIT_FAILED outcome means the Result is in
            # the pending_submissions queue. The transient case will retry
            # on the next tick; the terminal case sits for operator review.
            # In neither case do we double-refuse: the coordinator may or
            # may not have received the result, and an explicit refuse here
            # would tell it to free the slot we may have already filled.
            return
        # Runner crashed or sandbox unavailable — tell coordinator to free
        # the slot.
        self._call_refuse_after_failure(response, action_tag, outcome.reason or "")

    def _call_refuse_after_failure(
        self,
        response: AssignmentResponse,
        kind: str,
        reason: str,
    ) -> None:
        """Tell the coordinator the assignment failed after acceptance so
        the unit can be re-offered. Network errors during the refuse log
        a warning but don't propagate."""
        assert response.work_unit is not None
        try:
            self._coordinator.refuse_assignment(
                worker_id=self._worker_id,
                unit_id=response.work_unit.unit_id,
                kind=kind,
                reason=reason,
            )
            self._stats.refuse_calls_succeeded += 1
        except (AssignmentNotFoundError, AssignmentAlreadyResolvedError) as exc:
            self._stats.refuse_calls_failed += 1
            logger.info(
                "post-failure refuse no-op (unit=%s): %s",
                response.work_unit.unit_id,
                exc,
            )
        except CoordinatorError as exc:
            self._stats.refuse_calls_failed += 1
            logger.warning(
                "post-failure refuse failed (unit=%s): %s",
                response.work_unit.unit_id,
                exc,
            )

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
