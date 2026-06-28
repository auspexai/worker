"""Heartbeat loop.

Posts to `POST /api/v0/workers/{worker_id}/heartbeat` at the configured
interval. Uses `threading.Event.wait(timeout=...)` so signal handlers can
short-circuit a long sleep instead of waiting out the interval.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from auspexai_worker.capabilities import Capabilities
from auspexai_worker.coordinator import CoordinatorClient, CoordinatorError
from auspexai_worker.models import download_progress
from auspexai_worker.state import WorkerSelfRepository

logger = logging.getLogger(__name__)

CapabilityCollector = Callable[[], Capabilities]


@dataclass
class HeartbeatStats:
    """Counters tracked across the lifetime of a HeartbeatLoop. Useful for
    `--max-ticks`, observability, and tests."""

    ticks_attempted: int = 0
    ticks_succeeded: int = 0
    ticks_failed: int = 0
    last_error: str | None = None
    last_success_at: datetime | None = None
    errors: list[str] = field(default_factory=list)


class HeartbeatLoop:
    """Continuous heartbeat poster.

    Stops cleanly when `.stop()` is called or `.run(max_ticks=N)` reaches the
    cap (tests use the latter).
    """

    def __init__(
        self,
        *,
        coordinator: CoordinatorClient,
        repo: WorkerSelfRepository,
        worker_id: str,
        capability_collector: CapabilityCollector,
        interval_seconds: float,
        stop_event: threading.Event | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._repo = repo
        self._worker_id = worker_id
        self._collect_capabilities = capability_collector
        self._interval = float(interval_seconds)
        self._stop_event = stop_event or threading.Event()
        self._stats = HeartbeatStats()

    @property
    def stats(self) -> HeartbeatStats:
        return self._stats

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def stop(self) -> None:
        self._stop_event.set()

    def run(self, *, max_ticks: int | None = None) -> HeartbeatStats:
        """Loop until `stop()` is called or `max_ticks` is reached.

        Coordinator errors do not stop the loop — they're logged and the
        loop continues. Returns the populated `HeartbeatStats`.
        """
        logger.info(
            "heartbeat loop starting (worker_id=%s, interval=%.0fs)",
            self._worker_id,
            self._interval,
        )
        while not self._stop_event.is_set():
            if max_ticks is not None and self._stats.ticks_attempted >= max_ticks:
                break
            self._tick_once()
            # Interruptible sleep — stop() can wake us early.
            self._stop_event.wait(timeout=self._interval)
        logger.info(
            "heartbeat loop stopped (attempted=%d succeeded=%d failed=%d)",
            self._stats.ticks_attempted,
            self._stats.ticks_succeeded,
            self._stats.ticks_failed,
        )
        return self._stats

    # ---- internals ------------------------------------------------------

    def _tick_once(self) -> None:
        self._stats.ticks_attempted += 1
        try:
            capabilities = self._collect_capabilities().to_dict()
            # D12 5c: in-flight model-download progress (bytes/total) so a queued
            # researcher sees the pull advance. Empty → None (omit from the body).
            downloads = download_progress.snapshot()
            status = self._coordinator.heartbeat(
                worker_id=self._worker_id,
                capabilities=capabilities,
                model_downloads=downloads or None,
            )
            now = datetime.now(UTC)
            # Refresh the locally-cached trust_tier from the coordinator's
            # response so `status` / the dashboard reflect the worker's live
            # network standing (a coord-side promotion/demotion is otherwise
            # invisible locally).
            self._repo.record_heartbeat(now, trust_tier=status.trust_tier)
            # §9 #46: cache the release announcement (if any) for the status/
            # dashboard surfaces. Informational only — never acted on.
            if status.latest_release is not None:
                self._repo.record_latest_release(
                    version=status.latest_release.version,
                    notes=status.latest_release.notes,
                    url=status.latest_release.url,
                    at=now,
                )
            self._stats.ticks_succeeded += 1
            self._stats.last_success_at = now
            logger.debug("heartbeat ok (tick=%d)", self._stats.ticks_attempted)
        except CoordinatorError as exc:
            self._stats.ticks_failed += 1
            self._stats.last_error = str(exc)
            self._stats.errors.append(str(exc))
            logger.warning("heartbeat failed (tick=%d): %s", self._stats.ticks_attempted, exc)
        except Exception as exc:
            # Defense-in-depth: a monitoring tick (capability collection, etc.)
            # must NEVER kill the heartbeat loop — a crashed loop drops the worker
            # off the network silently (no heartbeats → coordinator marks offline).
            # Record and continue rather than letting the loop thread die.
            # (Root-caused once: a thermal sysfs read raised TypeError inside
            # capability collection and killed the loop on every Jetson.)
            self._stats.ticks_failed += 1
            self._stats.last_error = str(exc)
            self._stats.errors.append(str(exc))
            logger.exception(
                "heartbeat tick raised an unexpected error (tick=%d); continuing",
                self._stats.ticks_attempted,
            )
