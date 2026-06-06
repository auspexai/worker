"""Pre-stage loop (M3b — eager download conductor, worker side).

Polls `GET /workers/{id}/prestage` at a relaxed interval and pulls each model the
conductor directs this worker to acquire (via the same M3 auto-acquire path /
`StoreModelAcquirer`). Runs on its OWN thread — a pre-stage pull can take a long
time and must never block the heartbeat (liveness) or the assignment poller. Only
started when the worker opts into auto-acquire under a `provisioned` policy; a pull
failure is logged and the next directive is still attempted (the conductor re-offers
on the next poll until the model lands or it's no longer needed).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from auspexai_worker.coordinator import CoordinatorClient, CoordinatorError

logger = logging.getLogger(__name__)


@dataclass
class PrestageStats:
    ticks: int = 0
    directives_seen: int = 0
    pulls_succeeded: int = 0
    pulls_failed: int = 0
    last_error: str | None = None
    errors: list[str] = field(default_factory=list)


class PrestageLoop:
    """Polls for pre-stage directives and fulfils them with `acquirer.acquire`."""

    def __init__(
        self,
        *,
        coordinator: CoordinatorClient,
        worker_id: str,
        acquirer,  # provisioning.ModelAcquirer (StoreModelAcquirer)
        interval_seconds: float,
        stop_event: threading.Event | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._worker_id = worker_id
        self._acquirer = acquirer
        self._interval = float(interval_seconds)
        self._stop_event = stop_event or threading.Event()
        self._stats = PrestageStats()

    @property
    def stats(self) -> PrestageStats:
        return self._stats

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def stop(self) -> None:
        self._stop_event.set()

    def run(self, *, max_ticks: int | None = None) -> PrestageStats:
        logger.info(
            "prestage loop starting (worker_id=%s, interval=%.0fs)",
            self._worker_id,
            self._interval,
        )
        while not self._stop_event.is_set():
            if max_ticks is not None and self._stats.ticks >= max_ticks:
                break
            self._tick_once()
            self._stop_event.wait(timeout=self._interval)
        logger.info(
            "prestage loop stopped (ticks=%d pulled=%d failed=%d)",
            self._stats.ticks,
            self._stats.pulls_succeeded,
            self._stats.pulls_failed,
        )
        return self._stats

    def _tick_once(self) -> None:
        self._stats.ticks += 1
        try:
            directives = self._coordinator.get_prestage(worker_id=self._worker_id)
        except CoordinatorError as exc:
            self._stats.last_error = str(exc)
            self._stats.errors.append(str(exc))
            logger.info("prestage poll failed (tick=%d): %s", self._stats.ticks, exc)
            return
        except Exception as exc:  # a poll must never kill the loop thread
            self._stats.last_error = str(exc)
            self._stats.errors.append(str(exc))
            logger.exception("prestage poll raised (tick=%d); continuing", self._stats.ticks)
            return

        for d in directives:
            if self._stop_event.is_set():
                break
            self._stats.directives_seen += 1
            try:
                self._acquirer.acquire(
                    model_id=d.model_id, hf_repo=d.hf_repo, hf_filename=d.hf_filename
                )
                self._stats.pulls_succeeded += 1
                logger.info("pre-staged model %s (%s)", d.model_id, d.hf_repo)
            except Exception as exc:
                self._stats.pulls_failed += 1
                self._stats.last_error = str(exc)
                self._stats.errors.append(str(exc))
                logger.warning("pre-stage pull failed for %s: %s", d.model_id, exc)
