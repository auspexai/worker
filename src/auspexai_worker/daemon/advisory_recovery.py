"""Advisory auto-recovery — clear the serve-advisory card once the volunteer's
recommended fix takes effect, WITHOUT waiting for the coordinator to assign the next
unit.

The worker is the volunteer's machine to manage, so their action (drop the page cache,
update/restart Ollama) must get a worker-UI response. Previously the card cleared only
on the next SUCCESSFUL serve — so after a volunteer fixed the problem it could linger
until work happened to be routed there again, which reads as "still broken."

Runs on the heartbeat tick with CHEAP PROBES ONLY (the heartbeat is liveness-critical
— never re-serve here). The next real dispatch is the definitive confirmation and
re-raises the advisory if the serve still fails, so an optimistic clear is safe."""

from __future__ import annotations

import logging

from auspexai_worker.inference.server import ADVISORY_GPU_OOM, ADVISORY_STALE_BACKEND
from auspexai_worker.updates import ollama_update_recommended

logger = logging.getLogger(__name__)

# A GPU-OOM card clears when free memory rises at least this many GB above its
# raise-time baseline — enough to be confident the volunteer freed memory
# (sync + drop_caches), not just measurement noise.
_GPU_OOM_RECOVERY_MARGIN_GB = 0.5


def advisory_recovered(
    row, *, backend_version: str | None, available_memory_gb: float | None
) -> bool:
    """True when the condition behind `row` looks resolved by the volunteer's action:
      - stale backend → Ollama updated to >= the recommended floor,
      - GPU-OOM → free memory rose above the raise-time baseline by the margin.
    A generic serve error has no cheap recovery signal → clears on the next successful
    serve (handled by the ModelServer), so this returns False for it."""
    if row.kind == ADVISORY_STALE_BACKEND:
        return backend_version is not None and not ollama_update_recommended(backend_version)
    if row.kind == ADVISORY_GPU_OOM:
        base = row.available_at_raise_gb
        if base is None or available_memory_gb is None:
            return False
        return available_memory_gb >= base + _GPU_OOM_RECOVERY_MARGIN_GB
    return False


def run_recovery_check(repo, *, backend, available_memory_gb: float | None) -> bool:
    """If a serve-advisory is active and looks recovered by the volunteer's fix, CLEAR
    it (the dashboard card then vanishes). Returns True if cleared. Best-effort — never
    raises, so it's safe to call from the heartbeat tick."""
    try:
        row = repo.get()
        if row is None:
            return False
        version: str | None = None
        if row.kind == ADVISORY_STALE_BACKEND:
            probe = getattr(backend, "version", None)
            if callable(probe):
                try:
                    version = probe()
                except Exception:
                    version = None
        if advisory_recovered(
            row, backend_version=version, available_memory_gb=available_memory_gb
        ):
            repo.clear()
            logger.info(
                "serve-advisory (%s, model=%s) auto-cleared — the recommended fix took effect",
                row.kind,
                row.model_id,
            )
            return True
    except Exception:
        logger.debug("advisory recovery check failed (ignored)", exc_info=True)
    return False
