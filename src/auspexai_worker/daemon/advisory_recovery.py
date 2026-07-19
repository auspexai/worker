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
from collections.abc import Callable

from auspexai_worker.inference.server import ADVISORY_GPU_OOM, ADVISORY_STALE_BACKEND
from auspexai_worker.updates import ollama_update_recommended

logger = logging.getLogger(__name__)

# A GPU-OOM card clears when free memory rises at least this many GB above its
# raise-time baseline — enough to be confident the volunteer freed memory
# (sync + drop_caches), not just measurement noise. This is the RELATIVE signal, and
# it's responsive for a classic system-RAM exhaustion (raised near-starved → freeing
# memory clears it immediately, even below the absolute floor below).
_GPU_OOM_RECOVERY_MARGIN_GB = 0.5
# ...but the relative signal ALONE is wrong for the recurring Jetson case: a CUDA/GPU
# allocation OOM that starves the GPU while SYSTEM RAM is plentiful (page cache), so the
# raise-time baseline is already HIGH (~6 GB on a 7.4 GB box). Freeing page cache +
# restarting the serve stack can't push system-free-RAM 0.5 GB above an already-high
# baseline, so the card would pin forever after a correct remediation. So a GPU-OOM card
# ALSO clears once free memory is comfortably healthy in ABSOLUTE terms — this covers both
# the baseline-less orphaned card AND the high-baseline GPU/page-cache case. A genuine
# RAM-exhaustion OOM happens near-starved (~1-2 GB free), so this floor sits clearly above
# that, and the next real dispatch re-raises WITH a baseline if the serve still fails — an
# optimistic clear is safe.
_GPU_OOM_STALE_CLEAR_GB = 3.0


def advisory_recovered(
    row, *, backend_version: str | None, available_memory_gb: float | None
) -> bool:
    """True when the condition behind `row` looks resolved by the volunteer's action:
      - stale backend → Ollama updated to >= the recommended floor,
      - GPU-OOM (with a raise-time baseline) → free memory rose above it by the margin,
      - GPU-OOM (no baseline: legacy/orphaned card) → free memory is comfortably
        healthy in absolute terms (the dispatch backstop re-raises if still broken).
    A generic serve error has no cheap recovery signal → clears on the next successful
    serve (handled by the ModelServer), so this returns False for it."""
    if row.kind == ADVISORY_STALE_BACKEND:
        return backend_version is not None and not ollama_update_recommended(backend_version)
    if row.kind == ADVISORY_GPU_OOM:
        if available_memory_gb is None:
            return False
        base = row.available_at_raise_gb
        if base is None:
            return available_memory_gb >= _GPU_OOM_STALE_CLEAR_GB
        # Relative (freed above the raise point) OR absolute-healthy — the latter covers a
        # GPU/page-cache OOM whose baseline is already high (system RAM was fine), where the
        # relative test can never be met after a correct remediation.
        return (
            available_memory_gb >= base + _GPU_OOM_RECOVERY_MARGIN_GB
            or available_memory_gb >= _GPU_OOM_STALE_CLEAR_GB
        )
    return False


def run_recovery_check(
    repo,
    *,
    backend,
    available_memory_gb: float | None,
    on_cleared: Callable[[str, str], None] | None = None,
) -> bool:
    """If a serve-advisory is active and looks recovered by the volunteer's fix, CLEAR
    it (the dashboard card then vanishes). Returns True if cleared. Best-effort — never
    raises, so it's safe to call from the heartbeat tick.

    `on_cleared(kind, model_id)` fires when an advisory is cleared, so the caller can
    ALSO tell the coordinator the node recovered — a locally-cleared GPU-OOM card doesn't
    lift the coordinator's model-level serve exclusion, so without this the remediated node
    stays benched the full 6h (the exact remote-volunteer confusion this closes)."""
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
            if on_cleared is not None and row.model_id:
                try:
                    on_cleared(row.kind, row.model_id)
                except Exception:
                    logger.debug("serve-recovered signal failed (ignored)", exc_info=True)
            return True
    except Exception:
        logger.debug("advisory recovery check failed (ignored)", exc_info=True)
    return False
