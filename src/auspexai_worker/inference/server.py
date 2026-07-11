"""ModelServer — the supply↔serving bridge (W-S §2.1).

BYOM (W-M) and serving connect through ONE key, `model_id`: the manifest's
`models[].id` == the store directory name == the #30 routing match key ==
the executor's authorized model. Given a `model_id`, the server:

  1. locates the single sha256-verified GGUF in `ModelStore.path_for(model_id)`,
  2. registers it in the backend under the deterministic handle
     `auspex-<model_id>` via a Modelfile that references the file IN PLACE
     (BYOM stays the content-addressed source of truth). The Modelfile pins
     only num_ctx (a resource default) — generation params (temperature /
     seed / sampling knobs) are POLICY-NEUTRAL at the served handle and
     applied per-request by the broker from the unit's declared policy
     (v0.2 M1 §3b), so experiments sharing a served model never collide
     on baked-in params,
  3. warms it (one throwaway generation) so the first real unit isn't cold,
  4. records the GGUF's sha256 so `op:"info"` can return supply-chain
     provenance (the same digest the W-M fetch verified) for the executor
     to stamp into its result payload.

`served_ids()` feeds the heartbeat's `served_models` declaration — the
routing predicate sharpens from "holds the model" (BYOM inventory, #30) to
"holds it AND has it loaded".
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

from auspexai_worker.inference.backend import BackendError, InferenceBackend
from auspexai_worker.models.hf_browse import memory_fits
from auspexai_worker.models.store import ModelStore

logger = logging.getLogger(__name__)

# Worker-enforced defaults (§4). DEFAULT_SEED is the broker's greedy-mode seed
# default (v0.2 M1: a sampling unit's seed comes from its manifest-declared
# policy instead). Every replica gets identical params because the broker sets
# them explicitly ON EVERY REQUEST — the served handle itself is policy-neutral.
DEFAULT_SEED = 0
DEFAULT_NUM_CTX = 4096

# Hosts at or below this usable-memory line hold ~one model at a time and get the
# memory hygiene below (unload others before a load; free VRAM + retry once on a
# GPU out-of-memory failure). Above it — a 24 GB desktop / the Mac worker — the
# guard is off and multiple models stay warm. The Jetson-class boxes that hit this
# in practice report ~5-7 GB usable, well under the line.
CONSTRAINED_USABLE_GB = 12.0

# Substrings that mark a serve failure as a GPU/host memory shortage (vs a missing
# binary, a corrupt GGUF, or a down daemon). Matched case-insensitively against the
# backend error: Ollama surfaces the runner's "cudaMalloc failed: out of memory" /
# "unable to allocate CUDA0 buffer" / Jetson "NvMap" text through the /api/chat 500.
# Broad on purpose — the exact string varies by backend and driver.
_GPU_OOM_MARKERS: tuple[str, ...] = (
    "out of memory",
    "cudamalloc",
    "unable to allocate",
    "cuda0 buffer",
    "cuda buffer",
    "nvmap",
    "cannot allocate memory",
    "insufficient memory",
    "vram",
)


def _looks_like_gpu_oom(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _GPU_OOM_MARKERS)


class ModelServeError(Exception):
    """The model can't be served (missing/ambiguous GGUF, backend down,
    create failed). Dispatch maps this to a refusal — never an echo."""


@dataclass(frozen=True)
class ServedModel:
    """A model loaded into the backend and ready to broker."""

    model_id: str  # the store/manifest/routing id
    handle: str  # the backend-side name (auspex-<model_id>)
    gguf_sha256: str  # supply-chain digest of the served file
    gguf_path: Path


@dataclass(frozen=True)
class ServeAdvisory:
    """A serve failure the host operator might clear by hand. The worker NEVER
    runs the remedies itself — they need privileges (drop the OS page cache,
    restart the model server) a sandboxed/volunteer worker must not assume. It's
    logged and surfaced on the local dashboard (copy-to-run, never auto-run) so
    the operator can act if they choose; the coordinator routes the unit elsewhere
    meanwhile."""

    model_id: str
    reason: str
    commands: tuple[str, ...]
    at: datetime


def backend_handle(model_id: str) -> str:
    return f"auspex-{model_id}"


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelServer:
    """Per-daemon model-serving lifecycle over (store, backend).

    `serve()` is idempotent and cached — the first call for a model pays the
    load + digest + warm cost (can be tens of seconds on volunteer hardware;
    callers sit under the assignment, not the runner timeout); subsequent
    calls return the cached `ServedModel`. Thread-safe: dispatch and the
    heartbeat read concurrently.
    """

    def __init__(
        self,
        store: ModelStore,
        backend: InferenceBackend,
        *,
        seed: int = DEFAULT_SEED,
        num_ctx: int = DEFAULT_NUM_CTX,
        usable_memory_gb: float | None = None,
        advisory_sink: Callable[[ServeAdvisory | None], None] | None = None,
    ) -> None:
        self._store = store
        self._backend = backend
        self._seed = seed
        self._num_ctx = num_ctx
        # Sink for the operator-actionable serve state: called with a ServeAdvisory
        # when a GPU-OOM persists after in-process recovery, and with None to CLEAR
        # it when serving next succeeds. The daemon wires this to persist so the local
        # dashboard can show/hide the card; None sink ⇒ log-only (tests, headless runs).
        self._advisory_sink = advisory_sink
        # RAM guard (BYOM requirement): a model must FIT this host's memory to be
        # served. This is the LAST-LINE, non-bypassable gate — it catches a model
        # that reached the store any way at all, including a raw side-load that
        # skipped the pull-time guard. None ⇒ budget unknown, gate off (backstop).
        self._usable_memory_gb = usable_memory_gb
        self._served: dict[str, ServedModel] = {}
        self._lock = threading.Lock()

    def served_ids(self) -> list[str]:
        """Model ids currently loaded — the heartbeat `served_models` value."""
        with self._lock:
            return sorted(self._served)

    def served_digests(self) -> dict[str, str]:
        """{model_id: served-GGUF sha256} for the loaded models — the heartbeat
        `served_model_digests` value (v0_2 #13a; feeds #13b enforcement)."""
        with self._lock:
            return {mid: s.gguf_sha256 for mid, s in self._served.items()}

    def get_served(self, model_id: str) -> ServedModel | None:
        with self._lock:
            return self._served.get(model_id)

    def serve(self, model_id: str) -> ServedModel:
        """Ensure `model_id` is loaded in the backend; return its ServedModel.

        Raises ModelServeError on any failure — the dispatch gate turns that
        into a refusal (refuse-don't-echo, same posture as provisioning).
        """
        with self._lock:
            cached = self._served.get(model_id)
        if cached is not None:
            return cached

        gguf = self._locate_gguf(model_id)
        # RAM guard: refuse to serve a model this host can't fit — even one that was
        # side-loaded straight into the store (bypassing the pull-time guard). A
        # clean refusal (turned into a dispatch refusal upstream) beats letting the
        # backend OOM mid-load. Unknown budget ⇒ not gating (the backstop).
        if self._usable_memory_gb is not None:
            size = gguf.stat().st_size
            if not memory_fits(size, self._usable_memory_gb):
                raise ModelServeError(
                    f"{model_id!r} (~{size / 1e9:.1f} GB load footprint) exceeds this "
                    f"host's usable memory (~{self._usable_memory_gb:.1f} GB) — cannot serve"
                )
        digest = _sha256_of(gguf)
        handle = backend_handle(model_id)

        if not self._backend.is_healthy():
            raise ModelServeError("inference backend is not reachable")

        # A — memory hygiene on a tight host: hold ~one model at a time, so this
        # load isn't fighting a previously-loaded model for the GPU's memory.
        if self._is_constrained():
            self._free_other_loaded(handle)

        def _load() -> None:
            if not self._backend.has_model(handle):
                self._backend.create_model(handle, self._modelfile(gguf))
            # Warm: one throwaway single-token generation so the first real unit
            # doesn't pay the cold-load latency — and where a GPU-memory failure
            # actually surfaces (the model loads on the first generation).
            self._backend.chat(
                handle,
                [{"role": "user", "content": "ok"}],
                {"temperature": 0, "seed": self._seed, "num_predict": 1},
            )

        try:
            _load()
        except BackendError as exc:
            # B — recover from a GPU-memory failure WITHOUT privileges: unload every
            # loaded model to free VRAM, then retry once. A non-memory BackendError
            # (missing binary, corrupt GGUF, daemon down) is not retried — those
            # don't heal by freeing memory.
            if not _looks_like_gpu_oom(exc):
                raise ModelServeError(f"failed to serve {model_id}: {exc}") from exc
            logger.warning(
                "serve %s: GPU-memory failure (%s) — freeing VRAM and retrying once",
                model_id,
                exc,
            )
            self._free_all_loaded()
            try:
                _load()
            except BackendError as retry_exc:
                self._raise_gpu_oom(model_id, retry_exc)

        served = ServedModel(model_id=model_id, handle=handle, gguf_sha256=digest, gguf_path=gguf)
        with self._lock:
            self._served[model_id] = served
        logger.info("serving model %s as %s (gguf sha256 %s…)", model_id, handle, digest[:12])
        # Serving succeeded — clear any stale GPU-OOM advisory so the dashboard card
        # goes away once the operator (or the recovery above) freed the memory.
        self._emit_advisory(None)
        return served

    # ---- memory hygiene (§ GPU-OOM guard) -----------------------------------

    def _is_constrained(self) -> bool:
        """A tight-memory host that should hold ~one model at a time. Keyed on the
        same usable-memory budget the RAM guard uses; unknown budget ⇒ not tight
        (a big box would just be told to unload for nothing)."""
        return (
            self._usable_memory_gb is not None and self._usable_memory_gb <= CONSTRAINED_USABLE_GB
        )

    def _free_other_loaded(self, keep_handle: str) -> None:
        """Unload every model except `keep_handle` (best-effort), so a new load has
        the GPU to itself. Drops the freed models from the served-cache so heartbeat
        `served_models` stays truthful."""
        try:
            loaded = self._backend.loaded_models()
        except Exception:
            loaded = []
        for other in loaded:
            if other != keep_handle:
                self._backend.unload(other)
        with self._lock:
            self._served = {mid: s for mid, s in self._served.items() if s.handle == keep_handle}

    def _free_all_loaded(self) -> None:
        """Unload everything from VRAM (recovery step). Best-effort; clears the cache."""
        try:
            loaded = self._backend.loaded_models()
        except Exception:
            loaded = []
        for handle in loaded:
            self._backend.unload(handle)
        with self._lock:
            self._served = {}

    def _raise_gpu_oom(self, model_id: str, exc: BackendError) -> NoReturn:
        """Freeing VRAM + retry didn't clear it — refuse with a clear reason and
        emit an operator advisory (manual, privileged remedies; the worker never
        runs them). Whatever else, this always raises."""
        commands = (
            "sudo sync && sudo sysctl vm.drop_caches=3",
            "sudo systemctl restart ollama",
        )
        logger.error(
            "serve %s: insufficient GPU memory after freeing VRAM and retrying. "
            "Manual recovery on this host (needs admin — skip on a sandboxed worker): %s",
            model_id,
            " ; ".join(commands),
        )
        self._emit_advisory(
            ServeAdvisory(
                model_id=model_id,
                reason=(
                    f"GPU out of memory serving {model_id}. The worker freed what it "
                    "could and retried; it still failed, so this unit was declined."
                ),
                commands=commands,
                at=datetime.now(UTC),
            )
        )
        raise ModelServeError(
            f"insufficient GPU memory to serve {model_id} (freed VRAM and retried, still failed)"
        ) from exc

    def _emit_advisory(self, advisory: ServeAdvisory | None) -> None:
        """Push a new advisory (persistent GPU-OOM) or None (clear) to the sink.
        Best-effort — surfacing must never fail the serve/refusal path."""
        sink = self._advisory_sink
        if sink is None:
            return
        try:
            sink(advisory)
        except Exception:
            logger.debug("serve advisory sink failed (ignored)", exc_info=True)

    # ---- internals ----------------------------------------------------------

    def _locate_gguf(self, model_id: str) -> Path:
        """Thin slice: exactly one GGUF per store dir (W-M invariant)."""
        model_dir = self._store.path_for(model_id)
        if not model_dir.is_dir():
            raise ModelServeError(f"model {model_id} is not in the local store")
        ggufs = sorted(model_dir.glob("*.gguf"))
        if not ggufs:
            raise ModelServeError(f"model {model_id} has no .gguf in the store")
        if len(ggufs) > 1:
            raise ModelServeError(
                f"model {model_id} has {len(ggufs)} .gguf files; expected exactly one"
            )
        return ggufs[0]

    def _modelfile(self, gguf: Path) -> str:
        """The policy-neutral Modelfile (§2.1 step 2). References the BYOM file
        in place — no copy, no second store. Deliberately carries NO
        temperature/seed pin (v0.2 M1 §3b): the served handle is shared across
        experiments, so generation params baked here would collide across
        differing declared policies — the broker sets them per-request instead
        (and always did; request options override Modelfile params, so dropping
        the pin does not change greedy behavior)."""
        return f"FROM {gguf}\nPARAMETER num_ctx {self._num_ctx}\n"
