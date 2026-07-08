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
from dataclasses import dataclass
from pathlib import Path

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
    ) -> None:
        self._store = store
        self._backend = backend
        self._seed = seed
        self._num_ctx = num_ctx
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

        try:
            if not self._backend.has_model(handle):
                modelfile = self._modelfile(gguf)
                self._backend.create_model(handle, modelfile)
            # Warm: one throwaway single-token generation so the first real
            # unit doesn't pay the cold-load latency.
            self._backend.chat(
                handle,
                [{"role": "user", "content": "ok"}],
                {"temperature": 0, "seed": self._seed, "num_predict": 1},
            )
        except BackendError as exc:
            raise ModelServeError(f"failed to serve {model_id}: {exc}") from exc

        served = ServedModel(model_id=model_id, handle=handle, gguf_sha256=digest, gguf_path=gguf)
        with self._lock:
            self._served[model_id] = served
        logger.info("serving model %s as %s (gguf sha256 %s…)", model_id, handle, digest[:12])
        return served

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
