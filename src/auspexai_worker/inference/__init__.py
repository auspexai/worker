"""W-S (§9 #43) — worker model-serving + sandbox inference broker.

The worker serves the experiment's declared model out of the BYOM store
(W-M) and brokers deterministic inference to the sandboxed executor over a
per-unit unix-domain socket — a filesystem object that crosses the
`--unshare-net` boundary, so the executor reaches the model without any
network. Three pieces:

- `backend`  — the model runtime the worker manages (Ollama for D6).
- `server`   — the supply↔serving bridge: `model_id` → BYOM GGUF →
               deterministic backend handle (`auspex-<model_id>`).
- `broker`   — the per-unit unix-socket server the executor talks to;
               enforces model authorization + determinism + caps.

Dormant unless `[inference] backend = "ollama"` is set — the default
(`"none"`) changes no behavior at all.
"""

from auspexai_worker.inference.backend import (
    BackendError,
    InferenceBackend,
    OllamaBackend,
)
from auspexai_worker.inference.broker import (
    BROKER_SOCKET_NAME,
    UnitInferenceSession,
    open_unit_session,
)
from auspexai_worker.inference.server import (
    ModelServeError,
    ModelServer,
    ServedModel,
)

__all__ = [
    "BROKER_SOCKET_NAME",
    "BackendError",
    "InferenceBackend",
    "ModelServeError",
    "ModelServer",
    "OllamaBackend",
    "ServedModel",
    "UnitInferenceSession",
    "open_unit_session",
]
