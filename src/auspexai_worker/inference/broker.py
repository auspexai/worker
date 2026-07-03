"""Per-unit inference broker — the ONLY thing the sandboxed executor can
reach (W-S §2b/§3/§5).

One unix-domain socket per dispatched unit (the D6 authorization model:
the socket IS the capability — bound only for that unit, authorized for
only that unit's model). The daemon opens it before spawning the runner,
the wrapper binds it into the sandbox, the executor talks line-delimited
JSON over it, and the daemon closes it when the unit finishes.

Wire protocol (one JSON object per line, one reply line per request —
mirrored by the tenant-sdk / stdlib `InferenceClient`):

  {"op":"generate","model":"<model_id>","messages":[{"role":"user","content":"…"}],
   "options":{"seed":0,"num_predict":256}}
  → {"ok":true,"message":{"role":"assistant","content":"…"},"eval_count":N,
     "model":"<model_id>"}

  {"op":"info"}
  → {"ok":true,"model":"<model_id>","gguf_sha256":"…","backend_handle":"auspex-…"}

  errors → {"ok":false,"error":"<code>","detail":"…"}
  codes: bad_request | unauthorized_model | params_rejected | caps_exceeded
         | backend_error

Generation-policy enforcement (§4 / v0.2 M1): the broker enforces what the
SIGNED MANIFEST declared, never a free-for-all
(inference_determinism_scoping_memo.md §3b):

  - greedy (the default — temperature 0 or no `inference_determinism`
    block): only `seed`/`num_predict`/`num_ctx` (ints, capped) and
    `temperature` if exactly 0 are accepted; temperature 0 + the pinned
    seed default are FORCED on every backend call. Byte-for-byte the
    pre-M1 behavior.
  - seeded-sampling (declared temperature > 0 + pinned seed): the
    executor may request `temperature` up to the DECLARED value and the
    whitelist knobs `top_p`/`top_k`/`min_p` at exactly their DECLARED
    values (declared-but-unrequested knobs are injected; an undeclared
    knob is rejected). Every request still carries an explicit seed
    (default = the declared pin), so no request is ever unseeded.

Policy is applied PER-REQUEST — the served backend handle stays
policy-neutral (no Modelfile pin), so experiments sharing a served model
can never collide on baked-in params.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from pathlib import Path
from typing import Any

from auspexai_worker.inference.backend import BackendError, InferenceBackend
from auspexai_worker.inference.policy import GenerationPolicy
from auspexai_worker.inference.server import DEFAULT_SEED, ServedModel

logger = logging.getLogger(__name__)

BROKER_SOCKET_NAME = "inference.sock"

# Per-unit caps (§2b iii). Generous for a single work unit; a runaway
# executor hits these long before it hurts the host.
DEFAULT_MAX_REQUESTS = 256
DEFAULT_MAX_LINE_BYTES = 1024 * 1024  # 1 MiB request line
MAX_NUM_PREDICT = 4096

# Linux AF_UNIX sun_path limit (108 incl. NUL); refuse early with a clear
# error instead of a cryptic bind() failure.
_MAX_SOCKET_PATH = 100

_ALLOWED_OPTION_KEYS = frozenset({"seed", "num_predict", "num_ctx", "temperature"})
# v0.2 M1 (memo Q2): the seeded-sampling whitelist — requestable ONLY when the
# signed manifest declares the knob (see sanitize_options).
_SAMPLING_OPTION_KEYS = frozenset({"top_p", "top_k", "min_p"})


def _error(code: str, detail: str) -> dict[str, Any]:
    return {"ok": False, "error": code, "detail": detail}


def sanitize_options(
    raw: Any,
    *,
    policy: GenerationPolicy | None = None,
    default_seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Validate + pin generation options against the DECLARED generation policy
    (v0.2 M1 §3b — enforce the declaration, not a constant).

    Returns the options to send to the backend. Raises ValueError (mapped to
    `params_rejected`) on anything outside what the signed manifest declared.
    `policy=None` (or a greedy policy) is byte-for-byte the pre-M1 hard-0 path.
    """
    if policy is None:
        policy = GenerationPolicy()
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("options must be an object")
    declared_knobs = policy.knobs() if policy.is_sampling else {}
    allowed = _ALLOWED_OPTION_KEYS | set(declared_knobs)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"options not permitted (undeclared or unknown): {sorted(unknown)}")
    out: dict[str, Any] = {}
    if policy.is_sampling:
        # The executor may request LESS sampling than declared, never more.
        temp = raw.get("temperature", policy.temperature)
        if isinstance(temp, bool) or not isinstance(temp, (int, float)) or temp < 0:
            raise ValueError("temperature must be a number >= 0")
        if float(temp) > policy.temperature:
            raise ValueError(
                f"temperature {temp} exceeds the manifest-declared {policy.temperature}"
            )
        out["temperature"] = float(temp)
        # Declared knobs apply at exactly their declared values (injected when
        # unrequested); a differing request is rejected — the envelope is the
        # declaration, not a negotiation.
        for key, declared in declared_knobs.items():
            requested = raw.get(key, declared)
            if requested != declared:
                raise ValueError(
                    f"{key} {requested!r} differs from the manifest-declared {declared!r}"
                )
            out[key] = declared
    else:
        if "temperature" in raw and raw["temperature"] not in (0, 0.0):
            raise ValueError("temperature must be 0 (this experiment declared greedy decoding)")
        out["temperature"] = 0
    # Every request carries an explicit seed — no request is ever unseeded.
    # Sampling: default = the declared pin (from_manifest guarantees it exists).
    # Greedy: the worker default, as before. An executor may pass explicit seeds
    # (a deterministic seed-stream derived in content-addressed executor code).
    if policy.is_sampling:
        default_seed = policy.seed  # type: ignore[assignment]  # required under sampling
    seed = raw.get("seed", default_seed)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    out["seed"] = seed
    if "num_predict" in raw:
        np_ = raw["num_predict"]
        if not isinstance(np_, int) or isinstance(np_, bool) or np_ < 1:
            raise ValueError("num_predict must be a positive integer")
        out["num_predict"] = min(np_, MAX_NUM_PREDICT)
    if "num_ctx" in raw:
        nc = raw["num_ctx"]
        if not isinstance(nc, int) or isinstance(nc, bool) or nc < 1:
            raise ValueError("num_ctx must be a positive integer")
        out["num_ctx"] = nc
    return out


def _validate_messages(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("messages must be a non-empty array")
    for m in raw:
        if (
            not isinstance(m, dict)
            or not isinstance(m.get("role"), str)
            or not isinstance(m.get("content"), str)
        ):
            raise ValueError("each message must be {role: str, content: str}")
    return raw


class UnitInferenceSession:
    """A live per-unit broker socket. Construct via `open_unit_session`;
    always `close()` (dispatch does so in its finally block)."""

    def __init__(
        self,
        *,
        served: ServedModel,
        backend: InferenceBackend,
        socket_path: Path,
        policy: GenerationPolicy | None = None,
        max_requests: int = DEFAULT_MAX_REQUESTS,
        max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
    ) -> None:
        self._served = served
        self._backend = backend
        # v0.2 M1: the unit's manifest-declared generation policy; None ⇒ greedy.
        self._policy = policy
        self.socket_path = socket_path
        self._max_requests = max_requests
        self._max_line_bytes = max_line_bytes
        self._requests_handled = 0
        self._closing = threading.Event()
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(str(socket_path))
        self._listener.listen(2)
        self._listener.settimeout(0.5)  # so close() is prompt
        self._thread = threading.Thread(
            target=self._accept_loop,
            name=f"inference-broker-{socket_path.parent.name}",
            daemon=True,
        )
        self._thread.start()

    @property
    def model_id(self) -> str:
        return self._served.model_id

    @property
    def served_gguf_sha256(self) -> str:
        """The served file's supply-chain digest — the trusted-daemon view the
        dispatcher binds into the v1 signed result (§9 #13a `served_weights`)."""
        return self._served.gguf_sha256

    def close(self) -> None:
        """Stop serving and remove the socket file. Idempotent."""
        if self._closing.is_set():
            return
        self._closing.set()
        try:
            self._listener.close()
        except OSError:
            pass
        self._thread.join(timeout=2.0)
        try:
            self.socket_path.unlink(missing_ok=True)
        except OSError:
            pass

    # ---- serving ------------------------------------------------------------

    def _accept_loop(self) -> None:
        while not self._closing.is_set():
            try:
                conn, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return  # listener closed
            try:
                self._serve_connection(conn)
            except Exception:
                logger.exception("inference broker connection handler failed")
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def _serve_connection(self, conn: socket.socket) -> None:
        buf = b""
        while not self._closing.is_set():
            try:
                chunk = conn.recv(65536)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            if len(buf) > self._max_line_bytes:
                self._send(conn, _error("bad_request", "request line too large"))
                return
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                self._send(conn, self._handle_line(line))

    def _handle_line(self, line: bytes) -> dict[str, Any]:
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            return _error("bad_request", f"invalid JSON: {exc}")
        if not isinstance(request, dict):
            return _error("bad_request", "request must be a JSON object")

        op = request.get("op")
        if op == "info":
            return {
                "ok": True,
                "model": self._served.model_id,
                "gguf_sha256": self._served.gguf_sha256,
                "backend_handle": self._served.handle,
            }
        if op != "generate":
            return _error("bad_request", f"unknown op {op!r}")

        if self._requests_handled >= self._max_requests:
            return _error(
                "caps_exceeded",
                f"per-unit request cap ({self._max_requests}) reached",
            )

        model = request.get("model")
        # The session is the capability: only this unit's model is authorized.
        if model != self._served.model_id:
            return _error(
                "unauthorized_model",
                f"this unit is authorized for {self._served.model_id!r} only",
            )
        try:
            messages = _validate_messages(request.get("messages"))
        except ValueError as exc:
            return _error("bad_request", str(exc))
        try:
            options = sanitize_options(request.get("options"), policy=self._policy)
        except ValueError as exc:
            return _error("params_rejected", str(exc))

        self._requests_handled += 1
        try:
            resp = self._backend.chat(self._served.handle, messages, options)
        except BackendError as exc:
            return _error("backend_error", str(exc))
        return {
            "ok": True,
            "message": resp.get("message", {}),
            "eval_count": resp.get("eval_count", 0),
            "model": self._served.model_id,
        }

    @staticmethod
    def _send(conn: socket.socket, reply: dict[str, Any]) -> None:
        try:
            conn.sendall(json.dumps(reply).encode("utf-8") + b"\n")
        except OSError:
            pass  # client went away; the unit is ending anyway


def open_unit_session(
    *,
    served: ServedModel,
    backend: InferenceBackend,
    socket_dir: Path,
    policy: GenerationPolicy | None = None,
    max_requests: int = DEFAULT_MAX_REQUESTS,
) -> UnitInferenceSession:
    """Open the per-unit broker socket at `<socket_dir>/inference.sock`.

    `socket_dir` is normally the unit workspace — already bind-mounted into
    the sandbox, so the socket file is visible there with no extra mount.
    `policy` is the unit's manifest-declared generation policy (None ⇒ greedy).
    Raises ValueError when the resulting path would exceed the AF_UNIX
    limit (dispatch maps that to a refusal).
    """
    socket_path = socket_dir / BROKER_SOCKET_NAME
    if len(str(socket_path)) > _MAX_SOCKET_PATH:
        raise ValueError(
            f"socket path too long for AF_UNIX ({len(str(socket_path))} chars): {socket_path}"
        )
    return UnitInferenceSession(
        served=served,
        backend=backend,
        socket_path=socket_path,
        policy=policy,
        max_requests=max_requests,
    )
