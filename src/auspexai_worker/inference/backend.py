"""Inference backend — the model runtime the worker manages (W-S §6).

Ollama is the D6 backend (the Sentinel-proven path): the worker talks to a
host-side Ollama daemon over localhost HTTP. The daemon lives in the HOST
net namespace — the sandboxed executor can never reach it directly; only
the broker (also host-side) forwards to it. An embedded llama-cpp backend
is a future drop-in behind the same protocol.

Model creation goes through the `ollama` CLI (`ollama create <handle> -f
Modelfile`) rather than the HTTP create API — the CLI is the stable
interface across Ollama versions for Modelfile registration. Chat/show/
health use the HTTP API (ported from `sentinel/ollama.py::OllamaClient`,
near-mechanically per the W-S design).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

# Generation can be legitimately slow on volunteer hardware (Jetson-class
# boxes); the daemon's runner timeout is the hard wall-clock bound, this is
# just the per-HTTP-call ceiling under it.
CHAT_TIMEOUT_SECONDS = 600.0
CREATE_TIMEOUT_SECONDS = 600.0

# Well-known `ollama` CLI install locations, searched when the binary is not on
# PATH. macOS launchd (and the strict sandbox) hand the worker daemon a minimal
# PATH (/usr/bin:/bin:/usr/sbin:/sbin) that omits Homebrew + the desktop-app
# bundle — so a bare `ollama` is `[Errno 2] No such file or directory` at
# create-time even though the HTTP server (which create-time provenance and the
# capability probe both reach) is perfectly healthy. That exact split stranded
# the first macOS worker: it advertised the model off the HTTP probe, then
# refused every matched unit because `ollama create` couldn't find the binary.
_OLLAMA_FALLBACK_PATHS: tuple[str, ...] = (
    "/opt/homebrew/bin/ollama",  # macOS Apple-Silicon Homebrew
    "/usr/local/bin/ollama",  # macOS Intel Homebrew / Linux install.sh
    "/Applications/Ollama.app/Contents/Resources/ollama",  # macOS desktop-app bundle
    "/usr/bin/ollama",  # Linux distro packages
    os.path.expanduser("~/.local/bin/ollama"),  # user-local install
)


def resolve_ollama_bin(explicit: str | None = None) -> str:
    """Resolve the `ollama` CLI to a concrete path for the create-time subprocess.

    Order: an explicit operator override (`[inference] ollama_bin`) wins as-is;
    otherwise PATH (`shutil.which`); otherwise the well-known install locations
    above. Falls back to the bare name `"ollama"` so a genuinely-absent install
    still produces a legible error. This is what lets a macOS worker whose
    `ollama` lives in /opt/homebrew/bin work under launchd's minimal PATH without
    the operator hand-editing their launch agent.
    """
    if explicit:
        return explicit
    found = shutil.which("ollama")
    if found:
        return found
    for candidate in _OLLAMA_FALLBACK_PATHS:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return "ollama"


class BackendError(Exception):
    """The backend refused or failed a request (daemon down, create failed,
    generation error). The broker maps this to an `{"ok": false}` reply —
    it must never crash the worker daemon."""


class InferenceBackend(Protocol):
    """What the ModelServer + broker need from a runtime. Kept minimal so a
    fake (tests) or an embedded llama-cpp backend (future) drops in."""

    def is_healthy(self) -> bool: ...

    def has_model(self, handle: str) -> bool: ...

    def create_model(self, handle: str, modelfile: str) -> None: ...

    def chat(
        self, handle: str, messages: list[dict[str, Any]], options: dict[str, Any]
    ) -> dict[str, Any]: ...

    def loaded_models(self) -> list[str]: ...

    def unload(self, handle: str) -> None: ...


class OllamaBackend:
    """Ollama over localhost HTTP + the `ollama` CLI for Modelfile creation.

    `cli_runner` is an injectable seam for tests (signature matches
    `subprocess.run`); default runs the real CLI.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        *,
        ollama_bin: str | None = None,
        cli_runner=subprocess.run,
        transport: httpx.BaseTransport | None = None,
        keep_alive: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # Resolve to a concrete path so create-time works under a minimal PATH
        # (macOS launchd). An explicit override is honored as-is; None auto-resolves.
        self._ollama_bin = resolve_ollama_bin(ollama_bin)
        self._cli_runner = cli_runner
        self._transport = transport
        # §9 #46 serving policy: how long Ollama keeps the model loaded after
        # a request. None = Ollama's default (~5m). "0" = Sentinel's
        # unload-always posture (their fix for model release/reload wedging —
        # sentinel run_batch.py/worker.py send keep_alive:0); a long value
        # ("30m"/"24h") trades memory for warm latency (D6 measured ~7s
        # reload per idle gap on a Jetson 1B-f16).
        self._keep_alive = keep_alive

    # ---- HTTP plumbing ----------------------------------------------------

    def _post(self, path: str, body: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=timeout, transport=self._transport) as client:
                r = client.post(f"{self.base_url}{path}", json=body)
                r.raise_for_status()
                return r.json()
        except httpx.HTTPError as exc:
            raise BackendError(f"ollama {path} failed: {exc}") from exc

    def _get(self, path: str, *, timeout: float = 10.0) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=timeout, transport=self._transport) as client:
                r = client.get(f"{self.base_url}{path}")
                r.raise_for_status()
                return r.json()
        except httpx.HTTPError as exc:
            raise BackendError(f"ollama {path} failed: {exc}") from exc

    # ---- InferenceBackend ---------------------------------------------------

    def is_healthy(self) -> bool:
        try:
            self._get("/api/tags")
            return True
        except BackendError:
            return False

    def cli_available(self) -> bool:
        """Whether the `ollama` CLI (needed by create_model) actually resolves.

        Distinct from is_healthy() (the HTTP server): a worker can have a healthy
        server but an unresolvable CLI — the macOS launchd PATH gap. The daemon
        checks this at start so it can refuse to advertise inference rather than
        accept-then-refuse every matched unit."""
        if os.path.isabs(self._ollama_bin):
            return os.path.isfile(self._ollama_bin) and os.access(self._ollama_bin, os.X_OK)
        return shutil.which(self._ollama_bin) is not None

    @property
    def ollama_bin(self) -> str:
        """The resolved CLI path (for startup diagnostics)."""
        return self._ollama_bin

    def version(self) -> str | None:
        """The serving Ollama's version (GET /api/version), or None when
        unreachable/odd. Determinism provenance (§9 #46): the runtime version
        affects inference outputs, so the daemon probes it once at start and
        declares it in heartbeat capabilities."""
        try:
            v = self._get("/api/version").get("version")
        except BackendError:
            return None
        return v if isinstance(v, str) and v else None

    def has_model(self, handle: str) -> bool:
        try:
            self._post("/api/show", {"model": handle}, timeout=10.0)
            return True
        except BackendError:
            return False

    def create_model(self, handle: str, modelfile: str) -> None:
        """Register `handle` from a Modelfile via the CLI. The Modelfile
        REFERENCES the BYOM GGUF in place (`FROM <path>`) — no copy, no second
        model store; Ollama is a runtime view of the content-addressed store."""
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".Modelfile", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(modelfile)
            modelfile_path = fh.name
        try:
            result = self._cli_runner(
                [self._ollama_bin, "create", handle, "-f", modelfile_path],
                capture_output=True,
                text=True,
                timeout=CREATE_TIMEOUT_SECONDS,
                check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise BackendError(f"ollama create {handle} failed to run: {exc}") from exc
        finally:
            Path(modelfile_path).unlink(missing_ok=True)
        if result.returncode != 0:
            tail = (result.stderr or "").strip()[-400:]
            raise BackendError(f"ollama create {handle} exit={result.returncode}: {tail}")
        logger.info("ollama: created model %s", handle)

    def chat(
        self, handle: str, messages: list[dict[str, Any]], options: dict[str, Any]
    ) -> dict[str, Any]:
        """One non-streamed chat generation. Returns the raw Ollama response
        (the broker shapes the wire reply)."""
        return self._post(
            "/api/chat",
            {
                "model": handle,
                "messages": messages,
                "stream": False,
                "options": options,
                **({"keep_alive": self._keep_alive} if self._keep_alive is not None else {}),
            },
            timeout=CHAT_TIMEOUT_SECONDS,
        )

    def loaded_models(self) -> list[str]:
        """Handles currently resident in VRAM (GET /api/ps). Best-effort — an
        empty list on any error, since callers use this only to free memory."""
        try:
            data = self._get("/api/ps")
        except BackendError:
            return []
        models = data.get("models") if isinstance(data, dict) else None
        return [
            m["name"]
            for m in (models or [])
            if isinstance(m, dict) and isinstance(m.get("name"), str)
        ]

    def unload(self, handle: str) -> None:
        """Evict `handle` from VRAM now (keep_alive:0 — the Sentinel run_batch.py
        posture). Best-effort and never raises: freeing memory is opportunistic,
        so an unreachable/odd backend just leaves it loaded rather than failing
        the serve that asked for the room."""
        try:
            self._post("/api/generate", {"model": handle, "keep_alive": 0}, timeout=30.0)
        except BackendError as exc:
            logger.debug("ollama: unload %s failed (ignored): %s", handle, exc)
