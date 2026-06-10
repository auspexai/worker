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


class OllamaBackend:
    """Ollama over localhost HTTP + the `ollama` CLI for Modelfile creation.

    `cli_runner` is an injectable seam for tests (signature matches
    `subprocess.run`); default runs the real CLI.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        *,
        ollama_bin: str = "ollama",
        cli_runner=subprocess.run,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._ollama_bin = ollama_bin
        self._cli_runner = cli_runner
        self._transport = transport

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
            {"model": handle, "messages": messages, "stream": False, "options": options},
            timeout=CHAT_TIMEOUT_SECONDS,
        )
