"""W-S (§9 #43) ModelServer — the BYOM supply↔serving bridge, and the
OllamaBackend HTTP/CLI plumbing (mocked; no real Ollama)."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

from auspexai_worker.inference.backend import BackendError, OllamaBackend
from auspexai_worker.inference.server import ModelServeError, ModelServer, backend_handle
from auspexai_worker.models.store import ModelStore


class FakeBackend:
    def __init__(self, *, healthy: bool = True, preloaded: bool = False) -> None:
        self.healthy = healthy
        self.preloaded = preloaded
        self.created: list[tuple[str, str]] = []
        self.chats: list[tuple[str, list, dict]] = []

    def is_healthy(self) -> bool:
        return self.healthy

    def has_model(self, handle: str) -> bool:
        return self.preloaded or any(h == handle for h, _ in self.created)

    def create_model(self, handle: str, modelfile: str) -> None:
        self.created.append((handle, modelfile))

    def chat(self, handle: str, messages: list, options: dict) -> dict[str, Any]:
        self.chats.append((handle, messages, options))
        return {"message": {"role": "assistant", "content": "ok"}, "eval_count": 1}


def _store_with_model(tmp_path: Path, model_id: str = "tiny-q4") -> tuple[ModelStore, bytes]:
    content = b"GGUF fake weights bytes"
    model_dir = tmp_path / "models" / model_id
    model_dir.mkdir(parents=True)
    (model_dir / "weights.gguf").write_bytes(content)
    return ModelStore(tmp_path / "models"), content


def test_serve_creates_pinned_modelfile_warms_and_caches(tmp_path: Path):
    store, content = _store_with_model(tmp_path)
    backend = FakeBackend()
    server = ModelServer(store, backend, seed=123, num_ctx=2048)

    served = server.serve("tiny-q4")

    assert served.model_id == "tiny-q4"
    assert served.handle == backend_handle("tiny-q4") == "auspex-tiny-q4"
    # Supply-chain provenance: the digest IS the file's sha256.
    assert served.gguf_sha256 == hashlib.sha256(content).hexdigest()

    # Modelfile references the BYOM file in place + pins determinism (§4).
    handle, modelfile = backend.created[0]
    assert handle == "auspex-tiny-q4"
    assert f"FROM {served.gguf_path}" in modelfile
    assert "PARAMETER temperature 0" in modelfile
    assert "PARAMETER seed 123" in modelfile
    assert "PARAMETER num_ctx 2048" in modelfile

    # Warmed: one throwaway generation.
    assert len(backend.chats) == 1

    # Cached: second serve is free (no re-create, no re-warm).
    again = server.serve("tiny-q4")
    assert again is served
    assert len(backend.created) == 1
    assert len(backend.chats) == 1
    assert server.served_ids() == ["tiny-q4"]


def test_serve_skips_create_when_backend_already_has_model(tmp_path: Path):
    store, _ = _store_with_model(tmp_path)
    backend = FakeBackend(preloaded=True)
    server = ModelServer(store, backend)
    server.serve("tiny-q4")
    assert backend.created == []  # registered in a prior daemon run
    assert len(backend.chats) == 1  # still warmed


def test_serve_refuses_missing_model(tmp_path: Path):
    server = ModelServer(ModelStore(tmp_path / "models"), FakeBackend())
    with pytest.raises(ModelServeError, match="not in the local store"):
        server.serve("never-pulled")


def test_serve_refuses_no_gguf_and_ambiguous_gguf(tmp_path: Path):
    store, _ = _store_with_model(tmp_path)
    no_gguf = tmp_path / "models" / "empty-model"
    no_gguf.mkdir()
    (no_gguf / "README.txt").write_text("nothing here")
    server = ModelServer(store, FakeBackend())
    with pytest.raises(ModelServeError, match=r"no \.gguf"):
        server.serve("empty-model")

    two = tmp_path / "models" / "two-quants"
    two.mkdir()
    (two / "a.gguf").write_bytes(b"a")
    (two / "b.gguf").write_bytes(b"b")
    with pytest.raises(ModelServeError, match="expected exactly one"):
        server.serve("two-quants")


def test_serve_refuses_unhealthy_backend(tmp_path: Path):
    store, _ = _store_with_model(tmp_path)
    server = ModelServer(store, FakeBackend(healthy=False))
    with pytest.raises(ModelServeError, match="not reachable"):
        server.serve("tiny-q4")


# ---- OllamaBackend plumbing -------------------------------------------------


def test_ollama_chat_and_health(tmp_path: Path):
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/chat":
            seen["body"] = request.read()
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "hi"}, "eval_count": 3},
            )
        return httpx.Response(404)

    backend = OllamaBackend(transport=httpx.MockTransport(handler))
    assert backend.is_healthy() is True
    resp = backend.chat("auspex-m", [{"role": "user", "content": "x"}], {"temperature": 0})
    assert resp["message"]["content"] == "hi"
    import json as _json

    body = _json.loads(seen["body"])
    assert body["model"] == "auspex-m"
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0


def test_ollama_unhealthy_when_down():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    backend = OllamaBackend(transport=httpx.MockTransport(handler))
    assert backend.is_healthy() is False
    with pytest.raises(BackendError):
        backend.chat("h", [{"role": "user", "content": "x"}], {})


def test_ollama_create_model_via_cli(tmp_path: Path):
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        # The Modelfile must exist at CLI time.
        assert Path(argv[4]).read_text().startswith("FROM ")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    backend = OllamaBackend(cli_runner=fake_run)
    backend.create_model("auspex-m", "FROM /store/m/w.gguf\nPARAMETER temperature 0\n")
    assert calls[0][:3] == ["ollama", "create", "auspex-m"]


def test_ollama_create_model_failure_raises():
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="no such file")

    backend = OllamaBackend(cli_runner=fake_run)
    with pytest.raises(BackendError, match="exit=1"):
        backend.create_model("auspex-m", "FROM /nope\n")
