"""W-S (§9 #43) ModelServer — the BYOM supply↔serving bridge, and the
OllamaBackend HTTP/CLI plumbing (mocked; no real Ollama)."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

from auspexai_worker.inference.backend import BackendError, OllamaBackend, resolve_ollama_bin
from auspexai_worker.inference.server import (
    ModelServeError,
    ModelServer,
    ServeAdvisory,
    backend_handle,
)
from auspexai_worker.models.store import ModelStore


class FakeBackend:
    def __init__(
        self,
        *,
        healthy: bool = True,
        preloaded: bool = False,
        loaded: list[str] | None = None,
        chat_fail_times: int = 0,
        chat_error: str = "cudaMalloc failed: out of memory",
    ) -> None:
        self.healthy = healthy
        self.preloaded = preloaded
        self.created: list[tuple[str, str]] = []
        self.chats: list[tuple[str, list, dict]] = []
        self._loaded = list(loaded or [])
        self.unloaded: list[str] = []
        self._chat_fail_times = chat_fail_times
        self._chat_error = chat_error

    def is_healthy(self) -> bool:
        return self.healthy

    def has_model(self, handle: str) -> bool:
        return self.preloaded or any(h == handle for h, _ in self.created)

    def create_model(self, handle: str, modelfile: str) -> None:
        self.created.append((handle, modelfile))

    def chat(self, handle: str, messages: list, options: dict) -> dict[str, Any]:
        if self._chat_fail_times > 0:
            self._chat_fail_times -= 1
            raise BackendError(self._chat_error)
        self.chats.append((handle, messages, options))
        return {"message": {"role": "assistant", "content": "ok"}, "eval_count": 1}

    def loaded_models(self) -> list[str]:
        return list(self._loaded)

    def unload(self, handle: str) -> None:
        self.unloaded.append(handle)
        if handle in self._loaded:
            self._loaded.remove(handle)


def _store_with_model(tmp_path: Path, model_id: str = "tiny-q4") -> tuple[ModelStore, bytes]:
    content = b"GGUF fake weights bytes"
    model_dir = tmp_path / "models" / model_id
    model_dir.mkdir(parents=True)
    (model_dir / "weights.gguf").write_bytes(content)
    return ModelStore(tmp_path / "models"), content


def test_serve_creates_policy_neutral_modelfile_warms_and_caches(tmp_path: Path):
    store, content = _store_with_model(tmp_path)
    backend = FakeBackend()
    server = ModelServer(store, backend, seed=123, num_ctx=2048)

    served = server.serve("tiny-q4")

    assert served.model_id == "tiny-q4"
    assert served.handle == backend_handle("tiny-q4") == "auspex-tiny-q4"
    # Supply-chain provenance: the digest IS the file's sha256.
    assert served.gguf_sha256 == hashlib.sha256(content).hexdigest()

    # Modelfile references the BYOM file in place and is POLICY-NEUTRAL
    # (v0.2 M1 §3b): the served handle is shared across experiments, so no
    # temperature/seed is baked in — the broker sets them per-request from
    # the unit's declared policy. Only num_ctx (a resource default) is pinned.
    handle, modelfile = backend.created[0]
    assert handle == "auspex-tiny-q4"
    assert f"FROM {served.gguf_path}" in modelfile
    assert "PARAMETER temperature" not in modelfile
    assert "PARAMETER seed" not in modelfile
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

    # Pin the bin explicitly so the command structure is host-independent
    # (auto-resolve would pick an absolute path on a host with ollama installed).
    backend = OllamaBackend(cli_runner=fake_run, ollama_bin="ollama")
    backend.create_model("auspex-m", "FROM /store/m/w.gguf\nPARAMETER temperature 0\n")
    assert calls[0][:3] == ["ollama", "create", "auspex-m"]


def test_resolve_ollama_bin_explicit_override_wins(monkeypatch):
    # An explicit path is honored as-is — never re-searched.
    monkeypatch.setattr("shutil.which", lambda _: "/somewhere/else/ollama")
    assert resolve_ollama_bin("/custom/ollama") == "/custom/ollama"


def test_resolve_ollama_bin_prefers_path(monkeypatch):
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/local/bin/ollama" if name == "ollama" else None
    )
    assert resolve_ollama_bin(None) == "/usr/local/bin/ollama"


def test_resolve_ollama_bin_falls_back_to_known_location(monkeypatch):
    # PATH miss (the macOS launchd case) → search well-known install locations.
    monkeypatch.setattr("shutil.which", lambda _: None)
    brew = "/opt/homebrew/bin/ollama"
    monkeypatch.setattr("os.path.isfile", lambda p: p == brew)
    monkeypatch.setattr("os.access", lambda p, _mode: p == brew)
    assert resolve_ollama_bin(None) == brew


def test_resolve_ollama_bin_last_resort_bare_name(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr("os.path.isfile", lambda _p: False)
    assert resolve_ollama_bin(None) == "ollama"


def test_cli_available_reflects_resolution(monkeypatch):
    brew = "/opt/homebrew/bin/ollama"
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr("os.path.isfile", lambda p: p == brew)
    monkeypatch.setattr("os.access", lambda p, _mode: p == brew)
    assert OllamaBackend(ollama_bin=brew).cli_available() is True
    # An unresolvable bare name (PATH miss) → not available.
    assert OllamaBackend(ollama_bin="ollama").cli_available() is False


def test_ollama_create_model_failure_raises():
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="no such file")

    backend = OllamaBackend(cli_runner=fake_run)
    with pytest.raises(BackendError, match="exit=1"):
        backend.create_model("auspex-m", "FROM /nope\n")


def test_ollama_version_probe():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.6.5"})
        return httpx.Response(404)

    assert OllamaBackend(transport=httpx.MockTransport(handler)).version() == "0.6.5"


def test_ollama_version_probe_tolerates_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    # §9 #46: provenance probe must never raise — None when unreachable.
    assert OllamaBackend(transport=httpx.MockTransport(handler)).version() is None


def test_ollama_chat_sends_keep_alive_when_configured():
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen["body"] = _json.loads(request.read())
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "x"}})

    backend = OllamaBackend(transport=httpx.MockTransport(handler), keep_alive="0")
    backend.chat("h", [{"role": "user", "content": "x"}], {})
    assert seen["body"]["keep_alive"] == "0"  # Sentinel unload-always posture


def test_ollama_chat_omits_keep_alive_by_default():
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen["body"] = _json.loads(request.read())
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "x"}})

    OllamaBackend(transport=httpx.MockTransport(handler)).chat(
        "h", [{"role": "user", "content": "x"}], {}
    )
    assert "keep_alive" not in seen["body"]  # Ollama default applies


def _store_with_sized_model(
    tmp_path: Path, size_bytes: int, model_id: str = "big-q4"
) -> ModelStore:
    model_dir = tmp_path / "models" / model_id
    model_dir.mkdir(parents=True)
    (model_dir / "weights.gguf").write_bytes(b"\0" * size_bytes)
    return ModelStore(tmp_path / "models")


def test_serve_refuses_model_too_big_for_ram(tmp_path: Path):
    # BYOM RAM guard (last line): a model whose footprint exceeds usable memory is
    # refused at SERVE time — even one side-loaded into the store without `model
    # pull`, so no path can end up serving what the host can't run.
    store = _store_with_sized_model(tmp_path, 2_000_000)  # ~2 MB → footprint ~0.0024 GB
    server = ModelServer(store, FakeBackend(), usable_memory_gb=0.001)  # 0.001 GB budget
    with pytest.raises(ModelServeError, match="exceeds this host's usable memory"):
        server.serve("big-q4")
    assert server.served_ids() == []  # never loaded


def test_serve_allows_fitting_model_with_budget(tmp_path: Path):
    store = _store_with_sized_model(tmp_path, 2_000_000)
    server = ModelServer(store, FakeBackend(), usable_memory_gb=1.0)  # ample budget
    assert server.serve("big-q4").model_id == "big-q4"


def test_serve_no_budget_does_not_gate(tmp_path: Path):
    # Unknown budget ⇒ the serve guard is a backstop, not a wall (never blocks when
    # it can't judge). Preserves the pre-guard behavior on RAM-unknown hosts.
    store = _store_with_sized_model(tmp_path, 2_000_000)
    server = ModelServer(store, FakeBackend())  # usable_memory_gb defaults to None
    assert server.serve("big-q4").model_id == "big-q4"


# ---- GPU-memory guard: A (pre-serve hygiene) + B (recover, then advise) ------


def test_constrained_host_unloads_other_models_before_serving(tmp_path: Path):
    store, _ = _store_with_model(tmp_path)
    backend = FakeBackend(loaded=["auspex-other-model"])
    server = ModelServer(store, backend, usable_memory_gb=6.0)  # at/below the tight line

    server.serve("tiny-q4")

    assert "auspex-other-model" in backend.unloaded  # freed the other model first
    assert "auspex-tiny-q4" not in backend.unloaded  # never unloads the one being served


def test_roomy_host_keeps_other_models_loaded(tmp_path: Path):
    store, _ = _store_with_model(tmp_path)
    backend = FakeBackend(loaded=["auspex-other-model"])
    server = ModelServer(store, backend, usable_memory_gb=64.0)  # well above the line

    server.serve("tiny-q4")

    assert backend.unloaded == []  # no one-model-at-a-time hygiene on a big host


def test_gpu_oom_recovers_by_freeing_vram_and_retrying(tmp_path: Path):
    store, _ = _store_with_model(tmp_path)
    backend = FakeBackend(loaded=["auspex-hog"], chat_fail_times=1)  # first load OOMs, then OK
    server = ModelServer(store, backend, usable_memory_gb=6.0)

    served = server.serve("tiny-q4")  # must not raise

    assert served.model_id == "tiny-q4"
    assert "auspex-hog" in backend.unloaded  # freed VRAM in recovery
    assert len(backend.chats) == 1  # the successful retry's warm chat landed


def test_gpu_oom_persists_refuses_and_emits_advisory(tmp_path: Path):
    store, _ = _store_with_model(tmp_path)
    backend = FakeBackend(chat_fail_times=2)  # OOMs on both the first load and the retry
    advisories: list[ServeAdvisory] = []
    server = ModelServer(store, backend, usable_memory_gb=6.0, advisory_sink=advisories.append)

    with pytest.raises(ModelServeError, match="insufficient GPU memory"):
        server.serve("tiny-q4")

    assert len(advisories) == 1
    adv = advisories[0]
    assert adv.model_id == "tiny-q4"
    # the manual (privileged) remedies are surfaced for the operator, never auto-run
    assert any("drop_caches" in c for c in adv.commands)
    assert any("restart ollama" in c for c in adv.commands)


def test_non_memory_serve_error_is_not_retried_and_no_advisory(tmp_path: Path):
    store, _ = _store_with_model(tmp_path)
    backend = FakeBackend(chat_fail_times=1, chat_error="ollama create failed: no such file")
    advisories: list[ServeAdvisory] = []
    server = ModelServer(store, backend, usable_memory_gb=6.0, advisory_sink=advisories.append)

    with pytest.raises(ModelServeError, match="failed to serve"):
        server.serve("tiny-q4")

    assert backend.chats == []  # never succeeded
    assert advisories == []  # a non-memory failure is not a GPU-OOM advisory


def test_successful_serve_clears_any_prior_advisory(tmp_path: Path):
    store, _ = _store_with_model(tmp_path)
    emitted: list[ServeAdvisory | None] = []
    server = ModelServer(store, FakeBackend(), advisory_sink=emitted.append)

    server.serve("tiny-q4")

    # A clean serve pushes a clear (None) so the dashboard card self-dismisses once
    # the memory pressure is gone.
    assert emitted == [None]


def test_gpu_oom_recovery_then_success_clears_the_slot(tmp_path: Path):
    store, _ = _store_with_model(tmp_path)
    backend = FakeBackend(chat_fail_times=1)  # OOMs once, succeeds on the retry
    emitted: list[ServeAdvisory | None] = []
    server = ModelServer(store, backend, usable_memory_gb=6.0, advisory_sink=emitted.append)

    server.serve("tiny-q4")

    # Recovered by freeing VRAM in-process — no advisory raised, and the success
    # still clears the slot.
    assert emitted == [None]
