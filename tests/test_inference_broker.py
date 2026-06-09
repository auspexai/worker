"""W-S (§9 #43) inference broker — protocol, authorization, determinism, caps.

Exercises the real unix-socket server with a fake backend: the wire protocol
here is the contract the tenant-sdk / stdlib InferenceClient mirrors.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest

from auspexai_worker.inference.backend import BackendError
from auspexai_worker.inference.broker import (
    BROKER_SOCKET_NAME,
    open_unit_session,
    sanitize_options,
)
from auspexai_worker.inference.server import ServedModel


class FakeBackend:
    """Records chat calls; canned deterministic reply."""

    def __init__(self, *, fail: bool = False) -> None:
        self.chats: list[tuple[str, list, dict]] = []
        self.fail = fail

    def is_healthy(self) -> bool:
        return True

    def has_model(self, handle: str) -> bool:
        return True

    def create_model(self, handle: str, modelfile: str) -> None:  # pragma: no cover
        pass

    def chat(self, handle: str, messages: list, options: dict) -> dict[str, Any]:
        if self.fail:
            raise BackendError("backend exploded")
        self.chats.append((handle, messages, options))
        return {
            "message": {"role": "assistant", "content": "deterministic reply"},
            "eval_count": 7,
        }


def _served(tmp_path: Path) -> ServedModel:
    return ServedModel(
        model_id="tiny-model-q4",
        handle="auspex-tiny-model-q4",
        gguf_sha256="ab" * 32,
        gguf_path=tmp_path / "tiny.gguf",
    )


def _request(socket_path: Path, body: dict) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(5.0)
        s.connect(str(socket_path))
        s.sendall(json.dumps(body).encode() + b"\n")
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.split(b"\n", 1)[0])


@pytest.fixture
def session(tmp_path: Path):
    backend = FakeBackend()
    sess = open_unit_session(served=_served(tmp_path), backend=backend, socket_dir=tmp_path)
    sess._test_backend = backend  # convenience for assertions
    yield sess
    sess.close()


def test_generate_happy_path(session, tmp_path: Path):
    reply = _request(
        session.socket_path,
        {
            "op": "generate",
            "model": "tiny-model-q4",
            "messages": [{"role": "user", "content": "hello"}],
            "options": {"seed": 42, "num_predict": 16},
        },
    )
    assert reply["ok"] is True
    assert reply["message"]["content"] == "deterministic reply"
    assert reply["eval_count"] == 7
    assert reply["model"] == "tiny-model-q4"
    # The backend was called on the HANDLE with pinned deterministic options.
    handle, _messages, options = session._test_backend.chats[0]
    assert handle == "auspex-tiny-model-q4"
    assert options["temperature"] == 0
    assert options["seed"] == 42


def test_unauthorized_model_rejected(session):
    reply = _request(
        session.socket_path,
        {
            "op": "generate",
            "model": "some-other-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert reply["ok"] is False
    assert reply["error"] == "unauthorized_model"
    assert not session._test_backend.chats  # never reached the backend


def test_nondeterministic_params_rejected(session):
    reply = _request(
        session.socket_path,
        {
            "op": "generate",
            "model": "tiny-model-q4",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"temperature": 0.9},
        },
    )
    assert reply["ok"] is False
    assert reply["error"] == "params_rejected"

    reply = _request(
        session.socket_path,
        {
            "op": "generate",
            "model": "tiny-model-q4",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"top_p": 0.5},
        },
    )
    assert reply["ok"] is False
    assert reply["error"] == "params_rejected"


def test_info_returns_provenance(session):
    reply = _request(session.socket_path, {"op": "info"})
    assert reply["ok"] is True
    assert reply["model"] == "tiny-model-q4"
    assert reply["gguf_sha256"] == "ab" * 32
    assert reply["backend_handle"] == "auspex-tiny-model-q4"


def test_bad_request_shapes(session):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(5.0)
        s.connect(str(session.socket_path))
        s.sendall(b"this is not json\n")
        reply = json.loads(s.recv(65536).split(b"\n", 1)[0])
    assert reply["ok"] is False and reply["error"] == "bad_request"

    reply = _request(session.socket_path, {"op": "transmogrify"})
    assert reply["ok"] is False and reply["error"] == "bad_request"

    reply = _request(
        session.socket_path,
        {"op": "generate", "model": "tiny-model-q4", "messages": "not-a-list"},
    )
    assert reply["ok"] is False and reply["error"] == "bad_request"


def test_request_cap_enforced(tmp_path: Path):
    backend = FakeBackend()
    sess = open_unit_session(
        served=_served(tmp_path), backend=backend, socket_dir=tmp_path, max_requests=2
    )
    try:
        body = {
            "op": "generate",
            "model": "tiny-model-q4",
            "messages": [{"role": "user", "content": "hi"}],
        }
        assert _request(sess.socket_path, body)["ok"] is True
        assert _request(sess.socket_path, body)["ok"] is True
        reply = _request(sess.socket_path, body)
        assert reply["ok"] is False and reply["error"] == "caps_exceeded"
    finally:
        sess.close()


def test_backend_error_maps_not_crashes(tmp_path: Path):
    sess = open_unit_session(
        served=_served(tmp_path), backend=FakeBackend(fail=True), socket_dir=tmp_path
    )
    try:
        reply = _request(
            sess.socket_path,
            {
                "op": "generate",
                "model": "tiny-model-q4",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert reply["ok"] is False and reply["error"] == "backend_error"
    finally:
        sess.close()


def test_close_removes_socket(tmp_path: Path):
    sess = open_unit_session(served=_served(tmp_path), backend=FakeBackend(), socket_dir=tmp_path)
    path = sess.socket_path
    assert path.exists() and path.name == BROKER_SOCKET_NAME
    sess.close()
    assert not path.exists()
    sess.close()  # idempotent


def test_socket_path_length_guard(tmp_path: Path):
    deep = tmp_path / ("d" * 120)
    deep.mkdir()
    with pytest.raises(ValueError, match="socket path too long"):
        open_unit_session(served=_served(tmp_path), backend=FakeBackend(), socket_dir=deep)


def test_sanitize_options_whitelist():
    out = sanitize_options({"seed": 7, "num_predict": 99999, "num_ctx": 2048})
    assert out["temperature"] == 0
    assert out["seed"] == 7
    assert out["num_predict"] == 4096  # capped
    assert out["num_ctx"] == 2048

    assert sanitize_options(None)["seed"] == 0  # defaults pinned
    assert sanitize_options({"temperature": 0})["temperature"] == 0

    with pytest.raises(ValueError):
        sanitize_options({"temperature": 0.7})
    with pytest.raises(ValueError):
        sanitize_options({"mirostat": 2})
    with pytest.raises(ValueError):
        sanitize_options({"seed": "not-an-int"})
    with pytest.raises(ValueError):
        sanitize_options({"seed": True})  # bools are not seeds
    with pytest.raises(ValueError):
        sanitize_options("not-a-dict")
