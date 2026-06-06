"""Tests for the sandbox argv builder."""

from __future__ import annotations

import os
import sys

import pytest

from auspexai_worker.sandbox import (
    SandboxConfig,
    SandboxNotAvailableError,
    build_argv,
    check_bubblewrap_available,
    probe_bubblewrap,
)
from auspexai_worker.sandbox.wrapper import resolve_runner_bin


def _config(*, use_bubblewrap: bool, bwrap_path: str = "bwrap") -> SandboxConfig:
    return SandboxConfig(
        use_bubblewrap=use_bubblewrap,
        runner_bin="auspexai-worker-runner",
        workspace_path="/tmp/work/u-1",
        output_path="/tmp/work/u-1/output.json",
        unit_id="u-1",
        manifest_sha256="a" * 64,
        bwrap_path=bwrap_path,
    )


class TestPassthrough:
    def test_passthrough_argv_is_just_runner(self) -> None:
        argv = build_argv(_config(use_bubblewrap=False))
        assert argv == ["auspexai-worker-runner"]


class TestBubblewrap:
    def test_bwrap_argv_includes_env_setenv(self) -> None:
        if not check_bubblewrap_available():
            pytest.skip("bubblewrap not installed on this host")
        argv = build_argv(_config(use_bubblewrap=True))
        # Basic structural assertions.
        assert argv[0] == "bwrap"
        assert "--die-with-parent" in argv
        assert "--new-session" in argv
        # Env vars present.
        for key in ("AUSPEXAI_UNIT_ID", "AUSPEXAI_MANIFEST_SHA256", "AUSPEXAI_OUTPUT_PATH"):
            assert key in argv
        # The runner is the last argv element after "--", resolved to an
        # absolute path so bwrap's execvp finds it (the sandbox PATH lacks the
        # venv bin dir; a bare name fails with "No such file or directory").
        dashdash = argv.index("--")
        runner = argv[dashdash + 1 :]
        assert len(runner) == 1
        assert runner[0].endswith("auspexai-worker-runner")

    def test_missing_bwrap_raises(self) -> None:
        with pytest.raises(SandboxNotAvailableError):
            build_argv(_config(use_bubblewrap=True, bwrap_path="bwrap-that-does-not-exist"))


class TestResolveRunnerBin:
    def test_absolute_passes_through(self) -> None:
        p = "/opt/auspexai-worker/bin/auspexai-worker-runner"
        assert resolve_runner_bin(p) == p

    def test_colocated_with_interpreter_resolves_absolute(self, tmp_path, monkeypatch) -> None:
        # Simulate a venv bin dir: a fake interpreter + a colocated runner.
        bindir = tmp_path / "bin"
        bindir.mkdir()
        (bindir / "fake-runner").write_text("#!/bin/sh\n")
        monkeypatch.setattr(sys, "executable", str(bindir / "python"))
        resolved = resolve_runner_bin("fake-runner")
        assert resolved == str(bindir / "fake-runner")
        assert os.path.isabs(resolved)

    def test_unresolvable_name_passes_through(self, monkeypatch) -> None:
        # Not colocated with the interpreter and not on PATH → unchanged.
        monkeypatch.setattr(sys, "executable", "/nonexistent/python")
        monkeypatch.setattr("shutil.which", lambda _name: None)
        assert resolve_runner_bin("not-a-real-binary-xyz") == "not-a-real-binary-xyz"


class TestProbeBubblewrap:
    def test_missing_binary_returns_not_ok(self) -> None:
        result = probe_bubblewrap(bwrap_path="bwrap-that-does-not-exist")
        assert result.ok is False
        assert "not found on PATH" in (result.reason or "")

    def test_present_binary_probes_real_namespace(self) -> None:
        if not check_bubblewrap_available():
            pytest.skip("bubblewrap not installed on this host")
        result = probe_bubblewrap()
        # On the test host this should succeed (sysctl was flipped during
        # M4 verification). If the test fails on a CI host without the
        # workaround, the user will see exactly the same actionable error
        # the daemon would surface.
        assert result.ok is True, f"bwrap probe failed unexpectedly: {result.reason}"
