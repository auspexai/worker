"""Tests for the sandbox argv builder."""

from __future__ import annotations

import os
import sys

import pytest

from auspexai_worker.sandbox import (
    SandboxConfig,
    SandboxNotAvailableError,
    SandboxPolicy,
    build_argv,
    check_bubblewrap_available,
    probe_bubblewrap,
)
from auspexai_worker.sandbox.wrapper import _strict_fs_argv, resolve_runner_bin


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


_RUNNER = "/opt/auspexai-worker/bin/auspexai-worker-runner"


def _strict_config(**kw) -> SandboxConfig:
    base = dict(
        use_bubblewrap=True,
        policy=SandboxPolicy.STRICT,
        runner_bin=_RUNNER,
        workspace_path="/var/lib/auspexai-worker/work/u-1",
        output_path="/var/lib/auspexai-worker/work/u-1/output.json",
        unit_id="u-1",
        manifest_sha256="a" * 64,
    )
    base.update(kw)
    return SandboxConfig(**base)


class TestStrictPolicy:
    def test_strict_fs_argv_drops_host_fs_and_narrows(self) -> None:
        """§41(a): STRICT replaces --dev-bind / / with narrow read-only system +
        venv binds, a tmpfs, and the workspace as the only host-writable path."""
        args = _strict_fs_argv(_strict_config(), _RUNNER)
        assert "--dev-bind" not in args  # the whole-host hole is gone
        assert "--ro-bind" in args and "/usr" in args
        assert "/opt/auspexai-worker" in args  # the worker venv (runner + python)
        assert "--tmpfs" in args
        i = args.index("--setenv")
        assert args[i : i + 3] == ["--setenv", "HOME", "/tmp"]
        # the per-unit workspace IS bound (writable output path).
        assert "--bind" in args
        assert "/var/lib/auspexai-worker/work/u-1" in args

    def test_strict_build_argv_isolates_namespaces_and_binds_executor(self) -> None:
        if not check_bubblewrap_available():
            pytest.skip("bubblewrap not installed on this host")
        argv = build_argv(
            _strict_config(
                executor_command=["python", "exec.py"],
                executor_package_dir="/srv/pkg",
                models_dir="/srv/models",
                inference_socket="/var/lib/auspexai-worker/work/u-1/broker.sock",
            ),
            seccomp_fd=7,  # AUD-9: STRICT now requires a seccomp fd
        )
        assert "--dev-bind" not in argv  # host-fs hole closed
        for ns in ("--unshare-net", "--unshare-pid", "--unshare-ipc", "--unshare-uts"):
            assert ns in argv
        # executor reaches its package + model store + broker socket — and only
        # those, plus the system dirs + workspace.
        assert "/srv/pkg" in argv
        assert "/srv/models" in argv
        # No sensitive host path is bound into the sandbox.
        joined = " ".join(argv)
        assert "/root" not in joined
        assert ".config/auspexai-worker" not in joined

    def test_permissive_default_still_shares_host_fs(self) -> None:
        if not check_bubblewrap_available():
            pytest.skip("bubblewrap not installed on this host")
        argv = build_argv(_config(use_bubblewrap=True))  # default = PERMISSIVE
        assert "--dev-bind" in argv
        assert "--unshare-pid" not in argv


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
