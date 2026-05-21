"""Tests for the sandbox argv builder."""

from __future__ import annotations

import pytest

from auspexai_worker.sandbox import (
    SandboxConfig,
    SandboxNotAvailableError,
    build_argv,
    check_bubblewrap_available,
    probe_bubblewrap,
)


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
        # The runner is the last argv element after "--".
        dashdash = argv.index("--")
        assert argv[dashdash + 1 :] == ["auspexai-worker-runner"]

    def test_missing_bwrap_raises(self) -> None:
        with pytest.raises(SandboxNotAvailableError):
            build_argv(_config(use_bubblewrap=True, bwrap_path="bwrap-that-does-not-exist"))


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
