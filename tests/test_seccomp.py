"""§41(a) seccomp-bpf — filter construction, STRICT argv wiring, and a real
bwrap escape check (the "escape via syscall" gate)."""

from __future__ import annotations

import os
import subprocess

import pytest

from auspexai_worker.sandbox import (
    SandboxConfig,
    SandboxPolicy,
    build_argv,
    check_bubblewrap_available,
)
from auspexai_worker.sandbox.seccomp import (
    DENIED_SYSCALLS,
    open_seccomp_fd,
    seccomp_bpf,
)


def _strict_config() -> SandboxConfig:
    return SandboxConfig(
        use_bubblewrap=True,
        runner_bin="auspexai-worker-runner",
        workspace_path="/tmp/work/u-1",
        output_path="/tmp/work/u-1/output.json",
        unit_id="u-1",
        manifest_sha256="a" * 64,
        policy=SandboxPolicy.STRICT,
    )


class TestFilter:
    def test_denylist_covers_the_escape_vectors(self) -> None:
        for s in ("ptrace", "mount", "unshare", "setns", "bpf", "kexec_load", "init_module"):
            assert s in DENIED_SYSCALLS

    def test_bpf_builds_nonempty_and_cached(self) -> None:
        a = seccomp_bpf()
        b = seccomp_bpf()
        assert isinstance(a, bytes) and len(a) > 0
        assert a is b  # cached

    def test_open_seccomp_fd_holds_the_program(self) -> None:
        fd = open_seccomp_fd()
        try:
            data = os.read(fd, 1 << 20)
            assert data == seccomp_bpf()
        finally:
            os.close(fd)

    def test_open_seccomp_fd_falls_back_without_memfd(self, monkeypatch) -> None:
        # uv's standalone CPython lacks os.memfd_create (CI hit this); the fd path
        # must still work via the unlinked-tempfile fallback or STRICT breaks there.
        monkeypatch.delattr(os, "memfd_create", raising=False)
        fd = open_seccomp_fd()
        try:
            assert os.read(fd, 1 << 20) == seccomp_bpf()
        finally:
            os.close(fd)


class TestStrictArgv:
    def test_strict_adds_capdrop_and_cgroup_ns(self) -> None:
        if not check_bubblewrap_available():
            pytest.skip("bwrap not installed")
        argv = build_argv(_strict_config())
        assert "--cap-drop" in argv and argv[argv.index("--cap-drop") + 1] == "ALL"
        assert "--unshare-cgroup-try" in argv
        # No fd given → no --seccomp flag (dispatch supplies the fd in prod).
        assert "--seccomp" not in argv

    def test_strict_wires_seccomp_fd(self) -> None:
        if not check_bubblewrap_available():
            pytest.skip("bwrap not installed")
        argv = build_argv(_strict_config(), seccomp_fd=7)
        assert "--seccomp" in argv and argv[argv.index("--seccomp") + 1] == "7"


class TestRealBwrapEscape:
    """The gate proof for one vector: a denied syscall is EPERM inside STRICT,
    while ordinary fork still works (the denylist doesn't break the workload)."""

    def test_unshare_blocked_fork_allowed(self) -> None:
        if not check_bubblewrap_available():
            pytest.skip("bwrap not installed")
        fd = open_seccomp_fd()
        prog = (
            "import ctypes,os,errno\n"
            "libc=ctypes.CDLL(None,use_errno=True)\n"
            "assert libc.unshare(0x00020000)==-1 and ctypes.get_errno()==errno.EPERM\n"
            "pid=os.fork()\n"
            "if pid==0: os._exit(7)\n"
            "assert os.WEXITSTATUS(os.waitpid(pid,0)[1])==7\n"
            "print('OK')\n"
        )
        argv = [
            "bwrap",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--unshare-user",
            "--unshare-pid",
            "--seccomp",
            str(fd),
            "--",
            "python3",
            "-c",
            prog,
        ]
        try:
            p = subprocess.run(argv, pass_fds=(fd,), capture_output=True, text=True, timeout=30)
        finally:
            os.close(fd)
        assert p.returncode == 0, p.stderr
        assert p.stdout.strip() == "OK"
