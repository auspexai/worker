"""§41(a) adversarial red-team — the gate-condition-2 proof.

Runs attacker-controlled code INSIDE a real STRICT sandbox (the production
`build_argv` + the seccomp fd + the resource caps) and asserts all five escape
vectors fail:

  1. read a neighbour's data    → filesystem binds (no $HOME / cross-tenant dirs)
  2. reach the identity keystore → filesystem binds (keystore dir not bound)
  3. open a socket / exfiltrate  → --unshare-net (no network)
  4. escape via a syscall        → seccomp denylist (unshare/mount/ptrace → EPERM)
  5. exhaust resources           → rlimit fsize (always) + cgroup memory/pids
                                    (when a delegated subtree exists)

The whole module skips where bwrap can't construct a namespace (CI without
user-ns). The cgroup vectors (5b/5c) need systemd `Delegate=yes`; when no
delegated subtree is present they SKIP WITH REASON rather than passing silently —
run the module under `systemd-run --user -p Delegate=yes -- pytest …` to exercise
them (proven green that way; see the roadmap B1 row)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from auspexai_worker.sandbox import (
    ResourceLimits,
    SandboxConfig,
    SandboxPolicy,
    build_argv,
    probe_bubblewrap,
)
from auspexai_worker.sandbox import resources as res
from auspexai_worker.sandbox.seccomp import open_seccomp_fd

# Run the in-sandbox attacks via the *concrete* interpreter under /usr (which
# STRICT binds read-only), not the venv symlink: build_argv resolves symlinks to
# pick the bind root, so a venv python (symlinked to /usr) would leave the venv
# path itself unbound. The real runner is a venv console script (a real file), so
# production binds correctly; this just mirrors that for the test's stdlib-only
# attack scripts.
_PYTHON = os.path.realpath(sys.executable)

# Full-proof mode (`make redteam` / the self-hosted `sandbox-redteam` runner):
# an unmet precondition (no bwrap, no cgroup delegation) must FAIL LOUDLY rather
# than silently skip — a green run then means all five vectors actually fired.
# In an ordinary `pytest` run the same preconditions just skip.
_REQUIRE_FULL = os.environ.get("AUSPEXAI_REDTEAM_REQUIRE_FULL") == "1"

_probe = probe_bubblewrap()
pytestmark = pytest.mark.skipif(
    not _probe.ok and not _REQUIRE_FULL,
    reason=f"bwrap cannot build a namespace here: {_probe.reason}",
)


def _strict_config(workspace: Path, runner: str) -> SandboxConfig:
    return SandboxConfig(
        use_bubblewrap=True,
        runner_bin=runner,
        workspace_path=str(workspace),
        output_path=str(workspace / "output.json"),
        unit_id="redteam",
        manifest_sha256="a" * 64,
        policy=SandboxPolicy.STRICT,
    )


def _run_in_strict(
    attack: str,
    workspace: Path,
    *,
    rlimits: ResourceLimits | None = None,
    cgroup: ResourceLimits | None = None,
    stdin: str | None = None,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess:
    """Execute `python -c <attack>` inside a real STRICT sandbox built by the
    production argv builder. `rlimits` prepends the prlimit floor; `cgroup` (when
    its subtree is delegated) wraps the runner tree in a per-unit cgroup adopted
    just before the workload is released via stdin."""
    fd = open_seccomp_fd()
    argv = build_argv(_strict_config(workspace, _PYTHON), seccomp_fd=fd)
    argv = [*argv, "-c", attack]
    if rlimits is not None:
        argv = [*rlimits.prlimit_prefix(), *argv]
    unit_cg = cgroup.open_cgroup("redteam") if cgroup is not None else None
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            pass_fds=(fd,),
        )
    finally:
        os.close(fd)
    try:
        if unit_cg is not None:
            unit_cg.adopt(proc.pid)
        out, err = proc.communicate(input=stdin if stdin is not None else "", timeout=timeout)
    finally:
        if unit_cg is not None:
            unit_cg.destroy()
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


# --------------------------------------------------------------- vectors 1 & 2


def test_v1_cannot_read_a_neighbours_home_secret(tmp_path) -> None:
    """A secret in the real $HOME is unreachable: STRICT never binds $HOME."""
    secret = Path.home() / ".auspexai-redteam-secret"
    secret.write_text("TOP-SECRET-NEIGHBOUR-DATA")
    try:
        attack = (
            f"import pathlib\n"
            f"try:\n"
            f"    d = pathlib.Path({str(secret)!r}).read_text()\n"
            f"    print('LEAK:' + d)\n"
            f"except OSError:\n"
            f"    print('BLOCKED')\n"
        )
        r = _run_in_strict(attack, tmp_path)
        assert "BLOCKED" in r.stdout, r
        assert "LEAK" not in r.stdout, r
    finally:
        secret.unlink(missing_ok=True)


def test_v2_cannot_reach_the_identity_keystore(tmp_path) -> None:
    """The worker keystore dir (under XDG data home) is not among the STRICT
    binds, so its contents can't be opened from inside the sandbox."""
    keystore_dir = Path.home() / ".local" / "share" / "auspexai-worker"
    probe = keystore_dir / ".redteam-keystore-probe"
    keystore_dir.mkdir(parents=True, exist_ok=True)
    probe.write_text("ED25519-PRIVATE-KEY-MATERIAL")
    try:
        attack = (
            f"import pathlib\n"
            f"try:\n"
            f"    pathlib.Path({str(probe)!r}).read_text(); print('LEAK')\n"
            f"except OSError:\n"
            f"    print('BLOCKED')\n"
        )
        r = _run_in_strict(attack, tmp_path)
        assert "BLOCKED" in r.stdout and "LEAK" not in r.stdout, r
    finally:
        probe.unlink(missing_ok=True)


# ------------------------------------------------------------------- vector 3


def test_v3_cannot_open_an_exfiltration_socket(tmp_path) -> None:
    """--unshare-net leaves only a down loopback: an outbound connect fails."""
    attack = (
        "import socket\n"
        "try:\n"
        "    s = socket.socket(); s.settimeout(5)\n"
        "    s.connect(('1.1.1.1', 80)); print('LEAK')\n"
        "except OSError:\n"
        "    print('BLOCKED')\n"
    )
    r = _run_in_strict(attack, tmp_path)
    assert "BLOCKED" in r.stdout and "LEAK" not in r.stdout, r


# ------------------------------------------------------------------- vector 4


def test_v4_cannot_escape_via_syscall(tmp_path) -> None:
    """The seccomp denylist EPERMs the namespace/mount escape syscalls while
    ordinary work (fork) still runs."""
    attack = (
        "import ctypes, os, errno\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "blocked = []\n"
        "for name, num in (('unshare', 0x00020000),):  # CLONE_NEWNS\n"
        "    r = libc.unshare(num)\n"
        "    blocked.append(r == -1 and ctypes.get_errno() == errno.EPERM)\n"
        "ctypes.set_errno(0)\n"
        "# mount() must also be denied\n"
        "r = libc.mount(b'none', b'/', b'tmpfs', 0, None)\n"
        "blocked.append(r == -1 and ctypes.get_errno() in (errno.EPERM,))\n"
        "# ...but fork still works (denylist doesn't break the workload)\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os._exit(0)\n"
        "os.waitpid(pid, 0)\n"
        "print('BLOCKED' if all(blocked) else 'LEAK')\n"
    )
    r = _run_in_strict(attack, tmp_path)
    assert "BLOCKED" in r.stdout and "LEAK" not in r.stdout, r


# -------------------------------------------------------------- vector 5 (a-c)


def test_v5a_file_size_cap_kills_a_disk_bomb(tmp_path) -> None:
    """The rlimit floor (portable, no cgroup needed) caps a single-file write:
    SIGXFSZ kills the writer before it fills the disk."""
    limits = ResourceLimits(rlimit_fsize_bytes=65536, rlimit_nofile=None, rlimit_core_bytes=0)
    attack = (
        "import sys\n"
        "print('START', flush=True)\n"
        "f = open('/tmp/bomb', 'wb')\n"
        "f.write(b'a' * 10_000_000); f.flush()\n"
        "print('WROTE', flush=True)\n"
    )
    r = _run_in_strict(attack, tmp_path, rlimits=limits)
    assert "START" in r.stdout, r
    assert "WROTE" not in r.stdout, r  # the write never completed
    assert r.returncode != 0, r  # killed (SIGXFSZ) or errored out


@pytest.mark.skipif(
    res._delegated_parent() is None and not _REQUIRE_FULL,
    reason="no cgroup delegation in test env (run under systemd-run -p Delegate=yes)",
)
def test_v5b_memory_cap_ooms_an_alloc_bomb(tmp_path) -> None:
    """cgroup memory.max OOM-kills a memory bomb inside its own cgroup —
    neighbours untouched."""
    limits = ResourceLimits(memory_max_bytes=64 * 1024 * 1024, pids_max=None)
    attack = (
        "import sys\n"
        "sys.stdin.readline()\n"  # released after the daemon adopts us
        "b = bytearray(256 * 1024 * 1024)\n"
        "for i in range(0, len(b), 4096): b[i] = 1\n"
        "print('ALLOC_OK')\n"
    )
    r = _run_in_strict(attack, tmp_path, cgroup=limits, stdin="go\n")
    assert "ALLOC_OK" not in r.stdout, r
    assert r.returncode != 0, r  # OOM-killed


@pytest.mark.skipif(
    res._delegated_parent() is None and not _REQUIRE_FULL,
    reason="no cgroup delegation in test env (run under systemd-run -p Delegate=yes)",
)
def test_v5c_pids_cap_stops_a_fork_bomb(tmp_path) -> None:
    """cgroup pids.max caps the process count — a fork loop hits the ceiling
    instead of taking the host down."""
    limits = ResourceLimits(memory_max_bytes=None, pids_max=24)
    attack = (
        "import os, sys\n"
        "sys.stdin.readline()\n"
        "kids = 0\n"
        "try:\n"
        "    for _ in range(500):\n"
        "        pid = os.fork()\n"
        "        if pid == 0:\n"
        "            import time; time.sleep(10); os._exit(0)\n"
        "        kids += 1\n"
        "    print('FORKED_ALL', kids)\n"
        "except OSError:\n"
        "    print('CAPPED', kids)\n"
    )
    r = _run_in_strict(attack, tmp_path, cgroup=limits, stdin="go\n")
    assert "FORKED_ALL" not in r.stdout, r
    assert "CAPPED" in r.stdout, r
