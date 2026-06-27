"""Construct the argv that launches the runner subprocess.

Two modes:
- `passthrough` — direct exec of `auspexai-worker-runner`. No isolation.
  Used by tests + dev hosts without bubblewrap installed. CI uses this.
- `bubblewrap` — wrap the runner in `bwrap` with §5.17 bind-mounts. PERMISSIVE
  (default) shares the host filesystem (--dev-bind / /). STRICT (§41(a))
  replaces that with narrow read-only system binds + the worker venv + a
  tmpfs, and adds full namespace isolation (net/pid/ipc/uts) — so a
  malicious/buggy executor can't reach the keystore, $HOME, prior receipts, or
  cross-tenant data. Selected by the `policy` enum from `[sandbox] policy`,
  not daemon code. seccomp + cgroup caps are the next hardening increments.

Environment variables passed through to the runner regardless of mode:
  AUSPEXAI_UNIT_ID
  AUSPEXAI_MANIFEST_SHA256
  AUSPEXAI_OUTPUT_PATH
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class SandboxPolicy(Enum):
    """PERMISSIVE shares the host fs (`--dev-bind / /`) — fine only under the
    Phase-1 trust model (vetted tenants, signed packages, operator-owned hosts).
    STRICT (§41(a)) replaces that with narrow read-only binds + full namespace
    isolation, so a malicious/buggy executor can't reach the identity keystore,
    `$HOME`, prior receipts, other tenants' staged packages, or the model store —
    only a cooperating kernel. seccomp + cgroup caps are the next hardening
    layers (§41(a) follow-on increments)."""

    PERMISSIVE = "permissive"
    STRICT = "strict"


class SandboxNotAvailableError(Exception):
    """Raised when `use_bubblewrap=True` but `bwrap` isn't on PATH."""


@dataclass(frozen=True)
class SandboxConfig:
    """Resolved sandbox configuration for a single runner invocation.

    `workspace_path`: per-unit workspace dir. Bind-mounted into the runner
        (writable) so the runner can write output.json there.
    `output_path`: path inside the workspace where the runner writes the
        result body. Surfaced to the runner via $AUSPEXAI_OUTPUT_PATH.
    """

    use_bubblewrap: bool
    runner_bin: str
    workspace_path: str
    output_path: str
    unit_id: str
    manifest_sha256: str
    policy: SandboxPolicy = SandboxPolicy.PERMISSIVE
    bwrap_path: str = "bwrap"
    # §9 #37 real-executor dispatch. When set (the daemon resolved + consented
    # to a tenant package), the runner runs the tenant executor instead of the
    # synthetic echo. `executor_package_dir` / `models_dir` are bind-mounted
    # read-only (Phase 2 STRICT); reachable via --dev-bind / / under Phase 1.
    executor_command: list[str] | None = None
    executor_package_dir: str | None = None
    models_dir: str | None = None
    executor_timeout_seconds: float | None = None
    # W-S (§9 #43): the per-unit inference broker socket. A unix socket is a
    # filesystem object, so it crosses --unshare-net — the executor reaches
    # the worker-served model with the external network still cut. The socket
    # normally lives in the workspace (already bound); the explicit bind of
    # its parent dir keeps this working when Phase-2 STRICT drops --dev-bind.
    inference_socket: str | None = None
    inference_model: str | None = None


def check_bubblewrap_available(bwrap_path: str = "bwrap") -> bool:
    """Return True if the bubblewrap binary is on PATH."""
    return shutil.which(bwrap_path) is not None


def resolve_runner_bin(runner_bin: str) -> str:
    """Resolve the runner command to an absolute path for `bwrap` exec.

    bwrap's `execvp` resolves a bare command name against the SANDBOX's PATH,
    which does NOT include the venv bin dir — the daemon runs as a systemd
    `--user` service whose PATH is `/usr/bin:/bin:...`, not
    `/opt/auspexai-worker/bin`. So a bare `auspexai-worker-runner` is present
    under the permissive `--dev-bind / /` but not *found* (execvp: No such file
    or directory). Resolve it to an absolute path: prefer the console script
    colocated with the running interpreter (the venv bin dir), then PATH.
    Absolute paths and unresolvable names pass through unchanged.
    """
    if os.path.isabs(runner_bin):
        return runner_bin
    candidate = Path(sys.executable).parent / runner_bin
    if candidate.exists():
        return str(candidate)
    return shutil.which(runner_bin) or runner_bin


@dataclass(frozen=True)
class BubblewrapProbeResult:
    """Outcome of `probe_bubblewrap`. `ok=True` means bwrap can actually
    construct a user-namespace and run a no-op subprocess on this host.
    `ok=False` includes a human-readable `reason` suitable for surfacing
    to operators."""

    ok: bool
    reason: str | None = None


def probe_bubblewrap(bwrap_path: str = "bwrap") -> BubblewrapProbeResult:
    """Probe whether bwrap actually works on this host.

    Runs a minimal `bwrap --dev-bind / / -- /bin/true` and reports
    whether the namespace setup succeeded. This catches the common
    Ubuntu 24.04 failure mode where the binary is installed but
    AppArmor restricts unprivileged user namespaces — bwrap exits
    non-zero with `bwrap: setting up uid map: Permission denied`.

    The daemon calls this at startup when `use_bubblewrap=true` so
    operators see a clear, actionable error instead of a cryptic per-
    unit failure that only surfaces after assignments start arriving.
    """
    if not check_bubblewrap_available(bwrap_path):
        return BubblewrapProbeResult(
            ok=False,
            reason=f"bwrap binary {bwrap_path!r} not found on PATH",
        )
    try:
        result = subprocess.run(
            [bwrap_path, "--dev-bind", "/", "/", "--", "/bin/true"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return BubblewrapProbeResult(ok=False, reason=f"bwrap probe raised: {exc}")
    if result.returncode == 0:
        return BubblewrapProbeResult(ok=True)
    stderr_tail = result.stderr.decode("utf-8", errors="replace").strip()[-400:]
    return BubblewrapProbeResult(
        ok=False,
        reason=f"bwrap probe exit={result.returncode}: {stderr_tail}",
    )


SANDBOX_EXEC_BIN = "/usr/bin/sandbox-exec"


@dataclass(frozen=True)
class SeatbeltProbeResult:
    """Outcome of `probe_seatbelt` — whether macOS `sandbox-exec` can launch a command
    on this host. Mirrors BubblewrapProbeResult for the daemon's STRICT startup gate."""

    ok: bool
    reason: str | None = None


def probe_seatbelt() -> SeatbeltProbeResult:
    """Probe whether macOS `sandbox-exec` (Seatbelt) can launch a command — the mechanism
    behind STRICT on macOS. Runs a permissive profile around `/usr/bin/true`; the per-unit
    STRICT profile is built separately in `_seatbelt_profile`."""
    if not Path(SANDBOX_EXEC_BIN).exists() and shutil.which("sandbox-exec") is None:
        return SeatbeltProbeResult(ok=False, reason=f"{SANDBOX_EXEC_BIN} not found")
    try:
        result = subprocess.run(
            [SANDBOX_EXEC_BIN, "-p", "(version 1)(allow default)", "/usr/bin/true"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return SeatbeltProbeResult(ok=False, reason=f"sandbox-exec probe raised: {exc}")
    if result.returncode == 0:
        return SeatbeltProbeResult(ok=True)
    stderr_tail = result.stderr.decode("utf-8", errors="replace").strip()[-400:]
    return SeatbeltProbeResult(
        ok=False, reason=f"sandbox-exec probe exit={result.returncode}: {stderr_tail}"
    )


def build_argv(config: SandboxConfig, *, seccomp_fd: int | None = None) -> list[str]:
    """Construct the argv used to spawn the runner.

    `seccomp_fd` (§41(a), STRICT only) is an open fd holding the BPF program
    (see `sandbox.seccomp.open_seccomp_fd`); when given, `--seccomp <fd>` is
    added and the caller must pass the fd via `Popen(pass_fds=[fd])`.

    Raises:
        SandboxNotAvailableError: when `use_bubblewrap=True` but bwrap is
            not on PATH. Caller must either install bwrap or switch to
            passthrough mode.
    """
    env_args = _env_argv(config)
    # macOS STRICT — Seatbelt (sandbox-exec). On macOS use_bubblewrap is always False, so
    # the daemon sets env on the subprocess (passthrough-style); Seatbelt only wraps the
    # argv with the per-unit profile (no bwrap, no seccomp — Seatbelt mediates directly).
    if sys.platform == "darwin" and config.policy is SandboxPolicy.STRICT:
        return _seatbelt_argv(config)
    if not config.use_bubblewrap:
        return [config.runner_bin]
    if not check_bubblewrap_available(config.bwrap_path):
        raise SandboxNotAvailableError(
            f"bubblewrap binary {config.bwrap_path!r} not found on PATH; "
            "install `bubblewrap` or set `[sandbox] use_bubblewrap = false` "
            "(NOT recommended for production)"
        )
    resolved_runner = resolve_runner_bin(config.runner_bin)
    argv = [
        config.bwrap_path,
        # Die when the parent (worker daemon) exits — don't leak runner
        # subprocesses on daemon crash.
        "--die-with-parent",
        # New process session — signals to the daemon don't propagate to the
        # runner unintentionally; our explicit SIGTERM-via-PID-file path stays
        # the only abort channel.
        "--new-session",
    ]
    if config.policy is SandboxPolicy.STRICT:
        argv += _strict_fs_argv(config, resolved_runner)
        # §41(a): full namespace isolation — no network, private pid/ipc/uts,
        # and a private cgroup view (the executor can't see the host cgroup
        # layout). With --unshare-pid the executor is pid 1 in a private
        # namespace and cannot see or signal host processes.
        argv += [
            "--unshare-net",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup-try",
        ]
        # Drop every capability (belt-and-suspenders over the unprivileged
        # user-ns) so the executor holds no privilege even nominally.
        argv += ["--cap-drop", "ALL"]
        # The "escape via syscall" gate: a seccomp denylist of the escape /
        # kernel-attack syscalls. STRICT without it is refused upstream
        # (dispatch fails closed), so seccomp_fd is normally present here.
        # AUD-9 (A9 audit): fail CLOSED at the builder too — never emit a STRICT
        # sandbox without the seccomp gate (the invariant previously lived only in
        # the dispatch caller; this is defense-in-depth against a future caller).
        if seccomp_fd is None:
            raise SandboxNotAvailableError(
                "STRICT policy requires a seccomp fd; refusing to build a "
                "seccomp-less STRICT sandbox"
            )
        argv += ["--seccomp", str(seccomp_fd)]
    else:
        argv += _permissive_fs_argv(config)
        # §5.17: real tenant code gets NO network (the inference broker socket is
        # a filesystem object, so it survives --unshare-net). The synthetic echo
        # keeps host net — a pure no-op for the existing path.
        if config.executor_command is not None:
            argv.append("--unshare-net")
    # Real-executor resource binds — the tenant package + the served-model store
    # + the per-unit broker socket. Under PERMISSIVE these are redundant
    # (--dev-bind / / already exposes them) but explicit; under STRICT they are
    # the ONLY paths the executor can reach besides the system dirs + workspace.
    if config.executor_command is not None:
        if config.executor_package_dir:
            argv += ["--ro-bind", config.executor_package_dir, config.executor_package_dir]
        if config.models_dir:
            argv += ["--ro-bind-try", config.models_dir, config.models_dir]
        if config.inference_socket:
            sock_dir = str(Path(config.inference_socket).parent)
            argv += ["--bind", sock_dir, sock_dir]
    # Absolute path so bwrap's execvp finds the runner (the sandbox PATH won't
    # include the venv bin dir). Passthrough mode (above) leaves resolution to
    # the daemon's own PATH, unchanged.
    argv += [*env_args, "--", resolved_runner]
    return argv


def _permissive_fs_argv(config: SandboxConfig) -> list[str]:
    """Phase-1 PERMISSIVE filesystem view: the whole host fs, read-write. A
    separate mount namespace (so binds/tmpfs work) but no real containment —
    acceptable only under the Phase-1 trust model."""
    return [
        "--dev-bind",
        "/",
        "/",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--bind",
        config.workspace_path,
        config.workspace_path,
    ]


def _strict_fs_argv(config: SandboxConfig, resolved_runner: str) -> list[str]:
    """§41(a) STRICT filesystem view: NO `--dev-bind / /`. Narrow read-only
    system dirs + the worker venv (so the runner and its python resolve), a
    private tmpfs `/tmp` (with HOME pointed at it so library caches land on the
    ephemeral mount), a fresh `/proc` + minimal `/dev`, and the per-unit
    workspace as the SOLE host-writable path. The identity keystore, `$HOME`,
    prior receipts, and other tenants' staged packages live OUTSIDE these binds,
    so the executor cannot read them."""
    # <venv>/bin/<runner> -> <venv>; binds the python + auspexai_worker + deps.
    venv_root = str(Path(resolved_runner).resolve().parents[1])
    return [
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind-try",
        "/bin",
        "/bin",
        "--ro-bind-try",
        "/sbin",
        "/sbin",
        "--ro-bind-try",
        "/lib",
        "/lib",
        "--ro-bind-try",
        "/lib64",
        "/lib64",
        "--ro-bind-try",
        "/etc",
        "/etc",
        "--ro-bind",
        venv_root,
        venv_root,
        "--tmpfs",
        "/tmp",
        "--setenv",
        "HOME",
        "/tmp",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--bind",
        config.workspace_path,
        config.workspace_path,
    ]


def _seatbelt_quote(path: str) -> str:
    """Quote a path as a Seatbelt string literal (double-quoted; backslash-escaped)."""
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _seatbelt_profile(config: SandboxConfig) -> str:
    """A Seatbelt (sandbox-exec) profile for macOS STRICT.

    The integrity-critical controls match the Linux strict sandbox: NO external network
    (only the inference broker unix socket survives), writes confined to the per-unit
    workspace, and the worker's signing key + common host secrets kept UNREADABLE — so
    tenant code can't forge receipts, tamper the host, or exfiltrate over the network.

    Reads are otherwise BROAD. A tight read-allowlist proved brittle across macOS python
    layouts (framework dylibs, the dyld shared cache, Homebrew cellars differ per host
    and silently SIGKILL the runner when one is missed), so v1 trades full read-isolation
    for reliability while still protecting the integrity-critical assets. Tightening the
    read surface back to an allowlist is a tracked refinement."""
    # Seatbelt matches on REAL paths: macOS /var, /tmp, /etc are symlinks into /private,
    # so a rule written against the symlinked path never matches the resolved access and
    # the op is denied. realpath() every path that goes into a path-specific rule.
    home = os.path.realpath(os.path.expanduser("~"))
    # The signing key lives under the worker state dir — it MUST stay unreadable, or
    # tenant code could read it and forge receipts. Plus the obvious host secret stores.
    state_dir = os.path.realpath(
        os.environ.get("AUSPEXAI_WORKER_STATE_DIR")
        or os.path.join(home, ".local", "state", "auspexai-worker")
    )
    protected = [
        state_dir,
        os.path.join(home, ".ssh"),
        os.path.join(home, ".aws"),
        os.path.join(home, ".gnupg"),
        os.path.join(home, ".config", "gcloud"),
        os.path.join(home, "Library", "Keychains"),
    ]
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-exec*)",
        "(allow process-fork)",
        "(allow mach-lookup)",
        # python/openssl probe these for hw crypto accel + page size; non-fatal if denied
        # (software fallback) but allow them so the runner runs clean and fast.
        "(allow sysctl-read)",
        "(allow ipc-posix-shm-read-data)",
        "(allow file-read*)",
    ]
    # Later, more-specific rules win in Seatbelt — so these denials override the broad
    # read above for exactly the secret paths.
    lines += [f"(deny file-read* (subpath {_seatbelt_quote(p)}))" for p in protected]
    workspace = os.path.realpath(config.workspace_path)
    lines.append(
        f'(allow file-write* (subpath {_seatbelt_quote(workspace)}) (literal "/dev/null"))'
    )
    lines.append("(deny network*)")
    if config.inference_socket:
        socket = os.path.realpath(config.inference_socket)
        lines.append(f"(allow network-outbound (literal {_seatbelt_quote(socket)}))")
    return "\n".join(lines)


def _seatbelt_argv(config: SandboxConfig) -> list[str]:
    """macOS STRICT argv: `sandbox-exec -p <profile> <runner>`. Env is set on the
    subprocess by the daemon (passthrough-style on macOS), so it isn't in the argv."""
    runner = resolve_runner_bin(config.runner_bin)
    return [SANDBOX_EXEC_BIN, "-p", _seatbelt_profile(config), runner]


def _env_argv(config: SandboxConfig) -> list[str]:
    """Emit `--setenv KEY VALUE` pairs for bwrap, or env var prefix for
    passthrough callers. For passthrough we expect the caller to set env
    on the subprocess directly; this is bwrap-specific."""
    if not config.use_bubblewrap:
        return []
    args = [
        "--setenv",
        "AUSPEXAI_UNIT_ID",
        config.unit_id,
        "--setenv",
        "AUSPEXAI_MANIFEST_SHA256",
        config.manifest_sha256,
        "--setenv",
        "AUSPEXAI_OUTPUT_PATH",
        config.output_path,
    ]
    if config.executor_command is not None:
        args += ["--setenv", "AUSPEXAI_EXECUTOR_COMMAND", json.dumps(config.executor_command)]
        if config.executor_package_dir:
            args += ["--setenv", "AUSPEXAI_EXECUTOR_DIR", config.executor_package_dir]
        if config.models_dir:
            args += ["--setenv", "AUSPEXAI_MODELS_DIR", config.models_dir]
        if config.executor_timeout_seconds is not None:
            args += [
                "--setenv",
                "AUSPEXAI_EXECUTOR_TIMEOUT",
                str(int(config.executor_timeout_seconds)),
            ]
        if config.inference_socket:
            args += ["--setenv", "AUSPEXAI_INFERENCE_SOCKET", config.inference_socket]
        if config.inference_model:
            args += ["--setenv", "AUSPEXAI_INFERENCE_MODEL", config.inference_model]
    return args
