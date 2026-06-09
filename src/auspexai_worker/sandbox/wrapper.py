"""Construct the argv that launches the runner subprocess.

Two modes:
- `passthrough` — direct exec of `auspexai-worker-runner`. No isolation.
  Used by tests + dev hosts without bubblewrap installed. CI uses this.
- `bubblewrap` — wrap the runner in `bwrap` with §5.17 bind-mounts. Phase 1
  default policy is permissive (--dev-bind / / shares the host filesystem;
  no network namespace; no capability filtering). Phase 2 will tighten
  by changing the `policy` enum value, not the daemon code.

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
    """Phase 1 ships PERMISSIVE; Phase 2 will add STRICT (no-net, narrow
    binds, resource caps enforced) without changing this enum's interface."""

    PERMISSIVE = "permissive"
    # STRICT = "strict"  # Phase 2


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


def build_argv(config: SandboxConfig) -> list[str]:
    """Construct the argv used to spawn the runner.

    Raises:
        SandboxNotAvailableError: when `use_bubblewrap=True` but bwrap is
            not on PATH. Caller must either install bwrap or switch to
            passthrough mode.
    """
    env_args = _env_argv(config)
    if not config.use_bubblewrap:
        return [config.runner_bin]
    if not check_bubblewrap_available(config.bwrap_path):
        raise SandboxNotAvailableError(
            f"bubblewrap binary {config.bwrap_path!r} not found on PATH; "
            "install `bubblewrap` or set `[sandbox] use_bubblewrap = false` "
            "(NOT recommended for production)"
        )
    argv = [
        config.bwrap_path,
        # Die when the parent (worker daemon) exits — don't leak runner
        # subprocesses on daemon crash.
        "--die-with-parent",
        # New process session — signals to the daemon don't propagate to
        # the runner unintentionally, and our explicit SIGTERM-via-PID-file
        # path stays the only abort channel.
        "--new-session",
        # Permissive Phase 1 host-fs view: --dev-bind / / shares everything
        # read-write, but is still a separate mount namespace so binds and
        # tmpfs work. Phase 2 will replace this with narrow --ro-bind /usr
        # /etc /lib... and a tmpfs for /tmp. Same call site; config flag
        # controls.
        "--dev-bind",
        "/",
        "/",
        # /proc must be mounted in the new pid namespace (when STRICT
        # adds --unshare-pid). Phase 1 permissive doesn't unshare pid, so
        # this is a no-op safety net.
        "--proc",
        "/proc",
        # /dev — needed for /dev/null, /dev/random, GPU device files
        # (when present). Phase 2 will narrow to specific device files.
        "--dev",
        "/dev",
        # Ensure the workspace dir is visible to the runner (already
        # accessible via --dev-bind but make it explicit so STRICT keeps
        # working when --dev-bind is dropped).
        "--bind",
        config.workspace_path,
        config.workspace_path,
    ]
    if config.executor_command is not None:
        # §5.17: the inference subprocess gets NO network. Real tenant code
        # runs in a network namespace with no interface. (Synthetic-only
        # dispatch keeps host net so it stays a pure no-op change for the
        # existing echo path.) Phase 2 STRICT replaces --dev-bind / / with
        # narrow --ro-bind of the package + models dirs explicitly bound here.
        argv.append("--unshare-net")
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
    argv += [*env_args, "--", resolve_runner_bin(config.runner_bin)]
    return argv


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
