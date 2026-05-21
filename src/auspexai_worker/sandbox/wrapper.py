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

import shutil
from dataclasses import dataclass
from enum import Enum


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


def check_bubblewrap_available(bwrap_path: str = "bwrap") -> bool:
    """Return True if the bubblewrap binary is on PATH."""
    return shutil.which(bwrap_path) is not None


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
    return [
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
        *env_args,
        "--",
        config.runner_bin,
    ]


def _env_argv(config: SandboxConfig) -> list[str]:
    """Emit `--setenv KEY VALUE` pairs for bwrap, or env var prefix for
    passthrough callers. For passthrough we expect the caller to set env
    on the subprocess directly; this is bwrap-specific."""
    if not config.use_bubblewrap:
        return []
    return [
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
