"""§41(a) — resource caps for the STRICT sandbox (the "exhaust resources" gate).

Defense-in-depth, two composing layers so the cap degrades cleanly rather than
vanishing on hosts that lack one mechanism:

1. **POSIX rlimits** via a `prlimit(1)` argv-prefix. PORTABLE (every Linux ships
   util-linux) and RACE-FREE: prlimit sets the limits on itself then `exec`s the
   target, so *every* descendant of the bwrap tree inherits them at fork/exec
   with no window. This is the floor: a fork bomb hits `RLIMIT_NPROC`, a runaway
   file write hits `RLIMIT_FSIZE`, a core dump is suppressed. We deliberately do
   NOT set this via `preexec_fn` — the daemon is multi-threaded and a post-fork
   Python callback can deadlock on a lock held by another thread at fork time.

2. **cgroup v2** caps (memory.max / pids.max / cpu.max) when the worker runs
   under a *delegated* subtree (systemd `Delegate=yes`, task #22). This is the
   AUTHORITATIVE layer: `RLIMIT_AS` caps only *virtual* address space (footgunny
   against threads/mmap and useless as an RSS bound), whereas `memory.max` caps
   real memory and OOM-kills the offender; `pids.max` is an unambiguous process
   cap where `RLIMIT_NPROC` is per-uid and muddy across the user namespace.

Degradation: no `prlimit` → layer 1 skipped (logged); no delegated cgroup →
layer 2 skipped (logged), rlimit floor still applies. STRICT never *fails* for
want of these — the seccomp + namespace + bind layers are the hard gate; these
bound the blast radius of resource abuse.

Defaults are generous: the STRICT executor is a *light* inference harness (the
heavy model serving lives in the worker's own Ollama, OUTSIDE the sandbox, reached
over the broker socket), so a handful of processes and well under a GiB is the
norm. The caps stop a bomb without tripping legitimate work; operators tune them
via `[sandbox]` in worker.toml.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# The cgroup v2 unified hierarchy mount. (Hybrid/v1 hosts won't have the v2
# control files we need; detection below degrades to rlimits-only there.)
_CGROUP_ROOT = Path("/sys/fs/cgroup")
# Controllers we need delegated to apply the authoritative caps. cpu is optional
# (we only set cpu.max when a bandwidth cap is configured); memory + pids are the
# load-bearing ones for the "exhaust resources" vector.
_AUTHORITATIVE_CONTROLLERS = ("memory", "pids")
_OPTIONAL_CONTROLLERS = ("cpu",)
# cgroup names must be filesystem-safe; unit ids are coordinator-issued but we
# sanitize defensively (a path separator here would be an escape).
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")

_MiB = 1024 * 1024


@dataclass(frozen=True)
class ResourceLimits:
    """Resolved STRICT resource caps. `enabled=False` makes every method a
    no-op (the default for callers/tests that don't wire containment)."""

    enabled: bool = True
    # ---- cgroup v2 (authoritative; applied when a delegated subtree exists) ----
    # Real-memory (RSS) ceiling. The executor that exceeds it is OOM-killed by
    # the kernel inside its own cgroup — neighbors are unaffected.
    memory_max_bytes: int | None = 4096 * _MiB  # 4 GiB
    # Hard process/thread count — the unambiguous fork-bomb cap.
    pids_max: int | None = 512
    # CPU bandwidth as a percent of ONE core (100 = one full core, 200 = two).
    # None = no bandwidth cap: the wall-clock runner timeout already bounds a CPU
    # spin, and throttling can needlessly slow legitimate bursty work.
    cpu_max_percent: int | None = None
    # ---- rlimits (portable floor; inherited by the whole tree at exec) ----
    rlimit_fsize_bytes: int | None = 2048 * _MiB  # 2 GiB single-file write cap
    rlimit_nofile: int | None = 4096
    rlimit_core_bytes: int | None = 0  # no core dumps (don't spill memory to disk)
    rlimit_cpu_seconds: int | None = None  # None ⇒ wall-clock runner timeout governs
    # NPROC is OFF by default — and deliberately so. RLIMIT_NPROC is enforced
    # against the launching uid's *system-wide* process count, applied to the
    # prlimit→bwrap exec chain BEFORE the user namespace is entered; a low value
    # on a busy host would make bwrap itself fail to fork (EAGAIN) rather than
    # cap the executor. The safe, precise fork-bomb cap is cgroup `pids.max`.
    # Operators who know their host can still opt in.
    rlimit_nproc: int | None = None
    # Virtual address-space cap. OFF by default: it bounds VIRT not RSS and trips
    # threaded/mmap-heavy runtimes — cgroup memory.max is the real memory bound.
    # Set it as a coarse fallback for hosts with no cgroup delegation.
    rlimit_as_bytes: int | None = None

    # ------------------------------------------------------------------ rlimit
    def prlimit_prefix(self) -> list[str]:
        """The `prlimit … --` argv prefix to prepend to the runner argv, or `[]`
        when disabled / nothing to set / prlimit is absent."""
        if not self.enabled:
            return []
        prlimit = shutil.which("prlimit")
        if prlimit is None:
            logger.warning(
                "prlimit(1) not found on PATH — rlimit resource floor unavailable; "
                "cgroup caps (if delegated) still apply"
            )
            return []
        # prlimit size limits are in BYTES (it accepts K/M/G suffixes; we pass raw
        # bytes); cpu is seconds; nofile/nproc are counts. A single value sets both
        # the soft and hard limit.
        opts: list[str] = []
        if self.rlimit_as_bytes is not None:
            opts.append(f"--as={self.rlimit_as_bytes}")
        if self.rlimit_cpu_seconds is not None:
            opts.append(f"--cpu={self.rlimit_cpu_seconds}")
        if self.rlimit_fsize_bytes is not None:
            opts.append(f"--fsize={self.rlimit_fsize_bytes}")
        if self.rlimit_nofile is not None:
            opts.append(f"--nofile={self.rlimit_nofile}")
        if self.rlimit_nproc is not None:
            opts.append(f"--nproc={self.rlimit_nproc}")
        if self.rlimit_core_bytes is not None:
            opts.append(f"--core={self.rlimit_core_bytes}")
        if not opts:
            return []
        return [prlimit, *opts, "--"]

    # ------------------------------------------------------------------ cgroup
    def open_cgroup(self, unit_id: str) -> UnitCgroup | None:
        """Create + configure a per-unit cgroup under the delegated subtree, or
        return None when no delegated subtree is usable (degrade to rlimits)."""
        if not self.enabled:
            return None
        if self.memory_max_bytes is None and self.pids_max is None and self.cpu_max_percent is None:
            return None  # nothing to enforce via cgroup
        parent = _delegated_parent()
        if parent is None:
            return None
        name = "unit-" + _SAFE_NAME.sub("_", unit_id)[:200]
        path = parent / name
        try:
            path.mkdir(exist_ok=True)
        except OSError as exc:
            logger.warning(
                "cgroup: could not create %s (%s); rlimit floor still applies", path, exc
            )
            return None
        self._write_caps(path)
        return UnitCgroup(path)

    def _write_caps(self, path: Path) -> None:
        if self.memory_max_bytes is not None:
            _write_cgroup_file(path / "memory.max", str(self.memory_max_bytes))
            # Refuse to let a runaway dip into swap to dodge memory.max.
            _write_cgroup_file(path / "memory.swap.max", "0", required=False)
        if self.pids_max is not None:
            _write_cgroup_file(path / "pids.max", str(self.pids_max))
        if self.cpu_max_percent is not None:
            period = 100_000
            quota = max(1000, self.cpu_max_percent * period // 100)
            _write_cgroup_file(path / "cpu.max", f"{quota} {period}", required=False)


class UnitCgroup:
    """A live per-unit cgroup. `adopt` moves the runner tree in (before it does
    real work); `destroy` kills any stragglers and removes the cgroup."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def adopt(self, pid: int, *, settle_timeout: float = 2.0, poll_interval: float = 0.02) -> None:
        """Move `pid` and its whole subtree into the cgroup, WAITING for the
        runner subtree to materialize first.

        bwrap doesn't fork its `--unshare-pid` init child + exec the runner until
        ~100-200ms after `Popen` returns; adopting only the top pid at t=0 leaves
        the actual workload outside the cap (a fork/alloc bomb then escapes — the
        red-team caught exactly this). The runner blocks reading its work envelope
        from stdin, so the caller MUST adopt BEFORE writing that stdin: we poll
        until the subtree appears and two consecutive sweeps add nothing (settled),
        move every member, and only then does the workload get released to run —
        inside the cap. Bounded by `settle_timeout` so a bwrap that never spawns
        (its error surfaces on communicate) doesn't hang dispatch. Idempotent:
        each pid is written once."""
        procs_file = self._path / "cgroup.procs"
        moved: set[int] = set()
        deadline = time.monotonic() + settle_timeout
        stable = 0
        while True:
            added = 0
            for p in (pid, *_descendants(pid)):
                if p in moved:
                    continue
                try:
                    procs_file.write_text(str(p))
                    moved.add(p)
                    added += 1
                except OSError as exc:
                    # A pid that exited between enumeration and write is fine.
                    logger.debug("cgroup: could not adopt pid %d into %s: %s", p, self._path, exc)
            # Settled once the runner subtree is captured (≥1 descendant beyond
            # the top pid) and a clean sweep added nothing new.
            if len(moved) > 1:
                stable = stable + 1 if added == 0 else 0
                if stable >= 2:
                    return
            if time.monotonic() >= deadline:
                if len(moved) <= 1:
                    logger.warning(
                        "cgroup: runner subtree never appeared under pid %d within %.1fs — "
                        "only the top process is capped",
                        pid,
                        settle_timeout,
                    )
                return
            time.sleep(poll_interval)

    def destroy(self) -> None:
        """Kill any remaining members and remove the cgroup. Best-effort: a leak
        here is a stray empty cgroup dir, not a containment failure."""
        kill_file = self._path / "cgroup.kill"
        try:
            if kill_file.exists():
                kill_file.write_text("1")  # atomic tree-wide SIGKILL (kernel ≥5.14)
            else:
                _sigkill_members(self._path)
        except OSError as exc:
            logger.debug("cgroup: kill on %s failed: %s", self._path, exc)
        try:
            self._path.rmdir()
        except OSError as exc:
            # Non-empty (a member is still dying) or already gone — let systemd /
            # the next prune reclaim it; not worth blocking dispatch.
            logger.debug("cgroup: rmdir %s deferred: %s", self._path, exc)


# --------------------------------------------------------------------- helpers

_subtree_lock = threading.Lock()
# Tri-state cache: unset (compute), None (degraded — checked, unusable), Path.
_subtree_cache: Path | None = None
_subtree_computed = False


def _delegated_parent() -> Path | None:
    """The cgroup under which per-unit child cgroups may be created, or None.

    Under systemd `Delegate=yes` the worker owns its service cgroup subtree. To
    apply controllers to children, that cgroup must have no member processes of
    its own (the cgroup-v2 "no internal processes" rule), so we move our own
    process(es) into a `supervisor` leaf once and enable the controllers in the
    service cgroup's `subtree_control`. Result cached: prep runs once, on the
    first STRICT unit (dispatch is sequential, so no runner is competing)."""
    global _subtree_cache, _subtree_computed
    with _subtree_lock:
        if _subtree_computed:
            return _subtree_cache
        _subtree_computed = True
        _subtree_cache = _prepare_subtree()
        return _subtree_cache


def _prepare_subtree() -> Path | None:
    own = _own_cgroup_dir()
    if own is None or not own.is_dir():
        return None
    try:
        controllers = (own / "cgroup.controllers").read_text().split()
    except OSError:
        return None
    if not all(c in controllers for c in _AUTHORITATIVE_CONTROLLERS):
        logger.info(
            "cgroup: delegated controllers %s lack %s — resource caps fall back to "
            "rlimits only (enable systemd Delegate=yes for memory/pids caps)",
            controllers,
            _AUTHORITATIVE_CONTROLLERS,
        )
        return None
    supervisor = own / "supervisor"
    try:
        supervisor.mkdir(exist_ok=True)
        # Drain every process out of `own` into the leaf so `own` has no internal
        # processes and can enable controllers for its children.
        for pid in (own / "cgroup.procs").read_text().split():
            _write_cgroup_file(supervisor / "cgroup.procs", pid, required=False)
        enable = " ".join(
            f"+{c}"
            for c in (*_AUTHORITATIVE_CONTROLLERS, *_OPTIONAL_CONTROLLERS)
            if c in controllers
        )
        (own / "cgroup.subtree_control").write_text(enable)
    except OSError as exc:
        # No write permission (no delegation) or a busy hierarchy — degrade.
        logger.info("cgroup: subtree not delegated/usable (%s); rlimit floor only", exc)
        return None
    logger.info("cgroup: delegated subtree ready at %s (controllers: %s)", own, controllers)
    return own


def _own_cgroup_dir() -> Path | None:
    """This process's cgroup-v2 directory from /proc/self/cgroup (the `0::` line)."""
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                rel = line.split("::", 1)[1].lstrip("/")
                return _CGROUP_ROOT / rel
    except OSError:
        return None
    return None


def _descendants(pid: int) -> list[int]:
    """All live descendant pids of `pid`, via a /proc ppid scan (no dependency on
    CONFIG_PROC_CHILDREN). Order is unspecified; callers move them all."""
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            # comm may contain spaces/parens; ppid is the field after the final ')'.
            ppid = int(stat[stat.rfind(")") + 1 :].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(entry.name))
    out: list[int] = []
    stack = list(children.get(pid, []))
    while stack:
        cur = stack.pop()
        out.append(cur)
        stack.extend(children.get(cur, []))
    return out


def _sigkill_members(path: Path) -> None:
    try:
        pids = (path / "cgroup.procs").read_text().split()
    except OSError:
        return
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except (OSError, ValueError):
            continue


def _write_cgroup_file(path: Path, value: str, *, required: bool = True) -> None:
    try:
        path.write_text(value)
    except OSError as exc:
        log = logger.warning if required else logger.debug
        log("cgroup: write %s=%r failed: %s", path, value, exc)
