"""§41(a) — seccomp-bpf for the STRICT sandbox (the "escape via syscall" gate).

A *denylist* of the syscalls that enable sandbox escape or expand the kernel
attack surface, compiled to classic-BPF via libseccomp (the `pyseccomp` ctypes
binding — pure Python over the host `libseccomp.so`) and handed to bwrap as
`--seccomp <fd>`.

Why a denylist, not an allowlist: the executor runs *arbitrary tenant inference
code*, whose full syscall set is unknown and varies by kernel/glibc. An
allowlist would SIGSYS legitimate code; the denylist blocks the dangerous
syscalls and lets the rest through. Denied syscalls return EPERM — and crucially
seccomp filters BEFORE the kernel implementation runs, so a bug in that
implementation can't be triggered (which is the whole point of reducing the
escape surface).

The program is built once for the worker's native arch (a worker only runs on
its own arch) and cached; each spawn gets a fresh `memfd` holding it, so there
is no on-disk BPF blob and the fd is inherited only by the bwrap child.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading

logger = logging.getLogger(__name__)

# Linux clone(2) flag — creating a new USER namespace is the classic route to
# acquiring capabilities inside the sandbox, so it is arg-filtered below.
CLONE_NEWUSER = 0x10000000
_EPERM = 1
_ENOSYS = 38

# Denied in STRICT, grouped by escape vector. A syscall absent on the worker's
# arch/kernel is skipped (best-effort denylist).
DENIED_SYSCALLS: tuple[str, ...] = (
    # inspect / inject other processes
    "ptrace",
    "process_vm_readv",
    "process_vm_writev",
    "pidfd_getfd",
    "pidfd_open",
    "pidfd_send_signal",
    # mount / filesystem-namespace manipulation
    "mount",
    "umount2",
    "pivot_root",
    "move_mount",
    "fsopen",
    "fsconfig",
    "fsmount",
    "open_tree",
    "mount_setattr",
    # join / create namespaces (escape the pid/net/mount/user jail)
    "unshare",
    "setns",
    # kernel modules + boot
    "init_module",
    "finit_module",
    "delete_module",
    "kexec_load",
    "kexec_file_load",
    # kernel attack surface / exploit primitives
    "bpf",
    "perf_event_open",
    "userfaultfd",
    "io_uring_setup",
    "io_uring_enter",
    "io_uring_register",
    # kernel keyring
    "add_key",
    "keyctl",
    "request_key",
    # misc privileged / host-global state
    "swapon",
    "swapoff",
    "reboot",
    "acct",
    "nfsservctl",
    "quotactl",
    "ioperm",
    "iopl",
    "sethostname",
    "setdomainname",
    "settimeofday",
    "clock_settime",
    "clock_adjtime",
    "adjtimex",
)

_lock = threading.Lock()
_cached_bpf: bytes | None = None


class SeccompUnavailableError(Exception):
    """libseccomp / pyseccomp could not build the filter — STRICT fails closed."""


def _build_bpf() -> bytes:
    try:
        import pyseccomp as seccomp
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise SeccompUnavailableError(
            "pyseccomp is not importable (is libseccomp.so installed?). STRICT "
            "sandbox requires it — install libseccomp2 + the pyseccomp package."
        ) from exc

    flt = seccomp.SyscallFilter(seccomp.ALLOW)
    eperm = seccomp.ERRNO(_EPERM)
    for name in DENIED_SYSCALLS:
        try:
            flt.add_rule(eperm, name)
        except (ValueError, RuntimeError):
            # Unknown on this arch/kernel — fine, nothing to deny.
            continue
    # Block creating a nested USER namespace via clone() while leaving ordinary
    # fork/thread clones working (arg-filter on flags). clone3() takes a struct
    # pointer (un-filterable by seccomp), so force ENOSYS — glibc then falls
    # back to clone(), which IS arg-filtered.
    try:
        flt.add_rule(
            eperm, "clone", seccomp.Arg(0, seccomp.MASKED_EQ, CLONE_NEWUSER, CLONE_NEWUSER)
        )
        flt.add_rule(seccomp.ERRNO(_ENOSYS), "clone3")
    except (ValueError, RuntimeError, AttributeError):  # pragma: no cover
        logger.warning(
            "seccomp: clone-newuser arg-filter unavailable; namespace-via-clone not blocked"
        )

    with tempfile.TemporaryFile() as fh:
        flt.export_bpf(fh)
        fh.seek(0)
        return fh.read()


def seccomp_bpf() -> bytes:
    """The cached denylist BPF program for this worker's native arch."""
    global _cached_bpf
    with _lock:
        if _cached_bpf is None:
            _cached_bpf = _build_bpf()
        return _cached_bpf


def _anonymous_fd() -> int:
    """An anonymous, writable fd to hold the BPF program. Prefer `memfd_create`
    (never touches disk), but fall back to an immediately-unlinked tempfile where
    it's absent: `os.memfd_create` is missing on some CPython builds — notably
    uv's standalone interpreters (compiled against an old glibc). The worker's own
    installs use the system Python, which has it, but CI and portable installs may
    not, and STRICT must not hard-fail there. The unlinked tempfile leaves no
    lingering on-disk blob, and bwrap reads it through the inherited fd the same."""
    memfd = getattr(os, "memfd_create", None)
    if memfd is not None:
        return memfd("auspexai-seccomp", os.MFD_CLOEXEC)
    fd, path = tempfile.mkstemp(prefix="auspexai-seccomp-")
    os.unlink(path)
    return fd


def open_seccomp_fd() -> int:
    """A fresh fd (read position 0) holding the BPF program, for bwrap
    `--seccomp <fd>`. The caller passes it via `Popen(pass_fds=[fd])` and closes
    it after spawn. Raises SeccompUnavailableError if the filter can't be built."""
    bpf = seccomp_bpf()
    fd = _anonymous_fd()
    try:
        os.write(fd, bpf)
        os.lseek(fd, 0, os.SEEK_SET)
    except OSError:
        os.close(fd)
        raise
    return fd
