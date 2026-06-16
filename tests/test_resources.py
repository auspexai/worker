"""§41(a) resource caps — rlimit prefix construction + real prlimit enforcement,
cgroup cap-writing against a faked delegated subtree, and descendant enumeration.

The full "exhaust resources fails under STRICT" proof (real bwrap + a delegated
cgroup OOM-killing an alloc bomb) lives in the red-team harness; cgroup
enforcement needs systemd Delegate=yes, which CI hosts don't have. Here we prove
the pieces: prlimit really truncates a write, and the cgroup writer lays down the
right control files.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from auspexai_worker.sandbox import ResourceLimits, UnitCgroup
from auspexai_worker.sandbox import resources as res

_MiB = 1024 * 1024


class TestPrlimitPrefix:
    def test_disabled_is_noop(self) -> None:
        assert ResourceLimits(enabled=False).prlimit_prefix() == []

    def test_defaults_set_the_safe_floor(self) -> None:
        if shutil.which("prlimit") is None:
            pytest.skip("prlimit not installed")
        prefix = ResourceLimits().prlimit_prefix()
        assert prefix[0].endswith("prlimit")
        assert prefix[-1] == "--"
        joined = " ".join(prefix)
        # Safe-by-default floor: file-size, fd, core. NOT nproc/as (footguns).
        assert f"--fsize={2048 * _MiB}" in joined
        assert "--nofile=4096" in joined
        assert "--core=0" in joined
        assert "--nproc=" not in joined
        assert "--as=" not in joined

    def test_opt_in_limits_appear(self) -> None:
        if shutil.which("prlimit") is None:
            pytest.skip("prlimit not installed")
        joined = " ".join(
            ResourceLimits(
                rlimit_cpu_seconds=30, rlimit_nproc=64, rlimit_as_bytes=8 * _MiB
            ).prlimit_prefix()
        )
        assert "--cpu=30" in joined
        assert "--nproc=64" in joined
        assert f"--as={8 * _MiB}" in joined

    def test_no_rlimits_means_no_prefix(self) -> None:
        # Only cgroup caps set, every rlimit None → nothing for prlimit to do.
        limits = ResourceLimits(
            rlimit_fsize_bytes=None,
            rlimit_nofile=None,
            rlimit_core_bytes=None,
            rlimit_cpu_seconds=None,
            rlimit_nproc=None,
            rlimit_as_bytes=None,
        )
        assert limits.prlimit_prefix() == []


class TestPrlimitEnforces:
    """Host-independent proof the rlimit layer actually bites: a file-size cap
    truncates/kills an over-limit write."""

    def test_fsize_cap_blocks_a_large_write(self, tmp_path) -> None:
        if shutil.which("prlimit") is None:
            pytest.skip("prlimit not installed")
        target = tmp_path / "big.bin"
        limits = ResourceLimits(
            rlimit_fsize_bytes=4096,
            rlimit_nofile=None,
            rlimit_core_bytes=0,
        )
        prog = f"f=open({str(target)!r},'wb'); f.write(b'a'*1_000_000); f.flush()"
        argv = [*limits.prlimit_prefix(), sys.executable, "-c", prog]
        proc = subprocess.run(argv, capture_output=True)
        # SIGXFSZ kills the writer (or it errors) — either way non-zero, and the
        # file never reaches the attempted 1 MB.
        assert proc.returncode != 0
        assert (not target.exists()) or target.stat().st_size <= 4096


class TestCgroupWriter:
    def test_disabled_opens_nothing(self) -> None:
        assert ResourceLimits(enabled=False).open_cgroup("u-1") is None

    def test_no_caps_opens_nothing(self) -> None:
        limits = ResourceLimits(memory_max_bytes=None, pids_max=None, cpu_max_percent=None)
        assert limits.open_cgroup("u-1") is None

    def test_degrades_when_no_delegated_subtree(self, monkeypatch) -> None:
        monkeypatch.setattr(res, "_delegated_parent", lambda: None)
        assert ResourceLimits().open_cgroup("u-1") is None

    def test_writes_caps_into_faked_subtree(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(res, "_delegated_parent", lambda: tmp_path)
        limits = ResourceLimits(memory_max_bytes=512 * _MiB, pids_max=128, cpu_max_percent=150)
        cg = limits.open_cgroup("u-7")
        assert isinstance(cg, UnitCgroup)
        d = tmp_path / "unit-u-7"
        assert (d / "memory.max").read_text() == str(512 * _MiB)
        assert (d / "memory.swap.max").read_text() == "0"
        assert (d / "pids.max").read_text() == "128"
        # 150% of one core at a 100ms period → 150ms quota.
        assert (d / "cpu.max").read_text() == "150000 100000"

    def test_unit_id_is_sanitized_no_path_escape(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(res, "_delegated_parent", lambda: tmp_path)
        cg = ResourceLimits().open_cgroup("../../etc/evil id")
        assert cg is not None
        # The cgroup dir stays a direct child of the parent — no traversal.
        assert cg.path.parent == tmp_path
        assert "/" not in cg.path.name[len("unit-") :].replace("unit-", "")
        assert cg.path.name.startswith("unit-")


class TestUnitCgroupLifecycle:
    def test_adopt_writes_pids_to_cgroup_procs(self, tmp_path) -> None:
        d = tmp_path / "unit-x"
        d.mkdir()
        cg = UnitCgroup(d)
        # A real, harmless child so descendant enumeration has something to find.
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            # Lone child has no descendants → short settle (the poll-until-subtree
            # logic is exercised end-to-end in the red-team under real bwrap).
            cg.adopt(child.pid, settle_timeout=0.1)
            written = (d / "cgroup.procs").read_text()
            assert str(child.pid) in written
        finally:
            child.kill()
            child.wait()

    def test_destroy_prefers_cgroup_kill(self, tmp_path) -> None:
        d = tmp_path / "unit-k"
        d.mkdir()
        (d / "cgroup.kill").write_text("")
        UnitCgroup(d).destroy()
        assert (d / "cgroup.kill").read_text() == "1"

    def test_destroy_is_safe_without_cgroup_kill(self, tmp_path) -> None:
        d = tmp_path / "unit-s"
        d.mkdir()
        # No cgroup.kill, no cgroup.procs → falls back, finds nothing, removes dir.
        UnitCgroup(d).destroy()
        assert not d.exists()


class TestDescendants:
    def test_finds_a_spawned_child(self) -> None:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            assert child.pid in res._descendants(os.getpid())
        finally:
            child.kill()
            child.wait()
