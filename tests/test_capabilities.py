"""Tests for capability detection."""

from __future__ import annotations

from pathlib import Path

from auspexai_worker.capabilities import (
    Capabilities,
    DeclaredCaps,
    collect,
    detect_gpus,
    detect_ram_total_gb,
)


class TestRamDetection:
    def test_parses_well_formed_meminfo(self, tmp_path: Path) -> None:
        path = tmp_path / "meminfo"
        path.write_text("MemTotal:     16777216 kB\nOther:        12345 kB\n")
        ram = detect_ram_total_gb(meminfo_path=path)
        assert ram is not None
        # 16777216 KiB = 16 GiB exactly.
        assert ram == 16.0

    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        ram = detect_ram_total_gb(meminfo_path=tmp_path / "nope")
        assert ram is None

    def test_returns_none_when_memtotal_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "meminfo"
        path.write_text("CommittedAs: 1234 kB\n")
        assert detect_ram_total_gb(meminfo_path=path) is None


class TestGpuDetection:
    def test_no_gpu_when_sysroot_empty(self, tmp_path: Path) -> None:
        # sysroot/dev/ does not exist → no device files → no GPU.
        gpus = detect_gpus(sysroot=tmp_path)
        assert gpus.nvidia == 0
        assert gpus.amd is False
        assert gpus.has_any() is False

    def test_nvidia_detected_via_device_file(self, tmp_path: Path) -> None:
        dev = tmp_path / "dev"
        dev.mkdir()
        (dev / "nvidia0").touch()
        (dev / "nvidia1").touch()
        (dev / "nvidiactl").touch()  # control file, not counted
        gpus = detect_gpus(sysroot=tmp_path)
        assert gpus.nvidia == 2
        assert gpus.amd is False

    def test_amd_detected_via_kfd(self, tmp_path: Path) -> None:
        dev = tmp_path / "dev"
        dev.mkdir()
        (dev / "kfd").touch()
        gpus = detect_gpus(sysroot=tmp_path)
        assert gpus.amd is True
        assert gpus.nvidia == 0


class TestCollect:
    def test_returns_capabilities_with_expected_shape(self, tmp_path: Path) -> None:
        # No meminfo path; CI may or may not have /proc/meminfo so we explicitly
        # pass tmp_path so the test is hermetic.
        meminfo = tmp_path / "meminfo"
        meminfo.write_text("MemTotal: 8388608 kB\n")
        caps = collect(meminfo_path=meminfo, sysroot=tmp_path)
        assert isinstance(caps, Capabilities)
        assert caps.os == caps.os.lower()
        assert caps.arch == caps.arch.lower()
        assert caps.ram_total_gb == 8.0
        assert caps.cpu_count is not None
        assert caps.gpus.has_any() is False

    def test_to_dict_includes_gpus(self, tmp_path: Path) -> None:
        caps = collect(sysroot=tmp_path)
        payload = caps.to_dict()
        assert "gpus" in payload
        assert payload["gpus"] == {"nvidia": 0, "amd": False}
        assert "os" in payload
        assert "python_version" in payload

    def test_to_dict_omits_unset_declared_caps(self, tmp_path: Path) -> None:
        caps = collect(sysroot=tmp_path, declared=DeclaredCaps())
        payload = caps.to_dict()
        assert "declared_caps" not in payload

    def test_to_dict_includes_set_declared_caps(self, tmp_path: Path) -> None:
        caps = collect(
            sysroot=tmp_path,
            declared=DeclaredCaps(max_ram_gb=12.5, max_cpu_cores=4),
        )
        payload = caps.to_dict()
        assert payload["declared_caps"] == {"max_ram_gb": 12.5, "max_cpu_cores": 4}
