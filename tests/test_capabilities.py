"""Tests for capability detection."""

from __future__ import annotations

from pathlib import Path

from auspexai_worker.capabilities import (
    Capabilities,
    DeclaredCaps,
    GpuDeclaration,
    GpuObservation,
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
        assert ram == 16.0

    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        assert detect_ram_total_gb(meminfo_path=tmp_path / "nope") is None

    def test_returns_none_when_memtotal_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "meminfo"
        path.write_text("CommittedAs: 1234 kB\n")
        assert detect_ram_total_gb(meminfo_path=path) is None


class TestGpuObservation:
    def test_no_gpu_when_sysroot_empty(self, tmp_path: Path) -> None:
        observed = detect_gpus(sysroot=tmp_path)
        assert observed.nvidia == 0
        assert observed.amd is False
        assert observed.has_any() is False

    def test_nvidia_counted_when_device_files_responsive(self, tmp_path: Path) -> None:
        dev = tmp_path / "dev"
        dev.mkdir()
        (dev / "nvidia0").touch()
        (dev / "nvidia1").touch()
        # Regular files in tmp_path open() just fine — exercises the
        # validated-probe happy path.
        observed = detect_gpus(sysroot=tmp_path)
        assert observed.nvidia == 2

    def test_nvidiactl_not_counted(self, tmp_path: Path) -> None:
        dev = tmp_path / "dev"
        dev.mkdir()
        (dev / "nvidiactl").touch()
        observed = detect_gpus(sysroot=tmp_path)
        assert observed.nvidia == 0

    def test_stale_device_file_not_counted_q_w2(self, tmp_path: Path) -> None:
        """Q-W2: a /dev/nvidia0 that doesn't respond to open() is excluded.

        Simulates the partial-driver-uninstall, broken container bind-mount,
        boot-race, and kernel-module-unloaded failure modes.
        """
        dev = tmp_path / "dev"
        dev.mkdir()
        (dev / "nvidia0").touch()
        (dev / "nvidia1").touch()

        # Probe that returns False for nvidia0 (stale), True for nvidia1.
        def selective(path: Path) -> bool:
            return path.name != "nvidia0"

        observed = detect_gpus(sysroot=tmp_path, probe=selective)
        assert observed.nvidia == 1

    def test_amd_present_only_when_kfd_responsive(self, tmp_path: Path) -> None:
        dev = tmp_path / "dev"
        dev.mkdir()
        (dev / "kfd").touch()
        # Default probe (open()) succeeds on a regular file.
        assert detect_gpus(sysroot=tmp_path).amd is True
        # Custom probe rejecting kfd.
        assert detect_gpus(sysroot=tmp_path, probe=lambda _p: False).amd is False

    def test_amd_not_present_when_kfd_missing(self, tmp_path: Path) -> None:
        assert detect_gpus(sysroot=tmp_path).amd is False

    def test_open_failure_treats_device_as_absent(self, tmp_path: Path) -> None:
        """Sanity-check the production probe path: a path that doesn't exist
        cannot be opened, so should not count."""
        from auspexai_worker.capabilities.detect import _device_responsive

        assert _device_responsive(tmp_path / "definitely-not-there") is False


class TestGpuDeclaration:
    def test_empty_declaration(self) -> None:
        assert GpuDeclaration().is_empty() is True

    def test_partial_declaration_is_not_empty(self) -> None:
        assert GpuDeclaration(nvidia=1).is_empty() is False
        assert GpuDeclaration(amd=True).is_empty() is False
        assert GpuDeclaration(nvidia_model="RTX 4090").is_empty() is False


class TestCollect:
    def test_returns_capabilities_with_expected_shape(self, tmp_path: Path) -> None:
        meminfo = tmp_path / "meminfo"
        meminfo.write_text("MemTotal: 8388608 kB\n")
        caps = collect(meminfo_path=meminfo, sysroot=tmp_path)
        assert isinstance(caps, Capabilities)
        assert caps.os == caps.os.lower()
        assert caps.ram_total_gb == 8.0
        assert isinstance(caps.gpus_observed, GpuObservation)
        assert caps.gpus_observed.has_any() is False
        assert caps.gpus_declared.is_empty() is True

    def test_to_dict_includes_observed_omits_empty_declared(self, tmp_path: Path) -> None:
        caps = collect(sysroot=tmp_path)
        payload = caps.to_dict()
        assert payload["gpus_observed"] == {"nvidia": 0, "amd": False}
        assert "gpus_declared" not in payload
        assert "declared_caps" not in payload

    def test_to_dict_includes_declared_gpus_when_set(self, tmp_path: Path) -> None:
        caps = collect(
            sysroot=tmp_path,
            declared_gpus=GpuDeclaration(
                nvidia=2,
                nvidia_model="RTX 4090",
                vram_total_gb=48.0,
            ),
        )
        payload = caps.to_dict()
        assert payload["gpus_declared"] == {
            "nvidia": 2,
            "nvidia_model": "RTX 4090",
            "vram_total_gb": 48.0,
        }

    def test_to_dict_includes_set_declared_caps(self, tmp_path: Path) -> None:
        caps = collect(
            sysroot=tmp_path,
            declared_caps=DeclaredCaps(max_ram_gb=12.5, max_cpu_cores=4),
        )
        payload = caps.to_dict()
        assert payload["declared_caps"] == {"max_ram_gb": 12.5, "max_cpu_cores": 4}

    def test_to_dict_omits_auto_acquire_when_off(self, tmp_path: Path) -> None:
        # M3: absent on the wire == disabled (the coordinator matcher checks `is True`).
        payload = collect(sysroot=tmp_path).to_dict()
        assert "auto_acquire" not in payload

    def test_to_dict_includes_auto_acquire_when_on(self, tmp_path: Path) -> None:
        payload = collect(sysroot=tmp_path, auto_acquire=True).to_dict()
        assert payload["auto_acquire"] is True

    def test_to_dict_includes_model_sizes_when_present(self, tmp_path: Path) -> None:
        # Fleet-fit: on-disk sizes ride the heartbeat so the coordinator can size a
        # PRESENT model directly (classify it available/too_big by its real footprint).
        payload = collect(
            sysroot=tmp_path, models=["m-a"], model_sizes={"m-a": 1_000_000_000}
        ).to_dict()
        assert payload["model_sizes"] == {"m-a": 1_000_000_000}

    def test_to_dict_omits_model_sizes_when_empty(self, tmp_path: Path) -> None:
        assert "model_sizes" not in collect(sysroot=tmp_path).to_dict()

    def test_to_dict_includes_usable_memory_gb_when_set(self, tmp_path: Path) -> None:
        # Fleet-fit: the coordinator gates routing on this (serve budget), not ram_total.
        assert (
            collect(sysroot=tmp_path, usable_memory_gb=5.44).to_dict()["usable_memory_gb"] == 5.44
        )

    def test_to_dict_omits_usable_memory_gb_when_none(self, tmp_path: Path) -> None:
        assert "usable_memory_gb" not in collect(sysroot=tmp_path).to_dict()

    def test_to_dict_omits_self_paused_when_off(self, tmp_path: Path) -> None:
        # §2.1 #11: absent == not self-paused (coordinator checks `is True`).
        assert "self_paused" not in collect(sysroot=tmp_path).to_dict()

    def test_to_dict_includes_self_paused_when_on(self, tmp_path: Path) -> None:
        assert collect(sysroot=tmp_path, self_paused=True).to_dict()["self_paused"] is True

    def test_execute_tenant_code_declared_default_synthetic(self, tmp_path: Path) -> None:
        # M9 leg 4: always present (informative + routing); default is synthetic.
        assert collect(sysroot=tmp_path).to_dict()["execute_tenant_code"] == "synthetic"

    def test_execute_tenant_code_declared_provisioned(self, tmp_path: Path) -> None:
        payload = collect(sysroot=tmp_path, execute_tenant_code="provisioned").to_dict()
        assert payload["execute_tenant_code"] == "provisioned"

    def test_sandbox_policy_declared_default_permissive(self, tmp_path: Path) -> None:
        # §41: always present so the coordinator can enforce the containment floor.
        assert collect(sysroot=tmp_path).to_dict()["sandbox_policy"] == "permissive"

    def test_sandbox_policy_declared_strict(self, tmp_path: Path) -> None:
        payload = collect(sysroot=tmp_path, sandbox_policy="strict").to_dict()
        assert payload["sandbox_policy"] == "strict"

    def test_declared_alias_back_compat(self, tmp_path: Path) -> None:
        """`declared=` kwarg is the old name; still accepted for back-compat
        so existing callers don't break during the M2-tail rename."""
        caps = collect(sysroot=tmp_path, declared=DeclaredCaps(max_cpu_cores=2))
        assert caps.declared_caps.max_cpu_cores == 2


class TestServedModels:
    """W-S (§9 #43): served_models heartbeat declaration."""

    def test_served_models_in_wire_payload(self):
        caps = collect(served_models=["tiny-q4"])
        assert caps.to_dict()["served_models"] == ["tiny-q4"]

    def test_served_models_omitted_when_empty(self):
        assert "served_models" not in collect().to_dict()
        assert "served_models" not in collect(served_models=[]).to_dict()


class TestFlavorAndOllamaVersion:
    """§9 #46: flavor bookkeeping + serving-runtime provenance on the wire."""

    def test_omitted_when_unset(self, tmp_path: Path) -> None:
        payload = collect(sysroot=tmp_path).to_dict()
        assert "flavor" not in payload
        assert "ollama_version" not in payload

    def test_included_when_set(self, tmp_path: Path) -> None:
        caps = collect(sysroot=tmp_path, flavor="inference", ollama_version="0.6.5")
        payload = caps.to_dict()
        assert payload["flavor"] == "inference"
        assert payload["ollama_version"] == "0.6.5"


class TestWorkerFeatures:
    """v0.2 M1: the build's software-feature declaration for mixed-fleet routing."""

    def test_generation_policy_always_declared(self, tmp_path: Path) -> None:
        payload = collect(sysroot=tmp_path).to_dict()
        assert "generation_policy" in payload["worker_features"]

    def test_to_dict_includes_downloads_when_active(self, tmp_path: Path) -> None:
        # D12 5c: an in-flight download rides the heartbeat so the UI can show progress.
        dl = {"m-x": {"bytes_downloaded": 500, "total_bytes": 1000}}
        assert collect(sysroot=tmp_path, downloads=dl).to_dict()["downloads"] == dl

    def test_to_dict_omits_downloads_when_idle(self, tmp_path: Path) -> None:
        assert "downloads" not in collect(sysroot=tmp_path).to_dict()
