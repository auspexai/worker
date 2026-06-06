"""General accelerator detection across platforms (v0.1.11)."""

from __future__ import annotations

from auspexai_worker.accelerator import AcceleratorKind, detect_accelerator


def _none() -> None:
    return None


def test_jetson_unified():
    a = detect_accelerator(
        device_tree_model="NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super",
        ram_total_gb=7.4,
        nvidia_smi=_none,
        kfd_present=False,
    )
    assert a.kind is AcceleratorKind.JETSON
    assert a.unified is True
    assert a.memory_budget_gb == 7.4  # unified -> RAM is the budget


def test_jetson_wins_over_nvidia_smi():
    # tegra has NVIDIA silicon; device-tree must classify it as unified, not discrete.
    a = detect_accelerator(
        device_tree_model="NVIDIA Jetson Orin Nano",
        ram_total_gb=8.0,
        nvidia_smi=lambda: 8.0,
    )
    assert a.kind is AcceleratorKind.JETSON
    assert a.unified is True


def test_apple_silicon_unified():
    a = detect_accelerator(
        system="Darwin",
        machine="arm64",
        ram_total_gb=32.0,
        device_tree_model="",
        nvidia_smi=_none,
        kfd_present=False,
    )
    assert a.kind is AcceleratorKind.APPLE
    assert a.unified is True
    assert a.memory_budget_gb == 32.0


def test_nvidia_discrete():
    a = detect_accelerator(
        system="Linux",
        machine="x86_64",
        ram_total_gb=64.0,
        device_tree_model="",
        nvidia_smi=lambda: 24.0,
        kfd_present=False,
    )
    assert a.kind is AcceleratorKind.NVIDIA
    assert a.unified is False
    assert a.memory_budget_gb == 24.0  # discrete VRAM, not RAM


def test_amd_discrete():
    a = detect_accelerator(
        system="Linux",
        machine="x86_64",
        ram_total_gb=64.0,
        device_tree_model="",
        nvidia_smi=_none,
        kfd_present=True,
        amd_vram=lambda: 16.0,
    )
    assert a.kind is AcceleratorKind.AMD
    assert a.unified is False
    assert a.memory_budget_gb == 16.0


def test_cpu_only():
    a = detect_accelerator(
        system="Linux",
        machine="x86_64",
        ram_total_gb=16.0,
        device_tree_model="",
        nvidia_smi=_none,
        kfd_present=False,
    )
    assert a.kind is AcceleratorKind.CPU
    assert a.unified is False
    assert a.memory_budget_gb == 16.0


def test_to_dict_shape():
    d = detect_accelerator(
        device_tree_model="NVIDIA Jetson Orin Nano", ram_total_gb=7.4, nvidia_smi=_none
    ).to_dict()
    assert d["kind"] == "jetson" and d["unified"] is True and d["memory_budget_gb"] == 7.4
