"""W-H — thermal monitor: zone discovery, thresholds, hysteresis, graceful no-op."""

from __future__ import annotations

from pathlib import Path

from auspexai_worker.health import (
    ThermalMonitor,
    ThermalState,
    discover_thermal_zones,
)


def _zone(root: Path, n: int, milli_c: int) -> Path:
    d = root / f"thermal_zone{n}"
    d.mkdir(parents=True)
    (d / "temp").write_text(str(milli_c))
    return d / "temp"


def _set(temp_file: Path, milli_c: int) -> None:
    temp_file.write_text(str(milli_c))


def test_discover_thermal_zones(tmp_path: Path):
    _zone(tmp_path, 0, 40000)
    _zone(tmp_path, 1, 55000)
    zones = discover_thermal_zones(tmp_path)
    assert len(zones) == 2


def test_discover_absent_root_is_empty(tmp_path: Path):
    assert discover_thermal_zones(tmp_path / "nope") == []


def test_read_temp_is_hottest_zone(tmp_path: Path):
    z0 = _zone(tmp_path, 0, 40000)  # 40°C
    _zone(tmp_path, 1, 71500)  # 71.5°C
    mon = ThermalMonitor(zones=[z0, tmp_path / "thermal_zone1" / "temp"])
    assert mon.read_temp() == 71.5  # governs on the hottest


def test_state_thresholds(tmp_path: Path):
    z = _zone(tmp_path, 0, 50000)
    mon = ThermalMonitor(zones=[z], warn_c=70, crit_c=82, resume_c=68)
    assert mon.state() is ThermalState.OK
    _set(z, 75000)
    assert mon.state() is ThermalState.WARM
    _set(z, 83000)
    assert mon.state() is ThermalState.CRITICAL


def test_hysteresis_stays_critical_until_resume(tmp_path: Path):
    z = _zone(tmp_path, 0, 83000)
    mon = ThermalMonitor(zones=[z], warn_c=70, crit_c=82, resume_c=68)
    assert mon.state() is ThermalState.CRITICAL
    # Dip below crit (80) but above resume (68): still critical (no flap).
    _set(z, 80000)
    assert mon.state() is ThermalState.CRITICAL
    # Cool below resume: clears to OK.
    _set(z, 67000)
    assert mon.state() is ThermalState.OK


def test_no_zones_is_graceful_noop(tmp_path: Path):
    mon = ThermalMonitor(zones=[])
    assert mon.enabled is False
    assert mon.read_temp() is None
    assert mon.state() is ThermalState.OK  # never blocks work without a sensor


def test_unreadable_sensor_is_ok(tmp_path: Path):
    mon = ThermalMonitor(zones=[tmp_path / "ghost" / "temp"])
    assert mon.read_temp() is None
    assert mon.state() is ThermalState.OK


def test_snapshot_shape(tmp_path: Path):
    z = _zone(tmp_path, 0, 75000)
    mon = ThermalMonitor(zones=[z], warn_c=70, crit_c=82)
    snap = mon.snapshot().to_dict()
    assert snap["state"] == "warm"
    assert snap["current_temp_c"] == 75.0
    assert snap["zone_count"] == 1
