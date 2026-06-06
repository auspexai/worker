"""Worker health governor (W-H) — thermal safety for unknown-payload execution.

Running real tenant code (§9 #37) physically stresses the volunteer's machine.
Two harms follow if we don't watch the host's physical state:

  1. **Hardware** — sustained load can overheat / destabilize a donated machine
     (the Jetsons throttle hard). Cooking a volunteer's box violates the
     volunteer-first posture.
  2. **Experiment integrity** — a thermally-throttled or unstable host produces
     slower and potentially numerically-divergent results, which masquerade as
     consensus *disagreement* (breaking the #33 determinism contract) or silently
     drift the science. A degraded worker is a faulty worker.

This is ported + generalized from the prior Sentinel `thermal.py` (which
hardcoded a Jetson zone). Here zone discovery is generic — scan
`/sys/class/thermal/thermal_zone*/temp` and govern on the hottest — so it works
across hardware and is a graceful no-op where sysfs thermal is absent (macOS,
containers): no readings → state OK → never blocks work.

The governor's lever is **refuse-when-hot**: a worker at/over the critical
threshold declines new units (the coordinator re-offers to a cooler worker, so
the science isn't lost), and the worker cools naturally by not running the heavy
executor. Hysteresis (resume below a lower threshold) prevents flapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# Jetson-friendly defaults (°C); overridable via [health] config.
DEFAULT_WARN_C = 70.0
DEFAULT_CRIT_C = 82.0
DEFAULT_RESUME_C = 68.0

_THERMAL_ROOT = Path("/sys/class/thermal")


class ThermalState(StrEnum):
    OK = "ok"
    WARM = "warm"  # advisory; work still runs
    CRITICAL = "critical"  # refuse new work until cooled to resume_c


def discover_thermal_zones(root: Path = _THERMAL_ROOT) -> list[Path]:
    """Return readable `.../thermal_zone*/temp` files, or [] where absent
    (non-Linux, containers without thermal sysfs)."""
    if not root.is_dir():
        return []
    return [z / "temp" for z in sorted(root.glob("thermal_zone*")) if (z / "temp").is_file()]


@dataclass(frozen=True)
class ThermalSnapshot:
    state: ThermalState
    current_temp_c: float | None
    max_temp_c: float
    zone_count: int

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "current_temp_c": (
                round(self.current_temp_c, 1) if self.current_temp_c is not None else None
            ),
            "max_temp_c": round(self.max_temp_c, 1),
            "zone_count": self.zone_count,
        }


class ThermalMonitor:
    """Governs on the hottest discovered thermal zone, with resume hysteresis."""

    def __init__(
        self,
        *,
        zones: list[Path] | None = None,
        warn_c: float = DEFAULT_WARN_C,
        crit_c: float = DEFAULT_CRIT_C,
        resume_c: float = DEFAULT_RESUME_C,
    ) -> None:
        self._zones = zones if zones is not None else discover_thermal_zones()
        self._warn_c = warn_c
        self._crit_c = crit_c
        self._resume_c = resume_c
        self._in_critical = False  # hysteresis latch
        self._max_temp = 0.0

    @property
    def enabled(self) -> bool:
        """True if any thermal zone is readable (else the governor no-ops)."""
        return bool(self._zones)

    def read_temp(self) -> float | None:
        """Hottest temperature across discovered zones (°C), or None if no zone
        is readable. sysfs reports millidegrees."""
        temps: list[float] = []
        for tpath in self._zones:
            try:
                temps.append(int(tpath.read_text().strip()) / 1000.0)
            except (OSError, ValueError):
                continue
        return max(temps) if temps else None

    def state(self) -> ThermalState:
        """Current governed state with hysteresis. Sensor failure → OK (never
        block work on an unreadable sensor — matches the Sentinel guard)."""
        temp = self.read_temp()
        if temp is None:
            return ThermalState.OK
        if temp > self._max_temp:
            self._max_temp = temp
        if self._in_critical:
            # Stay critical until we've cooled below the resume threshold.
            if temp <= self._resume_c:
                self._in_critical = False
            else:
                return ThermalState.CRITICAL
        if temp >= self._crit_c:
            self._in_critical = True
            return ThermalState.CRITICAL
        if temp >= self._warn_c:
            return ThermalState.WARM
        return ThermalState.OK

    def snapshot(self) -> ThermalSnapshot:
        st = self.state()
        return ThermalSnapshot(
            state=st,
            current_temp_c=self.read_temp(),
            max_temp_c=self._max_temp,
            zone_count=len(self._zones),
        )
