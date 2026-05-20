"""Worker configuration — defaults + TOML + env-var overrides.

Resolution order (low → high precedence):
  1. Built-in defaults.
  2. `/etc/auspexai-worker/worker.toml` (system-shipped defaults).
  3. `$XDG_CONFIG_HOME/auspexai-worker/worker.toml` (user override).
  4. Environment variables.
  5. Explicit kwargs to `WorkerConfig.load()`.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


def _xdg_dir(env_name: str, default_subpath: str) -> Path:
    value = os.environ.get(env_name)
    if value:
        return Path(value).expanduser()
    return Path.home() / default_subpath


def _xdg_state_home() -> Path:
    return _xdg_dir("XDG_STATE_HOME", ".local/state")


def _xdg_data_home() -> Path:
    return _xdg_dir("XDG_DATA_HOME", ".local/share")


def _xdg_config_home() -> Path:
    return _xdg_dir("XDG_CONFIG_HOME", ".config")


@dataclass(frozen=True)
class WorkerConfig:
    """Resolved worker configuration."""

    coordinator_url: str
    heartbeat_interval_seconds: int
    assignment_poll_interval_seconds: int
    state_dir: Path
    data_dir: Path
    keystore_backend: str | None  # None = auto-detect; "secret_service" / "encrypted_file"
    # [resources] caps — volunteer-declared ceilings. None means the worker
    # makes no declaration; coordinator scheduler treats the slot as unbounded
    # and local sandbox enforcement (M4) uses host-detected limits instead.
    max_ram_gb: float | None = None
    max_vram_gb: float | None = None
    max_cpu_cores: int | None = None
    network_quota_mb_per_hour: int | None = None

    @property
    def state_db_path(self) -> Path:
        return self.state_dir / "worker.db"

    @property
    def keystore_path(self) -> Path:
        return self.data_dir / "keystore.enc"

    @classmethod
    def load(
        cls,
        *,
        config_path: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> WorkerConfig:
        """Resolve configuration from defaults + TOML files + env.

        Args:
            config_path: Optional explicit TOML path. When provided, only this
                file is read (no system/user search). Used by tests.
            env: Optional environment dict to consult instead of `os.environ`.
        """
        env = dict(env if env is not None else os.environ)

        defaults: dict[str, object] = {
            "coordinator_url": "http://127.0.0.1:8080",
            "heartbeat_interval_seconds": 60,
            "assignment_poll_interval_seconds": 30,
            "state_dir": str(_xdg_state_home() / "auspexai-worker"),
            "data_dir": str(_xdg_data_home() / "auspexai-worker"),
            "keystore_backend": None,
            "max_ram_gb": None,
            "max_vram_gb": None,
            "max_cpu_cores": None,
            "network_quota_mb_per_hour": None,
        }

        if config_path is not None:
            files = [config_path]
        else:
            files = [
                Path("/etc/auspexai-worker/worker.toml"),
                _xdg_config_home() / "auspexai-worker" / "worker.toml",
            ]

        merged = dict(defaults)
        for path in files:
            if not path.is_file():
                continue
            with path.open("rb") as fh:
                data = tomllib.load(fh)
            coord = data.get("coordinator") or {}
            if "url" in coord:
                merged["coordinator_url"] = coord["url"]
            if "heartbeat_interval_seconds" in coord:
                merged["heartbeat_interval_seconds"] = coord["heartbeat_interval_seconds"]
            if "assignment_poll_interval_seconds" in coord:
                merged["assignment_poll_interval_seconds"] = coord[
                    "assignment_poll_interval_seconds"
                ]
            identity = data.get("identity") or {}
            if "keystore_backend" in identity:
                merged["keystore_backend"] = identity["keystore_backend"]
            resources = data.get("resources") or {}
            for cap_key in (
                "max_ram_gb",
                "max_vram_gb",
                "max_cpu_cores",
                "network_quota_mb_per_hour",
            ):
                if cap_key in resources:
                    merged[cap_key] = resources[cap_key]
            # [sandbox], [tenants], [models], [telemetry] are reserved for
            # M3+ milestones; tolerated here without consumption.

        # Env var overrides (highest precedence besides explicit kwargs).
        if "AUSPEXAI_COORDINATOR_URL" in env:
            merged["coordinator_url"] = env["AUSPEXAI_COORDINATOR_URL"]
        if "AUSPEXAI_WORKER_STATE_DIR" in env:
            merged["state_dir"] = env["AUSPEXAI_WORKER_STATE_DIR"]
        if "AUSPEXAI_WORKER_DATA_DIR" in env:
            merged["data_dir"] = env["AUSPEXAI_WORKER_DATA_DIR"]
        if "AUSPEXAI_WORKER_KEYSTORE_BACKEND" in env:
            merged["keystore_backend"] = env["AUSPEXAI_WORKER_KEYSTORE_BACKEND"] or None

        return cls(
            coordinator_url=str(merged["coordinator_url"]).rstrip("/"),
            heartbeat_interval_seconds=int(merged["heartbeat_interval_seconds"]),
            assignment_poll_interval_seconds=int(merged["assignment_poll_interval_seconds"]),
            state_dir=Path(str(merged["state_dir"])).expanduser(),
            data_dir=Path(str(merged["data_dir"])).expanduser(),
            keystore_backend=(
                None if merged["keystore_backend"] is None else str(merged["keystore_backend"])
            ),
            max_ram_gb=_opt_float(merged.get("max_ram_gb")),
            max_vram_gb=_opt_float(merged.get("max_vram_gb")),
            max_cpu_cores=_opt_int(merged.get("max_cpu_cores")),
            network_quota_mb_per_hour=_opt_int(merged.get("network_quota_mb_per_hour")),
        )


def _opt_float(raw: object) -> float | None:
    return None if raw is None else float(raw)  # type: ignore[arg-type]


def _opt_int(raw: object) -> int | None:
    return None if raw is None else int(raw)  # type: ignore[arg-type]
