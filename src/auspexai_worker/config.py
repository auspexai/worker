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
from dataclasses import dataclass, field
from pathlib import Path

from .capabilities import GpuDeclaration


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
    # [capabilities.gpus] — volunteer-declared GPU hardware (Q-W2 resolution).
    # The volunteer is the source of truth for what hardware is actually
    # usable; observed device-file probes ride alongside as diagnostic.
    declared_gpus: GpuDeclaration = field(default_factory=GpuDeclaration)
    # [sandbox] — Phase 1 ships PERMISSIVE; tests + dev hosts without bwrap
    # set use_bubblewrap=false. Production MUST use bwrap.
    sandbox_use_bubblewrap: bool = True
    # Maximum wall-clock seconds a runner subprocess can take. None = no
    # timeout (Phase 1 synthetic-tenant work is bounded by trivial logic).
    runner_timeout_seconds: float | None = None
    # [executor] — §9 #37 tenant code-execution consent + provisioning. The
    # resource owner's say on running third-party code: "synthetic" (default;
    # built-in echo only, NO tenant code), "provisioned" (run hash-verified
    # operator-staged executors, refuse unresolved), "off" (refuse all).
    # provisioning_dir (None -> data_dir/tenants) holds staged tenant packages
    # keyed by manifest_sha256.
    execute_tenant_code: str = "synthetic"
    provisioning_dir: Path | None = None
    # [models] store_dir (None -> data_dir/models): the worker-local BYOM model
    # store, laid out by model id (`<store>/<model_id>/`). The volunteer fills
    # it (the platform never distributes weights, §5.8); `--models` resolves
    # here. A model-acquisition onramp will populate it at install/upgrade.
    models_store_dir: Path | None = None
    # [health] — W-H thermal governor thresholds (°C). Auto-discovers
    # /sys/class/thermal zones; graceful no-op where absent. A host at/over
    # crit refuses new work until it cools below resume (hysteresis).
    thermal_warn_c: float = 70.0
    thermal_crit_c: float = 82.0
    thermal_resume_c: float = 68.0
    # NB: per-tenant §5.14 consent (allow/deny lists) is owned by the DB-backed
    # TenantListRepository + the `auspexai-worker tenant` CLI, enforced at the
    # poller's accept-time gate — NOT duplicated here as config.
    # [dashboard] — Phase 2 §5.14 "Layer B" local volunteer-transparency
    # surface. Default-on, localhost-only. Disable with
    # `[dashboard] enabled = false` if the volunteer doesn't want the
    # local HTTP server running.
    dashboard_enabled: bool = True
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 7799
    upgrade_prompt_enabled: bool = True
    upgrade_prompt_threshold: int = 10

    @property
    def state_db_path(self) -> Path:
        return self.state_dir / "worker.db"

    @property
    def keystore_path(self) -> Path:
        return self.data_dir / "keystore.enc"

    @property
    def provisioning_path(self) -> Path:
        """Where staged tenant packages live (keyed by manifest_sha256).
        Defaults to data_dir/tenants when not explicitly configured."""
        return (
            self.provisioning_dir
            if self.provisioning_dir is not None
            else self.data_dir / "tenants"
        )

    @property
    def models_store_path(self) -> Path:
        """The worker-local BYOM model store (keyed by model id). Defaults to
        data_dir/models."""
        return (
            self.models_store_dir if self.models_store_dir is not None else self.data_dir / "models"
        )

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
            # The public AuspexAI coordinator (Phase 2 closed-beta lab
            # deployment on rage, Cloudflare-Tunneled at coord.auspexai
            # .network). Lab operators running their own coordinator on
            # localhost should override via `[coordinator] url` in
            # worker.toml or AUSPEXAI_COORDINATOR_URL env. v0.1.0 +
            # v0.1.1 used http://127.0.0.1:8080 as the default per the
            # original Q-W8 lab-altitude resolution; v0.1.2 flips this
            # now that a publicly-reachable coord exists.
            "coordinator_url": "https://coord.auspexai.network",
            "heartbeat_interval_seconds": 60,
            "assignment_poll_interval_seconds": 30,
            "state_dir": str(_xdg_state_home() / "auspexai-worker"),
            "data_dir": str(_xdg_data_home() / "auspexai-worker"),
            "keystore_backend": None,
            "max_ram_gb": None,
            "max_vram_gb": None,
            "max_cpu_cores": None,
            "network_quota_mb_per_hour": None,
            "declared_gpu_nvidia": None,
            "declared_gpu_nvidia_model": None,
            "declared_gpu_vram_total_gb": None,
            "declared_gpu_amd": None,
            "declared_gpu_amd_model": None,
            "sandbox_use_bubblewrap": True,
            "runner_timeout_seconds": None,
            "execute_tenant_code": "synthetic",
            "provisioning_dir": None,
            "models_store_dir": None,
            "thermal_warn_c": 70.0,
            "thermal_crit_c": 82.0,
            "thermal_resume_c": 68.0,
            "dashboard_enabled": True,
            "dashboard_host": "127.0.0.1",
            "dashboard_port": 7799,
            "upgrade_prompt_enabled": True,
            "upgrade_prompt_threshold": 10,
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
            capabilities = data.get("capabilities") or {}
            gpus_block = capabilities.get("gpus") or {}
            for gpu_key in (
                "nvidia",
                "nvidia_model",
                "vram_total_gb",
                "amd",
                "amd_model",
            ):
                if gpu_key in gpus_block:
                    merged[f"declared_gpu_{gpu_key}"] = gpus_block[gpu_key]
            sandbox_block = data.get("sandbox") or {}
            if "use_bubblewrap" in sandbox_block:
                merged["sandbox_use_bubblewrap"] = sandbox_block["use_bubblewrap"]
            if "runner_timeout_seconds" in sandbox_block:
                merged["runner_timeout_seconds"] = sandbox_block["runner_timeout_seconds"]
            executor_block = data.get("executor") or {}
            # accept `execute_tenant_code` or the shorter alias `mode`
            for key in ("execute_tenant_code", "mode"):
                if key in executor_block:
                    merged["execute_tenant_code"] = executor_block[key]
            if "provisioning_dir" in executor_block:
                merged["provisioning_dir"] = executor_block["provisioning_dir"]
            models_block = data.get("models") or {}
            if "store_dir" in models_block:
                merged["models_store_dir"] = models_block["store_dir"]
            health_block = data.get("health") or {}
            for short, full in (
                ("warn_c", "thermal_warn_c"),
                ("crit_c", "thermal_crit_c"),
                ("resume_c", "thermal_resume_c"),
            ):
                if short in health_block:
                    merged[full] = health_block[short]
            dashboard_block = data.get("dashboard") or {}
            if "enabled" in dashboard_block:
                merged["dashboard_enabled"] = dashboard_block["enabled"]
            if "host" in dashboard_block:
                merged["dashboard_host"] = dashboard_block["host"]
            if "port" in dashboard_block:
                merged["dashboard_port"] = dashboard_block["port"]

            upgrade_block = data.get("upgrade_prompt") or {}
            if "enabled" in upgrade_block:
                merged["upgrade_prompt_enabled"] = upgrade_block["enabled"]
            if "threshold" in upgrade_block:
                merged["upgrade_prompt_threshold"] = upgrade_block["threshold"]
            # [tenants], [telemetry] are reserved for later milestones;
            # tolerated here without consumption.

        # Env var overrides (highest precedence besides explicit kwargs).
        if "AUSPEXAI_COORDINATOR_URL" in env:
            merged["coordinator_url"] = env["AUSPEXAI_COORDINATOR_URL"]
        if "AUSPEXAI_WORKER_STATE_DIR" in env:
            merged["state_dir"] = env["AUSPEXAI_WORKER_STATE_DIR"]
        if "AUSPEXAI_WORKER_DATA_DIR" in env:
            merged["data_dir"] = env["AUSPEXAI_WORKER_DATA_DIR"]
        if "AUSPEXAI_WORKER_KEYSTORE_BACKEND" in env:
            merged["keystore_backend"] = env["AUSPEXAI_WORKER_KEYSTORE_BACKEND"] or None
        if "AUSPEXAI_WORKER_EXECUTE_TENANT_CODE" in env:
            merged["execute_tenant_code"] = env["AUSPEXAI_WORKER_EXECUTE_TENANT_CODE"]
        if "AUSPEXAI_WORKER_PROVISIONING_DIR" in env:
            merged["provisioning_dir"] = env["AUSPEXAI_WORKER_PROVISIONING_DIR"]
        if "AUSPEXAI_WORKER_DASHBOARD_ENABLED" in env:
            merged["dashboard_enabled"] = env["AUSPEXAI_WORKER_DASHBOARD_ENABLED"].lower() in (
                "1",
                "true",
                "yes",
                "on",
            )

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
            declared_gpus=GpuDeclaration(
                nvidia=_opt_int(merged.get("declared_gpu_nvidia")),
                nvidia_model=_opt_str(merged.get("declared_gpu_nvidia_model")),
                vram_total_gb=_opt_float(merged.get("declared_gpu_vram_total_gb")),
                amd=_opt_bool(merged.get("declared_gpu_amd")),
                amd_model=_opt_str(merged.get("declared_gpu_amd_model")),
            ),
            sandbox_use_bubblewrap=bool(merged.get("sandbox_use_bubblewrap", True)),
            runner_timeout_seconds=_opt_float(merged.get("runner_timeout_seconds")),
            execute_tenant_code=_validate_policy(merged.get("execute_tenant_code", "synthetic")),
            provisioning_dir=(
                None
                if merged.get("provisioning_dir") is None
                else Path(str(merged["provisioning_dir"])).expanduser()
            ),
            models_store_dir=(
                None
                if merged.get("models_store_dir") is None
                else Path(str(merged["models_store_dir"])).expanduser()
            ),
            thermal_warn_c=float(merged.get("thermal_warn_c", 70.0)),
            thermal_crit_c=float(merged.get("thermal_crit_c", 82.0)),
            thermal_resume_c=float(merged.get("thermal_resume_c", 68.0)),
            dashboard_enabled=bool(merged.get("dashboard_enabled", True)),
            dashboard_host=str(merged.get("dashboard_host", "127.0.0.1")),
            dashboard_port=int(merged.get("dashboard_port", 7799)),
            upgrade_prompt_enabled=bool(merged.get("upgrade_prompt_enabled", True)),
            upgrade_prompt_threshold=int(merged.get("upgrade_prompt_threshold", 10)),
        )


_EXECUTE_POLICIES = ("synthetic", "provisioned", "off")


def _validate_policy(raw: object) -> str:
    val = str(raw)
    if val not in _EXECUTE_POLICIES:
        raise ValueError(f"execute_tenant_code must be one of {_EXECUTE_POLICIES}, got {val!r}")
    return val


def _opt_float(raw: object) -> float | None:
    return None if raw is None else float(raw)  # type: ignore[arg-type]


def _opt_int(raw: object) -> int | None:
    return None if raw is None else int(raw)  # type: ignore[arg-type]


def _opt_str(raw: object) -> str | None:
    return None if raw is None else str(raw)


def _opt_bool(raw: object) -> bool | None:
    return None if raw is None else bool(raw)
