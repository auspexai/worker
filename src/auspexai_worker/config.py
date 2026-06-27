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
import re
import sys
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


def default_worker_toml_path() -> Path:
    """The canonical user-writable worker.toml location (the XDG override path the
    loader checks last). Where `executor set` writes when no explicit --config."""
    return _xdg_config_home() / "auspexai-worker" / "worker.toml"


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
    # §41(a): [sandbox] policy = permissive | strict. STRICT replaces the host-fs
    # --dev-bind / / with narrow read-only binds + namespace isolation, so real
    # tenant code can't reach the keystore / $HOME / cross-tenant data. Default
    # permissive (the Phase-1 trust model). (STRICT-by-default was tried + reverted
    # 2026-06-27: it fails closed on hosts with restricted unprivileged user
    # namespaces, which would strand volunteers; strict stays a recommended opt-in.)
    sandbox_policy: str = "permissive"
    # Maximum wall-clock seconds a runner subprocess can take. None = no
    # timeout (Phase 1 synthetic-tenant work is bounded by trivial logic).
    runner_timeout_seconds: float | None = None
    # §41(a) STRICT resource caps (the "exhaust resources" gate). Defense-in-depth:
    # a portable rlimit floor (always) + cgroup v2 memory/pids caps (when systemd
    # Delegate=yes gives a writable subtree). Caps apply ONLY under STRICT; the
    # generous defaults stop a fork/alloc bomb without tripping a light inference
    # harness. `[sandbox] resource_limits = false` disables the layer.
    sandbox_resource_limits: bool = True
    sandbox_memory_max_mb: int | None = 4096  # cgroup memory.max (RSS); None = uncapped
    sandbox_pids_max: int | None = 512  # cgroup pids.max (fork-bomb cap); None = uncapped
    sandbox_cpu_seconds: int | None = None  # rlimit cpu-seconds; None ⇒ wall-clock governs
    # [executor] — §9 #37 tenant code-execution consent + provisioning. The
    # resource owner's say on running third-party code: "synthetic" (default;
    # built-in echo only, NO tenant code), "provisioned" (run hash-verified
    # operator-staged executors, refuse unresolved), "off" (refuse all).
    # provisioning_dir (None -> data_dir/tenants) holds staged tenant packages
    # keyed by manifest_sha256.
    execute_tenant_code: str = "synthetic"
    provisioning_dir: Path | None = None
    # [executor] auto_acquire (M3 lazy auto-acquire) — when True AND the policy is
    # `provisioned`, a unit whose locally-required model is missing is pulled
    # (from the manifest's hf_repo/hf_filename) then run, rather than refused.
    # Default False keeps the refuse-don't-echo posture; this is an explicit
    # opt-in to spend bandwidth+disk acquiring models on demand.
    auto_acquire: bool = False
    # [provisioning] auto_fetch (#40a executor-package auto-fetch) — when True
    # (the default) a dispatched unit whose package digest is NOT in the local
    # package store is fetched from the coordinator, verified (manifest hash +
    # executor.package_sha256 over the extracted tree, traversal-safe), and
    # installed content-addressed before running. Strictly verified, hence
    # default ON; operator staging still works and short-circuits the fetch.
    # `false` restores staged-only resolution.
    auto_fetch: bool = True
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
    # [inference] — W-S (§9 #43) worker model-serving + sandbox inference
    # broker. "none" (default) keeps the whole feature dormant — no backend
    # management, no broker sockets, no served_models declaration. "ollama"
    # opts this worker into serving its BYOM models to sandboxed executors
    # via the per-unit unix-socket broker (the operator's opt-in IS the
    # signal that this worker hosts inference tenants).
    inference_backend: str = "none"
    inference_ollama_url: str = "http://127.0.0.1:11434"
    # §9 #46 serving policy: Ollama keep_alive sent on every brokered chat.
    # None = Ollama default (~5m). "0" = unload-always (Sentinel's posture for
    # reload/wedging stability); "30m"/"24h" = pin warm (memory for latency).
    inference_keep_alive: str | None = None
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
    # [worker] flavor — the install profile the onramp applied (§9 #46:
    # lean / inference / full / future names). Bookkeeping + upgrade
    # preservation ONLY: scheduling stays capability-based; nothing gates on
    # the flavor name. Validated by shape (regex), NOT an enum — an old
    # worker must tolerate a future flavor name.
    flavor: str | None = None

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
            "sandbox_policy": "permissive",
            "runner_timeout_seconds": None,
            "sandbox_resource_limits": True,
            "sandbox_memory_max_mb": 4096,
            "sandbox_pids_max": 512,
            "sandbox_cpu_seconds": None,
            "execute_tenant_code": "synthetic",
            "provisioning_dir": None,
            "auto_fetch": True,
            "models_store_dir": None,
            "thermal_warn_c": 70.0,
            "thermal_crit_c": 82.0,
            "thermal_resume_c": 68.0,
            "inference_backend": "none",
            "inference_ollama_url": "http://127.0.0.1:11434",
            "dashboard_enabled": True,
            "dashboard_host": "127.0.0.1",
            "dashboard_port": 7799,
            "upgrade_prompt_enabled": True,
            "upgrade_prompt_threshold": 10,
            "flavor": None,
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
            if "policy" in sandbox_block:
                merged["sandbox_policy"] = sandbox_block["policy"]
            if "runner_timeout_seconds" in sandbox_block:
                merged["runner_timeout_seconds"] = sandbox_block["runner_timeout_seconds"]
            if "resource_limits" in sandbox_block:
                merged["sandbox_resource_limits"] = sandbox_block["resource_limits"]
            if "memory_max_mb" in sandbox_block:
                merged["sandbox_memory_max_mb"] = sandbox_block["memory_max_mb"]
            if "pids_max" in sandbox_block:
                merged["sandbox_pids_max"] = sandbox_block["pids_max"]
            if "cpu_seconds" in sandbox_block:
                merged["sandbox_cpu_seconds"] = sandbox_block["cpu_seconds"]
            executor_block = data.get("executor") or {}
            # accept `execute_tenant_code` or the shorter alias `mode`
            for key in ("execute_tenant_code", "mode"):
                if key in executor_block:
                    merged["execute_tenant_code"] = executor_block[key]
            if "provisioning_dir" in executor_block:
                merged["provisioning_dir"] = executor_block["provisioning_dir"]
            if "auto_acquire" in executor_block:
                merged["auto_acquire"] = executor_block["auto_acquire"]
            provisioning_block = data.get("provisioning") or {}
            if "auto_fetch" in provisioning_block:
                merged["auto_fetch"] = provisioning_block["auto_fetch"]
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
            inference_block = data.get("inference") or {}
            if "backend" in inference_block:
                merged["inference_backend"] = inference_block["backend"]
            if "ollama_url" in inference_block:
                merged["inference_ollama_url"] = inference_block["ollama_url"]
            if "keep_alive" in inference_block:
                merged["inference_keep_alive"] = inference_block["keep_alive"]
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
            worker_block = data.get("worker") or {}
            if "flavor" in worker_block:
                merged["flavor"] = worker_block["flavor"]
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
        if "AUSPEXAI_WORKER_AUTO_ACQUIRE" in env:
            merged["auto_acquire"] = env["AUSPEXAI_WORKER_AUTO_ACQUIRE"].lower() in (
                "1",
                "true",
                "yes",
            )
        if "AUSPEXAI_WORKER_AUTO_FETCH" in env:
            merged["auto_fetch"] = env["AUSPEXAI_WORKER_AUTO_FETCH"].lower() in (
                "1",
                "true",
                "yes",
            )
        if "AUSPEXAI_WORKER_DASHBOARD_ENABLED" in env:
            merged["dashboard_enabled"] = env["AUSPEXAI_WORKER_DASHBOARD_ENABLED"].lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        if "AUSPEXAI_WORKER_FLAVOR" in env:
            merged["flavor"] = env["AUSPEXAI_WORKER_FLAVOR"] or None

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
            # bubblewrap is Linux-only; on macOS/other it is forced OFF (passthrough),
            # and since strict containment is impossible there the policy is pinned to
            # permissive — the worker must never sign a strict containment it cannot
            # actually enforce. (macOS strict via sandbox-exec is a separate, future
            # capability.) On Linux both honor the config / the volunteer's choice.
            sandbox_use_bubblewrap=(
                bool(merged.get("sandbox_use_bubblewrap", True)) and sys.platform == "linux"
            ),
            sandbox_policy=(
                _validate_sandbox_policy(merged.get("sandbox_policy", "permissive"))
                if sys.platform == "linux"
                else "permissive"
            ),
            runner_timeout_seconds=_opt_float(merged.get("runner_timeout_seconds")),
            sandbox_resource_limits=bool(merged.get("sandbox_resource_limits", True)),
            sandbox_memory_max_mb=_opt_int(merged.get("sandbox_memory_max_mb")),
            sandbox_pids_max=_opt_int(merged.get("sandbox_pids_max")),
            sandbox_cpu_seconds=_opt_int(merged.get("sandbox_cpu_seconds")),
            execute_tenant_code=_validate_policy(merged.get("execute_tenant_code", "synthetic")),
            auto_acquire=bool(merged.get("auto_acquire", False)),
            auto_fetch=bool(merged.get("auto_fetch", True)),
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
            inference_backend=_validate_inference_backend(merged.get("inference_backend", "none")),
            inference_ollama_url=str(
                merged.get("inference_ollama_url", "http://127.0.0.1:11434")
            ).rstrip("/"),
            inference_keep_alive=(
                None
                if merged.get("inference_keep_alive") is None
                else str(merged["inference_keep_alive"])
            ),
            dashboard_enabled=bool(merged.get("dashboard_enabled", True)),
            dashboard_host=str(merged.get("dashboard_host", "127.0.0.1")),
            dashboard_port=int(merged.get("dashboard_port", 7799)),
            upgrade_prompt_enabled=bool(merged.get("upgrade_prompt_enabled", True)),
            upgrade_prompt_threshold=int(merged.get("upgrade_prompt_threshold", 10)),
            flavor=_validate_flavor(merged.get("flavor")),
        )


_EXECUTE_POLICIES = ("synthetic", "provisioned", "off")
_INFERENCE_BACKENDS = ("none", "ollama")


def _validate_policy(raw: object) -> str:
    val = str(raw)
    if val not in _EXECUTE_POLICIES:
        raise ValueError(f"execute_tenant_code must be one of {_EXECUTE_POLICIES}, got {val!r}")
    return val


_SANDBOX_POLICIES = ("permissive", "strict")


def _validate_sandbox_policy(raw: object) -> str:
    val = str(raw)
    if val not in _SANDBOX_POLICIES:
        raise ValueError(f"[sandbox] policy must be one of {_SANDBOX_POLICIES}, got {val!r}")
    return val


def _validate_inference_backend(raw: object) -> str:
    val = str(raw)
    if val not in _INFERENCE_BACKENDS:
        raise ValueError(f"[inference] backend must be one of {_INFERENCE_BACKENDS}, got {val!r}")
    return val


# Shape-validated, NOT an enum: flavors are installer profiles defined in
# install.sh's data block (§9 #46) — a worker binary must tolerate a flavor
# name minted after it shipped.
_FLAVOR_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


def _validate_flavor(raw: object) -> str | None:
    if raw is None:
        return None
    val = str(raw)
    if not _FLAVOR_RE.match(val):
        raise ValueError(
            f"[worker] flavor must match {_FLAVOR_RE.pattern} (lowercase token), got {val!r}"
        )
    return val


def _opt_float(raw: object) -> float | None:
    return None if raw is None else float(raw)  # type: ignore[arg-type]


def _opt_int(raw: object) -> int | None:
    return None if raw is None else int(raw)  # type: ignore[arg-type]


def _opt_str(raw: object) -> str | None:
    return None if raw is None else str(raw)


def _opt_bool(raw: object) -> bool | None:
    return None if raw is None else bool(raw)


def _upsert_toml_section(path: Path, section: str, updates: dict[str, str]) -> None:
    """Set `key = value_literal` for each (key, value_literal) in `updates` inside
    `[section]` of a TOML file, preserving everything else (comments + other
    sections). Replaces a key already present in the section, inserts it right
    after the section header otherwise, and appends a new `[section]` if absent.
    A targeted text edit (no TOML round-trip) so the volunteer's file stays intact.
    `value_literal` is the raw TOML value (e.g. '"provisioned"', 'true')."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.splitlines()

    header = f"[{section}]"
    start = next((i for i, ln in enumerate(lines) if ln.strip() == header), None)
    if start is None:
        block = [header] + [f"{k} = {v}" for k, v in updates.items()]
        if lines and lines[-1].strip() != "":
            lines.append("")
        lines.extend(block)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")),
        len(lines),
    )
    for key, value in updates.items():
        idx = next(
            (
                i
                for i in range(start + 1, end)
                if lines[i].lstrip().startswith((f"{key} ", f"{key}="))
            ),
            None,
        )
        if idx is not None:
            lines[idx] = f"{key} = {value}"
        else:
            lines.insert(start + 1, f"{key} = {value}")
            end += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_executor_policy(config_path: Path | None) -> tuple[str, bool]:
    """Freshly resolve `(execute_tenant_code, effective_auto_acquire)` from disk —
    the live owner-consent state, NOT a daemon-start snapshot. Used for hot-reload:
    the daemon re-reads this per-heartbeat (capability declaration) and per-dispatch
    (execution gate) so a policy change applies WITHOUT a daemon restart. Raises if
    the config can't be loaded (callers fail safe). `effective_auto_acquire` already
    folds in the "only under provisioned" rule, matching the daemon's wiring."""
    cfg = WorkerConfig.load(config_path=config_path)
    return cfg.execute_tenant_code, (cfg.auto_acquire and cfg.execute_tenant_code == "provisioned")


def set_executor_policy(config_path: Path, policy: str, *, auto_acquire: bool | None = None) -> str:
    """Persist `[executor] execute_tenant_code` (+ optional `auto_acquire`) to the
    worker.toml at `config_path`, preserving the rest of the file. Validates the
    policy (raises ValueError on an unknown value). Shared by the CLI `executor
    set` and the dashboard setter — the single owner-consent write path. Returns the
    normalized policy. The daemon hot-reloads it (per-heartbeat + per-dispatch), so
    it takes effect within one heartbeat — no restart needed."""
    policy = _validate_policy(policy)
    updates = {"execute_tenant_code": f'"{policy}"'}
    if auto_acquire is not None:
        updates["auto_acquire"] = "true" if auto_acquire else "false"
    _upsert_toml_section(config_path, "executor", updates)
    return policy


def set_auto_acquire(config_path: Path, enabled: bool) -> bool:
    """Persist ONLY `[executor] auto_acquire` to worker.toml, preserving
    `execute_tenant_code` and the rest of the file. The onramp's auto-acquire
    consent and the dashboard toggle write through here. Hot-reloaded per
    heartbeat + per dispatch (like `set_executor_policy`) — no restart. Effective
    only under `execute_tenant_code = "provisioned"` (the daemon folds that rule
    in via `effective_auto_acquire`)."""
    _upsert_toml_section(config_path, "executor", {"auto_acquire": "true" if enabled else "false"})
    return enabled


def set_worker_flavor(config_path: Path, name: str) -> str:
    """Persist `[worker] flavor` to worker.toml (§9 #46). Written by the
    onramp's apply_flavor step (and `flavor set`) so upgrades can preserve the
    volunteer's chosen profile. Validates by shape; returns the normalized
    name."""
    flavor = _validate_flavor(name)
    assert flavor is not None
    _upsert_toml_section(config_path, "worker", {"flavor": f'"{flavor}"'})
    return flavor


def set_inference_backend(config_path: Path, backend: str) -> str:
    """Persist `[inference] backend` to worker.toml. The flavor choice at the
    onramp IS the consent for `ollama`. NOT hot-reloaded — the daemon
    instantiates the backend at start, so a change needs a daemon restart
    (callers print that)."""
    backend = _validate_inference_backend(backend)
    _upsert_toml_section(config_path, "inference", {"backend": f'"{backend}"'})
    return backend


def set_sandbox_policy(config_path: Path, policy: str) -> str:
    """Persist `[sandbox] policy` to worker.toml (permissive|strict) — the
    volunteer's host-isolation choice for running tenant code (§41). NOT
    hot-reloaded: the daemon reads it at start, so a change needs a daemon
    restart (callers print that)."""
    policy = _validate_sandbox_policy(policy)
    _upsert_toml_section(config_path, "sandbox", {"policy": f'"{policy}"'})
    return policy
