"""Stdlib-only capability detection for the worker.

Returns the JSON-serializable payload the worker sends to the coordinator on
every heartbeat. Detection is intentionally cheap so calling on every tick
(default 60 s) is fine.

**GPU probe robustness (Q-W2 resolution):** the worker treats device-file
*existence* as a hint, not as authority. A device file that doesn't respond
to `open()` — stale node from a partial driver uninstall, container
bind-mount without a working CUDA runtime, race during boot, kernel module
unloaded at runtime — is excluded from the observed count. The volunteer's
declared GPU hardware in `[capabilities.gpus]` is the routing-relevant
signal (per §5.8 BYOM); the observed-probe travels alongside as
corroboration / mismatch diagnostic, not as the authoritative inventory.
"""

from __future__ import annotations

import glob
import os
import platform
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# v0.2 M1: the software features this worker BUILD supports (a static property
# of the code, not the host). The scheduler matches an experiment's derived
# `features` requirement against this list, so mixed-version fleets route
# feature-gated work only to workers that can honor it.
WORKER_FEATURES = ("generation_policy",)


@dataclass(frozen=True)
class GpuObservation:
    """What the worker *observed* by probing device files. Not authoritative
    — only the volunteer (via `[capabilities.gpus]` config / `GpuDeclaration`)
    is authoritative about what hardware is actually usable."""

    nvidia: int  # /dev/nvidia[0-9]* device files that responded to open()
    amd: bool  # /dev/kfd exists and responded to open()

    def has_any(self) -> bool:
        return self.nvidia > 0 or self.amd


@dataclass(frozen=True)
class GpuDeclaration:
    """Volunteer-declared GPU hardware from `[capabilities.gpus]` config.

    All fields are optional — a worker that wants the coordinator's
    scheduler to consider it for GPU work must declare at least the count
    and VRAM total of its accelerators. Authoritative VRAM/model
    auto-detection (via `nvidia-smi` shell-out or NVML) is deferred to a
    later milestone.
    """

    nvidia: int | None = None
    nvidia_model: str | None = None
    vram_total_gb: float | None = None
    amd: bool | None = None
    amd_model: str | None = None

    def is_empty(self) -> bool:
        return all(
            v is None
            for v in (
                self.nvidia,
                self.nvidia_model,
                self.vram_total_gb,
                self.amd,
                self.amd_model,
            )
        )


@dataclass(frozen=True)
class DeclaredCaps:
    """Volunteer-declared resource caps from `[resources]`. Coordinator
    scheduler may use these for capability-matched scheduling; the worker
    enforces them locally in M4 (sandbox)."""

    max_ram_gb: float | None = None
    max_vram_gb: float | None = None
    max_cpu_cores: int | None = None
    network_quota_mb_per_hour: int | None = None


@dataclass(frozen=True)
class Capabilities:
    """Full payload sent in heartbeat.capabilities."""

    os: str
    arch: str
    python_version: str
    ram_total_gb: float | None
    cpu_count: int | None
    gpus_observed: GpuObservation
    gpus_declared: GpuDeclaration = field(default_factory=GpuDeclaration)
    declared_caps: DeclaredCaps = field(default_factory=DeclaredCaps)
    # Running worker code version (hatch-vcs). Always reported so the operator
    # can tell which workers support which features across a mixed-version
    # fleet (who has §9 #37 executor dispatch / W-M / W-H). Part of the
    # version-surfacing epic; rides the opaque capabilities channel.
    worker_version: str | None = None
    # Locally-available model ids (the BYOM store inventory, W-M). The §5.8
    # capability the scheduler will route on (#30). Omitted from the wire when
    # empty. The coordinator stores capabilities as an opaque dict, so this is
    # forward-compatible — it consumes `models` only once #30 lands.
    models: list[str] = field(default_factory=list)
    # W-S (§9 #43): model ids currently LOADED in the inference backend (not
    # merely present on disk). Sharpens the routing predicate from "holds the
    # model" (`models`) to "holds it AND has it serve-ready" — the scheduler
    # routes inference experiments to serve-ready workers once it consumes
    # this. Omitted from the wire when empty (incl. `[inference] backend =
    # "none"` workers — absent == not an inference host).
    served_models: list[str] = field(default_factory=list)
    # v0_2 #13a: {model_id: served-GGUF sha256} for the LOADED models — the
    # coordinator-asserted served-weights digest. When a manifest pins
    # `expected_gguf_sha256`, the coordinator REJECTS a result whose served
    # digest differs (§9 #13b: the declared model provably ran). Omitted from
    # the wire when empty (absent == no served weights to attest).
    served_model_digests: dict[str, str] = field(default_factory=dict)
    # Fleet-fit: {model_id: on-disk bytes} for the BYOM store, so the coordinator
    # can size a PRESENT model directly (mark it available only if it FITS this
    # worker's RAM, too_big if not) instead of only sizing models it happens to
    # have in its HF catalog. Presence alone never implies runnable — this closes
    # the gap where a small locally-run model (or a huge stranded one) had no size.
    # Omitted from the wire when empty.
    model_sizes: dict[str, int] = field(default_factory=dict)
    # Fleet-fit: the memory actually AVAILABLE to load a model (`ram/vram_total -
    # OS/runtime headroom`), i.e. the exact budget this worker's serve-time guard
    # uses. The coordinator must gate routing on THIS, not raw `ram_total_gb` — on a
    # unified 8 GB box the ~2 GB headroom means an 8B-q4 model fits raw RAM but NOT
    # the serve budget, so a raw-RAM gate would route a model the worker then
    # refuses. None when the accelerator budget is unknown. Omitted when None.
    usable_memory_gb: float | None = None
    # In-flight model downloads (D12 5c): {model_id: {bytes_downloaded, total_bytes}}
    # while the worker is auto-acquiring a model. Lets the coordinator + operator UI
    # show "provisioning: downloading <model> NN%" instead of a silent "provisioned"
    # for the minutes a multi-GB pull takes. Omitted from the wire when nothing is
    # downloading (the common case).
    downloads: dict[str, dict] = field(default_factory=dict)
    # Current thermal/health snapshot (W-H), or None where no sensor exists.
    # Lets the coordinator route work away from a degraded/overheating worker
    # (forward-compatible; opaque until consumed).
    thermal: dict[str, Any] | None = None
    # M3 lazy auto-acquire: when True, the worker will pull a missing
    # locally-required model on assignment (from the manifest's hf_repo/
    # hf_filename) rather than refusing. The coordinator's #30 capability
    # matcher reads this so it may route a model-gated unit here even when the
    # model isn't in `models` yet. Omitted from the wire when False (the matcher
    # checks `is True`, so absent == disabled).
    auto_acquire: bool = False
    # §2.1 #11: the volunteer self-paused this worker (owner hold). The
    # coordinator routes around it (like a degraded/paused worker). Omitted from
    # the wire when False (the matcher checks `is True`).
    self_paused: bool = False
    # M9 leg 4: the owner's code-execution consent mode (synthetic/provisioned/off).
    # ALWAYS sent (short + informative): the scheduler routes real (model-gated)
    # experiments only to `provisioned` workers, and the operator console surfaces
    # the mode. Absent (an older worker) is read coordinator-side as not-provisioned
    # (conservative — no real work routed until the worker affirmatively declares).
    execute_tenant_code: str = "synthetic"
    # §41: the sandbox isolation this worker runs tenant code under (permissive |
    # strict). ALWAYS sent so the coordinator can route tenant code to workers
    # that meet the tier-derived containment floor and record which containment
    # produced the evidence (firewall #2). Absent (an older worker) reads
    # coordinator-side as permissive — fail-safe (excluded from strict-required work).
    sandbox_policy: str = "permissive"
    # §9 #46: the install profile the onramp applied (lean/inference/full/...).
    # Fleet bookkeeping for the console — NOT a routing key (scheduling stays
    # capability-based). Omitted when unrecorded (pre-flavor installs).
    flavor: str | None = None
    # §9 #46/W-S determinism provenance: the serving Ollama's version (probed
    # once at daemon start when [inference] backend="ollama"). The runtime
    # version affects inference outputs, so consensus debugging wants it
    # visible fleet-wide. Omitted when not serving / probe failed.
    ollama_version: str | None = None
    # v0.2 M1: software features THIS worker build supports, for mixed-fleet
    # routing (a volunteer fleet never rolls atomically). `generation_policy` =
    # the broker honors a manifest-declared seeded-sampling policy; the
    # scheduler routes a sampling experiment only to declaring workers — a
    # pre-M1 worker would burn its units with params_rejected at request time.
    # A declared FEATURE, not a heuristic (declarative-enforcement hygiene).
    worker_features: list[str] = field(default_factory=lambda: list(WORKER_FEATURES))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable shape.

        - `gpus_observed` is always included (operators rely on its presence
          for fleet-view diagnostics).
        - `gpus_declared` is omitted entirely when the volunteer hasn't
          declared anything, to keep the wire payload compact.
        - `declared_caps` is omitted when no caps are set.
        """
        d = asdict(self)
        d["gpus_observed"] = asdict(self.gpus_observed)
        if self.gpus_declared.is_empty():
            d.pop("gpus_declared", None)
        else:
            d["gpus_declared"] = {
                k: v for k, v in asdict(self.gpus_declared).items() if v is not None
            }
        declared = {k: v for k, v in asdict(self.declared_caps).items() if v is not None}
        if declared:
            d["declared_caps"] = declared
        else:
            d.pop("declared_caps", None)
        if not self.models:
            d.pop("models", None)  # compact wire when the store is empty
        if not self.served_models:
            d.pop("served_models", None)  # absent == not an inference host
        if not self.served_model_digests:
            d.pop("served_model_digests", None)  # absent == no served weights to attest
        if not self.model_sizes:
            d.pop("model_sizes", None)  # compact wire when the store is empty
        if self.usable_memory_gb is None:
            d.pop("usable_memory_gb", None)  # absent == budget unknown
        if not self.downloads:
            d.pop("downloads", None)  # compact wire when nothing is downloading
        if self.thermal is None:
            d.pop("thermal", None)  # omit where no sensor / health disabled
        if self.worker_version is None:
            d.pop("worker_version", None)
        if not self.auto_acquire:
            d.pop("auto_acquire", None)  # compact wire; absent == disabled
        if not self.self_paused:
            d.pop("self_paused", None)  # compact wire; absent == not self-paused
        if self.flavor is None:
            d.pop("flavor", None)  # absent == pre-flavor install
        if self.ollama_version is None:
            d.pop("ollama_version", None)  # absent == not serving / probe failed
        return d


# ---- probes ----------------------------------------------------------------


def detect_ram_total_gb(*, meminfo_path: Path | None = None) -> float | None:
    """Total physical RAM in GB (binary), cross-platform.

    Linux reads `/proc/meminfo` `MemTotal`; other Unixes fall back to
    `os.sysconf` (where SC_PHYS_PAGES is defined) and then, on Darwin/BSD,
    `sysctl hw.memsize`. Returns None if nothing resolves — the coordinator's
    supported-models overlay treats None as 'unknown capacity', never 'too
    small', so a null is always safe.

    When `meminfo_path` is passed explicitly (test mode) ONLY the meminfo parse
    runs — no host fallbacks — so a bogus/empty path deterministically yields None.
    """
    path = meminfo_path or Path("/proc/meminfo")
    try:
        content = path.read_text(encoding="ascii", errors="replace")
        for line in content.splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                # Standard form: "MemTotal:   12345678 kB"
                if len(parts) >= 2 and parts[1].isdigit():
                    return round(int(parts[1]) / (1024 * 1024), 2)
    except (FileNotFoundError, PermissionError, OSError):
        pass

    if meminfo_path is not None:
        return None  # explicit path given — don't fall back to the real host

    # Non-Linux fallbacks. Each is best-effort; ANY failure returns None (the
    # prior behavior), so this can never regress a host that already reported.
    try:
        if "SC_PHYS_PAGES" in os.sysconf_names and "SC_PAGE_SIZE" in os.sysconf_names:
            total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            if total > 0:
                return round(total / (1024**3), 2)
    except (ValueError, OSError):
        pass
    try:
        import subprocess

        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout.strip()
        if out.isdigit() and int(out) > 0:
            return round(int(out) / (1024**3), 2)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def detect_cpu_count() -> int | None:
    return os.cpu_count()


def _device_responsive(path: Path) -> bool:
    """Open the device file non-blocking; close immediately on success.

    Stale `/dev/nvidiaN` nodes (driver unloaded, partial uninstall,
    container bind-mount with no runtime, boot race) return ENXIO / ENODEV
    on open. `EACCES` (permission denied) is also treated as "not
    available" — if this worker user can't open the device, it can't
    use the GPU even if one exists.

    `O_NONBLOCK` is used so this can't hang on character devices with
    weird semantics.
    """
    try:
        fd = os.open(str(path), os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return False
    try:
        os.close(fd)
    except OSError:
        pass
    return True


GpuProbe = Callable[[Path], bool]


def detect_gpus(
    *,
    sysroot: Path | None = None,
    probe: GpuProbe = _device_responsive,
) -> GpuObservation:
    """Probe device files for GPU presence, with `open()` validation.

    Args:
        sysroot: Override for testing. Probes look for files under
            `sysroot / "dev" / "nvidiaN"` etc. instead of the real `/dev/`.
        probe: Injectable for tests. Returns True iff the device file is
            actually openable. Default validates via `os.open()`.
    """
    if sysroot is None:
        nvidia_glob = "/dev/nvidia[0-9]*"
        kfd_path = Path("/dev/kfd")
    else:
        nvidia_glob = str(sysroot / "dev" / "nvidia[0-9]*")
        kfd_path = sysroot / "dev" / "kfd"
    candidates = [Path(p) for p in glob.glob(nvidia_glob) if Path(p).name != "nvidiactl"]
    nvidia_responsive = sum(1 for path in candidates if probe(path))
    amd = kfd_path.exists() and probe(kfd_path)
    return GpuObservation(nvidia=nvidia_responsive, amd=amd)


# ---- top-level collect -----------------------------------------------------


def collect(
    *,
    declared_caps: DeclaredCaps | None = None,
    declared_gpus: GpuDeclaration | None = None,
    sysroot: Path | None = None,
    meminfo_path: Path | None = None,
    probe: GpuProbe = _device_responsive,
    # Locally-available model ids (BYOM store inventory). Caller-supplied — the
    # collector doesn't read the store itself (keeps detection store-agnostic).
    models: list[str] | None = None,
    # {model_id: on-disk bytes} for the same store (caller-supplied) — lets the
    # coordinator size a present model directly for the fleet-fit classification.
    model_sizes: dict[str, int] | None = None,
    # The worker's usable load budget (caller-supplied from the accelerator) — the
    # coordinator gates routing on this, not raw ram_total, to match serve-time fit.
    usable_memory_gb: float | None = None,
    # In-flight model downloads (caller-supplied from download_progress.snapshot()).
    downloads: dict[str, dict] | None = None,
    # W-S: model ids loaded in the inference backend (caller-supplied from the
    # daemon's ModelServer; empty/None on non-inference hosts).
    served_models: list[str] | None = None,
    # v0_2 #13a: {model_id: served-GGUF sha256} (caller-supplied from the
    # ModelServer). Feeds the coordinator's served-weights enforcement (#13b).
    served_model_digests: dict[str, str] | None = None,
    # Current thermal snapshot (W-H), caller-supplied (the daemon owns the
    # stateful monitor so hysteresis is shared with the dispatch gate).
    thermal: dict[str, Any] | None = None,
    # M3 lazy auto-acquire opt-in (from [executor] auto_acquire). Caller-supplied
    # so the collector stays config-agnostic.
    auto_acquire: bool = False,
    # §2.1 #11 volunteer self-pause flag (caller-supplied from local state).
    self_paused: bool = False,
    # M9 leg 4: the owner's code-execution consent mode (config.execute_tenant_code:
    # synthetic / provisioned / off). Declared so the coordinator can route real
    # (model-gated) experiments only to provisioned-mode workers — a synthetic
    # worker echoes, which would pollute consensus. Caller-supplied from config.
    execute_tenant_code: str = "synthetic",
    # §41: the sandbox isolation policy (caller-supplied from config.sandbox_policy:
    # permissive / strict). Lets the coordinator enforce the containment floor.
    sandbox_policy: str = "permissive",
    # §9 #46: the onramp-applied install profile (caller-supplied from config;
    # bookkeeping, not routing).
    flavor: str | None = None,
    # §9 #46/W-S determinism provenance: serving Ollama version (caller-supplied
    # from the daemon's start-time probe; None when not serving).
    ollama_version: str | None = None,
    # Back-compat alias kept for callers that still pass `declared=...`.
    declared: DeclaredCaps | None = None,
) -> Capabilities:
    """Top-level capability snapshot. Cheap; safe to call on every heartbeat."""
    resolved_caps = declared_caps if declared_caps is not None else (declared or DeclaredCaps())
    # Lazy import avoids any package-load cycle (detect is imported early).
    from auspexai_worker import __version__ as worker_version

    return Capabilities(
        os=platform.system().lower(),
        arch=platform.machine().lower(),
        python_version=platform.python_version(),
        ram_total_gb=detect_ram_total_gb(meminfo_path=meminfo_path),
        cpu_count=detect_cpu_count(),
        gpus_observed=detect_gpus(sysroot=sysroot, probe=probe),
        gpus_declared=declared_gpus or GpuDeclaration(),
        declared_caps=resolved_caps,
        models=models or [],
        model_sizes=model_sizes or {},
        usable_memory_gb=usable_memory_gb,
        downloads=downloads or {},
        served_models=served_models or [],
        served_model_digests=served_model_digests or {},
        thermal=thermal,
        worker_version=worker_version,
        auto_acquire=auto_acquire,
        self_paused=self_paused,
        execute_tenant_code=execute_tenant_code,
        sandbox_policy=sandbox_policy,
        flavor=flavor,
        ollama_version=ollama_version,
    )
