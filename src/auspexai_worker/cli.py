"""Click CLI for `auspexai-worker`.

M1 ships two commands: `status` (read-only) and `bootstrap` (first-run
enrollment). The daemon entry point exists as a placeholder; the actual
heartbeat / assignment loop arrives in M2 / M3.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import signal
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import click

from . import __version__
from .accelerator import detect_accelerator
from .bootstrap import bootstrap as bootstrap_worker
from .bootstrap import build_signer, initialize_state, open_keystore
from .capabilities import DeclaredCaps
from .capabilities import collect as collect_capabilities
from .config import (
    WorkerConfig,
    default_worker_toml_path,
    read_executor_policy,
    set_auto_acquire,
    set_executor_policy,
)
from .coordinator import (
    BindingTokenConsumedError,
    BindingTokenExpiredError,
    BindingTokenNotFoundError,
    CoordinatorClient,
    CoordinatorError,
    InvalidAccessTokenError,
    PubkeyAlreadyEnrolledError,
    PubkeyAlreadyTenantError,
    UnsupportedIdpError,
    WorkerNotFoundError,
)
from .daemon import AssignmentPoller, HeartbeatLoop, PrestageLoop
from .daemon.dispatch import RunnerDispatcher
from .health import ThermalMonitor
from .keystore import KeystoreError
from .models import ModelCatalog, ModelStore, recommend, survey_resources
from .models.catalog import FileCatalogSource
from .models.fetch import HfHubFetcher, ModelFetchError, StoreModelAcquirer, pull_quant
from .models.hf_browse import HfHubBrowser, quant_fits, runnable_models, usable_budget_gb
from .models.recommend import parse_selection
from .oauth import (
    AccessDeniedError,
    DeviceCode,
    DeviceFlowError,
    ExpiredTokenError,
    run_device_flow,
)
from .provisioning import AutoFetchResolver, ExecutePolicy, ProvisioningResolver
from .sandbox import ResourceLimits, SandboxPolicy, probe_bubblewrap, probe_seatbelt
from .state import (
    AcceptedSensitiveRepository,
    AssignmentAuditRepository,
    ManifestPinRepository,
    PendingSubmissionRepository,
    SubmittedResultRepository,
    TenantListRepository,
)
from .workspace import WorkspaceManager, workspace_runs_dir


class CoordinatorPackageFetcher:
    """`provisioning.PackageFetcher` over the worker's signed CoordinatorClient
    (#40a executor-package auto-fetch). Pure adapter: failures (network, 404)
    propagate and `install_fetched_package` classifies them as
    package_unavailable refusals."""

    def __init__(self, client: CoordinatorClient) -> None:
        self._client = client

    def fetch(self, manifest_sha256: str) -> bytes:
        return self._client.fetch_package(digest=manifest_sha256)


@click.group(help="AuspexAI volunteer worker.")
@click.version_option(version=__version__, prog_name="auspexai-worker")
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Explicit path to a worker.toml; overrides /etc and XDG search.",
)
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None) -> None:
    ctx.ensure_object(dict)
    ctx.obj["config"] = WorkerConfig.load(config_path=config_path)
    ctx.obj["config_path"] = config_path  # raw --config (for `executor set` to write)


@cli.command(help="Show worker identity, tier, progress, and configured coordinator URL.")
@click.pass_context
def status(ctx: click.Context) -> None:
    config: WorkerConfig = ctx.obj["config"]
    db, repo = initialize_state(config)
    try:
        worker = repo.get()
        if worker is not None:
            progress = SubmittedResultRepository(db).progress_summary()
        else:
            progress = {"completed_units": 0, "distinct_experiments": 0}
    finally:
        db.close()

    click.echo(f"coordinator: {config.coordinator_url}")
    click.echo(f"state-dir:   {config.state_dir}")
    if worker is None:
        click.echo("identity:    not enrolled")
        click.echo("")
        click.echo("Run `auspexai-worker bootstrap` to enroll as T0 anonymous.")
        return
    from datetime import UTC, datetime

    from auspexai_worker.worker_state import derive_self_state

    _tm = ThermalMonitor(
        warn_c=config.thermal_warn_c,
        crit_c=config.thermal_crit_c,
        resume_c=config.thermal_resume_c,
    )
    _snap = _tm.snapshot() if _tm.enabled else None
    _state = derive_self_state(
        worker,
        thermal_critical=(_snap is not None and _snap.state.value == "critical"),
        now=datetime.now(UTC),
    )
    click.echo(f"worker-id:   {worker.worker_id}")
    click.echo(f"state:       {_state.label}")
    click.echo(f"tier:        T{worker.trust_tier}")
    if config.flavor:
        click.echo(f"flavor:      {config.flavor}")
    click.echo(f"pubkey:      {worker.pubkey_hex[:16]}… ({worker.pubkey_hex})")
    click.echo(f"enrolled-at: {worker.enrolled_at.isoformat()}")
    if worker.last_heartbeat_at is not None:
        click.echo(f"last-beat:   {worker.last_heartbeat_at.isoformat()}")
    click.echo(
        f"progress:    {progress['completed_units']} units completed "
        f"across {progress['distinct_experiments']} experiments"
    )
    if _snap is not None:
        click.echo(f"thermal:     {_snap.current_temp_c}°C ({_snap.state.value})")
    # §2.1 #11: surface the holds — the volunteer's own self-pause and the
    # operator's pause/quarantine (with the operator's reason, cached from the
    # last assignment poll).
    if worker.self_paused:
        click.echo("self-paused: yes — run `auspexai-worker unpause` to resume")
    if worker.operator_hold_kind == "pause":
        click.echo(
            f"operator hold: PAUSED by operator (no-fault) "
            f"— reason: {worker.operator_hold_reason or '<none given>'}"
        )
    elif worker.operator_hold_kind == "quarantine":
        click.echo(
            f"operator hold: QUARANTINED by operator "
            f"— reason: {worker.operator_hold_reason or '<none given>'}"
        )
    if (
        worker.trust_tier == 0
        and config.upgrade_prompt_enabled
        and progress["completed_units"] >= config.upgrade_prompt_threshold
    ):
        click.echo("")
        click.echo(
            "You've contributed enough to build a portable track record. "
            "Run `auspexai-worker login` to claim your contributions."
        )
    # §9 #46: surface the coordinator's release announcement when it's newer
    # than this worker. Informational — upgrading is always YOUR election.
    from auspexai_worker import __version__ as _version
    from auspexai_worker.updates import is_newer_version, upgrade_command

    if worker.latest_release_version and is_newer_version(worker.latest_release_version, _version):
        click.echo("")
        click.echo(
            f"update available: v{worker.latest_release_version}"
            + (f" — {worker.latest_release_notes}" if worker.latest_release_notes else "")
        )
        if worker.latest_release_url:
            click.echo(f"  release notes: {worker.latest_release_url}")
        click.echo("  to upgrade (your choice — updates are never automatic):")
        click.echo(f"    {upgrade_command(config.flavor)}")


@cli.command(help="Self-pause this worker: keep enrolled + your tier, but stop receiving work.")
@click.pass_context
def pause(ctx: click.Context) -> None:
    """Volunteer self-pause (§2.1 #11) — a no-fault, owner-controlled hold. The
    daemon keeps heartbeating (you stay enrolled, your tier is preserved) but the
    coordinator stops sending work until you `unpause`. A softer alternative to
    `retire` (withdrawal). Takes effect within one heartbeat of a running daemon."""
    config: WorkerConfig = ctx.obj["config"]
    db, repo = initialize_state(config)
    try:
        if repo.get() is None:
            click.echo("not enrolled; nothing to pause", err=True)
            sys.exit(1)
        repo.set_self_pause(True)
    finally:
        db.close()
    click.echo("worker self-paused — the coordinator will stop sending it work.")
    click.echo("Run `auspexai-worker unpause` to resume.")


@cli.command(help="Resume this worker after a self-pause.")
@click.pass_context
def unpause(ctx: click.Context) -> None:
    config: WorkerConfig = ctx.obj["config"]
    db, repo = initialize_state(config)
    try:
        if repo.get() is None:
            click.echo("not enrolled; nothing to unpause", err=True)
            sys.exit(1)
        repo.set_self_pause(False)
    finally:
        db.close()
    click.echo("worker unpaused — it will resume receiving work within a heartbeat.")


@cli.group("executor", help="View/set the tenant code-execution policy.")
def executor() -> None:
    pass


@executor.command("show", help="Show the current code-execution policy.")
@click.pass_context
def executor_show(ctx: click.Context) -> None:
    config: WorkerConfig = ctx.obj["config"]
    click.echo(f"execute_tenant_code: {config.execute_tenant_code}")
    click.echo(f"auto_acquire:        {config.auto_acquire}")


@executor.command(
    "set", help="Set the code-execution policy (writes worker.toml; hot-reloaded, no restart)."
)
@click.argument("policy", type=click.Choice(["synthetic", "provisioned", "off"]))
@click.option(
    "--auto-acquire/--no-auto-acquire",
    "auto_acquire",
    default=None,
    help="Also set auto_acquire (M3 pull-then-run; only effective under provisioned).",
)
@click.pass_context
def executor_set(ctx: click.Context, policy: str, auto_acquire: bool | None) -> None:
    """Deliberate, owner-driven change to what third-party code this worker runs.
    Writes the `[executor]` block of worker.toml in place (preserving the rest of
    the file). A running daemon **hot-reloads** the policy — per dispatch (execution)
    and per heartbeat (the coordinator-facing capability) — so the change takes
    effect within one heartbeat, **no restart needed**. Also available on the
    localhost dashboard (M9 leg 4), where enabling `provisioned` is gated behind a
    confirm step."""
    target = ctx.obj.get("config_path") or default_worker_toml_path()
    try:
        set_executor_policy(target, policy, auto_acquire=auto_acquire)
    except OSError as e:
        click.echo(f"ERROR: could not write {target}: {e}", err=True)
        sys.exit(1)
    click.echo(
        f"set [executor] execute_tenant_code = {policy}"
        + (f", auto_acquire = {auto_acquire}" if auto_acquire is not None else "")
    )
    if policy == "provisioned":
        click.echo(
            "note: provisioned runs ONLY operator-staged executors whose hash matches "
            "the coordinator's manifest_sha256 (refuse-don't-echo otherwise)."
        )
    click.echo("a running daemon picks this up within one heartbeat — no restart needed.")


@executor.command(
    "auto-acquire",
    help="Allow/deny on-demand model downloads (writes worker.toml; hot-reloaded).",
)
@click.argument("setting", type=click.Choice(["on", "off"]))
@click.pass_context
def executor_auto_acquire(ctx: click.Context, setting: str) -> None:
    """The volunteer's consent for pulling models on demand: when ON and the
    policy is `provisioned`, a unit whose required model this worker lacks
    triggers an in-line download of that exact model (M3). Writes ONLY
    `[executor] auto_acquire` — the execution policy is left untouched.
    Hot-reloaded per heartbeat (no restart). Also set at the onramp (installer)
    and the localhost dashboard."""
    target = ctx.obj.get("config_path") or default_worker_toml_path()
    enabled = setting == "on"
    try:
        set_auto_acquire(target, enabled)
    except OSError as e:
        click.echo(f"ERROR: could not write {target}: {e}", err=True)
        sys.exit(1)
    click.echo(f"set [executor] auto_acquire = {enabled}")
    click.echo("a running daemon picks this up within one heartbeat — no restart needed.")


@cli.group("flavor", help="View/record this worker's install profile (§9 #46).")
def flavor() -> None:
    pass


@flavor.command("show", help="Show the recorded install flavor.")
@click.option("--raw", is_flag=True, help="Print the bare flavor token (for scripting).")
@click.pass_context
def flavor_show(ctx: click.Context, raw: bool) -> None:
    config: WorkerConfig = ctx.obj["config"]
    if raw:
        click.echo(config.flavor or "lean")
        return
    if config.flavor:
        click.echo(f"flavor: {config.flavor}")
    else:
        click.echo("flavor: lean (default — not recorded; pre-flavor install)")


@flavor.command(
    "set", help="Record the install flavor in worker.toml (normally written by the onramp)."
)
@click.argument("name")
@click.pass_context
def flavor_set(ctx: click.Context, name: str) -> None:
    """Bookkeeping only — recording a flavor does NOT install anything; the
    onramp's apply_flavor step does the installs. Recorded so upgrades preserve
    the volunteer's chosen profile and the fleet view can show it."""
    from .config import set_worker_flavor

    target = ctx.obj.get("config_path") or default_worker_toml_path()
    try:
        normalized = set_worker_flavor(target, name)
    except ValueError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)
    except OSError as e:
        click.echo(f"ERROR: could not write {target}: {e}", err=True)
        sys.exit(1)
    click.echo(f"set [worker] flavor = {normalized}")


@cli.group("inference", help="View/set the inference-serving backend (W-S, §9 #43).")
def inference() -> None:
    pass


@inference.command("show", help="Show the inference backend configuration.")
@click.pass_context
def inference_show(ctx: click.Context) -> None:
    config: WorkerConfig = ctx.obj["config"]
    click.echo(f"backend:    {config.inference_backend}")
    if config.inference_backend == "ollama":
        click.echo(f"ollama_url: {config.inference_ollama_url}")


@inference.command(
    "set-backend",
    help="Set [inference] backend in worker.toml (none|ollama). Needs a daemon restart.",
)
@click.argument("backend", type=click.Choice(["none", "ollama"]))
@click.pass_context
def inference_set_backend(ctx: click.Context, backend: str) -> None:
    """The owner's opt-in to serving local models to sandboxed executors.
    Written by the onramp's inference flavor; also available here directly.
    NOT hot-reloaded: the daemon builds the backend at start, so restart it
    (`systemctl --user restart auspexai-worker` or re-run the daemon)."""
    from .config import set_inference_backend

    target = ctx.obj.get("config_path") or default_worker_toml_path()
    try:
        set_inference_backend(target, backend)
    except OSError as e:
        click.echo(f"ERROR: could not write {target}: {e}", err=True)
        sys.exit(1)
    click.echo(f"set [inference] backend = {backend}")
    click.echo(
        "restart the daemon to apply — [inference] is read at daemon start, not hot-reloaded."
    )
    if backend == "ollama":
        click.echo(
            "note: serving needs Ollama installed + running (the inference-flavor "
            "onramp installs it), and GGUF models in the BYOM store."
        )


@cli.command(help="Tail the daemon log file.")
@click.option("--lines", "-n", type=int, default=50, help="Number of lines to show (default 50).")
@click.option("--follow", "-f", is_flag=True, help="Follow the log in real time (like tail -f).")
@click.pass_context
def logs(ctx: click.Context, lines: int, follow: bool) -> None:
    config: WorkerConfig = ctx.obj["config"]
    log_file = config.state_dir / "daemon.log"
    if not log_file.exists():
        click.echo(f"no log file at {log_file}")
        click.echo("the daemon has not run yet, or state_dir is misconfigured")
        sys.exit(1)
    cmd = ["tail", f"-n{lines}"]
    if follow:
        cmd.append("-f")
    cmd.append(str(log_file))
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass


@cli.group("model", help="Manage local inference models (BYOM model store).")
def model() -> None:
    pass


# M3 disk-exposure: leave a safety margin of free disk untouched when sizing a
# model-acquisition set, so a worker never fills its root pulling weights (the
# mayhem0 lesson — though that hang was a dead connection, not full disk).
_DISK_SAFETY_MARGIN_BYTES = 5_000_000_000


def _load_catalog(catalog_path: str | None) -> ModelCatalog:
    source = FileCatalogSource(Path(catalog_path)) if catalog_path else None
    return ModelCatalog.load(source)


_catalog_opt = click.option(
    "--catalog",
    "catalog_path",
    type=click.Path(dir_okay=False, path_type=str),
    default=None,
    help="Override the model catalog with a local JSON file.",
)


@model.command("list", help="List models in the local store (your declared inventory).")
@click.pass_context
def model_list(ctx: click.Context) -> None:
    store = ModelStore(ctx.obj["config"].models_store_path)
    models = store.list()
    if not models:
        click.echo(f"no models in {store.root}")
        click.echo("run `auspexai-worker model recommend` to see what fits this host.")
        return
    for m in models:
        click.echo(f"{m.id:32} {m.size_bytes / 1e9:6.2f} GB  {m.path}")


def _echo_installed_summary(store: ModelStore) -> None:
    """Surface what's already on this device — preserved across upgrades — so a
    re-run (especially the install/upgrade path) is never mistaken for
    re-downloading. The store persists across upgrades and `pull` is idempotent
    (`store.has()` short-circuits), so anything listed here is NOT re-fetched."""
    installed = store.list()
    if not installed:
        return
    total = sum(m.size_bytes for m in installed)
    click.echo(
        f"already available on this device ({len(installed)} model(s), "
        f"{total / 1e9:.1f} GB — preserved across upgrades, not re-downloaded):"
    )
    for m in installed:
        click.echo(f"  ✓ {m.id:40} {m.size_bytes / 1e9:6.2f} GB")
    click.echo("")


def _recommend_fallback(catalog_path: str | None, store: ModelStore) -> None:
    """Offline degraded view: the bundled catalog (HuggingFace unavailable)."""
    catalog = _load_catalog(catalog_path)
    resources = survey_resources(store.root)
    for r in recommend(catalog, store, resources):
        tag = "installed" if r.installed else ("fits" if r.fits else "too big")
        click.echo(f"[{tag:9}] {r.entry.id:24} {r.entry.disk_bytes / 1e9:5.1f} GB  {r.entry.note}")
        for b in r.blockers:
            click.echo(f"            ✗ {b}")


@model.command("recommend", help="Recommend HuggingFace models that fit this host.")
@click.option("--limit", default=30, help="How many popular HF models to consider.")
@_catalog_opt
@click.pass_context
def model_recommend(ctx: click.Context, limit: int, catalog_path: str | None) -> None:
    config = ctx.obj["config"]
    store = ModelStore(config.models_store_path)
    acc = detect_accelerator()
    disk_free = survey_resources(store.root).disk_free_bytes
    click.echo(f"host: {acc.label}  ·  {disk_free / 1e9:.0f} GB disk free\n")
    _echo_installed_summary(store)
    try:
        runnable = runnable_models(
            HfHubBrowser(),
            memory_budget_gb=acc.memory_budget_gb,
            unified=acc.unified,
            disk_free_bytes=disk_free,
            installed_ids=frozenset(store.inventory()),
            limit=limit,
        )
    except Exception as exc:
        click.echo(f"(HuggingFace unavailable: {exc})")
        click.echo("Bundled offline fallback:\n")
        _recommend_fallback(catalog_path, store)
        return
    if not runnable:
        click.echo("No HuggingFace GGUF text-generation models fit this host's budget.")
        return
    for r in runnable:
        q = r.quant
        tag = "installed" if r.installed else "fits"
        click.echo(f"[{tag:9}] {q.repo:48} {q.quant:10} {q.size_gb:5.1f} GB")
    click.echo("\nInstall one with:  auspexai-worker model pull <repo> --quant <Q>")


@model.command("pull", help="Download a HuggingFace GGUF model into the local store.")
@click.argument("repo")
@click.option(
    "--quant", default=None, help="Quant to pull (e.g. Q4_K_M); default = largest that fits."
)
@click.pass_context
def model_pull(ctx: click.Context, repo: str, quant: str | None) -> None:
    config = ctx.obj["config"]
    store = ModelStore(config.models_store_path)
    acc = detect_accelerator()
    disk_free = survey_resources(store.root).disk_free_bytes
    try:
        quants = HfHubBrowser().quants(repo)
    except Exception as exc:
        click.echo(f"could not query HuggingFace for {repo!r}: {exc}", err=True)
        sys.exit(1)
    if not quants:
        click.echo(f"no GGUF files found in {repo!r}", err=True)
        sys.exit(1)
    if quant:
        chosen = next((q for q in quants if q.quant.lower() == quant.lower()), None)
        if chosen is None:
            avail = ", ".join(sorted({q.quant for q in quants}))
            click.echo(f"quant {quant!r} not in {repo!r}; available: {avail}", err=True)
            sys.exit(1)
    else:
        usable = usable_budget_gb(acc.memory_budget_gb, unified=acc.unified)
        fitting = [q for q in quants if quant_fits(q, usable, disk_free)]
        chosen = (
            max(fitting, key=lambda q: q.size_bytes)
            if fitting
            else min(quants, key=lambda q: q.size_bytes)
        )
    click.echo(f"pulling {chosen.repo} [{chosen.quant}] (~{chosen.size_gb:.1f} GB) …")
    try:
        dest = pull_quant(chosen, store, HfHubFetcher(), disk_free_bytes=disk_free)
    except ModelFetchError as exc:
        click.echo(f"pull failed: {exc}", err=True)
        sys.exit(1)
    click.echo(f"installed {chosen.model_id} -> {dest}")


@model.command("setup", help="Interactively pick + download HF models that fit this host.")
@click.option("--limit", default=30, help="How many popular HF models to consider.")
@click.option(
    "--yes", is_flag=True, help="Non-interactive: pull ALL fitting, not-installed models."
)
@click.pass_context
def model_setup(ctx: click.Context, limit: int, yes: bool) -> None:
    config = ctx.obj["config"]
    store = ModelStore(config.models_store_path)
    acc = detect_accelerator()
    disk_free = survey_resources(store.root).disk_free_bytes
    click.echo(f"host: {acc.label}  ·  {disk_free / 1e9:.0f} GB disk free\n")
    _echo_installed_summary(store)
    try:
        candidates = [
            r
            for r in runnable_models(
                HfHubBrowser(),
                memory_budget_gb=acc.memory_budget_gb,
                unified=acc.unified,
                disk_free_bytes=disk_free,
                installed_ids=frozenset(store.inventory()),
                limit=limit,
            )
            if not r.installed
        ]
    except Exception as exc:
        click.echo(
            f"HuggingFace unavailable ({exc}); cannot set up models. "
            "Install the [models] extra and ensure network access.",
            err=True,
        )
        return
    if not candidates:
        click.echo("No new HF models fit this host (or all fitting ones are installed).")
        return

    click.echo("HuggingFace models that fit this host:\n")
    for i, r in enumerate(candidates, 1):
        q = r.quant
        click.echo(f"  {i}. {q.repo:48} {q.quant:10} {q.size_gb:5.1f} GB")

    if yes:
        chosen = candidates
    elif not sys.stdin.isatty():
        click.echo(
            "\nNon-interactive shell; run `auspexai-worker model pull <repo> --quant <Q>` "
            "(or re-run with --yes to pull all that fit)."
        )
        return
    else:
        sel = click.prompt(
            "\nSelect models to download (comma-separated numbers, 'all', or 'none')",
            default="none",
        )
        chosen = [candidates[i] for i in parse_selection(sel, len(candidates))]

    if not chosen:
        click.echo("Nothing selected.")
        return

    # M3 disk-exposure: never let the selected SET exceed free disk (minus a
    # safety margin) — each quant fits individually, but `setup` pulls a set, and
    # nothing summed across them before. Trim greedily in listed order and report
    # exactly what was dropped (no silent truncation).
    budget = max(0, disk_free - _DISK_SAFETY_MARGIN_BYTES)
    total = sum(r.quant.size_bytes for r in chosen)
    if total > budget:
        kept: list = []
        dropped: list = []
        running = 0
        for r in chosen:
            if running + r.quant.size_bytes <= budget:
                kept.append(r)
                running += r.quant.size_bytes
            else:
                dropped.append(r)
        click.echo(
            f"\n⚠ selected ~{total / 1e9:.1f} GB exceeds ~{budget / 1e9:.1f} GB usable "
            f"({disk_free / 1e9:.1f} GB free - {_DISK_SAFETY_MARGIN_BYTES / 1e9:.0f} GB margin):"
        )
        for r in dropped:
            click.echo(
                f"  skipping {r.quant.repo} [{r.quant.quant}] "
                f"(~{r.quant.size_gb:.1f} GB) - won't fit"
            )
        chosen = kept
        if not chosen:
            click.echo("Nothing fits the disk budget; free space or pick smaller quants.")
            return
        total = running

    if (
        not yes
        and sys.stdin.isatty()
        and not click.confirm(
            f"\nDownload {len(chosen)} model(s), ~{total / 1e9:.1f} GB "
            f"(of ~{budget / 1e9:.1f} GB usable)?",
            default=True,
        )
    ):
        click.echo("Aborted.")
        return

    failures = 0
    remaining = disk_free
    for r in chosen:
        if r.quant.size_bytes > max(0, remaining - _DISK_SAFETY_MARGIN_BYTES):
            click.echo(
                f"  skipping {r.quant.model_id}: ~{r.quant.size_gb:.1f} GB won't fit "
                f"{remaining / 1e9:.1f} GB remaining",
                err=True,
            )
            continue
        click.echo(f"pulling {r.quant.repo} [{r.quant.quant}] …")
        try:
            pull_quant(r.quant, store, HfHubFetcher(), disk_free_bytes=remaining)
            remaining -= r.quant.size_bytes
            click.echo(f"  installed {r.quant.model_id}")
        except ModelFetchError as exc:
            failures += 1
            click.echo(f"  failed: {exc}", err=True)
    if failures:
        sys.exit(1)


@model.command("rm", help="Remove a model from the local store.")
@click.argument("model_id")
@click.pass_context
def model_rm(ctx: click.Context, model_id: str) -> None:
    store = ModelStore(ctx.obj["config"].models_store_path)
    if store.remove(model_id):
        click.echo(f"removed {model_id}")
    else:
        click.echo(f"{model_id} not in store", err=True)
        sys.exit(1)


def _enable_and_start_service() -> None:
    """Enable and start the systemd user service."""
    import shutil
    import subprocess

    systemctl = shutil.which("systemctl")
    if not systemctl:
        click.echo("systemctl not found; enable the service manually.")
        return

    click.echo("enabling auspexai-worker.service …")
    r = subprocess.run(
        [systemctl, "--user", "enable", "--now", "auspexai-worker.service"],
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        click.echo("service started.")
    else:
        click.echo(f"service start failed: {r.stderr.strip()}", err=True)


@cli.command(help="Generate identity and enroll with the coordinator (T0 anonymous).")
@click.option(
    "--start", is_flag=True, help="Enable and start the systemd user service after enrollment."
)
@click.pass_context
def bootstrap(ctx: click.Context, start: bool) -> None:
    config: WorkerConfig = ctx.obj["config"]
    try:
        result = bootstrap_worker(config)
    except KeystoreError as exc:
        click.echo(f"ERROR: keystore initialization failed:\n\n{exc}", err=True)
        sys.exit(2)
    except PubkeyAlreadyTenantError as exc:
        click.echo(
            "ERROR: this worker's public key collides with a registered tenant maintainer.\n"
            "       Recovery: delete the local keystore + worker.db and re-run bootstrap.\n"
            f"       Detail: {exc}",
            err=True,
        )
        sys.exit(2)
    except PubkeyAlreadyEnrolledError as exc:
        click.echo(
            "ERROR: this public key is already enrolled with the coordinator but no\n"
            "       local identity was found. The previous worker's local state may have\n"
            "       been removed without retiring the key. Investigate before re-bootstrapping.\n"
            f"       Detail: {exc}",
            err=True,
        )
        sys.exit(2)
    except CoordinatorError as exc:
        click.echo(f"ERROR: coordinator call failed: {exc}", err=True)
        sys.exit(1)

    worker = result.worker_self
    if result.fresh_enrollment:
        click.echo(f"enrolled: {worker.worker_id} (T{worker.trust_tier})")
        click.echo(f"pubkey:   {worker.pubkey_hex}")
    else:
        click.echo(f"already enrolled: {worker.worker_id} (T{worker.trust_tier})")

    if start:
        _enable_and_start_service()


@cli.command(help="Run the worker daemon (heartbeat loop).")
@click.option(
    "--max-ticks",
    type=int,
    default=None,
    help="Stop after N heartbeats (debug / test only).",
)
@click.option(
    "--verbose",
    is_flag=True,
    envvar="AUSPEXAI_WORKER_VERBOSE",
    help="Restore httpx per-request logs at INFO. Default keeps them at WARNING "
    "so the per-minute heartbeat loop doesn't flood journald.",
)
@click.pass_context
def daemon(ctx: click.Context, max_ticks: int | None, verbose: bool) -> None:
    config: WorkerConfig = ctx.obj["config"]
    log_fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_fmt,
    )

    log_file = config.state_dir / "daemon.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
    )
    file_handler.setFormatter(logging.Formatter(log_fmt))
    file_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(file_handler)

    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)

    db, repo = initialize_state(config)
    worker = repo.get()
    if worker is None:
        click.echo(
            "ERROR: this worker is not enrolled yet. Run `auspexai-worker bootstrap` first.",
            err=True,
        )
        db.close()
        sys.exit(2)

    keystore = open_keystore(config)
    signer = build_signer(keystore)
    if signer.pubkey_hex != worker.pubkey_hex:
        click.echo(
            "ERROR: keystore pubkey does not match the enrolled worker's pubkey.\n"
            f"       keystore: {signer.pubkey_hex}\n"
            f"       enrolled: {worker.pubkey_hex}\n"
            "       The keystore may have been regenerated or the wrong backend selected.",
            err=True,
        )
        db.close()
        sys.exit(2)

    # Sandbox pre-check (Q-W10): when running with bubblewrap, verify it
    # can actually construct a user namespace on this host before we start
    # accepting work. On Ubuntu 24.04 with default AppArmor settings the
    # binary is present but unprivileged userns is blocked; without this
    # check, every assignment would fail with a cryptic per-unit error.
    if config.sandbox_use_bubblewrap:
        probe = probe_bubblewrap()
        if not probe.ok:
            click.echo(
                "ERROR: bubblewrap sandbox is not functional on this host.\n"
                f"       Probe failure: {probe.reason}\n"
                "\n"
                "       Most common cause on Ubuntu 24.04: AppArmor restricts\n"
                "       unprivileged user namespaces. To fix:\n"
                "\n"
                "       1. (Phase 1 / lab) Drop the AppArmor restriction host-wide:\n"
                "            echo 'kernel.apparmor_restrict_unprivileged_userns = 0' \\\n"
                "              | sudo tee /etc/sysctl.d/60-auspexai-userns.conf\n"
                "            sudo sysctl --system\n"
                "\n"
                "       2. (Phase 2 packaging, NOT YET AVAILABLE) An AppArmor\n"
                "          profile scoped to auspexai-worker-runner is the right\n"
                "          long-term fix; ships with the .deb in M7.\n"
                "\n"
                "       3. (last resort, DEGRADES SECURITY) Set\n"
                "          `[sandbox] use_bubblewrap = false` in worker.toml to\n"
                "          run the runner outside the §5.17 sandbox.\n"
                "\n"
                "       See Documentation/AuspexAI/v0.1.0/worker_daemon_design.md\n"
                "       §15 Q-W10 for the full resolution discussion.",
                err=True,
            )
            db.close()
            sys.exit(2)

    # macOS STRICT runs under Seatbelt (sandbox-exec); probe it the same way so a broken
    # sandbox-exec fails CLOSED at startup, not per-unit. (use_bubblewrap is always False
    # on macOS, so the bubblewrap probe above never fires there.)
    if sys.platform == "darwin" and config.sandbox_policy == "strict":
        sb = probe_seatbelt()
        if not sb.ok:
            click.echo(
                "ERROR: macOS STRICT sandbox (sandbox-exec / Seatbelt) is not functional "
                "on this host.\n"
                f"       Probe failure: {sb.reason}\n"
                "\n"
                "       Run `auspexai-worker sandbox set-policy permissive` to run without\n"
                "       strict isolation, or investigate sandbox-exec on this machine.",
                err=True,
            )
            db.close()
            sys.exit(2)

    declared_caps = DeclaredCaps(
        max_ram_gb=config.max_ram_gb,
        max_vram_gb=config.max_vram_gb,
        max_cpu_cores=config.max_cpu_cores,
        network_quota_mb_per_hour=config.network_quota_mb_per_hour,
    )

    manifest_pins = ManifestPinRepository(db)
    accepted_sensitive = AcceptedSensitiveRepository(db)
    tenant_lists = TenantListRepository(db)
    audit = AssignmentAuditRepository(db)
    submitted_results = SubmittedResultRepository(db)
    pending_submissions = PendingSubmissionRepository(db)

    runs_dir = workspace_runs_dir(config.state_dir)
    workspace_manager = WorkspaceManager(runs_dir)
    privkey = keystore.load()

    with CoordinatorClient(base_url=config.coordinator_url, signer=signer) as client:
        # W-H: one shared thermal monitor — its hysteresis is consistent across
        # the dispatch gate (refuse-when-hot) and the heartbeat snapshot.
        thermal_monitor = ThermalMonitor(
            warn_c=config.thermal_warn_c,
            crit_c=config.thermal_crit_c,
            resume_c=config.thermal_resume_c,
        )
        # Hot-reload of the §9 #37 consent gate: re-read the owner's executor
        # policy from disk so a worker.toml change (CLI or dashboard) applies
        # WITHOUT a daemon restart — per heartbeat (the coordinator-facing
        # capability) and per dispatch (the execution gate). Fail safe: a read
        # error → OFF (refuse the unit; it's re-offered) rather than risk running
        # under the wrong policy.
        _cfg_path = ctx.obj.get("config_path")

        def _live_executor() -> tuple[ExecutePolicy, bool]:
            try:
                pol, aa = read_executor_policy(_cfg_path)
                return ExecutePolicy(pol), aa
            except Exception:
                logging.getLogger(__name__).warning(
                    "could not re-read executor policy from disk; refusing this tick"
                )
                return ExecutePolicy.OFF, False

        # W-S (§9 #43): inference serving + per-unit broker — dormant unless
        # the operator opts this worker in via `[inference] backend = "ollama"`.
        # The session provider serves the unit's model out of the BYOM store
        # (eager: loaded + warm before the runner spawns) and opens the broker
        # socket in the unit workspace; dispatch closes it when the unit ends.
        model_server = None
        open_inference_session = None
        _ollama_version: str | None = None
        if config.inference_backend == "ollama":
            from .inference import ModelServer, OllamaBackend, open_unit_session

            _inference_backend = OllamaBackend(
                config.inference_ollama_url,
                ollama_bin=config.inference_ollama_bin,
                keep_alive=config.inference_keep_alive,
            )
            model_server = ModelServer(ModelStore(config.models_store_path), _inference_backend)
            # §9 #46 determinism provenance: probe the serving Ollama's version
            # ONCE at daemon start (no per-tick HTTP); declared in heartbeats.
            _ollama_version = _inference_backend.version()
            # Dog-food guard: the HTTP server can be healthy while the `ollama`
            # CLI (which create_model shells out to) is unresolvable — the macOS
            # launchd minimal-PATH gap. Without this the worker advertises models
            # off the HTTP probe and then refuses EVERY matched unit. Surface it
            # loudly at start, with the remediation, instead of silently looping.
            if _inference_backend.is_healthy() and not _inference_backend.cli_available():
                logging.getLogger(__name__).error(
                    "inference: Ollama HTTP server is reachable but the `ollama` CLI "
                    "could not be found (searched PATH + standard install locations). "
                    "Model creation will fail and every inference unit will be refused. "
                    "Set [inference] ollama_bin to your ollama binary's absolute path "
                    "(e.g. /opt/homebrew/bin/ollama on Apple Silicon)."
                )

            def open_inference_session(model_id: str, socket_dir, policy=None):
                # v0.2 M1: `policy` is the unit's manifest-declared generation
                # policy (dispatch parses + validates it; None ⇒ greedy). The
                # served handle stays policy-neutral — the broker applies the
                # policy per-request.
                served = model_server.serve(model_id)
                return open_unit_session(
                    served=served,
                    backend=_inference_backend,
                    socket_dir=socket_dir,
                    policy=policy,
                )

        dispatcher = RunnerDispatcher(
            coordinator=client,
            worker_id=worker.worker_id,
            worker_pubkey=worker.pubkey_hex,
            privkey=privkey,
            workspace_manager=workspace_manager,
            submitted_repo=submitted_results,
            pending_repo=pending_submissions,
            use_bubblewrap=config.sandbox_use_bubblewrap,
            sandbox_policy=SandboxPolicy(config.sandbox_policy),
            runner_timeout_seconds=config.runner_timeout_seconds,
            # §9 #37: tenant code-execution consent + provisioned-executor
            # resolution. Tenant allow/deny stays the poller's accept-time
            # gate (DB-backed §5.14), so no tenant lists wired here. The static
            # execute_policy/auto_acquire are the daemon-start values; live_executor
            # re-reads them per unit so a policy change applies without a restart.
            execute_policy=ExecutePolicy(config.execute_tenant_code),
            # #40a executor-package auto-fetch: with `[provisioning] auto_fetch`
            # (default ON) a unit whose package digest isn't in the local store
            # is fetched from the coordinator, verified (manifest hash +
            # executor package digest, traversal-safe extraction), and
            # installed content-addressed before running; pre-staged packages
            # short-circuit. `auto_fetch = false` restores staged-only
            # resolution (the pre-#40a behavior).
            executor_resolver=(
                AutoFetchResolver(config.provisioning_path, CoordinatorPackageFetcher(client))
                if config.auto_fetch
                else ProvisioningResolver(config.provisioning_path)
            ),
            model_store_dir=config.models_store_path,
            thermal_monitor=thermal_monitor,
            # M3 lazy auto-acquire: pull a missing locally-required model on
            # assignment (opt-in; only meaningful under `provisioned`).
            auto_acquire=config.auto_acquire,
            # Build the acquirer UNCONDITIONALLY: it only wraps the model store
            # (cheap, no side effects), and decide_execution already gates on the
            # LIVE `auto_acquire` flag. Constructing it only when auto_acquire was
            # true AT STARTUP made `executor set --auto-acquire` a no-op for
            # acquisition until a perfectly-timed restart — #44 surfaced this
            # (caps reported auto_acquire=true but acquirer was None → refuse).
            model_acquirer=StoreModelAcquirer(ModelStore(config.models_store_path)),
            live_executor=_live_executor,
            open_inference_session=open_inference_session,
            # M1 (v0_2): this worker's serving version, for the manifest's
            # serving_version_pin gate. None when not serving (probe failed).
            serving_version=(f"ollama/{_ollama_version}" if _ollama_version else None),
            # §41(a): STRICT resource caps (the "exhaust resources" gate). rlimit
            # floor + cgroup v2 memory/pids when delegated. STRICT-only; generous
            # defaults tunable via [sandbox] in worker.toml.
            resource_limits=ResourceLimits(
                enabled=config.sandbox_resource_limits,
                memory_max_bytes=(
                    config.sandbox_memory_max_mb * 1024 * 1024
                    if config.sandbox_memory_max_mb is not None
                    else None
                ),
                pids_max=config.sandbox_pids_max,
                rlimit_cpu_seconds=config.sandbox_cpu_seconds,
            ),
        )

        def _collect_capabilities():
            # Re-read the executor policy each beat so the coordinator-facing
            # capability tracks the live worker.toml (hot-reload, no restart).
            policy, auto_acquire = _live_executor()
            return collect_capabilities(
                declared_caps=declared_caps,
                declared_gpus=config.declared_gpus,
                # W-M: declare the BYOM store inventory so #30 can route on it.
                models=ModelStore(config.models_store_path).inventory(),
                # W-S: declare what's serve-ready (loaded in the backend) so the
                # scheduler can route inference experiments to warm workers.
                served_models=(model_server.served_ids() if model_server is not None else None),
                # v0_2 #13a: the served-weights digests, for #13b enforcement.
                served_model_digests=(
                    model_server.served_digests() if model_server is not None else None
                ),
                # W-H: report current thermal state so the coordinator can route
                # work away from a degraded/overheating worker.
                thermal=(thermal_monitor.snapshot().to_dict() if thermal_monitor.enabled else None),
                # M3: auto-acquire (already folded to provisioned-only by _live_executor).
                auto_acquire=auto_acquire,
                # §2.1 #11: declare the volunteer self-pause so the coordinator
                # routes around this worker (read fresh from local state each beat).
                self_paused=bool(getattr(repo.get(), "self_paused", False)),
                # M9 leg 4: declare the owner's code-execution consent mode so the
                # coordinator routes real (model-gated) experiments only to
                # provisioned-mode workers (a synthetic worker would echo). Live
                # value (hot-reload) so a policy change reaches the coordinator on
                # the next beat — no restart.
                execute_tenant_code=policy.value,
                # §41: declare the sandbox isolation policy so the coordinator can
                # enforce the containment floor + record what produced the evidence.
                sandbox_policy=config.sandbox_policy,
                # §9 #46: install-profile bookkeeping + serving-runtime provenance.
                flavor=config.flavor,
                ollama_version=_ollama_version,
            )

        heartbeat = HeartbeatLoop(
            coordinator=client,
            repo=repo,
            worker_id=worker.worker_id,
            capability_collector=_collect_capabilities,
            interval_seconds=config.heartbeat_interval_seconds,
        )
        poller = AssignmentPoller(
            coordinator=client,
            worker_id=worker.worker_id,
            manifest_pins=manifest_pins,
            accepted_sensitive=accepted_sensitive,
            tenant_lists=tenant_lists,
            audit=audit,
            interval_seconds=config.assignment_poll_interval_seconds,
            dispatcher=dispatcher,
            worker_self=repo,  # §2.1 #11: self-pause check + operator-hold cache
        )

        # M3b: pre-stage loop — pulls models the conductor directs this worker to
        # acquire ahead of assignment. Own thread (a pull can be slow; must not
        # block heartbeat/poller). Only when the worker opts into auto-acquire
        # under a provisioned policy (same gate as the heartbeat auto_acquire flag).
        prestage: PrestageLoop | None = None
        if config.auto_acquire and config.execute_tenant_code == "provisioned":
            prestage = PrestageLoop(
                coordinator=client,
                worker_id=worker.worker_id,
                acquirer=StoreModelAcquirer(ModelStore(config.models_store_path)),
                interval_seconds=config.heartbeat_interval_seconds * 2,
            )

        # Dashboard server — third thread alongside heartbeat + poller.
        # Localhost-only per §5.14; disabled if config.dashboard_enabled
        # is false OR if --max-ticks is set (treating bounded runs as
        # CI/test mode where the HTTP surface adds noise without value).
        dashboard: DashboardServer | None = None
        if config.dashboard_enabled and max_ticks is None:
            from .dashboard import DashboardServer, build_app

            dashboard_app = build_app(db=db, config=config, config_path=ctx.obj.get("config_path"))
            dashboard = DashboardServer(
                app=dashboard_app,
                host=config.dashboard_host,
                port=config.dashboard_port,
            )
            dashboard.start()

        def _on_signal(signum: int, _frame: object) -> None:
            click.echo(f"received signal {signum}, shutting down", err=True)
            heartbeat.stop()
            poller.stop()
            if prestage is not None:
                prestage.stop()
            if dashboard is not None:
                dashboard.stop()

        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT, _on_signal)

        heartbeat_thread = threading.Thread(
            target=heartbeat.run,
            kwargs={"max_ticks": max_ticks},
            name="auspexai-heartbeat",
            daemon=True,
        )
        poller_thread = threading.Thread(
            target=poller.run,
            kwargs={"max_polls": max_ticks},
            name="auspexai-assignment-poller",
            daemon=True,
        )
        prestage_thread: threading.Thread | None = None
        if prestage is not None:
            prestage_thread = threading.Thread(
                target=prestage.run,
                kwargs={"max_ticks": max_ticks},
                name="auspexai-prestage",
                daemon=True,
            )
        heartbeat_thread.start()
        poller_thread.start()
        if prestage_thread is not None:
            prestage_thread.start()
        heartbeat_thread.join()
        poller_thread.join()
        if prestage_thread is not None:
            prestage_thread.join()

        # For bounded --max-ticks runs the dashboard wasn't started; for
        # unbounded runs, stop it after the worker threads exit so the
        # process doesn't hang on uvicorn's thread.
        if dashboard is not None:
            dashboard.stop()

        hstats = heartbeat.stats
        pstats = poller.stats
    db.close()

    click.echo(
        f"heartbeat ticks=attempted:{hstats.ticks_attempted} "
        f"succeeded:{hstats.ticks_succeeded} failed:{hstats.ticks_failed}",
        err=True,
    )
    click.echo(
        f"assignment polls=attempted:{pstats.polls_attempted} "
        f"succeeded:{pstats.polls_succeeded} failed:{pstats.polls_failed} "
        f"accepted:{pstats.units_accepted} refused:{pstats.units_refused} "
        f"no_work:{pstats.no_work_polls}",
        err=True,
    )
    if pstats.quarantined_at is not None:
        # Maintainer paused this worker; show why (the reason is worker-visible
        # by design) so the volunteer isn't left guessing.
        click.echo(
            f"quarantined:  by maintainer at {pstats.quarantined_at} "
            f"(reason: {pstats.quarantine_reason or '<none given>'})",
            err=True,
        )

    failed_total = hstats.ticks_failed + pstats.polls_failed
    succeeded_total = hstats.ticks_succeeded + pstats.polls_succeeded
    if failed_total > 0:
        # Non-zero exit so systemd flags Restart=on-failure paths correctly,
        # but only when *everything* failed (intermittent failures shouldn't
        # restart-loop the daemon).
        sys.exit(1 if succeeded_total == 0 else 0)


@cli.command(help="List recent assignment-handling decisions (local audit).")
@click.option("--limit", type=int, default=20, help="Show the most recent N rows.")
@click.pass_context
def queue(ctx: click.Context, limit: int) -> None:
    config: WorkerConfig = ctx.obj["config"]
    db, _ = initialize_state(config)
    try:
        audit = AssignmentAuditRepository(db)
        rows = audit.recent(limit=limit)
        if not rows:
            click.echo("no assignment activity yet")
            return
        for row in rows:
            ts = row.occurred_at.isoformat(timespec="seconds")
            unit = row.unit_id or "-"
            tenant = row.tenant_id or "-"
            action = row.action
            reason = row.reason or ""
            line = f"{ts}  {action:<35}  unit={unit}  tenant={tenant}"
            if reason:
                line += f"  ({reason})"
            click.echo(line)
    finally:
        db.close()


@cli.command(help="Inspect the worker's local audit history for a given unit.")
@click.argument("unit_id")
@click.pass_context
def peek(ctx: click.Context, unit_id: str) -> None:
    config: WorkerConfig = ctx.obj["config"]
    db, _ = initialize_state(config)
    try:
        audit = AssignmentAuditRepository(db)
        rows = audit.by_unit(unit_id)
        if not rows:
            click.echo(f"no local record of unit {unit_id!r}")
            click.echo(
                "(M3 worker drops work-unit payloads after the gate decision; "
                "live execution lands in M4)"
            )
            return
        for row in rows:
            click.echo(f"  occurred_at:    {row.occurred_at.isoformat(timespec='seconds')}")
            click.echo(f"  action:         {row.action}")
            click.echo(f"  experiment_id:  {row.coordinator_experiment_id}")
            click.echo(f"  tenant_id:      {row.tenant_id}")
            click.echo(f"  manifest_sha:   {row.manifest_sha256}")
            click.echo(f"  assignment_id:  {row.assignment_id}")
            if row.reason:
                click.echo(f"  reason:         {row.reason}")
            click.echo("")
    finally:
        db.close()


@cli.command(help="Opt in to a sensitive-flagged experiment by coordinator experiment ID.")
@click.argument("coordinator_experiment_id")
@click.pass_context
def accept(ctx: click.Context, coordinator_experiment_id: str) -> None:
    config: WorkerConfig = ctx.obj["config"]
    db, _ = initialize_state(config)
    try:
        AcceptedSensitiveRepository(db).accept(coordinator_experiment_id)
        click.echo(f"accepted: {coordinator_experiment_id}")
        click.echo("future sensitive-flagged assignments under this experiment will be accepted.")
    finally:
        db.close()


@cli.command(help="Manually refuse a unit (local audit only; coordinator re-schedules on timeout).")
@click.argument("unit_id")
@click.option("--reason", default="manual refuse", help="Free-form reason recorded in audit.")
@click.pass_context
def refuse(ctx: click.Context, unit_id: str, reason: str) -> None:
    config: WorkerConfig = ctx.obj["config"]
    db, _ = initialize_state(config)
    try:
        AssignmentAuditRepository(db).append(
            action="refused_manual",
            unit_id=unit_id,
            reason=reason,
        )
        click.echo(f"recorded manual refusal of unit {unit_id}")
        click.echo(
            "(Phase 1 worker has no live refuse endpoint — the coordinator "
            "re-schedules the unit when this worker's assignment times out.)"
        )
    finally:
        db.close()


@cli.command(help="Abort a running unit by signaling its runner subprocess.")
@click.argument("unit_id")
@click.option(
    "--grace-seconds",
    type=float,
    default=5.0,
    help="Wait N seconds after SIGTERM before sending SIGKILL.",
)
@click.pass_context
def abort(ctx: click.Context, unit_id: str, grace_seconds: float) -> None:
    """Send SIGTERM (then SIGKILL after grace_seconds) to the runner
    subprocess executing `unit_id`. Reads the PID from the workspace
    `runner.pid` file. No-op if no workspace / no PID file / process
    already exited; always writes an audit row."""
    import os as _os
    import time as _time

    config: WorkerConfig = ctx.obj["config"]
    runs_dir = workspace_runs_dir(config.state_dir)
    manager = WorkspaceManager(runs_dir)

    db, _ = initialize_state(config)
    audit = AssignmentAuditRepository(db)
    try:
        try:
            workspace = manager.get_existing(unit_id)
        except Exception as exc:
            audit.append(action="abort_no_workspace", unit_id=unit_id, reason=str(exc))
            click.echo(f"no active runner for unit {unit_id} ({exc})")
            return
        pid = workspace.read_pid()
        if pid is None:
            audit.append(action="abort_no_pid", unit_id=unit_id, reason="runner.pid missing")
            click.echo(f"no PID file in workspace for unit {unit_id}")
            return
        try:
            _os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            audit.append(
                action="abort_already_exited",
                unit_id=unit_id,
                reason=f"pid {pid} already exited",
            )
            click.echo(f"pid {pid} already exited")
            return

        deadline = _time.monotonic() + grace_seconds
        while _time.monotonic() < deadline:
            try:
                _os.kill(pid, 0)
            except ProcessLookupError:
                audit.append(
                    action="aborted_sigterm",
                    unit_id=unit_id,
                    reason=f"pid {pid} exited within grace period",
                )
                click.echo(f"sent SIGTERM to pid {pid}; exited cleanly")
                return
            _time.sleep(0.1)

        try:
            _os.kill(pid, signal.SIGKILL)
            audit.append(
                action="aborted_sigkill",
                unit_id=unit_id,
                reason=f"pid {pid} did not exit within {grace_seconds}s of SIGTERM",
            )
            click.echo(f"sent SIGKILL to pid {pid} after {grace_seconds}s grace")
        except ProcessLookupError:
            audit.append(
                action="aborted_sigterm_race",
                unit_id=unit_id,
                reason=f"pid {pid} exited between SIGTERM grace check and SIGKILL",
            )
            click.echo(f"pid {pid} exited just before SIGKILL")
    finally:
        db.close()


@cli.group(help="Sandbox-related operations.")
def sandbox() -> None:
    pass


@sandbox.command("probe", help="Verify bubblewrap can create a user namespace on this host.")
def sandbox_probe() -> None:
    """Standalone bwrap probe.

    Exposes the same probe the daemon runs at startup, as a one-shot
    command suitable for invocation from the .deb postinst or by an
    operator debugging a sandbox failure. Exits 0 on success, 1 on
    failure (with the probe reason on stderr).
    """
    result = probe_bubblewrap()
    if not result.ok:
        click.echo(f"bubblewrap probe FAILED: {result.reason}", err=True)
        sys.exit(1)
    click.echo("bubblewrap probe OK")


@sandbox.command("show", help="Show the configured sandbox policy.")
@click.pass_context
def sandbox_show(ctx: click.Context) -> None:
    config: WorkerConfig = ctx.obj["config"]
    click.echo(f"use_bubblewrap: {config.sandbox_use_bubblewrap}")
    click.echo(f"policy:         {config.sandbox_policy}")


@sandbox.command(
    "set-policy",
    help="Set [sandbox] policy in worker.toml (permissive|strict). Needs a daemon restart.",
)
@click.argument("policy", type=click.Choice(["permissive", "strict"]))
@click.pass_context
def sandbox_set_policy(ctx: click.Context, policy: str) -> None:
    """The volunteer's host-isolation choice for running tenant code (§41).
    strict = narrow filesystem + namespace isolation; permissive = shares the
    host fs (only for fully-trusted setups). Written by the onramp prompt; also
    available here. NOT hot-reloaded — the daemon reads it at start, so restart
    it (`systemctl --user restart auspexai-worker`)."""
    from .config import set_sandbox_policy

    target = ctx.obj.get("config_path") or default_worker_toml_path()
    try:
        set_sandbox_policy(target, policy)
    except ValueError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)
    except OSError as e:
        click.echo(f"ERROR: could not write {target}: {e}", err=True)
        sys.exit(1)
    click.echo(f"set [sandbox] policy = {policy}")
    click.echo("restart the daemon to apply — [sandbox] is read at daemon start, not hot-reloaded.")


@sandbox.command(
    "self-test",
    help="Run a probe under the STRICT sandbox to verify it works (esp. the macOS Seatbelt profile).",
)
def sandbox_self_test() -> None:
    """Build the STRICT sandbox profile for a throwaway workspace and run the venv python
    under it — verifying the interpreter + native deps load and the workspace is writable.
    On macOS this exercises the exact Seatbelt read-allowlist the daemon generates; it's
    the iteration tool for the profile (no real unit needed). On failure, the macOS
    unified log names the blocked path:
        log show --last 1m --predicate 'sender == "Sandbox"' --info | tail -40
    """
    if sys.platform != "darwin":
        result = probe_bubblewrap()
        click.echo(
            "bubblewrap probe OK" if result.ok else f"bubblewrap probe FAILED: {result.reason}"
        )
        sys.exit(0 if result.ok else 1)

    import tempfile

    from .sandbox import SandboxConfig
    from .sandbox.wrapper import SANDBOX_EXEC_BIN, _seatbelt_profile

    probe = probe_seatbelt()
    if not probe.ok:
        click.echo(f"✗ sandbox-exec (Seatbelt) not functional: {probe.reason}", err=True)
        sys.exit(1)

    py = sys.executable
    with tempfile.TemporaryDirectory(prefix="auspexai-selftest-") as ws:
        cfg = SandboxConfig(
            use_bubblewrap=False,
            policy=SandboxPolicy.STRICT,
            runner_bin=py,
            workspace_path=ws,
            output_path=str(Path(ws) / "output.json"),
            unit_id="self-test",
            manifest_sha256="0" * 64,
        )
        profile = _seatbelt_profile(cfg)
        # Import the worker's actual native/runtime deps (cryptography is a C
        # extension — a good test of dylib loading under the read-allowlist).
        script = (
            "import cryptography, httpx, click;"
            f"open({json.dumps(str(Path(ws) / 'output.json'))}, 'w').write('ok');"
            "print('PROFILE_OK')"
        )
        proc = subprocess.run(
            [SANDBOX_EXEC_BIN, "-p", profile, py, "-c", script],
            capture_output=True,
            text=True,
        )
    if proc.returncode == 0 and "PROFILE_OK" in proc.stdout:
        click.echo(
            "✓ macOS STRICT (Seatbelt) profile works: interpreter + native deps load, "
            "workspace writable, network denied."
        )
        return
    click.echo(
        "✗ STRICT Seatbelt profile FAILED — the read-allowlist likely needs widening.", err=True
    )
    tail = (proc.stderr or proc.stdout or "").strip()[-1000:]
    if tail:
        click.echo(tail, err=True)
    click.echo(
        "\n  Find the blocked path(s):\n"
        "    log show --last 1m --predicate 'sender == \"Sandbox\"' --info | tail -40",
        err=True,
    )
    sys.exit(1)


@cli.group(help="Manage tenant allow/deny lists.")
def tenant() -> None:
    pass


@tenant.command("allow", help="Add a tenant to the allow list.")
@click.argument("tenant_id")
@click.pass_context
def tenant_allow(ctx: click.Context, tenant_id: str) -> None:
    config: WorkerConfig = ctx.obj["config"]
    db, _ = initialize_state(config)
    try:
        TenantListRepository(db).allow_add(tenant_id)
        click.echo(f"allow: {tenant_id}")
    finally:
        db.close()


@tenant.command("deny", help="Add a tenant to the deny list.")
@click.argument("tenant_id")
@click.pass_context
def tenant_deny(ctx: click.Context, tenant_id: str) -> None:
    config: WorkerConfig = ctx.obj["config"]
    db, _ = initialize_state(config)
    try:
        TenantListRepository(db).deny_add(tenant_id)
        click.echo(f"deny: {tenant_id}")
    finally:
        db.close()


@tenant.command("list", help="Show allow + deny lists.")
@click.pass_context
def tenant_list(ctx: click.Context) -> None:
    config: WorkerConfig = ctx.obj["config"]
    db, _ = initialize_state(config)
    try:
        repo = TenantListRepository(db)
        allow = repo.list_allow()
        deny = repo.list_deny()
        click.echo("allow:")
        if allow:
            for t in allow:
                click.echo(f"  {t}")
        else:
            click.echo("  (empty — all known tenants accepted)")
        click.echo("deny:")
        if deny:
            for t in deny:
                click.echo(f"  {t}")
        else:
            click.echo("  (empty)")
    finally:
        db.close()


_DATETIME_FORMATS = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"]


@cli.group(help="Inspect locally-stored receipts (one row per submitted result).")
def receipts() -> None:
    pass


@receipts.command("list", help="List receipts, optionally filtered by --since or --tenant.")
@click.option(
    "--since",
    type=click.DateTime(formats=_DATETIME_FORMATS),
    default=None,
    help="Only show receipts submitted at or after this timestamp.",
)
@click.option(
    "--tenant",
    "tenant_id",
    type=str,
    default=None,
    help="Filter to receipts associated with this tenant_id (looked up via assignment_audit).",
)
@click.option("--limit", type=int, default=50, help="Show at most N rows (default 50).")
@click.pass_context
def receipts_list(
    ctx: click.Context,
    since: datetime | None,
    tenant_id: str | None,
    limit: int,
) -> None:
    config: WorkerConfig = ctx.obj["config"]
    db, _ = initialize_state(config)
    try:
        rows = SubmittedResultRepository(db).list_receipts(
            since=since, tenant_id=tenant_id, limit=limit
        )
        if not rows:
            if since is None and tenant_id is None:
                click.echo("no receipts yet")
            else:
                click.echo("no receipts match the given filters")
            return
        for row in rows:
            ts = row.submitted_at.isoformat(timespec="seconds")
            click.echo(
                f"{ts}  unit={row.unit_id}  result={row.result_id}  "
                f"exit={row.exit_code}  status={row.receipt_status}"
            )
    finally:
        db.close()


@receipts.command(
    "show",
    help="Pretty-print one receipt. Identifier may be a result_id or a unit_id "
    "(result_id matched first; unit_id falls back to most-recent submission).",
)
@click.argument("identifier")
@click.pass_context
def receipts_show(ctx: click.Context, identifier: str) -> None:
    config: WorkerConfig = ctx.obj["config"]
    db, _ = initialize_state(config)
    try:
        repo = SubmittedResultRepository(db)
        match = repo.get_by_result_id(identifier)
        if match is None:
            by_unit = repo.get_by_unit(identifier)
            match = by_unit[0] if by_unit else None
        if match is None:
            click.echo(f"no receipt found for identifier {identifier!r}", err=True)
            sys.exit(1)

        click.echo(f"unit_id:        {match.unit_id}")
        click.echo(f"result_id:      {match.result_id}")
        click.echo(f"assignment_id:  {match.assignment_id or '-'}")
        click.echo(f"submitted_at:   {match.submitted_at.isoformat(timespec='seconds')}")
        click.echo(f"completed_at:   {match.completed_at}")
        click.echo(f"exit_code:      {match.exit_code}")
        click.echo(f"receipt_status: {match.receipt_status}")
        if match.canonical_format is not None:
            blob_bytes = len(match.canonical_blob) if match.canonical_blob else 0
            fetched = (
                match.canonical_fetched_at.isoformat(timespec="seconds")
                if match.canonical_fetched_at
                else "-"
            )
            click.echo(
                f"canonical:      format={match.canonical_format} "
                f"size={blob_bytes}B fetched={fetched}"
            )
        if match.coord_unit_status_after is not None:
            click.echo(
                f"coord_state:    unit_status={match.coord_unit_status_after} "
                f"completions={match.coord_completions_so_far}/"
                f"{match.coord_replication_target}"
            )
        click.echo("")
        click.echo("payload:")
        try:
            payload = json.loads(match.payload_json)
            click.echo(json.dumps(payload, indent=2, sort_keys=True))
        except json.JSONDecodeError:
            click.echo(match.payload_json)
    finally:
        db.close()


@receipts.command(
    "export",
    help="Export all receipts as a JSON archive. T0 placeholder receipts are "
    "local attestations — not coordinator-signed portable credentials.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    default=None,
    help="Write to file instead of stdout.",
)
@click.pass_context
def receipts_export(ctx: click.Context, output_path: str | None) -> None:
    config: WorkerConfig = ctx.obj["config"]
    db, repo = initialize_state(config)
    try:
        worker = repo.get()
        rows = SubmittedResultRepository(db).list_receipts(limit=100000)
        if not rows:
            click.echo("no receipts to export")
            return

        export = {
            "auspexai_receipt_export": {
                "version": 1,
                "worker_id": worker.worker_id if worker else None,
                "trust_tier": worker.trust_tier if worker else None,
                "is_account_linked": worker is not None and worker.trust_tier >= 1,
                "exported_at": datetime.now(UTC).isoformat(),
                "note": (
                    "These are local attestations of work performed. "
                    "T0 placeholder receipts are not coordinator-signed and are "
                    "not portable credentials. Login with `auspexai-worker login` "
                    "to link your contributions to an account."
                    if (worker is not None and worker.trust_tier == 0)
                    else "Account-linked receipts. Canonical receipts (receipt_status=canonical) "
                    "are coordinator-signed and independently verifiable."
                ),
            },
            "receipts": [
                {
                    "unit_id": r.unit_id,
                    "result_id": r.result_id,
                    "exit_code": r.exit_code,
                    "completed_at": r.completed_at,
                    "submitted_at": r.submitted_at.isoformat(),
                    "receipt_status": r.receipt_status,
                    "canonical_format": r.canonical_format,
                }
                for r in rows
            ],
        }

        output_text = json.dumps(export, indent=2)
        if output_path:
            Path(output_path).write_text(output_text)
            click.echo(f"exported {len(rows)} receipts to {output_path}")
        else:
            click.echo(output_text)
    finally:
        db.close()


@cli.command(help="Filtered audit-log query (assignment_audit table).")
@click.option(
    "--since",
    type=click.DateTime(formats=_DATETIME_FORMATS),
    default=None,
    help="Only show audit rows from at or after this timestamp.",
)
@click.option(
    "--unit",
    "unit_id",
    type=str,
    default=None,
    help="Filter to a specific unit_id.",
)
@click.option(
    "--action",
    type=str,
    default=None,
    help="Filter to a specific action label (e.g., 'assignment.accept').",
)
@click.option("--limit", type=int, default=100, help="Show at most N rows (default 100).")
@click.pass_context
def log(
    ctx: click.Context,
    since: datetime | None,
    unit_id: str | None,
    action: str | None,
    limit: int,
) -> None:
    config: WorkerConfig = ctx.obj["config"]
    db, _ = initialize_state(config)
    try:
        rows = AssignmentAuditRepository(db).query(
            since=since, unit_id=unit_id, action=action, limit=limit
        )
        if not rows:
            click.echo("no audit rows match the given filters")
            return
        for row in rows:
            ts = row.occurred_at.isoformat(timespec="seconds")
            unit = row.unit_id or "-"
            tenant = row.tenant_id or "-"
            line = f"{ts}  {row.action:<35}  unit={unit}  tenant={tenant}"
            if row.reason:
                line += f"  ({row.reason})"
            click.echo(line)
    finally:
        db.close()


def _print_device_code(code: DeviceCode) -> None:
    """Default on_code callback for the login flow."""
    click.echo("")
    click.echo("To complete login, open the following URL in any browser:")
    click.echo(f"    {code.verification_uri}")
    click.echo("")
    click.echo(f"Enter this code on the GitHub page:  {code.user_code}")
    click.echo("")
    click.echo("Waiting for authorization... (Ctrl+C to cancel)")


def _stdin_is_interactive() -> bool:
    """Whether stdin is a TTY — isolated behind a helper so the interactive
    public-credit prompt in `login` is testable (CliRunner's stdin is never a TTY)."""
    return sys.stdin.isatty()


@cli.command(help="Bind this worker to a GitHub account (T0 → T1).")
@click.pass_context
def login(ctx: click.Context) -> None:
    config: WorkerConfig = ctx.obj["config"]
    db, repo = initialize_state(config)
    try:
        worker = repo.get()
        if worker is None:
            click.echo("not enrolled yet — run `auspexai-worker bootstrap` first", err=True)
            sys.exit(1)
        if worker.trust_tier >= 1:
            click.echo(f"already T{worker.trust_tier}; nothing to do")
            return

        click.echo("starting GitHub Device Flow...")
        try:
            access_token = run_device_flow(on_code=_print_device_code)
        except AccessDeniedError:
            click.echo("login cancelled: GitHub authorization denied", err=True)
            sys.exit(1)
        except ExpiredTokenError as exc:
            click.echo(f"login timed out: {exc}", err=True)
            sys.exit(1)
        except DeviceFlowError as exc:
            click.echo(f"device flow failed: {exc}", err=True)
            sys.exit(1)

        click.echo("GitHub authorization received; exchanging with coordinator...")
        keystore = open_keystore(config)
        signer = build_signer(keystore)
        try:
            with CoordinatorClient(base_url=config.coordinator_url, signer=signer) as client:
                exchange = client.oauth_exchange(idp="github", access_token=access_token)
                status_after = client.upgrade_worker(
                    worker_id=worker.worker_id,
                    binding_token=exchange.binding_token,
                )
        except UnsupportedIdpError as exc:
            click.echo(f"coordinator does not accept GitHub IdP: {exc}", err=True)
            sys.exit(1)
        except InvalidAccessTokenError as exc:
            click.echo(f"coordinator rejected GitHub token: {exc}", err=True)
            sys.exit(1)
        except BindingTokenExpiredError as exc:
            click.echo(f"binding token expired before upgrade: {exc}", err=True)
            sys.exit(1)
        except BindingTokenNotFoundError as exc:
            click.echo(f"binding token unknown to coordinator: {exc}", err=True)
            sys.exit(1)
        except BindingTokenConsumedError as exc:
            click.echo(f"binding token already used: {exc}", err=True)
            sys.exit(1)
        except CoordinatorError as exc:
            click.echo(f"coordinator call failed: {exc}", err=True)
            sys.exit(1)

        binding_payload = json.dumps(
            {
                "idp": "github",
                "account_id": exchange.account_id,
                "bound_at": exchange.expires_at.isoformat(),
                "is_new_account": exchange.is_new_account,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        repo.update_after_upgrade(
            new_tier=status_after.trust_tier,
            account_binding_json=binding_payload,
        )

        click.echo("")
        click.echo(f"login successful: T0 → T{status_after.trust_tier}")
        click.echo(f"account_id: {exchange.account_id}")
        if exchange.is_new_account:
            click.echo("(new AuspexAI account created on first login)")

        # System B (D-inc4): the public-citation opt-in — a SEPARATE, explicit choice
        # from authentication. Linking GitHub is auth-consent, NOT consent to be named
        # publicly, so we ask the distinct question here at the identity moment. We
        # PRESERVE the account's standing choice across re-logins: read it first, make the
        # default match it, and only write when the answer actually CHANGES it — so a
        # routine re-login never silently re-anonymizes (nor wrongly tells an opted-in
        # contributor they're anonymous). Reversible later via `account attribution`.
        # Interactive only: a scripted / non-TTY login leaves the standing choice untouched.
        if _stdin_is_interactive():
            # Read the current opt-in so a re-login preserves it. On failure, fall back to a
            # safe anonymous default — combined with the write-only-on-change guard below, a
            # failed read can never overwrite an existing opt-in.
            current_public = False
            try:
                with CoordinatorClient(base_url=config.coordinator_url, signer=signer) as client:
                    state = client.get_attribution(account_id=exchange.account_id)
                current_public = bool(state.get("public_attribution", False))
            except CoordinatorError:
                current_public = False

            click.echo("")
            if current_public:
                click.echo(
                    "Public credit: you're currently credited under your verified GitHub "
                    "account in citations of research you contribute to."
                )
                new_public = click.confirm("Keep being publicly credited?", default=True)
            else:
                click.echo(
                    "Optional — public credit. Research you contribute compute to may be "
                    "published with a contributor acknowledgment, under your verified GitHub "
                    "identity. This is separate from signing in, and off unless you opt in."
                )
                new_public = click.confirm(
                    "Be publicly credited (as your GitHub account) in those citations?",
                    default=False,
                )

            # Write ONLY when the choice changes — preserves the standing opt-in, avoids a
            # redundant consent-audit row, and means False is written only on a deliberate,
            # informed opt-out (currently credited + an explicit "no").
            if new_public != current_public:
                try:
                    with CoordinatorClient(
                        base_url=config.coordinator_url, signer=signer
                    ) as client:
                        client.set_attribution(
                            account_id=exchange.account_id,
                            public_attribution=new_public,
                            # Credit always uses the verified GitHub login — no custom name.
                            attribution_name=None,
                        )
                except CoordinatorError as exc:
                    click.echo(
                        f"(couldn't update public credit now: {exc} — set it later with "
                        "`auspexai-worker account attribution`)",
                        err=True,
                    )
                    new_public = current_public

            if new_public:
                click.echo("You'll be credited under your GitHub account name.")
            else:
                click.echo("No public credit — your contributions stay anonymous in citations.")
    finally:
        db.close()


@cli.command(
    help="Log out: drop the GitHub-account binding, revert to T0 anonymous (worker keeps running)."
)
@click.pass_context
def logout(ctx: click.Context) -> None:
    """The inverse of `login`: the coordinator reverts this worker to T0-anonymous and the local
    binding is cleared, but the worker stays enrolled and running -- NOT the `retire` purge.
    Receipts you already earned stay credited to your account (it keeps its trust); run `login`
    again to re-bind the SAME account (keep building) or a NEW one (start fresh). The
    public-citation choice is re-offered at login."""
    config: WorkerConfig = ctx.obj["config"]
    db, repo = initialize_state(config)
    try:
        worker = repo.get()
        if worker is None:
            click.echo("not enrolled; nothing to log out of", err=True)
            sys.exit(1)
        if worker.account_binding_json is None:
            click.echo("already anonymous (T0) -- no account binding to drop")
            return
        keystore = open_keystore(config)
        signer = build_signer(keystore)
        click.echo("calling coordinator to log out (unbind)...")
        try:
            with CoordinatorClient(base_url=config.coordinator_url, signer=signer) as client:
                client.unbind_worker(worker_id=worker.worker_id)
        except WorkerNotFoundError:
            click.echo("coordinator has no record of this worker; clearing local binding anyway...")
        except CoordinatorError as exc:
            click.echo(f"coordinator unbind call failed: {exc}", err=True)
            sys.exit(1)
        repo.update_after_unbind()
    finally:
        db.close()
    click.echo("")
    click.echo("logged out -- reverted to T0 anonymous (still enrolled + running).")
    click.echo(
        "Receipts you earned stay credited to your account. "
        "Run `auspexai-worker login` to re-bind (same account or a new one)."
    )


@cli.group(help="Account-level settings for this worker's bound identity.")
def account() -> None:
    """Account-scoped actions for the bound GitHub identity (public-citation credit)."""


@account.command(
    "attribution",
    help="Show or change your public-citation credit (System B opt-in; reversible).",
)
@click.option(
    "--public/--anonymous",
    "public",
    default=None,
    help="Opt in (--public) or out (--anonymous). Omit to just show the current state. "
    "Credit always uses your verified GitHub account name — there is no custom name.",
)
@click.pass_context
def account_attribution(ctx: click.Context, public: bool | None) -> None:
    config: WorkerConfig = ctx.obj["config"]
    db, repo = initialize_state(config)
    try:
        worker = repo.get()
        if worker is None or worker.trust_tier < 1 or not worker.account_binding_json:
            click.echo("not bound to an account — run `auspexai-worker login` first", err=True)
            sys.exit(1)
        try:
            account_id = json.loads(worker.account_binding_json).get("account_id")
        except (ValueError, TypeError):
            account_id = None
        if not account_id:
            click.echo("no account_id in the local binding", err=True)
            sys.exit(1)
        signer = build_signer(open_keystore(config))
        try:
            with CoordinatorClient(base_url=config.coordinator_url, signer=signer) as client:
                if public is None:
                    state = client.get_attribution(account_id=account_id)
                else:
                    state = client.set_attribution(
                        account_id=account_id,
                        public_attribution=public,
                        # Credit always uses the verified GitHub login — no custom name.
                        attribution_name=None,
                    )
        except CoordinatorError as exc:
            click.echo(f"coordinator call failed: {exc}", err=True)
            sys.exit(1)
        if state.get("public_attribution"):
            click.echo("public credit: ON — credited under your verified GitHub account name")
        else:
            click.echo("public credit: OFF — anonymous in citations")
    finally:
        db.close()


@cli.command(help="Retire this worker, purge local state, optionally uninstall.")
@click.option(
    "--yes",
    "confirmed",
    is_flag=True,
    default=False,
    help="Skip the interactive confirmation prompt (for non-interactive use).",
)
@click.pass_context
def withdraw(ctx: click.Context, confirmed: bool) -> None:
    config: WorkerConfig = ctx.obj["config"]
    db, repo = initialize_state(config)
    db_path = config.state_db_path
    try:
        worker = repo.get()
        if worker is None:
            click.echo("not enrolled — nothing to withdraw")
            return

        if not confirmed:
            click.echo("")
            click.echo("Withdrawal will:")
            click.echo("  - Tell the coordinator to retire this worker")
            click.echo("  - Delete the local state DB (audit log + receipts)")
            click.echo("  - Delete the worker's Ed25519 keypair from the keystore")
            click.echo("")
            click.echo("Receipts already issued by the coordinator REMAIN in the")
            click.echo("coordinator's transparency log. Per §5.15, the receipts")
            click.echo("remain signed and verifiable but become unattributed.")
            click.echo("")
            confirm_input = click.prompt(
                "Type the word 'withdraw' to confirm", type=str, default="", show_default=False
            )
            if confirm_input.strip().lower() != "withdraw":
                click.echo("aborted")
                sys.exit(1)

        click.echo("calling coordinator to retire worker...")
        keystore = open_keystore(config)
        signer = build_signer(keystore)
        try:
            with CoordinatorClient(base_url=config.coordinator_url, signer=signer) as client:
                client.retire_worker(worker_id=worker.worker_id)
        except WorkerNotFoundError:
            click.echo(
                "coordinator already had no record of this worker; continuing with local purge"
            )
        except CoordinatorError as exc:
            click.echo(f"coordinator retire call failed: {exc}", err=True)
            click.echo(
                "Continuing with local purge anyway — withdrawal is volunteer-initiated and "
                "local state should be removed even if the coordinator is unreachable.",
                err=True,
            )
    finally:
        db.close()

    # Purge local state. Order matters: close DB connection first (done in
    # the finally above), then delete the file, then drop the keystore key.
    if db_path.exists():
        db_path.unlink()
    # WAL/SHM sidecars from sqlite WAL mode.
    for suffix in ("-wal", "-shm"):
        sidecar = db_path.with_name(db_path.name + suffix)
        if sidecar.exists():
            sidecar.unlink()

    keystore = open_keystore(config)
    try:
        keystore.delete()
    except Exception as exc:
        click.echo(f"keystore deletion failed: {exc}", err=True)
        click.echo(
            "Local DB has been purged. Manually remove the keystore entry if needed.",
            err=True,
        )

    click.echo("")
    click.echo("worker withdrawn. Local state purged.")
    click.echo("To complete uninstall, run your package manager's uninstall command")
    click.echo("(e.g., `apt remove auspexai-worker` or `pipx uninstall auspexai-worker`).")


def main() -> None:
    """Entry point for the `auspexai-worker` console script."""
    cli(prog_name="auspexai-worker")


if __name__ == "__main__":
    main()
