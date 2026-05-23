"""Click CLI for `auspexai-worker`.

M1 ships two commands: `status` (read-only) and `bootstrap` (first-run
enrollment). The daemon entry point exists as a placeholder; the actual
heartbeat / assignment loop arrives in M2 / M3.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import threading
from datetime import datetime
from pathlib import Path

import click

from . import __version__
from .bootstrap import bootstrap as bootstrap_worker
from .bootstrap import build_signer, initialize_state, open_keystore
from .capabilities import DeclaredCaps
from .capabilities import collect as collect_capabilities
from .config import WorkerConfig
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
from .daemon import AssignmentPoller, HeartbeatLoop
from .daemon.dispatch import RunnerDispatcher
from .oauth import (
    AccessDeniedError,
    DeviceCode,
    DeviceFlowError,
    ExpiredTokenError,
    run_device_flow,
)
from .sandbox import probe_bubblewrap
from .state import (
    AcceptedSensitiveRepository,
    AssignmentAuditRepository,
    ManifestPinRepository,
    PendingSubmissionRepository,
    SubmittedResultRepository,
    TenantListRepository,
)
from .workspace import WorkspaceManager, workspace_runs_dir


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


@cli.command(help="Show worker identity, tier, and configured coordinator URL.")
@click.pass_context
def status(ctx: click.Context) -> None:
    config: WorkerConfig = ctx.obj["config"]
    db, repo = initialize_state(config)
    try:
        worker = repo.get()
    finally:
        db.close()

    click.echo(f"coordinator: {config.coordinator_url}")
    click.echo(f"state-dir:   {config.state_dir}")
    if worker is None:
        click.echo("identity:    not enrolled")
        click.echo("")
        click.echo("Run `auspexai-worker bootstrap` to enroll as T0 anonymous.")
        return
    click.echo(f"worker-id:   {worker.worker_id}")
    click.echo(f"tier:        T{worker.trust_tier}")
    click.echo(f"pubkey:      {worker.pubkey_hex[:16]}… ({worker.pubkey_hex})")
    click.echo(f"enrolled-at: {worker.enrolled_at.isoformat()}")
    if worker.last_heartbeat_at is not None:
        click.echo(f"last-beat:   {worker.last_heartbeat_at.isoformat()}")


@cli.command(help="Generate identity and enroll with the coordinator (T0 anonymous).")
@click.pass_context
def bootstrap(ctx: click.Context) -> None:
    config: WorkerConfig = ctx.obj["config"]
    try:
        result = bootstrap_worker(config)
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Per Q-W6: httpx logs one INFO line per request; the heartbeat loop fires
    # every 60s by default, so unfiltered httpx INFO floods journald. Pin
    # httpx to WARNING unless --verbose is set.
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
        dispatcher = RunnerDispatcher(
            coordinator=client,
            worker_id=worker.worker_id,
            worker_pubkey=worker.pubkey_hex,
            privkey=privkey,
            workspace_manager=workspace_manager,
            submitted_repo=submitted_results,
            pending_repo=pending_submissions,
            use_bubblewrap=config.sandbox_use_bubblewrap,
            runner_timeout_seconds=config.runner_timeout_seconds,
        )
        heartbeat = HeartbeatLoop(
            coordinator=client,
            repo=repo,
            worker_id=worker.worker_id,
            capability_collector=lambda: collect_capabilities(
                declared_caps=declared_caps,
                declared_gpus=config.declared_gpus,
            ),
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
        )

        def _on_signal(signum: int, _frame: object) -> None:
            click.echo(f"received signal {signum}, shutting down", err=True)
            heartbeat.stop()
            poller.stop()

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
        heartbeat_thread.start()
        poller_thread.start()
        heartbeat_thread.join()
        poller_thread.join()

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
