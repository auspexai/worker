"""Click CLI for `auspexai-worker`.

M1 ships two commands: `status` (read-only) and `bootstrap` (first-run
enrollment). The daemon entry point exists as a placeholder; the actual
heartbeat / assignment loop arrives in M2 / M3.
"""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path

import click

from . import __version__
from .bootstrap import bootstrap as bootstrap_worker
from .bootstrap import build_signer, initialize_state, open_keystore
from .capabilities import DeclaredCaps
from .capabilities import collect as collect_capabilities
from .config import WorkerConfig
from .coordinator import (
    CoordinatorClient,
    CoordinatorError,
    PubkeyAlreadyEnrolledError,
    PubkeyAlreadyTenantError,
)
from .daemon import HeartbeatLoop


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

    declared_caps = DeclaredCaps(
        max_ram_gb=config.max_ram_gb,
        max_vram_gb=config.max_vram_gb,
        max_cpu_cores=config.max_cpu_cores,
        network_quota_mb_per_hour=config.network_quota_mb_per_hour,
    )

    with CoordinatorClient(base_url=config.coordinator_url, signer=signer) as client:
        loop = HeartbeatLoop(
            coordinator=client,
            repo=repo,
            worker_id=worker.worker_id,
            capability_collector=lambda: collect_capabilities(
                declared_caps=declared_caps,
                declared_gpus=config.declared_gpus,
            ),
            interval_seconds=config.heartbeat_interval_seconds,
        )

        def _on_signal(signum: int, _frame: object) -> None:
            click.echo(f"received signal {signum}, shutting down", err=True)
            loop.stop()

        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT, _on_signal)

        stats = loop.run(max_ticks=max_ticks)
    db.close()

    if stats.ticks_failed > 0:
        click.echo(
            f"heartbeat loop exited with {stats.ticks_failed} failed ticks "
            f"(succeeded={stats.ticks_succeeded}); last error: {stats.last_error}",
            err=True,
        )
        # Non-zero exit so systemd flags Restart=on-failure paths correctly.
        sys.exit(1 if stats.ticks_succeeded == 0 else 0)


def main() -> None:
    """Entry point for the `auspexai-worker` console script."""
    cli(prog_name="auspexai-worker")


if __name__ == "__main__":
    main()
