"""Click CLI for `auspexai-worker`.

M1 ships two commands: `status` (read-only) and `bootstrap` (first-run
enrollment). The daemon entry point exists as a placeholder; the actual
heartbeat / assignment loop arrives in M2 / M3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from . import __version__
from .bootstrap import bootstrap as bootstrap_worker
from .bootstrap import initialize_state
from .config import WorkerConfig
from .coordinator import CoordinatorError, PubkeyAlreadyEnrolledError, PubkeyAlreadyTenantError


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


def main() -> None:
    """Entry point for the `auspexai-worker` console script."""
    cli(prog_name="auspexai-worker")


if __name__ == "__main__":
    main()
