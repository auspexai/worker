"""First-run bootstrap: generate key (if needed), enroll with coordinator,
persist identity locally.

Idempotent — re-running after a successful enrollment is a no-op that simply
loads the existing identity. Idempotency is the load-bearing property
because the worker daemon may be restarted many times before M2's heartbeat
loop is in place.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from datetime import UTC
from typing import Any

from .config import WorkerConfig
from .coordinator import CoordinatorClient, EnrollmentResponse
from .keystore import Keystore, default_keystore
from .keystore.base import pubkey_hex as _pubkey_hex
from .signing import Rfc9421Signer
from .state import Database, MigrationRunner, WorkerSelf, WorkerSelfRepository


@dataclass(frozen=True)
class BootstrapResult:
    """Outcome of a bootstrap run."""

    worker_self: WorkerSelf
    fresh_enrollment: bool  # True if we just enrolled; False if already present


def collect_capabilities() -> dict[str, Any]:
    """M1 capabilities payload — OS + arch + Python version only.

    Real capability detection (RAM, CPU count, GPU model, available local
    models) lands in M2 as part of the heartbeat loop. The coordinator
    treats capabilities as opaque JSON in M6b; it does not require any
    particular shape.
    """
    return {
        "os": platform.system().lower(),
        "arch": platform.machine().lower(),
        "python_version": platform.python_version(),
    }


def initialize_state(config: WorkerConfig) -> tuple[Database, WorkerSelfRepository]:
    """Open the local DB, apply migrations, return (db, repo)."""
    db = Database(config.state_db_path)
    MigrationRunner(db).apply_all()
    return db, WorkerSelfRepository(db)


def open_keystore(config: WorkerConfig) -> Keystore:
    """Resolve the configured keystore backend."""
    return default_keystore(
        encrypted_file_path=config.keystore_path,
        force_backend=config.keystore_backend,
    )


def build_signer(keystore: Keystore) -> Rfc9421Signer:
    """Load the worker's keypair and construct an RFC 9421 signer."""
    private_key = keystore.load()
    return Rfc9421Signer(private_key, _pubkey_hex(private_key))


def bootstrap(
    config: WorkerConfig,
    *,
    keystore: Keystore | None = None,
    coordinator: CoordinatorClient | None = None,
    capabilities: dict[str, Any] | None = None,
) -> BootstrapResult:
    """Ensure the worker has an enrolled identity. Idempotent.

    Steps:
      1. Open state DB, apply migrations.
      2. If `worker_self` row exists → load + return (no network call).
      3. Otherwise: ensure a keypair exists in the keystore (generate one if
         not), then call `POST /workers/enroll` with the public key + a
         capabilities snapshot, persist the assigned worker_id, return.

    The `keystore` / `coordinator` / `capabilities` kwargs allow tests to
    inject fakes; production callers pass nothing.
    """
    db, repo = initialize_state(config)
    existing = repo.get()
    if existing is not None:
        if keystore is None:
            # Don't open a keystore we don't need — avoids spurious libsecret
            # prompts on a daemon-start path that already has an identity.
            pass
        return BootstrapResult(worker_self=existing, fresh_enrollment=False)

    ks = keystore if keystore is not None else open_keystore(config)
    private_key = ks.load() if ks.has_key() else ks.generate_and_store()
    pubkey = _pubkey_hex(private_key)
    caps = capabilities if capabilities is not None else collect_capabilities()

    if coordinator is None:
        with CoordinatorClient(base_url=config.coordinator_url) as client:
            enrollment = client.enroll(pubkey_hex=pubkey, capabilities=caps)
    else:
        enrollment = coordinator.enroll(pubkey_hex=pubkey, capabilities=caps)

    worker_self = _persist(repo, pubkey, enrollment)
    db.close()
    return BootstrapResult(worker_self=worker_self, fresh_enrollment=True)


def _persist(
    repo: WorkerSelfRepository,
    pubkey: str,
    enrollment: EnrollmentResponse,
) -> WorkerSelf:
    enrolled_at = enrollment.registered_at
    if enrolled_at.tzinfo is None:
        enrolled_at = enrolled_at.replace(tzinfo=UTC)
    return repo.insert(
        worker_id=enrollment.worker_id,
        trust_tier=enrollment.trust_tier,
        pubkey_hex=pubkey,
        enrolled_at=enrolled_at,
    )
