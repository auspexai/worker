"""Repositories for worker-local state. M1: just `worker_self`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .db import Database


@dataclass(frozen=True)
class WorkerSelf:
    """The worker's own enrolled identity. Singleton in the local DB."""

    worker_id: str
    trust_tier: int
    pubkey_hex: str
    enrolled_at: datetime
    last_heartbeat_at: datetime | None
    account_binding_json: str | None


class AlreadyEnrolledError(Exception):
    """Raised when `insert_self` is called and a row already exists."""


class WorkerSelfRepository:
    """Access to the singleton `worker_self` row."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def get(self) -> WorkerSelf | None:
        row = self._db.connection.execute(
            "SELECT worker_id, trust_tier, pubkey_hex, enrolled_at, "
            "last_heartbeat_at, account_binding_json "
            "FROM worker_self WHERE id = 1"
        ).fetchone()
        if row is None:
            return None
        return WorkerSelf(
            worker_id=row["worker_id"],
            trust_tier=row["trust_tier"],
            pubkey_hex=row["pubkey_hex"],
            enrolled_at=_parse_ts(row["enrolled_at"]),
            last_heartbeat_at=_parse_ts(row["last_heartbeat_at"]),
            account_binding_json=row["account_binding_json"],
        )

    def insert(
        self,
        *,
        worker_id: str,
        trust_tier: int,
        pubkey_hex: str,
        enrolled_at: datetime,
    ) -> WorkerSelf:
        if self.get() is not None:
            raise AlreadyEnrolledError(
                "worker_self row already exists; call delete() first or use update_tier"
            )
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO worker_self "
                "(id, worker_id, trust_tier, pubkey_hex, enrolled_at) "
                "VALUES (1, ?, ?, ?, ?)",
                (worker_id, trust_tier, pubkey_hex, _format_ts(enrolled_at)),
            )
        return WorkerSelf(
            worker_id=worker_id,
            trust_tier=trust_tier,
            pubkey_hex=pubkey_hex,
            enrolled_at=enrolled_at,
            last_heartbeat_at=None,
            account_binding_json=None,
        )

    def update_tier(self, new_tier: int) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE worker_self SET trust_tier = ? WHERE id = 1",
                (new_tier,),
            )

    def update_after_upgrade(
        self,
        *,
        new_tier: int,
        account_binding_json: str,
    ) -> None:
        """Promote the singleton worker row after a successful upgrade.

        Both columns move together in one transaction — there should never
        be a state where trust_tier was bumped but the binding is missing
        (or vice versa).
        """
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE worker_self SET trust_tier = ?, account_binding_json = ? WHERE id = 1",
                (new_tier, account_binding_json),
            )

    def record_heartbeat(self, at: datetime) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE worker_self SET last_heartbeat_at = ? WHERE id = 1",
                (_format_ts(at),),
            )

    def delete(self) -> None:
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM worker_self WHERE id = 1")


def _format_ts(ts: datetime) -> str:
    return ts.isoformat()


def _parse_ts(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    return datetime.fromisoformat(raw)
