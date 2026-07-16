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
    # §2.1 #11 holds (local cache for status/dashboard surfacing):
    self_paused: bool = False
    self_pause_reason: str | None = None
    operator_hold_kind: str | None = None  # 'pause' | 'quarantine' | None
    operator_hold_reason: str | None = None
    operator_hold_at: str | None = None
    # §9 #46: last release announcement relayed by the heartbeat response
    # (informational — upgrading is always the volunteer's election):
    latest_release_version: str | None = None
    latest_release_notes: str | None = None
    latest_release_url: str | None = None
    latest_release_at: str | None = None


class AlreadyEnrolledError(Exception):
    """Raised when `insert_self` is called and a row already exists."""


class WorkerSelfRepository:
    """Access to the singleton `worker_self` row."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def get(self) -> WorkerSelf | None:
        row = self._db.connection.execute(
            "SELECT worker_id, trust_tier, pubkey_hex, enrolled_at, "
            "last_heartbeat_at, account_binding_json, self_paused, self_pause_reason, "
            "operator_hold_kind, operator_hold_reason, operator_hold_at, "
            "latest_release_version, latest_release_notes, latest_release_url, "
            "latest_release_at "
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
            self_paused=bool(row["self_paused"]),
            self_pause_reason=row["self_pause_reason"],
            operator_hold_kind=row["operator_hold_kind"],
            operator_hold_reason=row["operator_hold_reason"],
            operator_hold_at=row["operator_hold_at"],
            latest_release_version=row["latest_release_version"],
            latest_release_notes=row["latest_release_notes"],
            latest_release_url=row["latest_release_url"],
            latest_release_at=row["latest_release_at"],
        )

    def set_self_pause(self, paused: bool) -> None:
        """§2.1 #11: the volunteer's own pause hold (resource-owner sovereignty).
        Persisted so it survives a daemon restart; declared to the coordinator as
        the `self_paused` capability + short-circuits the assignment poller. No
        reason is collected — pausing one's own worker needs no justification (the
        legacy `self_pause_reason` column is left dormant/NULL)."""
        with self._db.transaction() as cur:
            cur.execute(
                "UPDATE worker_self SET self_paused = ?, self_pause_reason = NULL WHERE id = 1",
                (1 if paused else 0,),
            )

    def record_operator_hold(
        self, kind: str | None, *, reason: str | None = None, at: str | None = None
    ) -> None:
        """Cache the operator hold the coordinator reported on the last poll (or
        clear it with kind=None on a 200), so `status` + the dashboard can show
        it. `kind` ∈ 'pause' | 'quarantine' | None."""
        with self._db.transaction() as cur:
            cur.execute(
                "UPDATE worker_self SET operator_hold_kind = ?, operator_hold_reason = ?, "
                "operator_hold_at = ? WHERE id = 1",
                (kind, reason, at),
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

    def update_after_unbind(self) -> None:
        """Drop the account binding on logout: clear account_binding_json + reset trust_tier to
        0 (T0 anonymous), in one transaction (the inverse of update_after_upgrade). The worker
        stays enrolled and running; only the GitHub identity is detached."""
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE worker_self SET trust_tier = 0, account_binding_json = NULL WHERE id = 1"
            )

    def record_heartbeat(self, at: datetime, *, trust_tier: int | None = None) -> None:
        """Record the heartbeat timestamp. When the coordinator's heartbeat
        response carries the worker's current `trust_tier`, refresh the
        locally-cached value too — otherwise `status` / the local dashboard
        stay stuck at the tier captured at enrollment and a coord-side
        promotion/demotion is invisible to the volunteer."""
        with self._db.transaction() as conn:
            if trust_tier is None:
                conn.execute(
                    "UPDATE worker_self SET last_heartbeat_at = ? WHERE id = 1",
                    (_format_ts(at),),
                )
            else:
                conn.execute(
                    "UPDATE worker_self SET last_heartbeat_at = ?, trust_tier = ? WHERE id = 1",
                    (_format_ts(at), trust_tier),
                )

    def record_latest_release(
        self, *, version: str, notes: str | None, url: str | None, at: datetime
    ) -> None:
        """Cache the coordinator's release announcement from the heartbeat
        response (§9 #46). Unconditional singleton UPDATE — the row is always
        just the LAST announcement; display-time version comparison decides
        whether to surface it. Nothing in the worker ever acts on it."""
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE worker_self SET latest_release_version = ?, "
                "latest_release_notes = ?, latest_release_url = ?, "
                "latest_release_at = ? WHERE id = 1",
                (version, notes, url, _format_ts(at)),
            )

    def delete(self) -> None:
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM worker_self WHERE id = 1")


@dataclass(frozen=True)
class ServeAdvisoryRow:
    """The worker's most recent operator-actionable serve failure (GPU out-of-memory,
    a stale model server, or a generic serve error). `headline` is the short bold
    banner; `commands` are copy-to-run recovery hints — never auto-run."""

    model_id: str
    kind: str
    headline: str
    reason: str
    commands: list[str]
    raised_at: datetime
    available_at_raise_gb: float | None = None


class ServeAdvisoryRepository:
    """The singleton `serve_advisory` row (id=1). The daemon records a persistent
    serve failure here via the ModelServer advisory sink and clears it when serving
    recovers; the local dashboard reads it to show/hide the recovery card."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def record(
        self,
        model_id: str,
        kind: str,
        headline: str,
        reason: str,
        commands: list[str],
        raised_at: datetime,
        available_at_raise_gb: float | None = None,
    ) -> None:
        with self._db.transaction() as cur:
            cur.execute(
                "INSERT INTO serve_advisory "
                "(id, model_id, kind, headline, reason, commands, raised_at, available_at_raise_gb) "
                "VALUES (1, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET model_id = excluded.model_id, "
                "kind = excluded.kind, headline = excluded.headline, reason = excluded.reason, "
                "commands = excluded.commands, raised_at = excluded.raised_at, "
                "available_at_raise_gb = excluded.available_at_raise_gb",
                (
                    model_id,
                    kind,
                    headline,
                    reason,
                    "\n".join(commands),
                    _format_ts(raised_at),
                    available_at_raise_gb,
                ),
            )

    def get(self) -> ServeAdvisoryRow | None:
        row = self._db.connection.execute(
            "SELECT model_id, kind, headline, reason, commands, raised_at, available_at_raise_gb "
            "FROM serve_advisory WHERE id = 1"
        ).fetchone()
        if row is None:
            return None
        raised = _parse_ts(row["raised_at"])
        assert raised is not None  # NOT NULL column
        return ServeAdvisoryRow(
            model_id=row["model_id"],
            kind=row["kind"],
            headline=row["headline"],
            reason=row["reason"],
            commands=[c for c in (row["commands"] or "").split("\n") if c],
            raised_at=raised,
            available_at_raise_gb=row["available_at_raise_gb"],
        )

    def clear(self) -> None:
        with self._db.transaction() as cur:
            cur.execute("DELETE FROM serve_advisory WHERE id = 1")


def _format_ts(ts: datetime) -> str:
    return ts.isoformat()


def _parse_ts(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    return datetime.fromisoformat(raw)
