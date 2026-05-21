"""M3 repositories — manifest pins, sensitive consent, tenant lists, audit log.

Kept in one file because the four repositories are small and conceptually
grouped (they all support the M3 assignment-handling pipeline). If any of
them grow substantially in later milestones, split per the coordinator's
`repositories/` convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .db import Database

# ---- manifest_pins --------------------------------------------------------


@dataclass(frozen=True)
class ManifestPin:
    coordinator_experiment_id: str
    manifest_sha256: str
    tenant_id: str
    tenant_experiment_label: str
    pinned_at: datetime


class PinResult(Enum):
    """Outcome of `check_or_pin`."""

    NEW_PIN = "new_pin"
    MATCHED = "matched"
    SWAP_DETECTED = "swap_detected"


class ManifestPinRepository:
    """Read/write the local manifest-pin table.

    The pin is the worker's defense against §5.14 manifest swaps. First
    assignment for an experiment records the manifest_sha256; later
    assignments under a different hash trip `SWAP_DETECTED` and must be
    refused.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def get(self, coordinator_experiment_id: str) -> ManifestPin | None:
        row = self._db.connection.execute(
            "SELECT coordinator_experiment_id, manifest_sha256, tenant_id, "
            "tenant_experiment_label, pinned_at "
            "FROM manifest_pins WHERE coordinator_experiment_id = ?",
            (coordinator_experiment_id,),
        ).fetchone()
        if row is None:
            return None
        return ManifestPin(
            coordinator_experiment_id=row["coordinator_experiment_id"],
            manifest_sha256=row["manifest_sha256"],
            tenant_id=row["tenant_id"],
            tenant_experiment_label=row["tenant_experiment_label"],
            pinned_at=_parse_ts(row["pinned_at"]),
        )

    def check_or_pin(
        self,
        *,
        coordinator_experiment_id: str,
        manifest_sha256: str,
        tenant_id: str,
        tenant_experiment_label: str,
    ) -> PinResult:
        """If no pin exists for the experiment, create one and return NEW_PIN.
        If one exists and matches, return MATCHED. If it exists and disagrees,
        return SWAP_DETECTED (caller must refuse the assignment).
        """
        existing = self.get(coordinator_experiment_id)
        if existing is None:
            with self._db.transaction() as conn:
                conn.execute(
                    "INSERT INTO manifest_pins "
                    "(coordinator_experiment_id, manifest_sha256, tenant_id, "
                    "tenant_experiment_label) VALUES (?, ?, ?, ?)",
                    (
                        coordinator_experiment_id,
                        manifest_sha256,
                        tenant_id,
                        tenant_experiment_label,
                    ),
                )
            return PinResult.NEW_PIN
        if existing.manifest_sha256 == manifest_sha256:
            return PinResult.MATCHED
        return PinResult.SWAP_DETECTED


# ---- accepted_sensitive_experiments --------------------------------------


class AcceptedSensitiveRepository:
    """Tracks the volunteer's explicit per-experiment opt-in to sensitive
    work (per §5.14)."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def accept(self, coordinator_experiment_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO accepted_sensitive_experiments "
                "(coordinator_experiment_id) VALUES (?)",
                (coordinator_experiment_id,),
            )

    def contains(self, coordinator_experiment_id: str) -> bool:
        row = self._db.connection.execute(
            "SELECT 1 FROM accepted_sensitive_experiments WHERE coordinator_experiment_id = ?",
            (coordinator_experiment_id,),
        ).fetchone()
        return row is not None

    def remove(self, coordinator_experiment_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "DELETE FROM accepted_sensitive_experiments WHERE coordinator_experiment_id = ?",
                (coordinator_experiment_id,),
            )


# ---- tenant_allow_list / tenant_deny_list --------------------------------


class TenantListRepository:
    """Volunteer's tenant allow/deny lists per §5.14.

    Semantics:
    - Empty allow-list: accept all known tenants
    - Non-empty allow-list: accept ONLY tenants in the allow-list
    - Tenants in the deny-list are always refused
    - Deny wins over allow if a tenant appears in both
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def allow_add(self, tenant_id: str) -> None:
        self._add("tenant_allow_list", tenant_id)

    def allow_remove(self, tenant_id: str) -> None:
        self._remove("tenant_allow_list", tenant_id)

    def deny_add(self, tenant_id: str) -> None:
        self._add("tenant_deny_list", tenant_id)

    def deny_remove(self, tenant_id: str) -> None:
        self._remove("tenant_deny_list", tenant_id)

    def list_allow(self) -> list[str]:
        return self._list("tenant_allow_list")

    def list_deny(self) -> list[str]:
        return self._list("tenant_deny_list")

    def is_blocked(self, tenant_id: str) -> tuple[bool, str | None]:
        """Returns (blocked, reason).

        reason is None when allowed, else `"tenant_deny"` or
        `"tenant_allow_list_miss"`.
        """
        deny = self._list("tenant_deny_list")
        if tenant_id in deny:
            return True, "tenant_deny"
        allow = self._list("tenant_allow_list")
        if allow and tenant_id not in allow:
            return True, "tenant_allow_list_miss"
        return False, None

    # ---- private --------------------------------------------------------

    def _add(self, table: str, tenant_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                f"INSERT OR IGNORE INTO {table} (tenant_id) VALUES (?)",
                (tenant_id,),
            )

    def _remove(self, table: str, tenant_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                f"DELETE FROM {table} WHERE tenant_id = ?",
                (tenant_id,),
            )

    def _list(self, table: str) -> list[str]:
        rows = self._db.connection.execute(
            f"SELECT tenant_id FROM {table} ORDER BY tenant_id"
        ).fetchall()
        return [r["tenant_id"] for r in rows]


# ---- assignment_audit -----------------------------------------------------


@dataclass(frozen=True)
class AssignmentAuditRow:
    id: int
    occurred_at: datetime
    assignment_id: str | None
    coordinator_experiment_id: str | None
    tenant_id: str | None
    unit_id: str | None
    manifest_sha256: str | None
    action: str
    reason: str | None


class AssignmentAuditRepository:
    """Local append-only audit log of assignment decisions."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def append(
        self,
        *,
        action: str,
        assignment_id: str | None = None,
        coordinator_experiment_id: str | None = None,
        tenant_id: str | None = None,
        unit_id: str | None = None,
        manifest_sha256: str | None = None,
        reason: str | None = None,
    ) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO assignment_audit "
                "(assignment_id, coordinator_experiment_id, tenant_id, unit_id, "
                "manifest_sha256, action, reason) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    assignment_id,
                    coordinator_experiment_id,
                    tenant_id,
                    unit_id,
                    manifest_sha256,
                    action,
                    reason,
                ),
            )

    def recent(self, *, limit: int = 50) -> list[AssignmentAuditRow]:
        rows = self._db.connection.execute(
            "SELECT id, occurred_at, assignment_id, coordinator_experiment_id, "
            "tenant_id, unit_id, manifest_sha256, action, reason "
            "FROM assignment_audit ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            AssignmentAuditRow(
                id=r["id"],
                occurred_at=_parse_ts(r["occurred_at"]),
                assignment_id=r["assignment_id"],
                coordinator_experiment_id=r["coordinator_experiment_id"],
                tenant_id=r["tenant_id"],
                unit_id=r["unit_id"],
                manifest_sha256=r["manifest_sha256"],
                action=r["action"],
                reason=r["reason"],
            )
            for r in rows
        ]

    def by_unit(self, unit_id: str) -> list[AssignmentAuditRow]:
        rows = self._db.connection.execute(
            "SELECT id, occurred_at, assignment_id, coordinator_experiment_id, "
            "tenant_id, unit_id, manifest_sha256, action, reason "
            "FROM assignment_audit WHERE unit_id = ? ORDER BY id DESC",
            (unit_id,),
        ).fetchall()
        return [
            AssignmentAuditRow(
                id=r["id"],
                occurred_at=_parse_ts(r["occurred_at"]),
                assignment_id=r["assignment_id"],
                coordinator_experiment_id=r["coordinator_experiment_id"],
                tenant_id=r["tenant_id"],
                unit_id=r["unit_id"],
                manifest_sha256=r["manifest_sha256"],
                action=r["action"],
                reason=r["reason"],
            )
            for r in rows
        ]


# ---- submitted_results (M4) ----------------------------------------------


@dataclass(frozen=True)
class SubmittedResult:
    id: int
    unit_id: str
    assignment_id: str | None
    result_id: str
    exit_code: int
    completed_at: str
    submitted_at: datetime
    coord_unit_status_after: str | None
    coord_completions_so_far: int | None
    coord_replication_target: int | None
    payload_json: str


class SubmittedResultRepository:
    """Local record of results the worker submitted to the coordinator (M4)."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def record(
        self,
        *,
        unit_id: str,
        assignment_id: str | None,
        result_id: str,
        exit_code: int,
        completed_at: str,
        coord_unit_status_after: str | None,
        coord_completions_so_far: int | None,
        coord_replication_target: int | None,
        payload_json: str,
    ) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO submitted_results "
                "(unit_id, assignment_id, result_id, exit_code, completed_at, "
                " coord_unit_status_after, coord_completions_so_far, "
                " coord_replication_target, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    unit_id,
                    assignment_id,
                    result_id,
                    exit_code,
                    completed_at,
                    coord_unit_status_after,
                    coord_completions_so_far,
                    coord_replication_target,
                    payload_json,
                ),
            )

    def get_by_unit(self, unit_id: str) -> list[SubmittedResult]:
        rows = self._db.connection.execute(
            "SELECT id, unit_id, assignment_id, result_id, exit_code, "
            "completed_at, submitted_at, coord_unit_status_after, "
            "coord_completions_so_far, coord_replication_target, payload_json "
            "FROM submitted_results WHERE unit_id = ? ORDER BY id DESC",
            (unit_id,),
        ).fetchall()
        return [_row_to_submitted_result(r) for r in rows]

    def recent(self, *, limit: int = 50) -> list[SubmittedResult]:
        rows = self._db.connection.execute(
            "SELECT id, unit_id, assignment_id, result_id, exit_code, "
            "completed_at, submitted_at, coord_unit_status_after, "
            "coord_completions_so_far, coord_replication_target, payload_json "
            "FROM submitted_results ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_submitted_result(r) for r in rows]


def _row_to_submitted_result(row) -> SubmittedResult:
    return SubmittedResult(
        id=row["id"],
        unit_id=row["unit_id"],
        assignment_id=row["assignment_id"],
        result_id=row["result_id"],
        exit_code=row["exit_code"],
        completed_at=row["completed_at"],
        submitted_at=_parse_ts(row["submitted_at"]),
        coord_unit_status_after=row["coord_unit_status_after"],
        coord_completions_so_far=row["coord_completions_so_far"],
        coord_replication_target=row["coord_replication_target"],
        payload_json=row["payload_json"],
    )


def _parse_ts(raw: str) -> datetime:
    # SQLite DEFAULT CURRENT_TIMESTAMP yields "YYYY-MM-DD HH:MM:SS" (no T).
    # Application-side inserts use isoformat(). Normalize both.
    if " " in raw and "T" not in raw:
        raw = raw.replace(" ", "T")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    return datetime.fromisoformat(raw)
