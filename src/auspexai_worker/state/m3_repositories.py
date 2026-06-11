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
        return [_row_to_audit_row(r) for r in rows]

    def by_unit(self, unit_id: str) -> list[AssignmentAuditRow]:
        rows = self._db.connection.execute(
            "SELECT id, occurred_at, assignment_id, coordinator_experiment_id, "
            "tenant_id, unit_id, manifest_sha256, action, reason "
            "FROM assignment_audit WHERE unit_id = ? ORDER BY id DESC",
            (unit_id,),
        ).fetchall()
        return [_row_to_audit_row(r) for r in rows]

    def query(
        self,
        *,
        since: datetime | None = None,
        unit_id: str | None = None,
        action: str | None = None,
        limit: int = 200,
    ) -> list[AssignmentAuditRow]:
        """Filtered query for the M5 `auspexai-worker log` CLI."""
        where_clauses: list[str] = []
        params: list[object] = []

        if since is not None:
            where_clauses.append("occurred_at >= ?")
            params.append(since.isoformat())

        if unit_id is not None:
            where_clauses.append("unit_id = ?")
            params.append(unit_id)

        if action is not None:
            where_clauses.append("action = ?")
            params.append(action)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        params.append(limit)

        rows = self._db.connection.execute(
            "SELECT id, occurred_at, assignment_id, coordinator_experiment_id, "
            "tenant_id, unit_id, manifest_sha256, action, reason "
            f"FROM assignment_audit {where_sql} ORDER BY occurred_at DESC, id DESC LIMIT ?",
            params,
        ).fetchall()
        return [_row_to_audit_row(r) for r in rows]


def _row_to_audit_row(r) -> AssignmentAuditRow:
    return AssignmentAuditRow(
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


# ---- submitted_results (M4 row, M5 receipt extensions) -------------------


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
    receipt_status: str
    canonical_blob: bytes | None
    canonical_format: str | None
    canonical_fetched_at: datetime | None


# All columns selected by every receipt-reading method, kept in one place so
# the schema stays in sync across queries.
_SUBMITTED_RESULTS_COLUMNS = (
    "id, unit_id, assignment_id, result_id, exit_code, completed_at, "
    "submitted_at, coord_unit_status_after, coord_completions_so_far, "
    "coord_replication_target, payload_json, "
    "receipt_status, canonical_blob, canonical_format, canonical_fetched_at"
)


class SubmittedResultRepository:
    """Local record of results the worker submitted to the coordinator.

    M4 introduced the table with one row per submitted result. M5 added the
    receipt-canonical columns (`receipt_status`, `canonical_blob`,
    `canonical_format`, `canonical_fetched_at`) so the same row is both the
    "I submitted this" record AND the worker's local receipt store — single
    source of truth per the 2026-05-22 design decision in
    `worker_daemon_design.md` §10. M7 will fill the canonical_* columns via
    `set_canonical()` once the coordinator ships canonical CBOR+COSE
    receipts.
    """

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
        # `receipt_status` defaults to 'placeholder' via the M5 migration's
        # NOT NULL DEFAULT; canonical_* columns default to NULL.
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
            f"SELECT {_SUBMITTED_RESULTS_COLUMNS} "
            "FROM submitted_results WHERE unit_id = ? ORDER BY id DESC",
            (unit_id,),
        ).fetchall()
        return [_row_to_submitted_result(r) for r in rows]

    def get_by_result_id(self, result_id: str) -> SubmittedResult | None:
        row = self._db.connection.execute(
            f"SELECT {_SUBMITTED_RESULTS_COLUMNS} FROM submitted_results WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        return _row_to_submitted_result(row) if row is not None else None

    def recent(self, *, limit: int = 50) -> list[SubmittedResult]:
        rows = self._db.connection.execute(
            f"SELECT {_SUBMITTED_RESULTS_COLUMNS} FROM submitted_results ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_submitted_result(r) for r in rows]

    def list_receipts(
        self,
        *,
        since: datetime | None = None,
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[SubmittedResult]:
        """List receipts ordered by submitted_at DESC.

        `since` filters to receipts submitted at or after that timestamp.
        `tenant_id` filters via a subquery against assignment_audit (the
        only place tenant_id lives on the worker side) — pre-M5 rows whose
        audit rows lack tenant_id will be excluded from tenant-filtered
        results, which is acceptable for the placeholder phase.
        """
        where_clauses: list[str] = []
        params: list[object] = []

        if since is not None:
            where_clauses.append("submitted_at >= ?")
            params.append(since.isoformat())

        if tenant_id is not None:
            where_clauses.append(
                "unit_id IN ("
                "  SELECT DISTINCT unit_id FROM assignment_audit "
                "  WHERE tenant_id = ? AND unit_id IS NOT NULL"
                ")"
            )
            params.append(tenant_id)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        params.append(limit)

        rows = self._db.connection.execute(
            f"SELECT {_SUBMITTED_RESULTS_COLUMNS} "
            f"FROM submitted_results {where_sql} "
            "ORDER BY submitted_at DESC, id DESC LIMIT ?",
            params,
        ).fetchall()
        return [_row_to_submitted_result(r) for r in rows]

    def list_pending_canonical(self, *, limit: int = 100) -> list[SubmittedResult]:
        """Rows whose canonical-receipt fetch has not yet succeeded (M7-tail).

        `receipt_status='placeholder'` is the M5 default; the M7-tail
        background fetch loop walks these rows and calls the coord's
        canonical-receipt endpoint for each.
        """
        rows = self._db.connection.execute(
            f"SELECT {_SUBMITTED_RESULTS_COLUMNS} FROM submitted_results "
            "WHERE receipt_status = 'placeholder' ORDER BY id ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_submitted_result(r) for r in rows]

    def progress_summary(self) -> dict[str, int]:
        """Count completed units and distinct experiments for T0 progress signal."""
        row = self._db.connection.execute(
            "SELECT COUNT(*) AS total, "
            "COUNT(DISTINCT aa.coordinator_experiment_id) AS experiments "
            "FROM submitted_results sr "
            "LEFT JOIN assignment_audit aa ON sr.unit_id = aa.unit_id "
            "  AND aa.action = 'assignment.accepted'",
        ).fetchone()
        return {
            "completed_units": row["total"] if row else 0,
            "distinct_experiments": row["experiments"] if row else 0,
        }

    def set_canonical(
        self,
        *,
        result_id: str,
        canonical_blob: bytes,
        canonical_format: str,
        fetched_at: datetime,
    ) -> bool:
        """Promote a placeholder receipt to canonical (M7 fetch path).

        Returns True if a row was updated, False if no row matched the
        given result_id.
        """
        with self._db.transaction() as conn:
            cur = conn.execute(
                "UPDATE submitted_results SET "
                "receipt_status = 'canonical', "
                "canonical_blob = ?, "
                "canonical_format = ?, "
                "canonical_fetched_at = ? "
                "WHERE result_id = ?",
                (canonical_blob, canonical_format, fetched_at.isoformat(), result_id),
            )
        return cur.rowcount > 0


def _row_to_submitted_result(row) -> SubmittedResult:
    canonical_fetched_at_raw = row["canonical_fetched_at"]
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
        receipt_status=row["receipt_status"],
        canonical_blob=row["canonical_blob"],
        canonical_format=row["canonical_format"],
        canonical_fetched_at=(
            _parse_ts(canonical_fetched_at_raw) if canonical_fetched_at_raw else None
        ),
    )


def _parse_ts(raw: str) -> datetime:
    # SQLite DEFAULT CURRENT_TIMESTAMP yields "YYYY-MM-DD HH:MM:SS" (no T).
    # Application-side inserts use isoformat(). Normalize both.
    if " " in raw and "T" not in raw:
        raw = raw.replace(" ", "T")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    return datetime.fromisoformat(raw)


# ---- pending_submissions (M6-tail) ----------------------------------------


@dataclass(frozen=True)
class PendingSubmission:
    id: int
    unit_id: str
    assignment_id: str | None
    completed_at: str
    exit_code: int
    payload_json: str
    worker_signature: str
    worker_pubkey: str
    queued_at: datetime
    last_attempt_at: datetime | None
    attempt_count: int
    failure_kind: str | None  # 'transient' / 'terminal' / None
    failure_reason: str | None


_PENDING_COLUMNS = (
    "id, unit_id, assignment_id, completed_at, exit_code, payload_json, "
    "worker_signature, worker_pubkey, queued_at, last_attempt_at, "
    "attempt_count, failure_kind, failure_reason"
)


class PendingSubmissionRepository:
    """Write-before-submit queue for result submissions.

    Per M6-tail design (2026-05-22): the worker writes the signed Result here
    BEFORE attempting `coord.submit_result`, so a coordinator outage doesn't
    drop the result on the floor. On submit success, the row is atomically
    moved to `submitted_results` (and removed from this table). On transient
    failure, the row remains for retry by the next dispatch tick. On terminal
    4xx (other than 409), the row is marked 'terminal' so it surfaces to the
    operator instead of being silently retried forever.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def queue(
        self,
        *,
        unit_id: str,
        assignment_id: str,
        completed_at: str,
        exit_code: int,
        payload_json: str,
        worker_signature: str,
        worker_pubkey: str,
    ) -> None:
        """Add a Result to the write-before-submit queue.

        Keyed by assignment_id (§9 #46 D6 fix, migration 0008): unit_ids are
        tenant-chosen and collide across experiments; the assignment_id is
        coordinator-unique. Raises on a duplicate assignment_id (one result
        per assignment — guards against logic bugs).
        """
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO pending_submissions "
                "(unit_id, assignment_id, completed_at, exit_code, payload_json, "
                " worker_signature, worker_pubkey) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    unit_id,
                    assignment_id,
                    completed_at,
                    exit_code,
                    payload_json,
                    worker_signature,
                    worker_pubkey,
                ),
            )

    def get_by_unit(self, unit_id: str) -> PendingSubmission | None:
        """Display/diagnostic lookup ONLY (unit_ids can collide across
        experiments — never use for integrity decisions; see get_by_assignment)."""
        row = self._db.connection.execute(
            f"SELECT {_PENDING_COLUMNS} FROM pending_submissions WHERE unit_id = ?",
            (unit_id,),
        ).fetchone()
        return _row_to_pending(row) if row is not None else None

    def get_by_assignment(self, assignment_id: str) -> PendingSubmission | None:
        row = self._db.connection.execute(
            f"SELECT {_PENDING_COLUMNS} FROM pending_submissions WHERE assignment_id = ?",
            (assignment_id,),
        ).fetchone()
        return _row_to_pending(row) if row is not None else None

    def list_retryable(self, *, limit: int = 10) -> list[PendingSubmission]:
        """Return up to N pending rows eligible for retry, oldest first.

        Eligible = `failure_kind IS NULL` (never tried) OR
        `failure_kind = 'transient'` (last attempt was retryable).
        `terminal` rows are excluded — operator handles those.
        """
        rows = self._db.connection.execute(
            f"SELECT {_PENDING_COLUMNS} FROM pending_submissions "
            "WHERE failure_kind IS NULL OR failure_kind = 'transient' "
            "ORDER BY queued_at ASC, id ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_pending(r) for r in rows]

    def list_all(self) -> list[PendingSubmission]:
        """All pending rows, including terminal — used by the CLI surface."""
        rows = self._db.connection.execute(
            f"SELECT {_PENDING_COLUMNS} FROM pending_submissions ORDER BY queued_at ASC, id ASC"
        ).fetchall()
        return [_row_to_pending(r) for r in rows]

    def mark_attempt(
        self,
        *,
        assignment_id: str,
        failure_kind: str,
        failure_reason: str,
        attempted_at: datetime,
    ) -> None:
        """Record an attempt that left the row pending — transient or terminal.
        Keyed by assignment_id (§9 #46 — unit_ids collide across experiments)."""
        if failure_kind not in ("transient", "terminal"):
            raise ValueError(
                f"failure_kind must be 'transient' or 'terminal', got {failure_kind!r}"
            )
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE pending_submissions SET "
                "last_attempt_at = ?, "
                "attempt_count = attempt_count + 1, "
                "failure_kind = ?, "
                "failure_reason = ? "
                "WHERE assignment_id = ?",
                (attempted_at.isoformat(), failure_kind, failure_reason, assignment_id),
            )

    def remove(self, assignment_id: str) -> None:
        """Delete a pending row by its coordinator-unique assignment_id. Used
        after the row has been drained to submitted_results (success or 409
        idempotent path)."""
        with self._db.transaction() as conn:
            conn.execute(
                "DELETE FROM pending_submissions WHERE assignment_id = ?",
                (assignment_id,),
            )


def _row_to_pending(row) -> PendingSubmission:
    return PendingSubmission(
        id=row["id"],
        unit_id=row["unit_id"],
        assignment_id=row["assignment_id"],
        completed_at=row["completed_at"],
        exit_code=row["exit_code"],
        payload_json=row["payload_json"],
        worker_signature=row["worker_signature"],
        worker_pubkey=row["worker_pubkey"],
        queued_at=_parse_ts(row["queued_at"]),
        last_attempt_at=_parse_ts(row["last_attempt_at"]) if row["last_attempt_at"] else None,
        attempt_count=row["attempt_count"],
        failure_kind=row["failure_kind"],
        failure_reason=row["failure_reason"],
    )
