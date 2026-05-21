"""Worker local state — SQLite at $XDG_STATE_HOME/auspexai-worker/worker.db.

M1 schema is a single `worker_self` row holding the enrolled identity. Later
milestones grow the schema: manifest pins (M3), receipts metadata (M5),
account binding (M6). Migration framework mirrors the coordinator's
`migrations_sql/NNNN_name.sql` layout so the pattern is familiar across
codebases.
"""

from __future__ import annotations

from .db import Database, MigrationError, MigrationRunner
from .m3_repositories import (
    AcceptedSensitiveRepository,
    AssignmentAuditRepository,
    AssignmentAuditRow,
    ManifestPin,
    ManifestPinRepository,
    PinResult,
    SubmittedResult,
    SubmittedResultRepository,
    TenantListRepository,
)
from .repository import WorkerSelf, WorkerSelfRepository

__all__ = [
    "AcceptedSensitiveRepository",
    "AssignmentAuditRepository",
    "AssignmentAuditRow",
    "Database",
    "ManifestPin",
    "ManifestPinRepository",
    "MigrationError",
    "MigrationRunner",
    "PinResult",
    "SubmittedResult",
    "SubmittedResultRepository",
    "TenantListRepository",
    "WorkerSelf",
    "WorkerSelfRepository",
]
