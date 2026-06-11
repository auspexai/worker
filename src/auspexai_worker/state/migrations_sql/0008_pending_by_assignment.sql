-- 0008_pending_by_assignment.sql — §9 #46 D6 finding (Task #17 worker half).
--
-- unit_ids are TENANT-CHOSEN and collide across experiments (the D6 re-runs
-- proved it live): UNIQUE(unit_id) on the pending queue would crash the
-- enqueue when two experiments name a unit identically, and the unit_id-keyed
-- lookups could mark/remove the WRONG experiment's row. The assignment_id is
-- coordinator-generated and globally unique — rekey on it.
--
-- SQLite can't alter constraints: rebuild the table. Pending rows are a
-- transient retry queue (normally empty); legacy rows copy across with
-- their assignment_id (set on every enqueue since the column existed).

CREATE TABLE pending_submissions_new (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id           TEXT    NOT NULL,
    assignment_id     TEXT    NOT NULL UNIQUE,
    completed_at      TEXT    NOT NULL,
    exit_code         INTEGER NOT NULL,
    payload_json      TEXT    NOT NULL,
    worker_signature  TEXT    NOT NULL,
    worker_pubkey     TEXT    NOT NULL,
    queued_at         TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_attempt_at   TEXT,
    attempt_count     INTEGER NOT NULL DEFAULT 0,
    failure_kind      TEXT,
    failure_reason    TEXT
);

-- Legacy rows with a NULL assignment_id (shouldn't exist — every queue()
-- call has set it) can't satisfy NOT NULL: synthesize a unique placeholder
-- from the row id so nothing is dropped silently.
INSERT INTO pending_submissions_new (
    id, unit_id, assignment_id, completed_at, exit_code, payload_json,
    worker_signature, worker_pubkey, queued_at, last_attempt_at,
    attempt_count, failure_kind, failure_reason
)
SELECT id, unit_id,
       COALESCE(assignment_id, 'legacy-null-' || id),
       completed_at, exit_code, payload_json,
       worker_signature, worker_pubkey, queued_at, last_attempt_at,
       attempt_count, failure_kind, failure_reason
FROM pending_submissions;

DROP TABLE pending_submissions;
ALTER TABLE pending_submissions_new RENAME TO pending_submissions;

CREATE INDEX pending_submissions_queued_idx ON pending_submissions(queued_at);
CREATE INDEX pending_submissions_failure_kind_idx ON pending_submissions(failure_kind);
CREATE INDEX pending_submissions_unit_idx ON pending_submissions(unit_id);
