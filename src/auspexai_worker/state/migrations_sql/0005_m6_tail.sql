-- M6-tail schema: pending submissions queue for write-before-submit resilience.
--
-- Closes the result-loss gap identified during M7 scoping. Previous behavior:
-- if the coordinator was unreachable between unit completion and the
-- submit_result POST, the worker's signed Result was dropped on the floor.
-- Write-before-submit posture: dispatch persists the Result here before
-- attempting the coord call, retries transient failures on the next dispatch
-- tick, and deletes on success (atomically with the insert into
-- submitted_results).
--
-- unit_id is UNIQUE because the M6d coordinator scheduler creates at most
-- one (unit_id, worker_id) assignment per worker — there should never be
-- two in-flight pending submissions for the same unit.
--
-- failure_kind values (enforced in application code, not as a CHECK):
--   NULL         — never attempted yet (just queued)
--   'transient'  — last attempt hit network/5xx; eligible for retry
--   'terminal'   — last attempt hit a 4xx error (other than 409) that won't
--                  resolve by retrying; surfaced to operator
--   (note: 409 result_already_submitted is treated as success — the row is
--   removed and a submitted_results row is inserted with the existing
--   result_id the coordinator returns in 409 details)

CREATE TABLE pending_submissions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id           TEXT    NOT NULL UNIQUE,
    assignment_id     TEXT,
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

CREATE INDEX pending_submissions_queued_idx ON pending_submissions(queued_at);
CREATE INDEX pending_submissions_failure_kind_idx ON pending_submissions(failure_kind);
