-- M4 schema: local record of results the worker submitted to the coordinator.
--
-- One row per submitted result. `coord_*` columns mirror the
-- ResultSubmissionResponse the coordinator returned at submit time;
-- they're a snapshot, not authoritative — the coordinator's per-job DB
-- is the canonical source of truth.

CREATE TABLE submitted_results (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id                  TEXT    NOT NULL,
    assignment_id            TEXT,
    result_id                TEXT    NOT NULL,
    exit_code                INTEGER NOT NULL,
    completed_at             TEXT    NOT NULL,
    submitted_at             TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    coord_unit_status_after  TEXT,
    coord_completions_so_far INTEGER,
    coord_replication_target INTEGER,
    payload_json             TEXT    NOT NULL
);

CREATE INDEX submitted_results_unit_idx ON submitted_results(unit_id);
CREATE INDEX submitted_results_submitted_idx ON submitted_results(submitted_at DESC);
