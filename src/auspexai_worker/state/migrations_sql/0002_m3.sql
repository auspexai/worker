-- M3 schema: manifest pins, sensitive-content consent, tenant lists,
-- local assignment audit log.

-- Manifest pins keyed by the coordinator's experiment_id (stable across
-- the experiment's lifetime). Per §5.14: the first assignment for an
-- experiment pins manifest_sha256; subsequent assignments with a different
-- hash are rejected as a manifest-swap attempt.
CREATE TABLE manifest_pins (
    coordinator_experiment_id TEXT PRIMARY KEY,
    manifest_sha256 TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    tenant_experiment_label TEXT NOT NULL,
    pinned_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Volunteer's explicit opt-in to sensitive-flagged experiments per §5.14.
-- Default is decline; an entry here flips the gate for that experiment.
CREATE TABLE accepted_sensitive_experiments (
    coordinator_experiment_id TEXT PRIMARY KEY,
    accepted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Volunteer's tenant allow/deny lists per §5.14.
-- Allow-list semantics: empty = accept all known tenants;
-- non-empty = accept ONLY tenants in this table.
-- Deny-list semantics: always reject tenants in this table.
-- If both are non-empty, deny wins on overlap.
CREATE TABLE tenant_allow_list (
    tenant_id TEXT PRIMARY KEY,
    added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE tenant_deny_list (
    tenant_id TEXT PRIMARY KEY,
    added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Local audit log of assignment-handling decisions. Append-only at the
-- application layer (DELETE reserved for `withdraw` which purges
-- everything per §5.15).
CREATE TABLE assignment_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    assignment_id TEXT,
    coordinator_experiment_id TEXT,
    tenant_id TEXT,
    unit_id TEXT,
    manifest_sha256 TEXT,
    action TEXT NOT NULL,
    reason TEXT
);

CREATE INDEX assignment_audit_occurred_idx ON assignment_audit(occurred_at DESC);
CREATE INDEX assignment_audit_action_idx ON assignment_audit(action);
CREATE INDEX assignment_audit_experiment_idx ON assignment_audit(coordinator_experiment_id);
