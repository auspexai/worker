-- M1 worker schema: single-row identity table.
--
-- `worker_self` is a singleton — the worker enrolls once per install and
-- carries one identity at a time. Re-enrollment requires `withdraw` (M6)
-- which deletes this row. CHECK constraint enforces single-row.

CREATE TABLE worker_self (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    worker_id TEXT NOT NULL,
    trust_tier INTEGER NOT NULL DEFAULT 0,
    pubkey_hex TEXT NOT NULL,
    enrolled_at TEXT NOT NULL,
    last_heartbeat_at TEXT,
    account_binding_json TEXT
);
