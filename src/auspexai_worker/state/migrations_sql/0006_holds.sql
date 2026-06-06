-- 0006_holds.sql — local cache of holds for surfacing (§2.1 #11).
--
-- Two distinct holds, surfaced to the volunteer via `auspexai-worker status` +
-- the local dashboard:
--   * volunteer SELF-pause — the resource owner withholding their machine
--     (owner's hold; persisted so it survives a daemon restart).
--   * operator hold cache — the latest pause/quarantine the coordinator reported
--     on the assignment poll (operator's hold; refreshed each poll, cleared on a
--     200). The reason travels in the /assignments 423 (the heartbeat strips the
--     OPERATOR_ONLY fields), so the poller caches it here for the read-only
--     surfaces to show.

ALTER TABLE worker_self ADD COLUMN self_paused INTEGER NOT NULL DEFAULT 0;
ALTER TABLE worker_self ADD COLUMN self_pause_reason TEXT;
ALTER TABLE worker_self ADD COLUMN operator_hold_kind TEXT;     -- 'pause' | 'quarantine' | NULL
ALTER TABLE worker_self ADD COLUMN operator_hold_reason TEXT;
ALTER TABLE worker_self ADD COLUMN operator_hold_at TEXT;
