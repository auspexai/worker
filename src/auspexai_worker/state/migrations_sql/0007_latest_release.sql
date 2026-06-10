-- 0007_latest_release.sql — local cache of the coordinator's release
-- announcement (§9 #46).
--
-- The heartbeat response relays the latest announced release for this
-- worker's channel; the loop caches it here so `status` + the dashboard can
-- show "update available" with the maintainer's headline. Display-time
-- version comparison hides the notice once the worker is current — no
-- clearing logic needed; the row is always just "the last announcement".
-- Upgrading is ALWAYS the volunteer's election; nothing acts on this row.

ALTER TABLE worker_self ADD COLUMN latest_release_version TEXT;
ALTER TABLE worker_self ADD COLUMN latest_release_notes TEXT;
ALTER TABLE worker_self ADD COLUMN latest_release_url TEXT;
ALTER TABLE worker_self ADD COLUMN latest_release_at TEXT;
