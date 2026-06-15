-- 0009_served_weights.sql — §9 #13a (worker half): bind the served-weights
-- digest into the signed result.
--
-- A v1 result signs two new fields alongside `payload`: a `schema_version`
-- and `served_weights` ({model_id: gguf_sha256} for the model(s) the worker
-- DAEMON actually brokered to THIS unit — the trusted-daemon view, never the
-- executor's self-report). These must survive a write-before-submit retry, so
-- the pending queue persists them.
--
-- Legacy rows (schema_version 0) keep the original 5-field signature:
-- `result_schema_version` defaults to 0 and `served_weights_json` stays NULL,
-- so the coordinator reconstructs the legacy canonical bytes for them. Plain
-- ADD COLUMN (no rebuild) — both columns are nullable/defaulted.

ALTER TABLE pending_submissions ADD COLUMN result_schema_version INTEGER NOT NULL DEFAULT 0;
ALTER TABLE pending_submissions ADD COLUMN served_weights_json TEXT;
