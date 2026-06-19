-- 0010_ran_under.sql — A2 #32 (worker half): bind the sandbox policy into the
-- signed result (v2).
--
-- A v2 result signs one new field alongside the v1 body: `ran_under` — the
-- sandbox policy the worker DAEMON actually ran THIS unit under ("strict" /
-- "permissive"). It is covered by the worker signature, so the coordinator's
-- equal-trust containment guard is worker-ATTESTED + accountable, not
-- heartbeat-self-reported. The value must survive a write-before-submit retry,
-- so the pending queue persists it (else a re-submitted v2 result would fail the
-- coordinator's signature check).
--
-- Legacy rows (schema_version 0/1) keep their original signature: `ran_under`
-- stays NULL, so the coordinator reconstructs the v0/v1 canonical bytes for them.
-- Plain ADD COLUMN (no rebuild) — nullable.

ALTER TABLE pending_submissions ADD COLUMN ran_under TEXT;
