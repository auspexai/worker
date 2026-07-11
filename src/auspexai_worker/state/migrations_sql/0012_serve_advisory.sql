-- 0012_serve_advisory.sql — the worker's most recent operator-actionable serve
-- failure: a persistent GPU out-of-memory that survived the worker's OWN recovery
-- (unload every loaded model to free VRAM, then retry once — inference/server.py
-- `serve()`). Surfaced on the local dashboard as a copy-to-run recovery command —
-- NEVER auto-run, since the remedies (drop the OS page cache, restart the model
-- server) need privileges a sandboxed/volunteer worker must not assume. Singleton
-- (id=1): the latest advisory replaces the prior one; the row is cleared the next
-- time serving succeeds. Informational only — the coordinator still routes the unit
-- elsewhere via the ordinary refusal.
CREATE TABLE IF NOT EXISTS serve_advisory (
    id        INTEGER PRIMARY KEY CHECK (id = 1),
    model_id  TEXT NOT NULL,
    reason    TEXT NOT NULL,
    commands  TEXT NOT NULL,  -- newline-joined shell commands (copy-to-run, never auto-run)
    raised_at TEXT NOT NULL   -- ISO-8601 UTC
);
