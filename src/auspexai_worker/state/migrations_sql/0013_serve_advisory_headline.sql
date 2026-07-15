-- 0013_serve_advisory_headline.sql — the serve advisory (0012) is no longer only a
-- GPU-out-of-memory notice: it now also surfaces a STALE model server (Ollama too
-- old to load a newer model's architecture — the recurring "installed once, never
-- updated" bite) and a generic serve error. Add the short bold banner so the
-- dashboard card reads accurately per cause instead of hardcoding "GPU out of
-- memory". Back-compat: an existing row (only ever a GPU-OOM) defaults to that
-- headline; the next advisory overwrites the singleton with its own.
ALTER TABLE serve_advisory
    ADD COLUMN headline TEXT NOT NULL DEFAULT 'Couldn''t load a model — GPU out of memory.';
