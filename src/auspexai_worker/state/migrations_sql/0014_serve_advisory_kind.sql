-- 0014_serve_advisory_kind.sql — the serve advisory (0012/0013) now AUTO-CLEARS when
-- the volunteer's recommended fix takes effect (daemon.advisory_recovery), so the
-- card doesn't linger until the coordinator happens to assign the next unit. The
-- daemon needs to know WHICH recovery signal clears it — free memory recovering
-- (GPU-OOM), Ollama updated to the floor (stale backend), or the next serve (generic
-- error). Add the kind. Back-compat: an existing row was only ever a GPU-OOM.
ALTER TABLE serve_advisory ADD COLUMN kind TEXT NOT NULL DEFAULT 'gpu_oom';
-- The live available memory when the advisory was raised — the GPU-OOM baseline. The
-- daemon clears the card when free memory rises above this by a margin (the volunteer
-- ran drop_caches / freed memory). NULL = unknown (no live-mem probe) → not used.
ALTER TABLE serve_advisory ADD COLUMN available_at_raise_gb REAL;
