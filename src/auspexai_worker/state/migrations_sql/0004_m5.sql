-- M5 schema: extend M4 submitted_results with receipt-canonical columns.
--
-- Per 2026-05-22 design-doc decision (worker_daemon_design.md §10
-- "Receipt storage decision" + §14 M5 entry): receipts live in worker.db,
-- not a separate filesystem tree. Single source of truth across
-- submitted_results (the receipt row) and assignment_audit (audit rows
-- that reference it) — atomic transactions cover both, no DB↔filesystem
-- drift risk, withdrawal is a single transaction with no orphan-file
-- cleanup.
--
-- Allowed values for receipt_status: 'placeholder' / 'canonical' / 'failed'.
-- Enforced in application code (M5 ships with NOT NULL + default; SQLite
-- ADD COLUMN doesn't accept CHECK constraints inline, and table-rebuild
-- for CHECK isn't worth the migration complexity at this volunteer-private
-- altitude).
--
-- M5 ships with receipt_status='placeholder' on every row; coordinator
-- M7 fills canonical_blob + canonical_format + canonical_fetched_at when
-- it ships canonical CBOR+COSE receipts via a post-submit fetch path.
-- Existing pre-M5 rows backfill automatically via the NOT NULL DEFAULT.

ALTER TABLE submitted_results ADD COLUMN receipt_status TEXT NOT NULL DEFAULT 'placeholder';
ALTER TABLE submitted_results ADD COLUMN canonical_blob BLOB;
ALTER TABLE submitted_results ADD COLUMN canonical_format TEXT;
ALTER TABLE submitted_results ADD COLUMN canonical_fetched_at TEXT;
