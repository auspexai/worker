-- 0011_no_receipt_note.sql — D22-B: a terminal "no canonical receipt" state.
--
-- Before D22-B the M7-tail backfill loop had only two receipt_status values it
-- ever reached — 'placeholder' (waiting) and 'canonical' (fetched) — so a
-- result whose receipt would NEVER issue (the unit's quorum reached consensus
-- without selecting this replica — a VALID non-consensus observation, not a
-- failure; or the experiment went terminal before consensus) stayed
-- 'placeholder' and was re-polled every tick forever (a perpetual 404).
--
-- The coordinator now answers such polls with 410 receipt_will_not_issue. The
-- worker records that terminal outcome as receipt_status='no_receipt' (a new
-- allowed value; receipt_status has no CHECK constraint — allowed values are
-- enforced in application code, see 0004_m5.sql) so the row leaves the
-- list_pending_canonical set and is never re-polled. `receipt_note` carries the
-- coordinator's non-pejorative reason (e.g. 'diverged_from_consensus',
-- 'non_consensus', 'experiment_aborted') so `auspexai-worker receipts show`
-- can explain WHY there is no consensus receipt — the result stays valid and
-- exportable; there is simply no receipt for it.

ALTER TABLE submitted_results ADD COLUMN receipt_note TEXT;
