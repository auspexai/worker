"""Tests for M6-tail: write-before-submit + pending_submissions retry queue.

Closes the result-loss gap: a worker that finished a unit and signed its
Result must not lose it if the coordinator is unreachable.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from auspexai_worker.coordinator import (
    AssignmentResponse,
    CoordinatorClient,
    WorkUnitEnvelope,
)
from auspexai_worker.daemon.dispatch import (
    DispatchOutcomeKind,
    RunnerDispatcher,
)
from auspexai_worker.signing import Rfc9421Signer
from auspexai_worker.state import (
    Database,
    MigrationRunner,
    PendingSubmissionRepository,
    SubmittedResultRepository,
)
from auspexai_worker.workspace import WorkspaceManager


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "worker.db")
    MigrationRunner(d).apply_all()
    return d


# ---- PendingSubmissionRepository ------------------------------------------


class TestPendingSubmissionRepository:
    def test_queue_and_get_by_unit(self, db: Database) -> None:
        repo = PendingSubmissionRepository(db)
        repo.queue(
            unit_id="u-1",
            assignment_id="asg-1",
            completed_at="2026-05-22T10:00:00",
            exit_code=0,
            payload_json='{"k":"v"}',
            worker_signature="sig-base64",
            worker_pubkey="a" * 64,
        )
        row = repo.get_by_unit("u-1")
        assert row is not None
        assert row.unit_id == "u-1"
        assert row.assignment_id == "asg-1"
        assert row.attempt_count == 0
        assert row.failure_kind is None

    def test_queue_duplicate_unit_id_raises(self, db: Database) -> None:
        repo = PendingSubmissionRepository(db)
        repo.queue(
            unit_id="u-1",
            assignment_id="asg-1",
            completed_at="2026-05-22T10:00:00",
            exit_code=0,
            payload_json="{}",
            worker_signature="sig",
            worker_pubkey="a" * 64,
        )
        with pytest.raises(Exception):  # noqa: B017 — sqlite IntegrityError or wrapped
            repo.queue(
                unit_id="u-1",
                assignment_id="asg-2",
                completed_at="2026-05-22T11:00:00",
                exit_code=0,
                payload_json="{}",
                worker_signature="sig",
                worker_pubkey="a" * 64,
            )

    def test_list_retryable_excludes_terminal(self, db: Database) -> None:
        repo = PendingSubmissionRepository(db)
        for i in range(3):
            repo.queue(
                unit_id=f"u-{i}",
                assignment_id=f"asg-{i}",
                completed_at="2026-05-22T10:00:00",
                exit_code=0,
                payload_json="{}",
                worker_signature="sig",
                worker_pubkey="a" * 64,
            )
        repo.mark_attempt(
            unit_id="u-1",
            failure_kind="terminal",
            failure_reason="some 4xx",
            attempted_at=datetime.now(UTC),
        )
        retryable = repo.list_retryable()
        retryable_ids = {r.unit_id for r in retryable}
        assert retryable_ids == {"u-0", "u-2"}

    def test_mark_attempt_increments_counter(self, db: Database) -> None:
        repo = PendingSubmissionRepository(db)
        repo.queue(
            unit_id="u-1",
            assignment_id="asg-1",
            completed_at="2026-05-22T10:00:00",
            exit_code=0,
            payload_json="{}",
            worker_signature="sig",
            worker_pubkey="a" * 64,
        )
        for _ in range(3):
            repo.mark_attempt(
                unit_id="u-1",
                failure_kind="transient",
                failure_reason="net error",
                attempted_at=datetime.now(UTC),
            )
        row = repo.get_by_unit("u-1")
        assert row is not None
        assert row.attempt_count == 3
        assert row.failure_kind == "transient"

    def test_mark_attempt_rejects_bad_failure_kind(self, db: Database) -> None:
        repo = PendingSubmissionRepository(db)
        repo.queue(
            unit_id="u-1",
            assignment_id=None,
            completed_at="2026-05-22T10:00:00",
            exit_code=0,
            payload_json="{}",
            worker_signature="sig",
            worker_pubkey="a" * 64,
        )
        with pytest.raises(ValueError, match="must be 'transient' or 'terminal'"):
            repo.mark_attempt(
                unit_id="u-1",
                failure_kind="bogus",
                failure_reason="x",
                attempted_at=datetime.now(UTC),
            )

    def test_remove(self, db: Database) -> None:
        repo = PendingSubmissionRepository(db)
        repo.queue(
            unit_id="u-1",
            assignment_id=None,
            completed_at="2026-05-22T10:00:00",
            exit_code=0,
            payload_json="{}",
            worker_signature="sig",
            worker_pubkey="a" * 64,
        )
        repo.remove("u-1")
        assert repo.get_by_unit("u-1") is None

    def test_list_retryable_oldest_first(self, db: Database) -> None:
        repo = PendingSubmissionRepository(db)
        for i in range(5):
            repo.queue(
                unit_id=f"u-{i}",
                assignment_id=None,
                completed_at="2026-05-22T10:00:00",
                exit_code=0,
                payload_json="{}",
                worker_signature="sig",
                worker_pubkey="a" * 64,
            )
        retryable = repo.list_retryable(limit=3)
        # SQLite CURRENT_TIMESTAMP at second resolution ties; we tie-break
        # by id ASC, so the first three inserted come back first.
        assert [r.unit_id for r in retryable] == ["u-0", "u-1", "u-2"]


# ---- Dispatcher write-before-submit + retry -------------------------------


def _make_key() -> tuple[Ed25519PrivateKey, str]:
    pk = Ed25519PrivateKey.generate()
    pub = pk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    return pk, pub


def _coord_client(handler, signer: Rfc9421Signer) -> CoordinatorClient:
    return CoordinatorClient(
        base_url="http://coord-test.invalid",
        signer=signer,
        transport=httpx.MockTransport(handler),
    )


def _envelope(unit_id: str = "u-1") -> AssignmentResponse:
    return AssignmentResponse(
        assignment_id="asg-1",
        assigned_at=datetime(2026, 5, 22, 10, 0, 0, tzinfo=UTC),
        coordinator_experiment_id="exp-1",
        work_unit=WorkUnitEnvelope(
            schema_version="0.1",
            unit_id=unit_id,
            tenant_id="t-1",
            experiment_id="exp-tenant-label",
            manifest_sha256="a" * 64,
            created_at=datetime(2026, 5, 22, 10, 0, 0, tzinfo=UTC),
            payload={"input": 1},
        ),
    )


def _runner_bin() -> str:
    import sys

    return str(Path(sys.executable).parent / "auspexai-worker-runner")


def _make_dispatcher(
    *,
    db: Database,
    privkey: Ed25519PrivateKey,
    pub: str,
    client: CoordinatorClient,
    workspace_dir: Path,
    max_pending_attempts: int = 100,
) -> RunnerDispatcher:
    return RunnerDispatcher(
        coordinator=client,
        worker_id="wkr-a",
        worker_pubkey=pub,
        privkey=privkey,
        workspace_manager=WorkspaceManager(workspace_dir),
        submitted_repo=SubmittedResultRepository(db),
        pending_repo=PendingSubmissionRepository(db),
        use_bubblewrap=False,
        runner_bin=_runner_bin(),
        max_pending_attempts=max_pending_attempts,
    )


class TestWriteBeforeSubmitHappyPath:
    def test_success_writes_submitted_then_clears_pending(
        self, tmp_path: Path, db: Database
    ) -> None:
        privkey, pub = _make_key()

        def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            return httpx.Response(
                201,
                json={
                    "result_id": "res-1",
                    "unit_id": body["unit_id"],
                    "unit_status_after": "in_progress",
                    "completions_so_far": 1,
                    "replication_target": 3,
                },
            )

        with _coord_client(handler, Rfc9421Signer(privkey, pub)) as client:
            dispatcher = _make_dispatcher(
                db=db,
                privkey=privkey,
                pub=pub,
                client=client,
                workspace_dir=tmp_path / "runs",
            )
            outcome = dispatcher.run_unit(_envelope())

        assert outcome.kind == DispatchOutcomeKind.SUBMITTED
        # pending_submissions row was removed on success.
        assert PendingSubmissionRepository(db).get_by_unit("u-1") is None
        # submitted_results row exists.
        rows = SubmittedResultRepository(db).recent()
        assert len(rows) == 1
        assert rows[0].result_id == "res-1"


class TestWriteBeforeSubmitTransientFailure:
    def test_5xx_leaves_pending_marked_transient(self, tmp_path: Path, db: Database) -> None:
        privkey, pub = _make_key()

        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="coord temporarily unavailable")

        with _coord_client(handler, Rfc9421Signer(privkey, pub)) as client:
            dispatcher = _make_dispatcher(
                db=db,
                privkey=privkey,
                pub=pub,
                client=client,
                workspace_dir=tmp_path / "runs",
            )
            outcome = dispatcher.run_unit(_envelope())

        assert outcome.kind == DispatchOutcomeKind.SUBMIT_FAILED_TRANSIENT
        # pending row remains, marked transient with attempt_count=1.
        pending = PendingSubmissionRepository(db).get_by_unit("u-1")
        assert pending is not None
        assert pending.failure_kind == "transient"
        assert pending.attempt_count == 1
        assert pending.failure_reason is not None
        # No submitted_results row yet.
        assert SubmittedResultRepository(db).recent() == []


class TestWriteBeforeSubmitTerminalFailure:
    def test_403_worker_pubkey_mismatch_marks_terminal(self, tmp_path: Path, db: Database) -> None:
        privkey, pub = _make_key()

        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                403,
                json={
                    "detail": {
                        "error": {
                            "code": "worker_pubkey_mismatch",
                            "message": "Result.worker_pubkey != signing credential",
                        }
                    }
                },
            )

        with _coord_client(handler, Rfc9421Signer(privkey, pub)) as client:
            dispatcher = _make_dispatcher(
                db=db,
                privkey=privkey,
                pub=pub,
                client=client,
                workspace_dir=tmp_path / "runs",
            )
            outcome = dispatcher.run_unit(_envelope())

        assert outcome.kind == DispatchOutcomeKind.SUBMIT_FAILED_TERMINAL
        pending = PendingSubmissionRepository(db).get_by_unit("u-1")
        assert pending is not None
        assert pending.failure_kind == "terminal"
        # No submitted_results row.
        assert SubmittedResultRepository(db).recent() == []


class TestWriteBeforeSubmit409Idempotent:
    def test_409_with_existing_result_id_reconciles(self, tmp_path: Path, db: Database) -> None:
        privkey, pub = _make_key()

        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                409,
                json={
                    "detail": {
                        "error": {
                            "code": "result_already_submitted",
                            "message": "this assignment already has a result",
                            "details": {
                                "assignment_id": "asg-1",
                                "existing_result_id": "res-already-existing",
                            },
                        }
                    }
                },
            )

        with _coord_client(handler, Rfc9421Signer(privkey, pub)) as client:
            dispatcher = _make_dispatcher(
                db=db,
                privkey=privkey,
                pub=pub,
                client=client,
                workspace_dir=tmp_path / "runs",
            )
            outcome = dispatcher.run_unit(_envelope())

        # Net effect identical to successful submit: pending cleared,
        # submitted_results row inserted with the coord's existing_result_id.
        assert outcome.kind == DispatchOutcomeKind.SUBMITTED
        assert "reconciled-via-409" in (outcome.reason or "")
        assert PendingSubmissionRepository(db).get_by_unit("u-1") is None
        rows = SubmittedResultRepository(db).recent()
        assert len(rows) == 1
        assert rows[0].result_id == "res-already-existing"

    def test_409_without_existing_result_id_marks_terminal(
        self, tmp_path: Path, db: Database
    ) -> None:
        # Defensive: if coord returns 409 without the detail field, the
        # worker can't reconcile and must surface to the operator.
        privkey, pub = _make_key()

        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                409,
                json={
                    "detail": {
                        "error": {
                            "code": "result_already_submitted",
                            "message": "already have result",
                        }
                    }
                },
            )

        with _coord_client(handler, Rfc9421Signer(privkey, pub)) as client:
            dispatcher = _make_dispatcher(
                db=db,
                privkey=privkey,
                pub=pub,
                client=client,
                workspace_dir=tmp_path / "runs",
            )
            outcome = dispatcher.run_unit(_envelope())

        assert outcome.kind == DispatchOutcomeKind.SUBMIT_FAILED_TERMINAL
        pending = PendingSubmissionRepository(db).get_by_unit("u-1")
        assert pending is not None
        assert pending.failure_kind == "terminal"


# ---- retry_pending --------------------------------------------------------


class TestRetryPending:
    def test_no_pending_returns_empty(self, tmp_path: Path, db: Database) -> None:
        privkey, pub = _make_key()

        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="should never be called")

        with _coord_client(handler, Rfc9421Signer(privkey, pub)) as client:
            dispatcher = _make_dispatcher(
                db=db,
                privkey=privkey,
                pub=pub,
                client=client,
                workspace_dir=tmp_path / "runs",
            )
            assert dispatcher.retry_pending() == []

    def test_transient_then_success_clears_pending(self, tmp_path: Path, db: Database) -> None:
        privkey, pub = _make_key()
        call_count = {"i": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            call_count["i"] += 1
            if call_count["i"] == 1:
                return httpx.Response(503, text="coord down")
            body = json.loads(req.content)
            return httpx.Response(
                201,
                json={
                    "result_id": "res-2",
                    "unit_id": body["unit_id"],
                    "unit_status_after": "in_progress",
                    "completions_so_far": 1,
                    "replication_target": 3,
                },
            )

        with _coord_client(handler, Rfc9421Signer(privkey, pub)) as client:
            dispatcher = _make_dispatcher(
                db=db,
                privkey=privkey,
                pub=pub,
                client=client,
                workspace_dir=tmp_path / "runs",
            )
            # First dispatch fails transiently.
            outcome1 = dispatcher.run_unit(_envelope())
            assert outcome1.kind == DispatchOutcomeKind.SUBMIT_FAILED_TRANSIENT
            assert PendingSubmissionRepository(db).get_by_unit("u-1") is not None

            # Retry tick succeeds.
            retry_outcomes = dispatcher.retry_pending()

        assert len(retry_outcomes) == 1
        assert retry_outcomes[0].kind == DispatchOutcomeKind.SUBMITTED
        assert PendingSubmissionRepository(db).get_by_unit("u-1") is None
        rows = SubmittedResultRepository(db).recent()
        assert len(rows) == 1
        assert rows[0].result_id == "res-2"

    def test_attempt_cap_promotes_to_terminal(self, tmp_path: Path, db: Database) -> None:
        privkey, pub = _make_key()

        # Pre-seed a pending row with attempt_count already at the cap.
        pending_repo = PendingSubmissionRepository(db)
        pending_repo.queue(
            unit_id="u-cap",
            assignment_id="asg-cap",
            completed_at="2026-05-22T10:00:00",
            exit_code=0,
            payload_json='{"input": 1}',
            worker_signature="sig-base64",
            worker_pubkey=pub,
        )
        # Bump attempt_count to the cap.
        for _ in range(3):
            pending_repo.mark_attempt(
                unit_id="u-cap",
                failure_kind="transient",
                failure_reason="net err",
                attempted_at=datetime.now(UTC),
            )

        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="should not be called")  # cap exceeded

        with _coord_client(handler, Rfc9421Signer(privkey, pub)) as client:
            dispatcher = _make_dispatcher(
                db=db,
                privkey=privkey,
                pub=pub,
                client=client,
                workspace_dir=tmp_path / "runs",
                max_pending_attempts=3,
            )
            outcomes = dispatcher.retry_pending()

        assert len(outcomes) == 1
        assert outcomes[0].kind == DispatchOutcomeKind.SUBMIT_FAILED_TERMINAL
        row = pending_repo.get_by_unit("u-cap")
        assert row is not None
        assert row.failure_kind == "terminal"
        assert "exceeded max_pending_attempts" in (row.failure_reason or "")


# ---- Restart resilience ---------------------------------------------------


class TestRestartSurvivesPending:
    def test_pending_row_picked_up_by_new_dispatcher(self, tmp_path: Path, db: Database) -> None:
        """Simulate worker process restart: pending row exists, new dispatcher
        instance retries from the queue."""
        privkey, pub = _make_key()

        # Pre-seed a pending row as if a prior dispatcher run had queued it.
        PendingSubmissionRepository(db).queue(
            unit_id="u-survives",
            assignment_id="asg-survives",
            completed_at="2026-05-22T10:00:00",
            exit_code=0,
            payload_json='{"input": 42}',
            worker_signature="signed-by-prior-process",
            worker_pubkey=pub,
        )

        # New process / new dispatcher: coord is back; retry succeeds.
        def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            assert body["unit_id"] == "u-survives"
            return httpx.Response(
                201,
                json={
                    "result_id": "res-survived",
                    "unit_id": body["unit_id"],
                    "unit_status_after": "completed",
                    "completions_so_far": 3,
                    "replication_target": 3,
                },
            )

        with _coord_client(handler, Rfc9421Signer(privkey, pub)) as client:
            dispatcher = _make_dispatcher(
                db=db,
                privkey=privkey,
                pub=pub,
                client=client,
                workspace_dir=tmp_path / "runs",
            )
            outcomes = dispatcher.retry_pending()

        assert len(outcomes) == 1
        assert outcomes[0].kind == DispatchOutcomeKind.SUBMITTED
        # submitted_results row carries the worker_signature from the
        # pending queue (the original signature from before the restart),
        # not a newly-computed one.
        rows = SubmittedResultRepository(db).recent()
        assert len(rows) == 1
        assert rows[0].result_id == "res-survived"
        assert PendingSubmissionRepository(db).get_by_unit("u-survives") is None
