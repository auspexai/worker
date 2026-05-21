"""Tests for the runner dispatcher (M4)."""

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
from auspexai_worker.signing import Rfc9421Signer, verify_result_signature
from auspexai_worker.state import (
    Database,
    MigrationRunner,
    SubmittedResultRepository,
)
from auspexai_worker.workspace import WorkspaceManager


def _runner_bin() -> str:
    """Absolute path to the installed runner script, so tests don't rely on PATH."""
    import sys

    return str(Path(sys.executable).parent / "auspexai-worker-runner")


def _make_key() -> tuple[Ed25519PrivateKey, str]:
    pk = Ed25519PrivateKey.generate()
    pub = pk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    return pk, pub


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "worker.db")
    MigrationRunner(d).apply_all()
    return d


def _envelope() -> AssignmentResponse:
    return AssignmentResponse(
        assignment_id="asg-1",
        assigned_at=datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC),
        coordinator_experiment_id="exp-coord-1",
        work_unit=WorkUnitEnvelope(
            schema_version="0.1",
            unit_id="u-1",
            tenant_id="t-1",
            experiment_id="exp-label",
            manifest_sha256="a" * 64,
            created_at=datetime(2026, 5, 20, 11, 0, 0, tzinfo=UTC),
            payload={"input": 42},
        ),
    )


def _make_client(handler, signer) -> CoordinatorClient:
    return CoordinatorClient(
        base_url="http://test-coord.invalid",
        signer=signer,
        transport=httpx.MockTransport(handler),
    )


class TestRunUnitHappyPath:
    def test_submitted_outcome_via_real_runner(self, tmp_path: Path, db: Database) -> None:
        """End-to-end: spawn the actual auspexai-worker-runner binary in
        passthrough mode, read its output, sign, submit, persist."""
        privkey, pub = _make_key()
        captured: dict[str, object] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            body = json.loads(req.content)
            return httpx.Response(
                201,
                json={
                    "result_id": "res-test-001",
                    "unit_id": body["unit_id"],
                    "unit_status_after": "in_progress",
                    "completions_so_far": 1,
                    "replication_target": 3,
                },
            )

        with _make_client(handler, Rfc9421Signer(privkey, pub)) as client:
            dispatcher = RunnerDispatcher(
                coordinator=client,
                worker_id="wkr-a",
                worker_pubkey=pub,
                privkey=privkey,
                workspace_manager=WorkspaceManager(tmp_path / "runs"),
                submitted_repo=SubmittedResultRepository(db),
                use_bubblewrap=False,
                runner_bin=_runner_bin(),
            )
            outcome = dispatcher.run_unit(_envelope())

        assert outcome.kind == DispatchOutcomeKind.SUBMITTED
        assert outcome.result_response is not None
        assert outcome.result_response.result_id == "res-test-001"

        # Signature in the submitted body verifies against pub.
        body = captured["body"]
        assert verify_result_signature(
            pubkey_hex=pub,
            unit_id=body["unit_id"],
            worker_pubkey=body["worker_pubkey"],
            completed_at=body["completed_at"],
            exit_code=body["exit_code"],
            payload=body["payload"],
            signature_b64=body["worker_signature"],
        )
        # Synthetic executor echoed our input back.
        assert body["payload"]["echo"] == {"input": 42}

        # Local record exists.
        rows = SubmittedResultRepository(db).recent()
        assert len(rows) == 1
        assert rows[0].unit_id == "u-1"
        assert rows[0].result_id == "res-test-001"


class TestRunUnitFailurePaths:
    def test_runner_failure_when_binary_missing(self, tmp_path: Path, db: Database) -> None:
        privkey, pub = _make_key()

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(500)  # never called

        with _make_client(handler, Rfc9421Signer(privkey, pub)) as client:
            dispatcher = RunnerDispatcher(
                coordinator=client,
                worker_id="wkr-a",
                worker_pubkey=pub,
                privkey=privkey,
                workspace_manager=WorkspaceManager(tmp_path / "runs"),
                submitted_repo=SubmittedResultRepository(db),
                use_bubblewrap=False,
                runner_bin="auspexai-worker-runner-DOES-NOT-EXIST",
            )
            outcome = dispatcher.run_unit(_envelope())
        assert outcome.kind == DispatchOutcomeKind.SANDBOX_UNAVAILABLE

    def test_submit_failure_returns_typed_outcome(self, tmp_path: Path, db: Database) -> None:
        privkey, pub = _make_key()

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="coord blew up")

        with _make_client(handler, Rfc9421Signer(privkey, pub)) as client:
            dispatcher = RunnerDispatcher(
                coordinator=client,
                worker_id="wkr-a",
                worker_pubkey=pub,
                privkey=privkey,
                workspace_manager=WorkspaceManager(tmp_path / "runs"),
                submitted_repo=SubmittedResultRepository(db),
                use_bubblewrap=False,
                runner_bin=_runner_bin(),
            )
            outcome = dispatcher.run_unit(_envelope())
        assert outcome.kind == DispatchOutcomeKind.SUBMIT_FAILED
        # Workspace got cleaned up despite the submit failure.
        assert not (tmp_path / "runs" / "u-1").exists()


class TestWorkspaceCleanup:
    def test_workspace_cleaned_up_on_success(self, tmp_path: Path, db: Database) -> None:
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

        with _make_client(handler, Rfc9421Signer(privkey, pub)) as client:
            dispatcher = RunnerDispatcher(
                coordinator=client,
                worker_id="wkr-a",
                worker_pubkey=pub,
                privkey=privkey,
                workspace_manager=WorkspaceManager(tmp_path / "runs"),
                submitted_repo=SubmittedResultRepository(db),
                use_bubblewrap=False,
                runner_bin=_runner_bin(),
            )
            dispatcher.run_unit(_envelope())
        assert not (tmp_path / "runs" / "u-1").exists()
