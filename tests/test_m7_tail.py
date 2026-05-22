"""Tests for M7-tail (worker side) — canonical-receipt fetch loop.

Covers:
- CoordinatorClient.get_canonical_receipt happy path + 404 + transport
  errors
- SubmittedResultRepository.list_pending_canonical filter behavior
- Dispatcher.fetch_pending_canonical promotes placeholder rows on 200,
  leaves them on 404, swallows transport errors
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from auspexai_worker.coordinator import CoordinatorClient
from auspexai_worker.coordinator.client import (
    CanonicalReceiptResponse,
    UnauthorizedError,
)
from auspexai_worker.daemon.dispatch import RunnerDispatcher
from auspexai_worker.signing import Rfc9421Signer
from auspexai_worker.state import Database, MigrationRunner
from auspexai_worker.state.m3_repositories import (
    PendingSubmissionRepository,
    SubmittedResultRepository,
)
from auspexai_worker.workspace import WorkspaceManager

# ---- fixtures ----------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "worker.db")
    MigrationRunner(database).apply_all()
    return database


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


def _seed_placeholder_row(repo: SubmittedResultRepository, *, result_id: str) -> None:
    """Insert a submitted_results row in 'placeholder' state ready for the
    M7-tail fetch loop to promote."""
    repo.record(
        unit_id="u-test",
        assignment_id="asg-test",
        result_id=result_id,
        exit_code=0,
        completed_at="2026-05-22T10:00:00+00:00",
        coord_unit_status_after="completed",
        coord_completions_so_far=3,
        coord_replication_target=3,
        payload_json=json.dumps({"answer": 42}),
    )


def _make_dispatcher(
    *,
    db: Database,
    privkey: Ed25519PrivateKey,
    pub: str,
    client: CoordinatorClient,
    workspace_dir: Path,
) -> RunnerDispatcher:
    import sys

    return RunnerDispatcher(
        coordinator=client,
        worker_id="wkr-tail",
        worker_pubkey=pub,
        privkey=privkey,
        workspace_manager=WorkspaceManager(workspace_dir),
        submitted_repo=SubmittedResultRepository(db),
        pending_repo=PendingSubmissionRepository(db),
        use_bubblewrap=False,
        runner_bin=str(Path(sys.executable).parent / "auspexai-worker-runner"),
        max_pending_attempts=100,
    )


# ---- CoordinatorClient.get_canonical_receipt --------------------------


class TestGetCanonicalReceiptClient:
    def test_200_returns_decoded_response(self) -> None:
        privkey, pub = _make_key()
        signer = Rfc9421Signer(privkey, pubkey_hex=pub)

        cose_bytes = b"\x84\x40\xa0\x40"  # arbitrary; tests parsing not COSE

        def handler(req: httpx.Request) -> httpx.Response:
            assert "/canonical-receipt" in str(req.url)
            return httpx.Response(
                200,
                json={
                    "receipt_id": "rcpt-abc",
                    "experiment_id": "exp-1",
                    "cose_signed_blob_b64": base64.b64encode(cose_bytes).decode(),
                    "signing_key_pubkey_hex": "ff" * 32,
                },
            )

        client = _coord_client(handler, signer)
        resp = client.get_canonical_receipt(worker_id="wkr-tail", result_id="res-1")
        assert resp is not None
        assert resp.receipt_id == "rcpt-abc"
        assert resp.experiment_id == "exp-1"
        assert resp.cose_signed_blob == cose_bytes
        assert resp.signing_key_pubkey_hex == "ff" * 32

    def test_404_returns_none(self) -> None:
        privkey, pub = _make_key()
        signer = Rfc9421Signer(privkey, pubkey_hex=pub)

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                json={"detail": {"error": {"code": "receipt_not_issued", "message": "x"}}},
            )

        client = _coord_client(handler, signer)
        resp = client.get_canonical_receipt(worker_id="wkr-tail", result_id="res-missing")
        assert resp is None

    def test_403_raises_unauthorized(self) -> None:
        privkey, pub = _make_key()
        signer = Rfc9421Signer(privkey, pubkey_hex=pub)

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                403,
                json={"detail": {"error": {"code": "forbidden", "message": "no"}}},
            )

        client = _coord_client(handler, signer)
        with pytest.raises(UnauthorizedError):
            client.get_canonical_receipt(worker_id="wkr-tail", result_id="res-x")


# ---- SubmittedResultRepository.list_pending_canonical -----------------


class TestListPendingCanonical:
    def test_returns_only_placeholder_rows(self, db: Database) -> None:
        repo = SubmittedResultRepository(db)
        _seed_placeholder_row(repo, result_id="res-1")
        _seed_placeholder_row(repo, result_id="res-2")
        _seed_placeholder_row(repo, result_id="res-3")

        # Promote one row to canonical.
        repo.set_canonical(
            result_id="res-2",
            canonical_blob=b"some-cose-bytes",
            canonical_format="cose-sign1-cbor-receipt-v0.1",
            fetched_at=datetime.now(UTC),
        )

        pending = repo.list_pending_canonical()
        assert {p.result_id for p in pending} == {"res-1", "res-3"}

    def test_empty_when_no_rows(self, db: Database) -> None:
        repo = SubmittedResultRepository(db)
        assert repo.list_pending_canonical() == []


# ---- Dispatcher.fetch_pending_canonical -------------------------------


class TestFetchPendingCanonical:
    def test_promotes_placeholder_rows_on_200(self, tmp_path: Path, db: Database) -> None:
        repo = SubmittedResultRepository(db)
        _seed_placeholder_row(repo, result_id="res-A")
        _seed_placeholder_row(repo, result_id="res-B")

        cose_bytes = b"\x84\x40\xa0\x40"
        privkey, pub = _make_key()
        signer = Rfc9421Signer(privkey, pubkey_hex=pub)

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "receipt_id": "rcpt-xyz",
                    "experiment_id": "exp-1",
                    "cose_signed_blob_b64": base64.b64encode(cose_bytes).decode(),
                    "signing_key_pubkey_hex": "ee" * 32,
                },
            )

        client = _coord_client(handler, signer)
        dispatcher = _make_dispatcher(
            db=db,
            privkey=privkey,
            pub=pub,
            client=client,
            workspace_dir=tmp_path / "wsp",
        )
        promoted = dispatcher.fetch_pending_canonical()
        assert promoted == 2

        # Both rows now canonical with the COSE bytes stored.
        for result_id in ("res-A", "res-B"):
            row = repo.get_by_result_id(result_id)
            assert row is not None
            assert row.receipt_status == "canonical"
            assert row.canonical_blob == cose_bytes
            assert row.canonical_format == "cose-sign1-cbor-receipt-v0.1"
            assert row.canonical_fetched_at is not None

    def test_404_leaves_row_as_placeholder(self, tmp_path: Path, db: Database) -> None:
        repo = SubmittedResultRepository(db)
        _seed_placeholder_row(repo, result_id="res-pending")

        privkey, pub = _make_key()
        signer = Rfc9421Signer(privkey, pubkey_hex=pub)

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                json={"detail": {"error": {"code": "receipt_not_issued", "message": "x"}}},
            )

        client = _coord_client(handler, signer)
        dispatcher = _make_dispatcher(
            db=db,
            privkey=privkey,
            pub=pub,
            client=client,
            workspace_dir=tmp_path / "wsp",
        )
        promoted = dispatcher.fetch_pending_canonical()
        assert promoted == 0

        row = repo.get_by_result_id("res-pending")
        assert row is not None
        assert row.receipt_status == "placeholder"
        assert row.canonical_blob is None

    def test_transport_error_swallowed(self, tmp_path: Path, db: Database) -> None:
        """A coord that's unreachable should not crash the loop — fetch
        returns 0 and the row stays placeholder."""
        repo = SubmittedResultRepository(db)
        _seed_placeholder_row(repo, result_id="res-tx-err")

        privkey, pub = _make_key()
        signer = Rfc9421Signer(privkey, pubkey_hex=pub)

        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("coord unreachable")

        client = _coord_client(handler, signer)
        dispatcher = _make_dispatcher(
            db=db,
            privkey=privkey,
            pub=pub,
            client=client,
            workspace_dir=tmp_path / "wsp",
        )
        # Doesn't raise; logs and continues.
        promoted = dispatcher.fetch_pending_canonical()
        assert promoted == 0

        row = repo.get_by_result_id("res-tx-err")
        assert row is not None
        assert row.receipt_status == "placeholder"

    def test_empty_pending_returns_zero(self, tmp_path: Path, db: Database) -> None:
        privkey, pub = _make_key()
        signer = Rfc9421Signer(privkey, pubkey_hex=pub)

        def handler(req: httpx.Request) -> httpx.Response:
            # Should not be called when there's nothing pending.
            raise AssertionError("coord called unexpectedly")

        client = _coord_client(handler, signer)
        dispatcher = _make_dispatcher(
            db=db,
            privkey=privkey,
            pub=pub,
            client=client,
            workspace_dir=tmp_path / "wsp",
        )
        promoted = dispatcher.fetch_pending_canonical()
        assert promoted == 0

    def test_mix_200_and_404_promotes_only_200(self, tmp_path: Path, db: Database) -> None:
        """Two pending rows; coord returns 200 for one and 404 for the
        other; only the first gets promoted."""
        repo = SubmittedResultRepository(db)
        _seed_placeholder_row(repo, result_id="res-ok")
        _seed_placeholder_row(repo, result_id="res-miss")

        cose_bytes = b"\x84\x40\xa0\x40"
        privkey, pub = _make_key()
        signer = Rfc9421Signer(privkey, pubkey_hex=pub)

        def handler(req: httpx.Request) -> httpx.Response:
            if "res-ok" in str(req.url):
                return httpx.Response(
                    200,
                    json={
                        "receipt_id": "rcpt-1",
                        "experiment_id": "exp-1",
                        "cose_signed_blob_b64": base64.b64encode(cose_bytes).decode(),
                        "signing_key_pubkey_hex": "aa" * 32,
                    },
                )
            return httpx.Response(404, json={"detail": {"error": {"code": "x", "message": "y"}}})

        client = _coord_client(handler, signer)
        dispatcher = _make_dispatcher(
            db=db,
            privkey=privkey,
            pub=pub,
            client=client,
            workspace_dir=tmp_path / "wsp",
        )
        promoted = dispatcher.fetch_pending_canonical()
        assert promoted == 1

        assert repo.get_by_result_id("res-ok").receipt_status == "canonical"
        assert repo.get_by_result_id("res-miss").receipt_status == "placeholder"


# ---- Sanity import: CanonicalReceiptResponse is exported --------------


def test_canonical_receipt_response_dataclass_shape() -> None:
    resp = CanonicalReceiptResponse(
        receipt_id="r",
        experiment_id="e",
        cose_signed_blob=b"x",
        signing_key_pubkey_hex="aa" * 32,
    )
    assert resp.cose_signed_blob == b"x"
