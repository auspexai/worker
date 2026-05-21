"""Tests for the local state DB + WorkerSelfRepository."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from auspexai_worker.state import Database, MigrationRunner, WorkerSelfRepository
from auspexai_worker.state.repository import AlreadyEnrolledError


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "worker.db")


@pytest.fixture
def repo(db: Database) -> WorkerSelfRepository:
    MigrationRunner(db).apply_all()
    return WorkerSelfRepository(db)


class TestMigrations:
    def test_apply_all_creates_worker_self_table(self, db: Database) -> None:
        applied = MigrationRunner(db).apply_all()
        # M1 = 0001_init, M3 = 0002_m3. Test the prefix shape rather than the
        # exact list so adding migrations doesn't break this assertion.
        assert applied[:2] == [1, 2]
        rows = db.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = [row["name"] for row in rows]
        assert "worker_self" in names
        assert "schema_migrations" in names
        # M3 tables also present.
        for t in (
            "manifest_pins",
            "accepted_sensitive_experiments",
            "tenant_allow_list",
            "tenant_deny_list",
            "assignment_audit",
        ):
            assert t in names, f"M3 table {t!r} missing after migrations"

    def test_re_apply_is_noop(self, db: Database) -> None:
        runner = MigrationRunner(db)
        runner.apply_all()
        assert runner.apply_all() == []

    def test_single_row_constraint(self, db: Database) -> None:
        MigrationRunner(db).apply_all()
        db.connection.execute(
            "INSERT INTO worker_self (id, worker_id, trust_tier, pubkey_hex, enrolled_at) "
            "VALUES (1, 'wkr-a', 0, 'aa', '2026-05-20T00:00:00+00:00')"
        )
        # Second row with id=1 → primary key conflict.
        with pytest.raises(sqlite3.IntegrityError):
            db.connection.execute(
                "INSERT INTO worker_self (id, worker_id, trust_tier, pubkey_hex, enrolled_at) "
                "VALUES (1, 'wkr-b', 0, 'bb', '2026-05-20T00:00:00+00:00')"
            )
        # id=2 → CHECK fails.
        with pytest.raises(sqlite3.IntegrityError):
            db.connection.execute(
                "INSERT INTO worker_self (id, worker_id, trust_tier, pubkey_hex, enrolled_at) "
                "VALUES (2, 'wkr-b', 0, 'bb', '2026-05-20T00:00:00+00:00')"
            )


class TestWorkerSelfRepository:
    def test_get_returns_none_when_empty(self, repo: WorkerSelfRepository) -> None:
        assert repo.get() is None

    def test_insert_then_get_roundtrips(self, repo: WorkerSelfRepository) -> None:
        now = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
        repo.insert(
            worker_id="wkr-abc",
            trust_tier=0,
            pubkey_hex="a" * 64,
            enrolled_at=now,
        )
        loaded = repo.get()
        assert loaded is not None
        assert loaded.worker_id == "wkr-abc"
        assert loaded.trust_tier == 0
        assert loaded.pubkey_hex == "a" * 64
        assert loaded.enrolled_at == now
        assert loaded.last_heartbeat_at is None

    def test_insert_twice_raises(self, repo: WorkerSelfRepository) -> None:
        now = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
        repo.insert(worker_id="wkr-a", trust_tier=0, pubkey_hex="a" * 64, enrolled_at=now)
        with pytest.raises(AlreadyEnrolledError):
            repo.insert(worker_id="wkr-b", trust_tier=0, pubkey_hex="b" * 64, enrolled_at=now)

    def test_update_tier(self, repo: WorkerSelfRepository) -> None:
        now = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
        repo.insert(worker_id="wkr-a", trust_tier=0, pubkey_hex="a" * 64, enrolled_at=now)
        repo.update_tier(1)
        assert repo.get().trust_tier == 1  # type: ignore[union-attr]

    def test_record_heartbeat(self, repo: WorkerSelfRepository) -> None:
        now = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
        repo.insert(worker_id="wkr-a", trust_tier=0, pubkey_hex="a" * 64, enrolled_at=now)
        beat_at = datetime(2026, 5, 20, 12, 30, 0, tzinfo=UTC)
        repo.record_heartbeat(beat_at)
        assert repo.get().last_heartbeat_at == beat_at  # type: ignore[union-attr]

    def test_delete_clears(self, repo: WorkerSelfRepository) -> None:
        now = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
        repo.insert(worker_id="wkr-a", trust_tier=0, pubkey_hex="a" * 64, enrolled_at=now)
        repo.delete()
        assert repo.get() is None
