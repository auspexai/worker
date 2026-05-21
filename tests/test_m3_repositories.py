"""Tests for M3 state repositories."""

from __future__ import annotations

from pathlib import Path

import pytest

from auspexai_worker.state import (
    AcceptedSensitiveRepository,
    AssignmentAuditRepository,
    Database,
    ManifestPinRepository,
    MigrationRunner,
    PinResult,
    TenantListRepository,
)


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "worker.db")
    MigrationRunner(d).apply_all()
    return d


class TestManifestPinRepository:
    def test_first_sighting_pins(self, db: Database) -> None:
        repo = ManifestPinRepository(db)
        result = repo.check_or_pin(
            coordinator_experiment_id="exp-1",
            manifest_sha256="a" * 64,
            tenant_id="t-1",
            tenant_experiment_label="my-exp",
        )
        assert result == PinResult.NEW_PIN
        pin = repo.get("exp-1")
        assert pin is not None
        assert pin.manifest_sha256 == "a" * 64

    def test_same_hash_matches(self, db: Database) -> None:
        repo = ManifestPinRepository(db)
        repo.check_or_pin(
            coordinator_experiment_id="exp-1",
            manifest_sha256="a" * 64,
            tenant_id="t-1",
            tenant_experiment_label="my-exp",
        )
        result = repo.check_or_pin(
            coordinator_experiment_id="exp-1",
            manifest_sha256="a" * 64,
            tenant_id="t-1",
            tenant_experiment_label="my-exp",
        )
        assert result == PinResult.MATCHED

    def test_different_hash_detects_swap(self, db: Database) -> None:
        repo = ManifestPinRepository(db)
        repo.check_or_pin(
            coordinator_experiment_id="exp-1",
            manifest_sha256="a" * 64,
            tenant_id="t-1",
            tenant_experiment_label="my-exp",
        )
        result = repo.check_or_pin(
            coordinator_experiment_id="exp-1",
            manifest_sha256="b" * 64,
            tenant_id="t-1",
            tenant_experiment_label="my-exp",
        )
        assert result == PinResult.SWAP_DETECTED
        # Original pin unchanged.
        pin = repo.get("exp-1")
        assert pin is not None
        assert pin.manifest_sha256 == "a" * 64


class TestAcceptedSensitiveRepository:
    def test_default_does_not_contain(self, db: Database) -> None:
        repo = AcceptedSensitiveRepository(db)
        assert repo.contains("exp-1") is False

    def test_accept_then_contains(self, db: Database) -> None:
        repo = AcceptedSensitiveRepository(db)
        repo.accept("exp-1")
        assert repo.contains("exp-1") is True

    def test_accept_is_idempotent(self, db: Database) -> None:
        repo = AcceptedSensitiveRepository(db)
        repo.accept("exp-1")
        repo.accept("exp-1")
        assert repo.contains("exp-1") is True

    def test_remove_clears(self, db: Database) -> None:
        repo = AcceptedSensitiveRepository(db)
        repo.accept("exp-1")
        repo.remove("exp-1")
        assert repo.contains("exp-1") is False


class TestTenantListRepository:
    def test_empty_lists_allow_everything(self, db: Database) -> None:
        repo = TenantListRepository(db)
        blocked, reason = repo.is_blocked("t-1")
        assert blocked is False
        assert reason is None

    def test_deny_blocks(self, db: Database) -> None:
        repo = TenantListRepository(db)
        repo.deny_add("t-1")
        blocked, reason = repo.is_blocked("t-1")
        assert blocked is True
        assert reason == "tenant_deny"

    def test_allow_list_blocks_others(self, db: Database) -> None:
        repo = TenantListRepository(db)
        repo.allow_add("t-1")
        blocked, reason = repo.is_blocked("t-2")
        assert blocked is True
        assert reason == "tenant_allow_list_miss"

    def test_allow_list_admits_listed(self, db: Database) -> None:
        repo = TenantListRepository(db)
        repo.allow_add("t-1")
        blocked, reason = repo.is_blocked("t-1")
        assert blocked is False
        assert reason is None

    def test_deny_wins_when_both(self, db: Database) -> None:
        repo = TenantListRepository(db)
        repo.allow_add("t-1")
        repo.deny_add("t-1")
        blocked, reason = repo.is_blocked("t-1")
        assert blocked is True
        assert reason == "tenant_deny"


class TestAssignmentAuditRepository:
    def test_append_and_recent(self, db: Database) -> None:
        repo = AssignmentAuditRepository(db)
        repo.append(action="accepted", unit_id="u-1", tenant_id="t-1")
        repo.append(action="refused_tenant_deny", unit_id="u-2", tenant_id="t-2")
        recent = repo.recent(limit=10)
        assert len(recent) == 2
        # Most recent first.
        assert recent[0].action == "refused_tenant_deny"
        assert recent[1].action == "accepted"

    def test_by_unit(self, db: Database) -> None:
        repo = AssignmentAuditRepository(db)
        repo.append(action="accepted", unit_id="u-1", manifest_sha256="aa")
        repo.append(action="refused_manifest_swap", unit_id="u-1", manifest_sha256="bb")
        repo.append(action="accepted", unit_id="u-2")
        rows = repo.by_unit("u-1")
        assert len(rows) == 2
        assert all(r.unit_id == "u-1" for r in rows)
