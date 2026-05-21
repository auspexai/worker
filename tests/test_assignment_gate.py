"""Tests for the M3 assignment gate pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from auspexai_worker.assignment import DecisionKind, GateContext, apply_gates
from auspexai_worker.coordinator import WorkUnitEnvelope
from auspexai_worker.state import (
    AcceptedSensitiveRepository,
    Database,
    ManifestPinRepository,
    MigrationRunner,
    TenantListRepository,
)


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "worker.db")
    MigrationRunner(d).apply_all()
    return d


def _envelope(
    *,
    unit_id: str = "u-1",
    tenant_id: str = "t-1",
    experiment_label: str = "exp-label",
    manifest_sha256: str = "a" * 64,
    payload: dict[str, Any] | None = None,
) -> WorkUnitEnvelope:
    return WorkUnitEnvelope(
        schema_version="0.1",
        unit_id=unit_id,
        tenant_id=tenant_id,
        experiment_id=experiment_label,
        manifest_sha256=manifest_sha256,
        created_at=datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC),
        payload=payload or {},
    )


def _ctx(db: Database, coordinator_experiment_id: str = "exp-coord-1") -> GateContext:
    return GateContext(
        coordinator_experiment_id=coordinator_experiment_id,
        manifest_pins=ManifestPinRepository(db),
        accepted_sensitive=AcceptedSensitiveRepository(db),
        tenant_lists=TenantListRepository(db),
    )


class TestManifestPin:
    def test_first_sighting_accepts(self, db: Database) -> None:
        decision = apply_gates(_envelope(), _ctx(db))
        assert decision.accepted

    def test_same_manifest_repeats_accept(self, db: Database) -> None:
        ctx = _ctx(db)
        apply_gates(_envelope(), ctx)
        decision = apply_gates(_envelope(), ctx)
        assert decision.accepted

    def test_swap_detected_rejects(self, db: Database) -> None:
        ctx = _ctx(db)
        apply_gates(_envelope(manifest_sha256="a" * 64), ctx)
        decision = apply_gates(_envelope(manifest_sha256="b" * 64), ctx)
        assert decision.kind == DecisionKind.REFUSED_MANIFEST_SWAP


class TestTenantGate:
    def test_deny_blocks(self, db: Database) -> None:
        TenantListRepository(db).deny_add("t-1")
        decision = apply_gates(_envelope(tenant_id="t-1"), _ctx(db))
        assert decision.kind == DecisionKind.REFUSED_TENANT_DENY

    def test_allow_list_miss_blocks(self, db: Database) -> None:
        TenantListRepository(db).allow_add("t-2")
        decision = apply_gates(_envelope(tenant_id="t-1"), _ctx(db))
        assert decision.kind == DecisionKind.REFUSED_TENANT_ALLOW_LIST_MISS

    def test_allow_list_hit_accepts(self, db: Database) -> None:
        TenantListRepository(db).allow_add("t-1")
        decision = apply_gates(_envelope(tenant_id="t-1"), _ctx(db))
        assert decision.accepted


class TestSensitiveContent:
    def test_sensitive_flags_default_decline(self, db: Database) -> None:
        env = _envelope(payload={"sensitive_content_flags": ["dual-use"]})
        decision = apply_gates(env, _ctx(db))
        assert decision.kind == DecisionKind.REFUSED_SENSITIVE

    def test_sensitive_with_explicit_accept_passes(self, db: Database) -> None:
        ctx = _ctx(db, "exp-coord-1")
        AcceptedSensitiveRepository(db).accept("exp-coord-1")
        env = _envelope(payload={"sensitive_content_flags": ["dual-use"]})
        decision = apply_gates(env, ctx)
        assert decision.accepted

    def test_empty_sensitive_list_is_not_sensitive(self, db: Database) -> None:
        env = _envelope(payload={"sensitive_content_flags": []})
        decision = apply_gates(env, _ctx(db))
        assert decision.accepted

    def test_missing_field_treated_as_not_sensitive(self, db: Database) -> None:
        # The coordinator's M6d envelope doesn't yet ship sensitive flags;
        # missing field should let the assignment through (with the gate
        # being a no-op until the coordinator catches up).
        env = _envelope(payload={"unrelated": "data"})
        decision = apply_gates(env, _ctx(db))
        assert decision.accepted


class TestGateOrdering:
    def test_swap_takes_priority_over_tenant(self, db: Database) -> None:
        """Swap is caught first because the pin lookup is keyed by
        coordinator_experiment_id, not by tenant_id; we want the operator
        to see the swap signal even when the tenant happens to be denied."""
        TenantListRepository(db).deny_add("t-1")
        ctx = _ctx(db)
        apply_gates(_envelope(manifest_sha256="a" * 64), ctx)
        decision = apply_gates(_envelope(tenant_id="t-1", manifest_sha256="b" * 64), ctx)
        assert decision.kind == DecisionKind.REFUSED_MANIFEST_SWAP
