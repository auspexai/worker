"""Decision logic for an incoming assignment.

`apply_gates(envelope, ctx)` runs the three checks in order:

  1. Manifest pin (§5.14): first sighting → pin; subsequent sighting under
     a different hash → refuse swap.
  2. Sensitive-content gate (§5.14 + §5.12): envelope-declared sensitive
     flags require an explicit `accept <experiment-id>` to proceed.
  3. Tenant allow/deny gate (§5.14): deny-listed tenants are refused;
     when an allow-list is configured, only tenants on it are accepted.

Returns `AssignmentDecision(kind, reason)`. The caller is responsible for
writing the audit row and handing off to the runner (M4+).

**Coordinator dependency for sensitive-content gate:** M3 reads sensitive
flags from `envelope.payload["sensitive_content_flags"]` (a list of strings;
non-empty means sensitive). The coordinator's M6d `WorkUnitEnvelopeOut`
does NOT yet ship this field, so the gate is currently a no-op against
the M3-era coordinator. When the coordinator starts emitting the flag —
likely as part of receipt/manifest plumbing in coordinator M7 or a
dedicated follow-up — this gate activates without further worker change.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from auspexai_worker.coordinator import WorkUnitEnvelope
from auspexai_worker.state import (
    AcceptedSensitiveRepository,
    ManifestPinRepository,
    PinResult,
    TenantListRepository,
)


class DecisionKind(Enum):
    ACCEPTED = "accepted"
    REFUSED_MANIFEST_SWAP = "refused_manifest_swap"
    REFUSED_SENSITIVE = "refused_sensitive"
    REFUSED_TENANT_DENY = "refused_tenant_deny"
    REFUSED_TENANT_ALLOW_LIST_MISS = "refused_tenant_allow_list_miss"


@dataclass(frozen=True)
class AssignmentDecision:
    kind: DecisionKind
    reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.kind == DecisionKind.ACCEPTED


@dataclass(frozen=True)
class GateContext:
    coordinator_experiment_id: str
    manifest_pins: ManifestPinRepository
    accepted_sensitive: AcceptedSensitiveRepository
    tenant_lists: TenantListRepository


def _sensitive_flags(envelope: WorkUnitEnvelope) -> list[str]:
    """Extract sensitive-content flags from the payload.

    See module docstring re: coordinator-side dependency. Until the
    coordinator ships these flags, this returns [] for every envelope and
    the gate is a no-op. The contract is in place for the day the
    coordinator catches up.
    """
    raw = envelope.payload.get("sensitive_content_flags")
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw if isinstance(x, str)]


def apply_gates(envelope: WorkUnitEnvelope, ctx: GateContext) -> AssignmentDecision:
    """Run the three gates against an envelope. Pure-ish — does write the
    manifest pin on first sighting (the pin is intrinsic to "have we
    accepted this manifest" and there's no useful intermediate state)."""
    # ---- 1. manifest pin ----------------------------------------------
    pin_result = ctx.manifest_pins.check_or_pin(
        coordinator_experiment_id=ctx.coordinator_experiment_id,
        manifest_sha256=envelope.manifest_sha256,
        tenant_id=envelope.tenant_id,
        tenant_experiment_label=envelope.experiment_id,
    )
    if pin_result == PinResult.SWAP_DETECTED:
        return AssignmentDecision(
            kind=DecisionKind.REFUSED_MANIFEST_SWAP,
            reason=(
                f"experiment {ctx.coordinator_experiment_id} previously pinned to a "
                f"different manifest hash; refusing assignment with hash "
                f"{envelope.manifest_sha256}"
            ),
        )

    # ---- 2. tenant allow/deny -----------------------------------------
    blocked, tenant_reason = ctx.tenant_lists.is_blocked(envelope.tenant_id)
    if blocked:
        if tenant_reason == "tenant_deny":
            return AssignmentDecision(
                kind=DecisionKind.REFUSED_TENANT_DENY,
                reason=f"tenant {envelope.tenant_id} is on the deny list",
            )
        if tenant_reason == "tenant_allow_list_miss":
            return AssignmentDecision(
                kind=DecisionKind.REFUSED_TENANT_ALLOW_LIST_MISS,
                reason=(f"tenant {envelope.tenant_id} is not on the configured allow list"),
            )

    # ---- 3. sensitive-content gate ------------------------------------
    flags = _sensitive_flags(envelope)
    if flags and not ctx.accepted_sensitive.contains(ctx.coordinator_experiment_id):
        return AssignmentDecision(
            kind=DecisionKind.REFUSED_SENSITIVE,
            reason=(
                f"experiment {ctx.coordinator_experiment_id} carries sensitive "
                f"flags {flags}; explicit `accept` required"
            ),
        )

    return AssignmentDecision(kind=DecisionKind.ACCEPTED)
