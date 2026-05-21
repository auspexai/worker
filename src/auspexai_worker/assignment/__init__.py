"""Assignment-handling pipeline (M3).

Three steps run on every received assignment:

  1. Manifest pin / swap defense (§5.14)
  2. Sensitive-content gate (§5.14, §5.12)
  3. Tenant allow/deny gate (§5.14)

The gate logic is a pure function over the envelope + repository snapshots,
returning an `AssignmentDecision`. The caller writes audit rows and decides
whether to drop the assignment (M3) or hand to a runner (M4+).
"""

from __future__ import annotations

from .gate import (
    AssignmentDecision,
    DecisionKind,
    GateContext,
    apply_gates,
)

__all__ = [
    "AssignmentDecision",
    "DecisionKind",
    "GateContext",
    "apply_gates",
]
