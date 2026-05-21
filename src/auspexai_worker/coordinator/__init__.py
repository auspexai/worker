"""Coordinator client — HTTP calls the worker makes to the AuspexAI coordinator.

M1 ships only `enroll(...)`; heartbeat/assignment/result methods land in M2/M3/M4.
"""

from __future__ import annotations

from .client import (
    AssignmentAlreadyResolvedError,
    AssignmentNotFoundError,
    AssignmentResponse,
    CoordinatorClient,
    CoordinatorError,
    EnrollmentResponse,
    PubkeyAlreadyEnrolledError,
    PubkeyAlreadyTenantError,
    RefuseResponse,
    UnauthorizedError,
    WorkerIdMismatchError,
    WorkerNotFoundError,
    WorkerStatusResponse,
    WorkUnitEnvelope,
)

__all__ = [
    "AssignmentAlreadyResolvedError",
    "AssignmentNotFoundError",
    "AssignmentResponse",
    "CoordinatorClient",
    "CoordinatorError",
    "EnrollmentResponse",
    "PubkeyAlreadyEnrolledError",
    "PubkeyAlreadyTenantError",
    "RefuseResponse",
    "UnauthorizedError",
    "WorkUnitEnvelope",
    "WorkerIdMismatchError",
    "WorkerNotFoundError",
    "WorkerStatusResponse",
]
