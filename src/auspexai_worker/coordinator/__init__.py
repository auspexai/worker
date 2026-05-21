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
    ResultAlreadySubmittedError,
    ResultSubmissionResponse,
    UnauthorizedError,
    UnitIdMismatchError,
    WorkerIdMismatchError,
    WorkerNotFoundError,
    WorkerPubkeyMismatchError,
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
    "ResultAlreadySubmittedError",
    "ResultSubmissionResponse",
    "UnauthorizedError",
    "UnitIdMismatchError",
    "WorkUnitEnvelope",
    "WorkerIdMismatchError",
    "WorkerNotFoundError",
    "WorkerPubkeyMismatchError",
    "WorkerStatusResponse",
]
