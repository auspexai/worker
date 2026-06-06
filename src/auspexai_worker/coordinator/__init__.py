"""Coordinator client — HTTP calls the worker makes to the AuspexAI coordinator.

M1 ships only `enroll(...)`; heartbeat/assignment/result methods land in M2/M3/M4.
"""

from __future__ import annotations

from .client import (
    AssignmentAlreadyResolvedError,
    AssignmentNotFoundError,
    AssignmentResponse,
    BindingTokenConsumedError,
    BindingTokenExpiredError,
    BindingTokenNotFoundError,
    CoordinatorClient,
    CoordinatorError,
    EnrollmentResponse,
    InvalidAccessTokenError,
    OAuthExchangeResponse,
    PubkeyAlreadyEnrolledError,
    PubkeyAlreadyTenantError,
    RefuseResponse,
    ResultAlreadySubmittedError,
    ResultSubmissionResponse,
    UnauthorizedError,
    UnitIdMismatchError,
    UnsupportedIdpError,
    WorkerIdMismatchError,
    WorkerNotFoundError,
    WorkerPausedError,
    WorkerPubkeyMismatchError,
    WorkerQuarantinedError,
    WorkerStatusResponse,
    WorkUnitEnvelope,
)

__all__ = [
    "AssignmentAlreadyResolvedError",
    "AssignmentNotFoundError",
    "AssignmentResponse",
    "BindingTokenConsumedError",
    "BindingTokenExpiredError",
    "BindingTokenNotFoundError",
    "CoordinatorClient",
    "CoordinatorError",
    "EnrollmentResponse",
    "InvalidAccessTokenError",
    "OAuthExchangeResponse",
    "PubkeyAlreadyEnrolledError",
    "PubkeyAlreadyTenantError",
    "RefuseResponse",
    "ResultAlreadySubmittedError",
    "ResultSubmissionResponse",
    "UnauthorizedError",
    "UnitIdMismatchError",
    "UnsupportedIdpError",
    "WorkUnitEnvelope",
    "WorkerIdMismatchError",
    "WorkerNotFoundError",
    "WorkerPausedError",
    "WorkerPubkeyMismatchError",
    "WorkerQuarantinedError",
    "WorkerStatusResponse",
]
