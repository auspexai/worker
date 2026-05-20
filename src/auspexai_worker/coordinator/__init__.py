"""Coordinator client — HTTP calls the worker makes to the AuspexAI coordinator.

M1 ships only `enroll(...)`; heartbeat/assignment/result methods land in M2/M3/M4.
"""

from __future__ import annotations

from .client import (
    CoordinatorClient,
    CoordinatorError,
    EnrollmentResponse,
    PubkeyAlreadyEnrolledError,
    PubkeyAlreadyTenantError,
)

__all__ = [
    "CoordinatorClient",
    "CoordinatorError",
    "EnrollmentResponse",
    "PubkeyAlreadyEnrolledError",
    "PubkeyAlreadyTenantError",
]
