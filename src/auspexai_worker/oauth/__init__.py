"""OAuth 2.0 Device Authorization Flow (RFC 8628) client for GitHub.

Used by `auspexai-worker login` (M6) to bind a worker's Ed25519 keypair to
a GitHub identity for the T0→T1 upgrade. Q5 in the principles doc ratified
a hand-rolled implementation rather than pulling in an OAuth library — the
flow is small (~80 LOC), public-client only (no client secret), and a thin
library wrapper would obscure more than it helps at this size.
"""

from __future__ import annotations

from .device_flow import (
    GITHUB_CLIENT_ID,
    GITHUB_SCOPE,
    AccessDeniedError,
    DeviceCode,
    DeviceFlowError,
    ExpiredTokenError,
    run_device_flow,
    start_device_flow,
)

__all__ = [
    "GITHUB_CLIENT_ID",
    "GITHUB_SCOPE",
    "AccessDeniedError",
    "DeviceCode",
    "DeviceFlowError",
    "ExpiredTokenError",
    "run_device_flow",
    "start_device_flow",
]
