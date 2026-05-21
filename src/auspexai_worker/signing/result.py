"""Result signing — canonical encoding the worker signs over.

The coordinator's M6d POST /result body carries a `worker_signature` field
that M6d stores verbatim but does NOT verify. M7 will reproduce the
canonical encoding to verify the body chain when issuing receipts. To
keep that future M7 work mechanical, the canonical encoding is defined
here once and exercised by tests with an inline verifier.

Canonical input for the signature:

    json.dumps(
        {
            "unit_id": str,
            "worker_pubkey": str,            # 64-char lowercase hex
            "completed_at": str,             # ISO 8601 UTC with offset
            "exit_code": int,
            "payload": dict,                 # opaque to platform / signer
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

The worker_signature on the wire is `base64.b64encode(ed25519_sign(...))`.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def canonical_result_bytes(
    *,
    unit_id: str,
    worker_pubkey: str,
    completed_at: datetime | str,
    exit_code: int,
    payload: dict[str, Any],
) -> bytes:
    """Produce the canonical encoding the worker signs.

    `completed_at` is accepted as either a datetime (will be `isoformat()`'d)
    or as a pre-formatted ISO 8601 string (so the worker can sign the same
    string it eventually serializes onto the wire).
    """
    ts = completed_at.isoformat() if isinstance(completed_at, datetime) else completed_at
    body = {
        "unit_id": unit_id,
        "worker_pubkey": worker_pubkey.lower(),
        "completed_at": ts,
        "exit_code": int(exit_code),
        "payload": payload,
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_result(
    *,
    privkey: Ed25519PrivateKey,
    pubkey_hex: str,
    unit_id: str,
    completed_at: datetime | str,
    exit_code: int,
    payload: dict[str, Any],
) -> str:
    """Sign the result body and return the base64 signature string."""
    sig_input = canonical_result_bytes(
        unit_id=unit_id,
        worker_pubkey=pubkey_hex,
        completed_at=completed_at,
        exit_code=exit_code,
        payload=payload,
    )
    sig = privkey.sign(sig_input)
    return base64.b64encode(sig).decode("ascii")


def verify_result_signature(
    *,
    pubkey_hex: str,
    unit_id: str,
    worker_pubkey: str,
    completed_at: datetime | str,
    exit_code: int,
    payload: dict[str, Any],
    signature_b64: str,
) -> bool:
    """Test-side verifier. NOT used in production worker code — the
    coordinator (M7) is the canonical verifier. Exposed for round-trip
    tests so we don't have to depend on the platform package.
    """
    pubkey = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex.lower()))
    sig_input = canonical_result_bytes(
        unit_id=unit_id,
        worker_pubkey=worker_pubkey,
        completed_at=completed_at,
        exit_code=exit_code,
        payload=payload,
    )
    try:
        pubkey.verify(base64.b64decode(signature_b64), sig_input)
        return True
    except InvalidSignature:
        return False
