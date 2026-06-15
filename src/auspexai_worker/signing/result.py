"""Result signing — canonical encoding the worker signs over.

The coordinator's M6d POST /result body carries a `worker_signature` field
that M6d stores verbatim but does NOT verify. M7 will reproduce the
canonical encoding to verify the body chain when issuing receipts. To
keep that future M7 work mechanical, the canonical encoding is defined
here once and exercised by tests with an inline verifier.

Canonical input for the signature (schema_version 0 — the legacy 5-field form):

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

schema_version 1 (§9 #13a) additionally signs two fields, so the served-weights
digest is worker-ATTESTED (anti-fabrication), not coordinator-asserted:

    { ...the five v0 fields...,
      "schema_version": 1,
      "served_weights": {model_id: gguf_sha256_hex, ...} }  # {} when no model served

`served_weights` is the digest of the model(s) the worker DAEMON brokered to
this unit (its own ModelServer view — never the executor's self-report). Version
is carried INSIDE the signed body so a downgrade-to-v0 can't strip the field
unnoticed. v0 reconstruction stays byte-identical, so already-signed historical
results remain verifiable through the fleet roll.

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

# §9 #13a: the result-signing canonical schema version. v0 = the legacy
# five-field body (no version key on the wire). v1 = v0 + `schema_version` +
# `served_weights`. Workers on this release sign every result as v1 (with an
# empty `served_weights` for non-inference units); the coordinator reconstructs
# per the declared version, so v0 (un-rolled fleet) and v1 verify side by side.
RESULT_SCHEMA_VERSION = 1


def canonical_result_bytes(
    *,
    unit_id: str,
    worker_pubkey: str,
    completed_at: datetime | str,
    exit_code: int,
    payload: dict[str, Any],
    schema_version: int = 0,
    served_weights: dict[str, str] | None = None,
) -> bytes:
    """Produce the canonical encoding the worker signs.

    `completed_at` is accepted as either a datetime (will be `isoformat()`'d)
    or as a pre-formatted ISO 8601 string (so the worker can sign the same
    string it eventually serializes onto the wire).

    `schema_version` selects the body shape. 0 (default) reproduces the legacy
    five-field encoding byte-for-byte. >= 1 adds the signed `schema_version`
    and `served_weights` fields ({model_id: gguf_sha256}; normalized to lower
    hex; `{}` when no model was served). `sort_keys` makes ordering canonical
    regardless of how the dict was built.
    """
    ts = completed_at.isoformat() if isinstance(completed_at, datetime) else completed_at
    body: dict[str, Any] = {
        "unit_id": unit_id,
        "worker_pubkey": worker_pubkey.lower(),
        "completed_at": ts,
        "exit_code": int(exit_code),
        "payload": payload,
    }
    if schema_version >= 1:
        body["schema_version"] = int(schema_version)
        body["served_weights"] = {str(k): str(v).lower() for k, v in (served_weights or {}).items()}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_result(
    *,
    privkey: Ed25519PrivateKey,
    pubkey_hex: str,
    unit_id: str,
    completed_at: datetime | str,
    exit_code: int,
    payload: dict[str, Any],
    schema_version: int = 0,
    served_weights: dict[str, str] | None = None,
) -> str:
    """Sign the result body and return the base64 signature string."""
    sig_input = canonical_result_bytes(
        unit_id=unit_id,
        worker_pubkey=pubkey_hex,
        completed_at=completed_at,
        exit_code=exit_code,
        payload=payload,
        schema_version=schema_version,
        served_weights=served_weights,
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
    schema_version: int = 0,
    served_weights: dict[str, str] | None = None,
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
        schema_version=schema_version,
        served_weights=served_weights,
    )
    try:
        pubkey.verify(base64.b64decode(signature_b64), sig_input)
        return True
    except InvalidSignature:
        return False
