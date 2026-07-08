"""RFC 9421 HTTP Message Signatures — worker-side signing.

The coordinator's verifier (platform `auspexai_platform.auth.signature`) accepts
a deliberately narrow subset:

  - Algorithm: ed25519 only.
  - Required covered components: @method, @path, @authority.
  - Conditional: content-digest (required when body is non-empty).
  - `created` (Unix timestamp) REQUIRED; ±5 min window.
  - `keyid` is the lowercase hex Ed25519 pubkey (64 chars). The coordinator
    resolves it to a worker (or tenant) via its CredentialResolver.
  - One signature per request, label "sig1".
  - Content-Digest is RFC 9530 `sha-256` only.

The worker side of this contract is exactly mirrored here so the same bytes
go on the wire. Verification stays in the coordinator; we don't ship a
verifier here in production. Tests in `tests/test_signing.py` use a small
inline verifier to confirm round-trips without depending on the platform
package.
"""

from __future__ import annotations

from .result import (
    RESULT_SCHEMA_VERSION,
    canonical_raw_bytes,
    canonical_result_bytes,
    sign_raw,
    sign_result,
    verify_result_signature,
)
from .signer import (
    SIGNATURE_LABEL,
    SUPPORTED_ALG,
    Rfc9421Signer,
    compute_content_digest,
    sign_request,
)

__all__ = [
    "RESULT_SCHEMA_VERSION",
    "SIGNATURE_LABEL",
    "SUPPORTED_ALG",
    "Rfc9421Signer",
    "canonical_raw_bytes",
    "canonical_result_bytes",
    "compute_content_digest",
    "sign_raw",
    "sign_request",
    "sign_result",
    "verify_result_signature",
]
