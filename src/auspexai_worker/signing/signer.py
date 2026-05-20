"""RFC 9421 signer for worker-credentialed coordinator requests.

This is the symmetric pair of the coordinator's `verify_request`. Kept in a
narrow self-contained module so the wire format is auditable in one place
and the worker has no runtime dependency on the platform package.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

SUPPORTED_ALG = "ed25519"
SIGNATURE_LABEL = "sig1"


def compute_content_digest(body: bytes) -> str:
    """Build an RFC 9530 Content-Digest header value (`sha-256` only)."""
    digest = hashlib.sha256(body).digest()
    return f"sha-256=:{base64.b64encode(digest).decode('ascii')}:"


def _build_signature_base(
    *,
    covered: tuple[str, ...],
    raw_covered_and_params: str,
    method: str,
    path: str,
    authority: str,
    content_digest_header: str | None,
) -> bytes:
    """Reconstruct the signature base per RFC 9421 §2.5."""
    lines: list[str] = []
    for component in covered:
        if component == "@method":
            lines.append(f'"@method": {method.upper()}')
        elif component == "@path":
            lines.append(f'"@path": {path}')
        elif component == "@authority":
            lines.append(f'"@authority": {authority.lower()}')
        elif component == "content-digest":
            if content_digest_header is None:
                raise ValueError(
                    "covered components include 'content-digest' but no header was provided"
                )
            lines.append(f'"content-digest": {content_digest_header.strip()}')
        else:
            raise ValueError(
                f"unsupported covered component {component!r}; coordinator v0 accepts "
                "@method, @path, @authority, content-digest"
            )
    lines.append(f'"@signature-params": {raw_covered_and_params}')
    return "\n".join(lines).encode("utf-8")


def sign_request(
    *,
    privkey: Ed25519PrivateKey,
    pubkey_hex: str,
    method: str,
    path: str,
    authority: str,
    body: bytes,
    created: int | None = None,
    covered: tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Produce Signature-Input + Signature (+ Content-Digest if body) headers.

    Args:
        privkey: Worker's Ed25519 private key.
        pubkey_hex: 64-char lowercase hex Ed25519 pubkey (the `keyid`).
        method: HTTP method ("GET", "POST", ...).
        path: Request path including any query string.
        authority: HTTP `host` (with port if non-default).
        body: Request body bytes; pass `b""` for empty.
        created: Optional Unix timestamp override (defaults to now).
        covered: Optional override; defaults to (@method, @path, @authority)
            plus content-digest when `body` is non-empty.

    Returns:
        Headers dict to merge into the outgoing request.
    """
    if created is None:
        created = int(datetime.now(UTC).timestamp())
    if covered is None:
        covered = ("@method", "@path", "@authority") + (("content-digest",) if body else ())

    covered_list = " ".join(f'"{c}"' for c in covered)
    raw_covered_and_params = (
        f'({covered_list});created={created};alg="{SUPPORTED_ALG}";keyid="{pubkey_hex.lower()}"'
    )
    signature_input_value = f"{SIGNATURE_LABEL}={raw_covered_and_params}"

    content_digest_header: str | None = None
    if "content-digest" in covered:
        content_digest_header = compute_content_digest(body)

    base = _build_signature_base(
        covered=covered,
        raw_covered_and_params=raw_covered_and_params,
        method=method,
        path=path,
        authority=authority,
        content_digest_header=content_digest_header,
    )
    sig = privkey.sign(base)
    sig_value = f"{SIGNATURE_LABEL}=:{base64.b64encode(sig).decode('ascii')}:"

    headers = {
        "Signature-Input": signature_input_value,
        "Signature": sig_value,
    }
    if content_digest_header:
        headers["Content-Digest"] = content_digest_header
    return headers


class Rfc9421Signer:
    """Thin wrapper holding the worker's keypair + pubkey hex.

    Designed to be passed into the coordinator client so the client can sign
    every authenticated request without re-loading the keystore.
    """

    def __init__(self, privkey: Ed25519PrivateKey, pubkey_hex: str) -> None:
        self._privkey = privkey
        self._pubkey_hex = pubkey_hex.lower()

    @property
    def pubkey_hex(self) -> str:
        return self._pubkey_hex

    def sign(
        self,
        *,
        method: str,
        path: str,
        authority: str,
        body: bytes,
        created: int | None = None,
    ) -> dict[str, str]:
        return sign_request(
            privkey=self._privkey,
            pubkey_hex=self._pubkey_hex,
            method=method,
            path=path,
            authority=authority,
            body=body,
            created=created,
        )
