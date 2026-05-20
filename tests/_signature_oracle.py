"""Inline RFC 9421 verifier for tests.

The worker only signs in production; the coordinator owns verification. To
test the signer end-to-end without taking a runtime dep on the platform
package, this helper reconstructs the signature base and verifies the
Ed25519 signature. If this file ever diverges from the coordinator's
verifier, the signer is the wrong thing — the wire format is the contract
and this oracle exists to enforce it from the worker side too.
"""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

_SIG_INPUT_HEAD = re.compile(
    r"^(?P<label>[A-Za-z0-9_-]+)=(?P<rest>\(.*\))(?P<params>(?:;[^;]+)*)\s*$"
)
_COVERED_LIST = re.compile(r'^\(\s*((?:"[^"]+"\s*)*)\)$')
_PARAM_KV = re.compile(r";\s*([A-Za-z0-9_-]+)\s*=\s*([^;]+?)\s*(?=$|;)")
_SIG_HEADER = re.compile(r"^(?P<label>[A-Za-z0-9_-]+)=:(?P<b64>[A-Za-z0-9+/=]+):\s*$")
_CONTENT_DIGEST = re.compile(r"^sha-256=:(?P<b64>[A-Za-z0-9+/=]+):\s*$")


@dataclass(frozen=True)
class Parsed:
    label: str
    covered: tuple[str, ...]
    created: int
    alg: str
    keyid: str
    raw_covered_and_params: str


def parse_signature_input(value: str) -> Parsed:
    m = _SIG_INPUT_HEAD.match(value.strip())
    assert m, f"malformed Signature-Input: {value!r}"
    covered_part = m.group("rest")
    params_part = m.group("params")
    cm = _COVERED_LIST.match(covered_part)
    assert cm, f"malformed covered list: {covered_part!r}"
    items = tuple(x.strip('"') for x in cm.group(1).split())
    params: dict[str, str] = {}
    for k, v in _PARAM_KV.findall(params_part):
        params[k.lower()] = v.strip().strip('"')
    return Parsed(
        label=m.group("label"),
        covered=items,
        created=int(params["created"]),
        alg=params["alg"].lower(),
        keyid=params["keyid"].lower(),
        raw_covered_and_params=covered_part + params_part,
    )


def parse_signature(value: str) -> tuple[str, bytes]:
    m = _SIG_HEADER.match(value.strip())
    assert m, f"malformed Signature: {value!r}"
    return m.group("label"), base64.b64decode(m.group("b64"), validate=True)


def assert_content_digest(value: str, body: bytes) -> None:
    m = _CONTENT_DIGEST.match(value.strip())
    assert m, f"malformed Content-Digest: {value!r}"
    claimed = base64.b64decode(m.group("b64"), validate=True)
    assert claimed == hashlib.sha256(body).digest()


def build_base(
    parsed: Parsed,
    *,
    method: str,
    path: str,
    authority: str,
    content_digest_header: str | None,
) -> bytes:
    lines: list[str] = []
    for c in parsed.covered:
        if c == "@method":
            lines.append(f'"@method": {method.upper()}')
        elif c == "@path":
            lines.append(f'"@path": {path}')
        elif c == "@authority":
            lines.append(f'"@authority": {authority.lower()}')
        elif c == "content-digest":
            assert content_digest_header is not None
            lines.append(f'"content-digest": {content_digest_header.strip()}')
        else:
            raise AssertionError(f"unexpected covered component {c!r}")
    lines.append(f'"@signature-params": {parsed.raw_covered_and_params}')
    return "\n".join(lines).encode("utf-8")


def verify_request(
    *,
    method: str,
    path: str,
    authority: str,
    body: bytes,
    headers: dict[str, str],
    expected_keyid_hex: str | None = None,
) -> Parsed:
    """Verify a signed request. Returns the parsed Signature-Input on success.

    Raises AssertionError or InvalidSignature on any contract violation.
    """
    sig_input = headers["Signature-Input"]
    sig = headers["Signature"]
    parsed = parse_signature_input(sig_input)
    label, sig_bytes = parse_signature(sig)
    assert label == parsed.label, f"label mismatch: {label!r} vs {parsed.label!r}"
    assert parsed.alg == "ed25519"
    if expected_keyid_hex is not None:
        assert parsed.keyid == expected_keyid_hex.lower()
    content_digest_header: str | None = None
    if "content-digest" in parsed.covered:
        content_digest_header = headers.get("Content-Digest")
        assert content_digest_header is not None, "covered content-digest but no header"
        assert_content_digest(content_digest_header, body)
    base = build_base(
        parsed,
        method=method,
        path=path,
        authority=authority,
        content_digest_header=content_digest_header,
    )
    pubkey = Ed25519PublicKey.from_public_bytes(bytes.fromhex(parsed.keyid))
    try:
        pubkey.verify(sig_bytes, base)
    except InvalidSignature as e:
        raise AssertionError(f"Ed25519 verify failed: {e}") from e
    return parsed
