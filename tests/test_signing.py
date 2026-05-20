"""Tests for the RFC 9421 signer.

Uses `tests/_signature_oracle.py` as an inline verifier so the wire format
is exercised end-to-end without taking a runtime dep on the platform
package. The oracle is the test side of the same contract the coordinator's
verifier implements.
"""

from __future__ import annotations

import base64
import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from auspexai_worker.signing import (
    SIGNATURE_LABEL,
    SUPPORTED_ALG,
    Rfc9421Signer,
    compute_content_digest,
    sign_request,
)
from tests._signature_oracle import (
    parse_signature_input,
)
from tests._signature_oracle import (
    verify_request as oracle_verify,
)


def _make_key() -> tuple[Ed25519PrivateKey, str]:
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    privkey = Ed25519PrivateKey.generate()
    pub = privkey.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    return privkey, pub


class TestContentDigest:
    def test_known_vector(self) -> None:
        # Pin the wire format: RFC 9530 sha-256 base64.
        body = b'{"hello":"world"}'
        digest = compute_content_digest(body)
        expected = base64.b64encode(hashlib.sha256(body).digest()).decode("ascii")
        assert digest == f"sha-256=:{expected}:"

    def test_empty_body_still_renders(self) -> None:
        assert compute_content_digest(b"").startswith("sha-256=:")


class TestSignRequestWireFormat:
    def test_includes_required_covered_components(self) -> None:
        privkey, pub = _make_key()
        headers = sign_request(
            privkey=privkey,
            pubkey_hex=pub,
            method="POST",
            path="/api/v0/workers/wkr-abc/heartbeat",
            authority="coord.example:8080",
            body=b'{"capabilities":{}}',
            created=1716000000,
        )
        assert set(headers) == {"Signature-Input", "Signature", "Content-Digest"}
        parsed = parse_signature_input(headers["Signature-Input"])
        assert parsed.label == SIGNATURE_LABEL
        assert parsed.alg == SUPPORTED_ALG
        assert parsed.keyid == pub
        assert parsed.created == 1716000000
        assert "@method" in parsed.covered
        assert "@path" in parsed.covered
        assert "@authority" in parsed.covered
        assert "content-digest" in parsed.covered

    def test_empty_body_omits_content_digest(self) -> None:
        privkey, pub = _make_key()
        headers = sign_request(
            privkey=privkey,
            pubkey_hex=pub,
            method="GET",
            path="/api/v0/workers/wkr-abc/assignments",
            authority="coord.example:8080",
            body=b"",
        )
        assert "Content-Digest" not in headers
        parsed = parse_signature_input(headers["Signature-Input"])
        assert "content-digest" not in parsed.covered

    def test_keyid_is_lowercased(self) -> None:
        privkey, pub = _make_key()
        headers = sign_request(
            privkey=privkey,
            pubkey_hex=pub.upper(),
            method="GET",
            path="/api/v0/workers/wkr-a",
            authority="c.example",
            body=b"",
        )
        parsed = parse_signature_input(headers["Signature-Input"])
        assert parsed.keyid == pub  # already lowercase


class TestSignRequestRoundTrip:
    def test_oracle_accepts_signed_request_with_body(self) -> None:
        privkey, pub = _make_key()
        body = b'{"capabilities":{"os":"linux"}}'
        headers = sign_request(
            privkey=privkey,
            pubkey_hex=pub,
            method="POST",
            path="/api/v0/workers/wkr-abc/heartbeat",
            authority="coord.example:8080",
            body=body,
        )
        oracle_verify(
            method="POST",
            path="/api/v0/workers/wkr-abc/heartbeat",
            authority="coord.example:8080",
            body=body,
            headers=headers,
            expected_keyid_hex=pub,
        )

    def test_oracle_accepts_signed_request_with_empty_body(self) -> None:
        privkey, pub = _make_key()
        headers = sign_request(
            privkey=privkey,
            pubkey_hex=pub,
            method="GET",
            path="/api/v0/workers/wkr-a/assignments",
            authority="c.example",
            body=b"",
        )
        oracle_verify(
            method="GET",
            path="/api/v0/workers/wkr-a/assignments",
            authority="c.example",
            body=b"",
            headers=headers,
            expected_keyid_hex=pub,
        )

    def test_oracle_rejects_tampered_body(self) -> None:
        privkey, pub = _make_key()
        body = b'{"capabilities":{"os":"linux"}}'
        headers = sign_request(
            privkey=privkey,
            pubkey_hex=pub,
            method="POST",
            path="/api/v0/workers/wkr-a/heartbeat",
            authority="c.example",
            body=body,
        )
        tampered = body.replace(b"linux", b"macos")
        with pytest.raises(AssertionError):
            oracle_verify(
                method="POST",
                path="/api/v0/workers/wkr-a/heartbeat",
                authority="c.example",
                body=tampered,
                headers=headers,
            )

    def test_oracle_rejects_tampered_path(self) -> None:
        privkey, pub = _make_key()
        headers = sign_request(
            privkey=privkey,
            pubkey_hex=pub,
            method="GET",
            path="/api/v0/workers/wkr-A/assignments",
            authority="c.example",
            body=b"",
        )
        with pytest.raises(AssertionError):
            oracle_verify(
                method="GET",
                path="/api/v0/workers/wkr-B/assignments",
                authority="c.example",
                body=b"",
                headers=headers,
            )


class TestSignerClass:
    def test_signer_class_wraps_correctly(self) -> None:
        privkey, pub = _make_key()
        signer = Rfc9421Signer(privkey, pub)
        assert signer.pubkey_hex == pub.lower()
        headers = signer.sign(
            method="POST",
            path="/api/v0/workers/wkr-a/heartbeat",
            authority="c.example",
            body=b'{"x":1}',
        )
        oracle_verify(
            method="POST",
            path="/api/v0/workers/wkr-a/heartbeat",
            authority="c.example",
            body=b'{"x":1}',
            headers=headers,
            expected_keyid_hex=pub,
        )
