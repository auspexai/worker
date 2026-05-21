"""Tests for the M4 result signer and canonical encoding."""

from __future__ import annotations

import json

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from auspexai_worker.signing import (
    canonical_result_bytes,
    sign_result,
    verify_result_signature,
)


def _make_key() -> tuple[Ed25519PrivateKey, str]:
    pk = Ed25519PrivateKey.generate()
    pub = pk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    return pk, pub


class TestCanonicalEncoding:
    def test_keys_sorted_and_compact(self) -> None:
        encoded = canonical_result_bytes(
            unit_id="u-1",
            worker_pubkey="A" * 64,
            completed_at="2026-05-21T00:00:00+00:00",
            exit_code=0,
            payload={"b": 2, "a": 1},
        )
        decoded = json.loads(encoded)
        # No whitespace in compact form.
        assert b" " not in encoded
        # sort_keys: top-level keys appear alphabetically.
        first_keys = list(decoded.keys())
        assert first_keys == sorted(first_keys)
        # pubkey lowercased.
        assert decoded["worker_pubkey"] == "a" * 64

    def test_payload_keys_also_sorted(self) -> None:
        # Make sure nested ordering is canonical too — important for
        # cross-language verifiability.
        encoded = canonical_result_bytes(
            unit_id="u-1",
            worker_pubkey="a" * 64,
            completed_at="2026-05-21T00:00:00+00:00",
            exit_code=0,
            payload={"z": 1, "a": 2},
        )
        # Slice the payload region and check 'a' comes before 'z'.
        text = encoded.decode()
        a_pos = text.index('"a"')
        z_pos = text.index('"z"')
        assert a_pos < z_pos


class TestSignAndVerify:
    def test_signature_roundtrips(self) -> None:
        privkey, pub = _make_key()
        sig = sign_result(
            privkey=privkey,
            pubkey_hex=pub,
            unit_id="u-1",
            completed_at="2026-05-21T00:00:00+00:00",
            exit_code=0,
            payload={"hello": "world"},
        )
        assert verify_result_signature(
            pubkey_hex=pub,
            unit_id="u-1",
            worker_pubkey=pub,
            completed_at="2026-05-21T00:00:00+00:00",
            exit_code=0,
            payload={"hello": "world"},
            signature_b64=sig,
        )

    def test_signature_rejects_tampered_payload(self) -> None:
        privkey, pub = _make_key()
        sig = sign_result(
            privkey=privkey,
            pubkey_hex=pub,
            unit_id="u-1",
            completed_at="2026-05-21T00:00:00+00:00",
            exit_code=0,
            payload={"hello": "world"},
        )
        assert not verify_result_signature(
            pubkey_hex=pub,
            unit_id="u-1",
            worker_pubkey=pub,
            completed_at="2026-05-21T00:00:00+00:00",
            exit_code=0,
            payload={"hello": "tampered"},
            signature_b64=sig,
        )

    def test_signature_rejects_wrong_pubkey(self) -> None:
        privkey, pub = _make_key()
        _, other_pub = _make_key()
        sig = sign_result(
            privkey=privkey,
            pubkey_hex=pub,
            unit_id="u-1",
            completed_at="2026-05-21T00:00:00+00:00",
            exit_code=0,
            payload={},
        )
        assert not verify_result_signature(
            pubkey_hex=other_pub,
            unit_id="u-1",
            worker_pubkey=pub,
            completed_at="2026-05-21T00:00:00+00:00",
            exit_code=0,
            payload={},
            signature_b64=sig,
        )

    def test_pubkey_case_normalized(self) -> None:
        privkey, pub = _make_key()
        sig = sign_result(
            privkey=privkey,
            pubkey_hex=pub.upper(),  # caller passes uppercase
            unit_id="u-1",
            completed_at="2026-05-21T00:00:00+00:00",
            exit_code=0,
            payload={},
        )
        # Verifier accepts the lowercase form.
        assert verify_result_signature(
            pubkey_hex=pub,
            unit_id="u-1",
            worker_pubkey=pub,
            completed_at="2026-05-21T00:00:00+00:00",
            exit_code=0,
            payload={},
            signature_b64=sig,
        )
