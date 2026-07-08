"""Tests for the M4 result signer and canonical encoding."""

from __future__ import annotations

import json
from typing import ClassVar

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from auspexai_worker.signing import (
    canonical_raw_bytes,
    canonical_result_bytes,
    sign_raw,
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


class TestServedWeightsV1:
    """§9 #13a: the versioned canonical that binds the served-weights digest."""

    _ARGS: ClassVar[dict] = {  # shared canonical inputs
        "unit_id": "u-1",
        "worker_pubkey": "a" * 64,
        "completed_at": "2026-06-15T00:00:00+00:00",
        "exit_code": 0,
        "payload": {"k": "v"},
    }

    def test_v0_is_byte_identical_to_legacy(self) -> None:
        # Backward-compat keystone: schema_version 0 (and the default) must
        # reproduce the original five-field encoding EXACTLY, so results signed
        # before #13a stay verifiable. No version/served_weights keys leak in.
        legacy = canonical_result_bytes(**self._ARGS)
        explicit_v0 = canonical_result_bytes(**self._ARGS, schema_version=0)
        v0_with_weights_ignored = canonical_result_bytes(
            **self._ARGS, schema_version=0, served_weights={"m": "abc"}
        )
        assert explicit_v0 == legacy
        assert v0_with_weights_ignored == legacy
        assert b"schema_version" not in legacy
        assert b"served_weights" not in legacy

    def test_v1_binds_version_and_digest(self) -> None:
        encoded = canonical_result_bytes(
            **self._ARGS, schema_version=1, served_weights={"GEMMA": "ABCDEF"}
        )
        decoded = json.loads(encoded)
        assert decoded["schema_version"] == 1
        # Digest value normalized to lower hex; model id preserved verbatim.
        assert decoded["served_weights"] == {"GEMMA": "abcdef"}
        # Top-level keys still canonical (sorted) for cross-language verify.
        assert list(decoded) == sorted(decoded)

    def test_v1_served_weights_ordering_canonical(self) -> None:
        a = canonical_result_bytes(
            **self._ARGS, schema_version=1, served_weights={"z": "11", "a": "22"}
        )
        b = canonical_result_bytes(
            **self._ARGS, schema_version=1, served_weights={"a": "22", "z": "11"}
        )
        assert a == b  # insertion order must not change the signed bytes

    def test_v1_empty_weights_roundtrips(self) -> None:
        # Non-inference units still sign v1 with an empty map.
        privkey, pub = _make_key()
        args = {**self._ARGS, "worker_pubkey": pub}
        sig = sign_result(privkey=privkey, pubkey_hex=pub, schema_version=1, **_drop_pub(args))
        assert verify_result_signature(
            pubkey_hex=pub, signature_b64=sig, schema_version=1, served_weights={}, **args
        )

    def test_v1_roundtrips_with_digest(self) -> None:
        privkey, pub = _make_key()
        args = {**self._ARGS, "worker_pubkey": pub}
        weights = {"gemma-3-1b-it-q4": "deadbeef"}
        sig = sign_result(
            privkey=privkey,
            pubkey_hex=pub,
            schema_version=1,
            served_weights=weights,
            **_drop_pub(args),
        )
        assert verify_result_signature(
            pubkey_hex=pub,
            signature_b64=sig,
            schema_version=1,
            served_weights=weights,
            **args,
        )

    def test_v1_signature_does_not_verify_as_v0(self) -> None:
        # Downgrade-strip detection: a v1 signature must NOT verify if the
        # verifier reconstructs the body as v0 (dropping the bound digest).
        privkey, pub = _make_key()
        args = {**self._ARGS, "worker_pubkey": pub}
        sig = sign_result(
            privkey=privkey,
            pubkey_hex=pub,
            schema_version=1,
            served_weights={"m": "abcd"},
            **_drop_pub(args),
        )
        assert not verify_result_signature(
            pubkey_hex=pub, signature_b64=sig, schema_version=0, **args
        )

    def test_v1_rejects_tampered_digest(self) -> None:
        privkey, pub = _make_key()
        args = {**self._ARGS, "worker_pubkey": pub}
        sig = sign_result(
            privkey=privkey,
            pubkey_hex=pub,
            schema_version=1,
            served_weights={"m": "abcd"},
            **_drop_pub(args),
        )
        assert not verify_result_signature(
            pubkey_hex=pub,
            signature_b64=sig,
            schema_version=1,
            served_weights={"m": "ffff"},  # swapped digest
            **args,
        )


class TestRanUnderV2:
    """A2 #32: v2 binds the worker-signed `ran_under` (sandbox policy). The
    known-vector MUST equal the platform's test_result_signature_v2.py vector —
    a byte-for-byte cross-codebase wire contract; if either side changes, both
    known-vector tests fail."""

    # MUST stay identical to platform tests/test_result_signature_v2.py vector.
    _V2_KNOWN_VECTOR: ClassVar[bytes] = (
        b'{"completed_at":"2026-06-19T00:00:00+00:00","exit_code":0,'
        b'"payload":{"k":"v"},"ran_under":"strict","schema_version":2,'
        b'"served_weights":{"m":"aabb"},"unit_id":"u","worker_pubkey":"ab"}'
    )

    def test_v2_known_vector_matches_platform(self) -> None:
        out = canonical_result_bytes(
            unit_id="u",
            worker_pubkey="ab",
            completed_at="2026-06-19T00:00:00+00:00",
            exit_code=0,
            payload={"k": "v"},
            schema_version=2,
            served_weights={"m": "AABB"},
            ran_under="STRICT",  # lower-cased in the canonical body
        )
        assert out == self._V2_KNOWN_VECTOR

    def test_v0_v1_unchanged_by_ran_under(self) -> None:
        args = {
            "unit_id": "u",
            "worker_pubkey": "ab",
            "completed_at": "2026-06-19T00:00:00+00:00",
            "exit_code": 0,
            "payload": {"k": "v"},
        }
        assert canonical_result_bytes(**args, schema_version=0, ran_under="strict") == (
            canonical_result_bytes(**args, schema_version=0)
        )
        assert canonical_result_bytes(
            **args, schema_version=1, served_weights={"m": "aabb"}, ran_under="strict"
        ) == canonical_result_bytes(**args, schema_version=1, served_weights={"m": "aabb"})

    def test_v2_roundtrips_and_rejects_tampered_ran_under(self) -> None:
        privkey, pub = _make_key()
        args = {
            "unit_id": "u-1",
            "worker_pubkey": pub,
            "completed_at": "2026-06-15T00:00:00+00:00",
            "exit_code": 0,
            "payload": {"k": "v"},
        }
        sig = sign_result(
            privkey=privkey,
            pubkey_hex=pub,
            schema_version=2,
            served_weights={},
            ran_under="strict",
            **_drop_pub(args),
        )
        assert verify_result_signature(
            pubkey_hex=pub,
            signature_b64=sig,
            schema_version=2,
            served_weights={},
            ran_under="strict",
            **args,
        )
        # Signed strict; a worker cannot have it verify as a permissive claim.
        assert not verify_result_signature(
            pubkey_hex=pub,
            signature_b64=sig,
            schema_version=2,
            served_weights={},
            ran_under="permissive",
            **args,
        )


def _drop_pub(args: dict) -> dict:
    """sign_result takes pubkey_hex, not worker_pubkey — strip the dup key."""
    return {k: v for k, v in args.items() if k != "worker_pubkey"}


# ── AUD-26: detached raw-content signature ─────────────────────────────────────

# MUST equal the coordinator's known vector (platform test_result_signature_v2.py):
# sha256("hi"), keys sorted, worker_pubkey lower-cased.
_RAW_KNOWN_VECTOR = (
    b'{"raw_response_sha256":'
    b'"8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4",'
    b'"unit_id":"u","worker_pubkey":"ab"}'
)


def test_raw_canonical_known_vector() -> None:
    """Byte-for-byte drift guard vs the coordinator's canonical_raw_bytes."""
    assert (
        canonical_raw_bytes(unit_id="u", worker_pubkey="AB", raw_response="hi") == _RAW_KNOWN_VECTOR
    )


def test_sign_raw_roundtrips() -> None:
    import base64

    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    pk, pub = _make_key()
    raw = "some untrusted model text"
    sig_b64 = sign_raw(privkey=pk, pubkey_hex=pub, unit_id="u1", raw_response=raw)
    pubkey = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub))
    # verifies over the canonical detached body
    pubkey.verify(
        base64.b64decode(sig_b64),
        canonical_raw_bytes(unit_id="u1", worker_pubkey=pub, raw_response=raw),
    )
    # a tampered raw no longer verifies
    try:
        pubkey.verify(
            base64.b64decode(sig_b64),
            canonical_raw_bytes(unit_id="u1", worker_pubkey=pub, raw_response=raw + "!"),
        )
        raise AssertionError("tampered raw should not verify")
    except InvalidSignature:
        pass
