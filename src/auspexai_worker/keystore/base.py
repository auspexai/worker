"""Keystore protocol + shared types."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


class KeystoreError(Exception):
    """Base exception for keystore failures."""


class KeyNotFoundError(KeystoreError):
    """Raised when `load()` is called and no key has been stored."""


@runtime_checkable
class Keystore(Protocol):
    """Abstract worker keystore.

    Backends store a single 32-byte Ed25519 private seed. The worker never
    asks the user to handle the key material — generation and storage are
    fully internal.
    """

    def has_key(self) -> bool:
        """Return True if a private key is already stored."""
        ...

    def generate_and_store(self) -> Ed25519PrivateKey:
        """Generate a fresh Ed25519 keypair, persist it, return the key.

        Raises:
            KeystoreError: if a key is already stored. Callers must check
                `has_key()` first or call `delete()` to overwrite.
        """
        ...

    def load(self) -> Ed25519PrivateKey:
        """Load the previously-stored private key.

        Raises:
            KeyNotFoundError: when no key has been stored yet.
            KeystoreError: on backend failures (e.g. corrupted file).
        """
        ...

    def delete(self) -> None:
        """Remove the stored key. No-op if none exists."""
        ...


def pubkey_hex(private_key: Ed25519PrivateKey) -> str:
    """Return the 64-char lowercase hex encoding of an Ed25519 public key.

    Matches the coordinator's `WorkerEnrollRequest.pubkey_hex` wire shape.
    """
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    raw = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return raw.hex()


def pubkey_fingerprint(private_key: Ed25519PrivateKey) -> str:
    """Short human-readable fingerprint for CLI display: first 16 hex chars."""
    return pubkey_hex(private_key)[:16]
