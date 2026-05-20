"""In-memory keystore. Test-only — never persisted."""

from __future__ import annotations

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .base import KeyNotFoundError, KeystoreError


class InMemoryKeystore:
    """Holds a single Ed25519 key in process memory.

    Tests use this to exercise the bootstrap + enroll path without touching
    libsecret or the filesystem.
    """

    def __init__(self) -> None:
        self._key: Ed25519PrivateKey | None = None

    def has_key(self) -> bool:
        return self._key is not None

    def generate_and_store(self) -> Ed25519PrivateKey:
        if self._key is not None:
            raise KeystoreError("key already present; call delete() first")
        self._key = Ed25519PrivateKey.generate()
        return self._key

    def load(self) -> Ed25519PrivateKey:
        if self._key is None:
            raise KeyNotFoundError("no key has been generated yet")
        return self._key

    def delete(self) -> None:
        self._key = None
