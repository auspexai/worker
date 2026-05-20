"""libsecret / Secret Service keystore (GNOME Keyring, KWallet, etc.).

This backend is lazy — `SecretStorage` is imported only when this module is
actually constructed, so a worker package installed without the
`secret-service` extra (or running in a container without a Secret Service
provider) still imports and runs the encrypted-file fallback cleanly.

The factory in `factory.py` probes for D-Bus + a running Secret Service before
constructing this backend; clients should normally call `default_keystore()`
rather than instantiating this directly.
"""

from __future__ import annotations

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

from .base import KeyNotFoundError, KeystoreError

_SCHEMA_ATTRIBUTES = {
    "service": "auspexai-worker",
    "key-type": "ed25519-private",
}
_ITEM_LABEL = "AuspexAI worker — Ed25519 private key"


class SecretServiceKeystore:
    """Stores the Ed25519 private seed as a libsecret item.

    The seed (32 bytes) is stored verbatim as the item's secret payload.
    Attributes act as the unique lookup key.
    """

    def __init__(self) -> None:
        try:
            import secretstorage  # type: ignore[import-not-found]
        except ImportError as exc:
            raise KeystoreError(
                "SecretStorage is not installed; install with the [secret-service] "
                "extra or use EncryptedFileKeystore"
            ) from exc

        self._secretstorage = secretstorage
        try:
            self._connection = secretstorage.dbus_init()
            self._collection = secretstorage.get_default_collection(self._connection)
            if self._collection.is_locked():
                self._collection.unlock()
        except Exception as exc:
            raise KeystoreError(f"could not connect to Secret Service: {exc}") from exc

    def _find_item(self):
        items = list(self._collection.search_items(_SCHEMA_ATTRIBUTES))
        return items[0] if items else None

    def has_key(self) -> bool:
        return self._find_item() is not None

    def generate_and_store(self) -> Ed25519PrivateKey:
        if self.has_key():
            raise KeystoreError("key already present in Secret Service; call delete() first")
        key = Ed25519PrivateKey.generate()
        seed = key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
        self._collection.create_item(_ITEM_LABEL, _SCHEMA_ATTRIBUTES, seed, replace=False)
        return key

    def load(self) -> Ed25519PrivateKey:
        item = self._find_item()
        if item is None:
            raise KeyNotFoundError("no AuspexAI worker key in Secret Service")
        seed = item.get_secret()
        if len(seed) != 32:
            raise KeystoreError(
                f"Secret Service returned a {len(seed)}-byte payload; expected 32-byte "
                "Ed25519 seed (keystore may be corrupted)"
            )
        return Ed25519PrivateKey.from_private_bytes(seed)

    def delete(self) -> None:
        item = self._find_item()
        if item is not None:
            item.delete()
