"""Backend selection for the worker keystore.

`default_keystore()` returns the Secret Service backend when reachable and
falls back to the encrypted-file backend otherwise. The selection is sticky
once a key is stored: if both backends are reachable and one already has a
key, the one with the key wins.
"""

from __future__ import annotations

from pathlib import Path

from .base import Keystore, KeystoreError
from .encrypted_file import EncryptedFileKeystore


def _try_secret_service() -> Keystore | None:
    """Construct SecretServiceKeystore or return None if unavailable.

    Unavailable cases (any of which falls back to encrypted-file):
    - SecretStorage not installed
    - D-Bus session not running (headless host, container)
    - No Secret Service provider running on the bus
    - Default collection unlock prompt failed
    """
    try:
        from .secret_service import SecretServiceKeystore
    except ImportError:
        return None
    try:
        return SecretServiceKeystore()
    except KeystoreError:
        return None


def default_keystore(
    *,
    encrypted_file_path: Path,
    force_backend: str | None = None,
) -> Keystore:
    """Return the appropriate keystore for this host.

    Args:
        encrypted_file_path: Where the encrypted-file backend stores its file
            when used. Provided by config; usually
            `$XDG_DATA_HOME/auspexai-worker/keystore.enc`.
        force_backend: Optional override — "secret_service" or "encrypted_file".
            Defaults to auto-detect.

    Returns:
        A Keystore instance.
    """
    if force_backend == "secret_service":
        ks = _try_secret_service()
        if ks is None:
            raise KeystoreError("Secret Service backend forced but unavailable on this host")
        return ks
    if force_backend == "encrypted_file":
        return EncryptedFileKeystore(encrypted_file_path)
    if force_backend is not None:
        raise KeystoreError(f"unknown keystore backend: {force_backend!r}")

    # Auto-detect. Prefer Secret Service when reachable AND it has a key, or
    # when reachable AND encrypted-file does not have a key (so the first
    # generated key lands in the keyring on a desktop host). Otherwise fall
    # back to encrypted-file.
    encrypted = EncryptedFileKeystore(encrypted_file_path)
    secret = _try_secret_service()
    if secret is not None:
        if secret.has_key():
            return secret
        if not encrypted.has_key():
            return secret
    return encrypted
