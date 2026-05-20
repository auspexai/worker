"""Worker keystore — generates, stores, and loads the worker's Ed25519 keypair.

Per principles doc §5.11. Phase 1 (M1) ships two backends on Linux:

- `SecretServiceKeystore` — libsecret via D-Bus (GNOME Keyring, KWallet,
  KeePassXC-Secret-Service). Preferred on desktop / interactive hosts.
- `EncryptedFileKeystore` — AEAD-encrypted file at
  `$XDG_DATA_HOME/auspexai-worker/keystore.enc`, with a key derived from
  `/etc/machine-id` + the running user's UID. Used on headless hosts and in
  containers / CI.

`InMemoryKeystore` is a test-only implementation.

The factory `default_keystore(...)` tries SecretService first and falls back
to EncryptedFile so a single worker package works across desktop and headless
Linux without configuration.
"""

from __future__ import annotations

from .base import KeyNotFoundError, Keystore, KeystoreError
from .encrypted_file import EncryptedFileKeystore
from .factory import default_keystore
from .in_memory import InMemoryKeystore

__all__ = [
    "EncryptedFileKeystore",
    "InMemoryKeystore",
    "KeyNotFoundError",
    "Keystore",
    "KeystoreError",
    "default_keystore",
]
