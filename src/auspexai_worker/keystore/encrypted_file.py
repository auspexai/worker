"""Encrypted-file keystore for headless Linux hosts and CI.

Stores the 32-byte Ed25519 private seed AEAD-encrypted with ChaCha20-Poly1305.
The encryption key is derived via HKDF-SHA256 from a host fingerprint:
`/etc/machine-id` (or `/var/lib/dbus/machine-id`) plus the running user's UID.

This is the §5.11 "encrypted file with passphrase derived from machine ID +
user" fallback. The volunteer never types a passphrase; the host fingerprint
binds the file to this machine + user so a copy of the file alone is
non-trivial to use elsewhere. Phase 1 lab-acceptable: a passive file-copy
attacker who also captured `/etc/machine-id` (e.g. via a `$HOME`-inclusive
backup) can decrypt offline. Phase 2 will replace this fallback with a
`systemd-creds`-backed Tier B keystore (root-only host-key derivation
closes the offline-decrypt gap) and `python-keyring` as the cross-platform
abstraction layer; an optional `[identity] keystore_passphrase` power-user
opt-in lands in the same pass.

File format: `magic(4) || version(1) || nonce(12) || ciphertext+tag`.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

from .base import KeyNotFoundError, KeystoreError

_MAGIC = b"AKWv"  # AuspexAI keystore wire
_VERSION = 0x01
_NONCE_LEN = 12
_KDF_INFO = b"auspexai-worker-keystore-v0"


def _read_machine_id() -> str:
    """Read the host's machine-id. Linux only.

    Raises:
        KeystoreError: if neither standard path is present. Real Ubuntu /
            Debian hosts always have one (created at first boot via
            systemd-machine-id-setup). The failure surfaces on minimal
            containers built without systemd ever running, and on some
            stripped embedded distributions. The error message includes
            the standard remediation.
    """
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            content = Path(path).read_text(encoding="ascii").strip()
        except FileNotFoundError:
            continue
        if content:
            return content
    raise KeystoreError(
        "no machine-id found at /etc/machine-id or /var/lib/dbus/machine-id "
        "(both empty or missing). Real Ubuntu / Debian hosts always have one "
        "— this typically surfaces only on minimal containers / dev images "
        "where systemd-machine-id-setup hasn't run.\n"
        "\n"
        "Fix (run once, as root):\n"
        "    sudo systemd-machine-id-setup\n"
        "  OR\n"
        "    cat /proc/sys/kernel/random/uuid | tr -d - | sudo tee /etc/machine-id\n"
        "\n"
        "The machine-id is the host-specific entropy that protects the "
        "encrypted-file keystore from being usable on a different host; "
        "we don't fall back to a non-host-specific value because that "
        "would invalidate the security property."
    )


def _derive_key(machine_id: str, uid: int) -> bytes:
    """Derive a 32-byte ChaCha20-Poly1305 key from host fingerprint."""
    secret_input = f"{machine_id}:{uid}".encode()
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=_KDF_INFO,
    )
    return hkdf.derive(secret_input)


class EncryptedFileKeystore:
    """Stores the Ed25519 private seed in an AEAD-encrypted file."""

    def __init__(
        self,
        path: Path,
        *,
        machine_id: str | None = None,
        uid: int | None = None,
    ) -> None:
        """
        Args:
            path: Filesystem path. Parent directory is created on store.
            machine_id: Override for testing. Defaults to host's machine-id.
            uid: Override for testing. Defaults to `os.geteuid()`.
        """
        self._path = path
        self._machine_id = machine_id if machine_id is not None else _read_machine_id()
        self._uid = uid if uid is not None else os.geteuid()

    @property
    def path(self) -> Path:
        return self._path

    def has_key(self) -> bool:
        return self._path.exists()

    def generate_and_store(self) -> Ed25519PrivateKey:
        if self.has_key():
            raise KeystoreError(f"key already present at {self._path}; call delete() first")
        key = Ed25519PrivateKey.generate()
        seed = key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
        self._write(seed)
        return key

    def load(self) -> Ed25519PrivateKey:
        if not self.has_key():
            raise KeyNotFoundError(f"no keystore file at {self._path}")
        seed = self._read()
        return Ed25519PrivateKey.from_private_bytes(seed)

    def delete(self) -> None:
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass

    # ---- private --------------------------------------------------------

    def _write(self, seed: bytes) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        nonce = secrets.token_bytes(_NONCE_LEN)
        cipher = ChaCha20Poly1305(_derive_key(self._machine_id, self._uid))
        ct = cipher.encrypt(nonce, seed, associated_data=_MAGIC)
        payload = _MAGIC + bytes([_VERSION]) + nonce + ct
        # Atomic write: temp file in same dir + rename.
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_bytes(payload)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self._path)

    def _read(self) -> bytes:
        data = self._path.read_bytes()
        if len(data) < len(_MAGIC) + 1 + _NONCE_LEN + 16:
            raise KeystoreError(f"keystore file at {self._path} is too short to be valid")
        if data[: len(_MAGIC)] != _MAGIC:
            raise KeystoreError(f"keystore file at {self._path} has wrong magic bytes")
        version = data[len(_MAGIC)]
        if version != _VERSION:
            raise KeystoreError(
                f"keystore file version {version:#x} is not supported by this worker"
            )
        offset = len(_MAGIC) + 1
        nonce = data[offset : offset + _NONCE_LEN]
        ct = data[offset + _NONCE_LEN :]
        cipher = ChaCha20Poly1305(_derive_key(self._machine_id, self._uid))
        try:
            return cipher.decrypt(nonce, ct, associated_data=_MAGIC)
        except Exception as exc:
            raise KeystoreError(
                f"failed to decrypt keystore at {self._path}; the host fingerprint "
                "(machine-id + uid) may have changed since the key was stored"
            ) from exc
