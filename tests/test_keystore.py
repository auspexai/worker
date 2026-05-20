"""Tests for keystore backends and selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from auspexai_worker.keystore import (
    EncryptedFileKeystore,
    InMemoryKeystore,
    KeyNotFoundError,
    KeystoreError,
    default_keystore,
)
from auspexai_worker.keystore.base import pubkey_fingerprint, pubkey_hex


class TestInMemoryKeystore:
    def test_generate_then_load_roundtrips(self) -> None:
        ks = InMemoryKeystore()
        assert ks.has_key() is False
        generated = ks.generate_and_store()
        assert ks.has_key() is True
        loaded = ks.load()
        # Same object (in-memory), same public bytes.
        assert pubkey_hex(generated) == pubkey_hex(loaded)

    def test_load_before_generate_raises(self) -> None:
        ks = InMemoryKeystore()
        with pytest.raises(KeyNotFoundError):
            ks.load()

    def test_double_generate_raises(self) -> None:
        ks = InMemoryKeystore()
        ks.generate_and_store()
        with pytest.raises(KeystoreError):
            ks.generate_and_store()

    def test_delete_clears(self) -> None:
        ks = InMemoryKeystore()
        ks.generate_and_store()
        ks.delete()
        assert ks.has_key() is False
        with pytest.raises(KeyNotFoundError):
            ks.load()


class TestEncryptedFileKeystore:
    def test_generate_then_reload_roundtrips(self, tmp_path: Path) -> None:
        path = tmp_path / "keystore.enc"
        ks = EncryptedFileKeystore(path, machine_id="test-machine-id", uid=1000)
        assert ks.has_key() is False
        generated = ks.generate_and_store()
        assert ks.has_key() is True
        assert path.exists()
        # Reopen from disk with the same fingerprint.
        ks2 = EncryptedFileKeystore(path, machine_id="test-machine-id", uid=1000)
        loaded = ks2.load()
        assert pubkey_hex(generated) == pubkey_hex(loaded)

    def test_file_is_user_only(self, tmp_path: Path) -> None:
        path = tmp_path / "keystore.enc"
        ks = EncryptedFileKeystore(path, machine_id="m", uid=1000)
        ks.generate_and_store()
        # 0o600 = read+write owner only.
        assert path.stat().st_mode & 0o777 == 0o600

    def test_wrong_fingerprint_fails_to_decrypt(self, tmp_path: Path) -> None:
        path = tmp_path / "keystore.enc"
        EncryptedFileKeystore(path, machine_id="m1", uid=1000).generate_and_store()
        attacker_view = EncryptedFileKeystore(path, machine_id="m2", uid=1000)
        with pytest.raises(KeystoreError):
            attacker_view.load()

    def test_uid_is_part_of_fingerprint(self, tmp_path: Path) -> None:
        path = tmp_path / "keystore.enc"
        EncryptedFileKeystore(path, machine_id="m", uid=1000).generate_and_store()
        with pytest.raises(KeystoreError):
            EncryptedFileKeystore(path, machine_id="m", uid=2000).load()

    def test_load_missing_file_raises_key_not_found(self, tmp_path: Path) -> None:
        ks = EncryptedFileKeystore(tmp_path / "nope.enc", machine_id="m", uid=1000)
        with pytest.raises(KeyNotFoundError):
            ks.load()

    def test_double_generate_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "keystore.enc"
        ks = EncryptedFileKeystore(path, machine_id="m", uid=1000)
        ks.generate_and_store()
        with pytest.raises(KeystoreError):
            ks.generate_and_store()

    def test_delete_removes_file(self, tmp_path: Path) -> None:
        path = tmp_path / "keystore.enc"
        ks = EncryptedFileKeystore(path, machine_id="m", uid=1000)
        ks.generate_and_store()
        ks.delete()
        assert not path.exists()
        ks.delete()  # second delete is a no-op

    def test_corrupted_file_raises_keystore_error(self, tmp_path: Path) -> None:
        path = tmp_path / "keystore.enc"
        ks = EncryptedFileKeystore(path, machine_id="m", uid=1000)
        ks.generate_and_store()
        path.write_bytes(b"not a keystore file")
        with pytest.raises(KeystoreError):
            ks.load()


class TestPubkeyHelpers:
    def test_pubkey_hex_is_64_lower_hex(self) -> None:
        ks = InMemoryKeystore()
        key = ks.generate_and_store()
        hex_str = pubkey_hex(key)
        assert len(hex_str) == 64
        assert hex_str == hex_str.lower()
        int(hex_str, 16)  # parses as hex

    def test_fingerprint_is_first_16(self) -> None:
        ks = InMemoryKeystore()
        key = ks.generate_and_store()
        full = pubkey_hex(key)
        assert pubkey_fingerprint(key) == full[:16]


class TestDefaultKeystoreFactory:
    def test_force_encrypted_file_returns_that_backend(self, tmp_path: Path) -> None:
        ks = default_keystore(
            encrypted_file_path=tmp_path / "k.enc",
            force_backend="encrypted_file",
        )
        assert isinstance(ks, EncryptedFileKeystore)

    def test_unknown_backend_raises(self, tmp_path: Path) -> None:
        with pytest.raises(KeystoreError):
            default_keystore(
                encrypted_file_path=tmp_path / "k.enc",
                force_backend="weird-backend",
            )

    def test_auto_select_falls_back_to_encrypted_file_when_no_dbus(self, tmp_path: Path) -> None:
        # CI / containers have no Secret Service. Auto-select must give us
        # the encrypted-file backend.
        ks = default_keystore(encrypted_file_path=tmp_path / "k.enc")
        assert isinstance(ks, EncryptedFileKeystore)
