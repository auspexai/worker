"""Tests for the M6 CLI verbs: `auspexai-worker login` and `auspexai-worker withdraw`.

The login flow has two external dependencies (the GitHub Device Flow and
the coordinator HTTP API); both are stubbed at module-attribute level so
the CLI runs end-to-end against real local state.
"""

from __future__ import annotations

import json as _json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from auspexai_worker import cli as cli_module
from auspexai_worker.cli import cli
from auspexai_worker.coordinator import (
    BindingTokenExpiredError,
    CoordinatorError,
    OAuthExchangeResponse,
    WorkerStatusResponse,
)
from auspexai_worker.oauth import AccessDeniedError, DeviceCode, ExpiredTokenError
from auspexai_worker.state import (
    Database,
    MigrationRunner,
    WorkerSelfRepository,
)


def _write_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "worker.toml"
    cfg.write_text(
        '[coordinator]\nurl = "http://m6-test.invalid"\n'
        '[identity]\nkeystore_backend = "encrypted_file"\n'
    )
    return cfg


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "AUSPEXAI_WORKER_STATE_DIR": str(tmp_path / "state"),
        "AUSPEXAI_WORKER_DATA_DIR": str(tmp_path / "data"),
    }


def _bootstrap_t0_identity(tmp_path: Path) -> Database:
    """Pre-create a T0 worker_self row so login has something to upgrade."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    db = Database(state_dir / "worker.db")
    MigrationRunner(db).apply_all()
    WorkerSelfRepository(db).insert(
        worker_id="wkr-test",
        trust_tier=0,
        pubkey_hex="a" * 64,
        enrolled_at=datetime(2026, 5, 20, 10, 0, 0, tzinfo=UTC),
    )
    return db


def _generate_real_keystore(tmp_path: Path) -> None:
    """Build a fresh keystore + keypair so the login flow's build_signer
    succeeds. The CLI opens the keystore directly so we can't fake this
    cheaply; using the actual EncryptedFileKeystore is faster than
    monkeypatching."""
    from auspexai_worker.keystore import EncryptedFileKeystore

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    keystore_path = data_dir / "keystore.enc"
    ks = EncryptedFileKeystore(keystore_path)
    ks.generate_and_store()


class _FakeCoordinatorClient:
    """Drop-in for CoordinatorClient. Records calls; lets per-test code raise."""

    def __init__(self, *, base_url: str, signer=None) -> None:
        self.calls: list[tuple[str, dict]] = []
        # Test-controlled response/exception attributes:
        self.exchange_response = OAuthExchangeResponse(
            account_id="acct-fake",
            binding_token="bnd-fake",
            expires_at=datetime(2026, 5, 22, 10, 5, 0, tzinfo=UTC),
            is_new_account=True,
        )
        self.exchange_exc: Exception | None = None
        self.upgrade_response = WorkerStatusResponse(
            worker_id="wkr-test",
            trust_tier=1,
            registered_at=datetime(2026, 5, 20, 10, 0, 0, tzinfo=UTC),
            last_heartbeat_at=None,
            retired_at=None,
        )
        self.upgrade_exc: Exception | None = None
        self.retire_response = WorkerStatusResponse(
            worker_id="wkr-test",
            trust_tier=0,
            registered_at=datetime(2026, 5, 20, 10, 0, 0, tzinfo=UTC),
            last_heartbeat_at=None,
            retired_at=datetime(2026, 5, 22, 11, 0, 0, tzinfo=UTC),
        )
        self.retire_exc: Exception | None = None
        self.attribution_state: dict = {
            "account_id": "acct-fake",
            "public_attribution": False,
            "attribution_name": None,
        }
        self.attribution_exc: Exception | None = None

    def __enter__(self) -> _FakeCoordinatorClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass

    def oauth_exchange(self, *, idp: str, access_token: str) -> OAuthExchangeResponse:
        self.calls.append(("oauth_exchange", {"idp": idp, "access_token": access_token}))
        if self.exchange_exc is not None:
            raise self.exchange_exc
        return self.exchange_response

    def upgrade_worker(self, *, worker_id: str, binding_token: str) -> WorkerStatusResponse:
        self.calls.append(
            ("upgrade_worker", {"worker_id": worker_id, "binding_token": binding_token})
        )
        if self.upgrade_exc is not None:
            raise self.upgrade_exc
        return self.upgrade_response

    def retire_worker(self, *, worker_id: str) -> WorkerStatusResponse:
        self.calls.append(("retire_worker", {"worker_id": worker_id}))
        if self.retire_exc is not None:
            raise self.retire_exc
        return self.retire_response

    def set_attribution(self, *, account_id, public_attribution, attribution_name=None) -> dict:
        self.calls.append(
            (
                "set_attribution",
                {
                    "account_id": account_id,
                    "public_attribution": public_attribution,
                    "attribution_name": attribution_name,
                },
            )
        )
        if self.attribution_exc is not None:
            raise self.attribution_exc
        self.attribution_state = {
            "account_id": account_id,
            "public_attribution": public_attribution,
            "attribution_name": attribution_name,
        }
        return self.attribution_state

    def get_attribution(self, *, account_id) -> dict:
        self.calls.append(("get_attribution", {"account_id": account_id}))
        if self.attribution_exc is not None:
            raise self.attribution_exc
        return self.attribution_state


@pytest.fixture
def fake_client_factory(monkeypatch: pytest.MonkeyPatch):
    """Replace cli_module.CoordinatorClient with a singleton fake for inspection.

    One instance is reused for every ``CoordinatorClient(...)`` call within a test, so
    calls accumulate and attribution state persists across the several short-lived
    clients the `login` flow opens (exchange/upgrade, then get, then set). Pre-built so a
    test can seed ``["instance"].attribution_state`` before invoking the command.
    """
    instance = _FakeCoordinatorClient(base_url="", signer=None)

    def factory(*args, **kwargs) -> _FakeCoordinatorClient:
        return instance

    monkeypatch.setattr(cli_module, "CoordinatorClient", factory)
    return {"instance": instance}


def _bootstrap_t1_bound(tmp_path: Path, account_id: str = "acct-fake") -> None:
    """A T0 identity upgraded to a T1 account-bound worker (local state only)."""
    db = _bootstrap_t0_identity(tmp_path)
    try:
        WorkerSelfRepository(db).update_after_upgrade(
            new_tier=1,
            account_binding_json=_json.dumps({"idp": "github", "account_id": account_id}),
        )
    finally:
        db.close()
    _generate_real_keystore(tmp_path)


def test_update_after_unbind_clears_binding_and_resets_tier(tmp_path: Path) -> None:
    """Worker logout (state side): the binding is dropped + tier reverts to 0, the inverse of
    update_after_upgrade. The worker row stays (not deleted) -- logout is not retire."""
    db = _bootstrap_t0_identity(tmp_path)
    try:
        repo = WorkerSelfRepository(db)
        repo.update_after_upgrade(
            new_tier=1,
            account_binding_json=_json.dumps({"idp": "github", "account_id": "acct-x"}),
        )
        bound = repo.get()
        assert bound is not None
        assert bound.trust_tier == 1
        assert bound.account_binding_json is not None
        repo.update_after_unbind()
        after = repo.get()
        assert after is not None
        assert after.trust_tier == 0
        assert after.account_binding_json is None
    finally:
        db.close()


class TestAccountAttribution:
    """D-inc4: `auspexai-worker account attribution` — the reversible opt-in surface."""

    def test_opt_in_uses_verified_identity(self, tmp_path: Path, fake_client_factory) -> None:
        """Opting in credits the verified GitHub account — no custom name is accepted."""
        _bootstrap_t1_bound(tmp_path)
        cfg = _write_config(tmp_path)
        result = CliRunner().invoke(
            cli,
            ["--config", str(cfg), "account", "attribution", "--public"],
            env=_env(tmp_path),
        )
        assert result.exit_code == 0, result.output
        assert "public credit: ON" in result.output
        fake = fake_client_factory["instance"]
        assert (
            "set_attribution",
            {
                "account_id": "acct-fake",
                "public_attribution": True,
                "attribution_name": None,  # always the verified GitHub identity
            },
        ) in fake.calls

    def test_name_option_is_removed(self, tmp_path: Path, fake_client_factory) -> None:
        """`--name` is gone — citation must use real credentials, not a typed name."""
        _bootstrap_t1_bound(tmp_path)
        cfg = _write_config(tmp_path)
        result = CliRunner().invoke(
            cli,
            ["--config", str(cfg), "account", "attribution", "--public", "--name", "Faker"],
            env=_env(tmp_path),
        )
        assert result.exit_code == 2  # click: no such option

    def test_opt_out(self, tmp_path: Path, fake_client_factory) -> None:
        _bootstrap_t1_bound(tmp_path)
        cfg = _write_config(tmp_path)
        result = CliRunner().invoke(
            cli,
            ["--config", str(cfg), "account", "attribution", "--anonymous"],
            env=_env(tmp_path),
        )
        assert result.exit_code == 0, result.output
        assert "public credit: OFF" in result.output

    def test_not_bound_errors(self, tmp_path: Path, fake_client_factory) -> None:
        _bootstrap_t0_identity(tmp_path).close()  # T0, never bound
        cfg = _write_config(tmp_path)
        result = CliRunner().invoke(
            cli,
            ["--config", str(cfg), "account", "attribution"],
            env=_env(tmp_path),
        )
        assert result.exit_code == 1
        assert "not bound" in result.output


def _stub_device_flow(
    monkeypatch: pytest.MonkeyPatch,
    *,
    token: str | None = "gho_test_token",
    raises: Exception | None = None,
) -> None:
    """Replace cli_module.run_device_flow with a deterministic stub."""

    def fake_run(
        *,
        on_code: Callable[[DeviceCode], None],
        **_kwargs,
    ) -> str:
        on_code(
            DeviceCode(
                device_code="DEV-test",
                user_code="WXYZ-1234",
                verification_uri="https://github.com/login/device",
                expires_in=900,
                interval=5,
            )
        )
        if raises is not None:
            raise raises
        assert token is not None
        return token

    monkeypatch.setattr(cli_module, "run_device_flow", fake_run)


# ---- login -----------------------------------------------------------------


class TestLoginCommand:
    def test_not_enrolled_errors(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(cfg), "login"], env=_env(tmp_path))
        assert result.exit_code == 1
        assert "not enrolled" in result.output

    def test_already_t1_is_noop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db = _bootstrap_t0_identity(tmp_path)
        WorkerSelfRepository(db).update_tier(1)
        db.close()
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(cfg), "login"], env=_env(tmp_path))
        assert result.exit_code == 0
        assert "already T1" in result.output

    def test_happy_path_promotes_t0_to_t1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_client_factory,
    ) -> None:
        db = _bootstrap_t0_identity(tmp_path)
        db.close()
        _generate_real_keystore(tmp_path)
        _stub_device_flow(monkeypatch)

        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(cfg), "login"], env=_env(tmp_path))
        assert result.exit_code == 0, result.output
        assert "login successful" in result.output
        assert "T0 → T1" in result.output

        # Verify the local DB was updated.
        db2 = Database((tmp_path / "state") / "worker.db")
        try:
            worker = WorkerSelfRepository(db2).get()
        finally:
            db2.close()
        assert worker is not None
        assert worker.trust_tier == 1
        assert worker.account_binding_json is not None
        binding = _json.loads(worker.account_binding_json)
        assert binding["idp"] == "github"
        assert binding["account_id"] == "acct-fake"

        # Verify the coordinator was called correctly.
        fake = fake_client_factory["instance"]
        assert fake.calls[0] == (
            "oauth_exchange",
            {"idp": "github", "access_token": "gho_test_token"},
        )
        assert fake.calls[1] == (
            "upgrade_worker",
            {"worker_id": "wkr-test", "binding_token": "bnd-fake"},
        )

    def test_user_denied_exits_with_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_client_factory,
    ) -> None:
        db = _bootstrap_t0_identity(tmp_path)
        db.close()
        _generate_real_keystore(tmp_path)
        _stub_device_flow(monkeypatch, raises=AccessDeniedError("user said no"))

        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(cfg), "login"], env=_env(tmp_path))
        assert result.exit_code == 1
        assert "authorization denied" in result.output

    def test_expired_token_exits_with_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_client_factory,
    ) -> None:
        db = _bootstrap_t0_identity(tmp_path)
        db.close()
        _generate_real_keystore(tmp_path)
        _stub_device_flow(monkeypatch, raises=ExpiredTokenError("expired before authorization"))

        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(cfg), "login"], env=_env(tmp_path))
        assert result.exit_code == 1
        assert "timed out" in result.output

    def test_binding_token_expired_exits_with_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_client_factory,
    ) -> None:
        db = _bootstrap_t0_identity(tmp_path)
        db.close()
        _generate_real_keystore(tmp_path)
        _stub_device_flow(monkeypatch)

        cfg = _write_config(tmp_path)
        runner = CliRunner()

        # Pre-arrange the upgrade call to fail. We need to access the fake
        # instance after the CLI builds it — set the exception on the factory
        # by capturing the instance before invoke. Simpler: patch the factory
        # to attach the exception before the CLI calls upgrade.
        def factory(*args, **kwargs) -> _FakeCoordinatorClient:
            inst = _FakeCoordinatorClient(*args, **kwargs)
            inst.upgrade_exc = BindingTokenExpiredError("token aged out")
            fake_client_factory["instance"] = inst
            return inst

        monkeypatch.setattr(cli_module, "CoordinatorClient", factory)

        result = runner.invoke(cli, ["--config", str(cfg), "login"], env=_env(tmp_path))
        assert result.exit_code == 1
        assert "binding token expired" in result.output


# ---- withdraw --------------------------------------------------------------


class TestLoginPublicCreditPrompt:
    """The interactive public-credit (System B) opt-in at the end of `login`.

    The prompt is STATE-AWARE: it reads the account's standing choice, defaults to it,
    and writes only when the answer CHANGES it — so a routine re-login never silently
    re-anonymizes a contributor (the citation-footgun fix). The prompt is gated on an
    interactive stdin, forced here via `_stdin_is_interactive` (CliRunner is never a TTY).
    """

    def _setup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        db = _bootstrap_t0_identity(tmp_path)
        db.close()
        _generate_real_keystore(tmp_path)
        _stub_device_flow(monkeypatch)
        monkeypatch.setattr(cli_module, "_stdin_is_interactive", lambda: True)
        return _write_config(tmp_path)

    def test_preserves_standing_opt_in(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_client_factory
    ) -> None:
        """Re-login while already credited + keeping the choice writes NOTHING (preserve)."""
        fake_client_factory["instance"].attribution_state = {
            "account_id": "acct-fake",
            "public_attribution": True,
            "attribution_name": None,
        }
        cfg = self._setup(tmp_path, monkeypatch)
        result = CliRunner().invoke(
            cli, ["--config", str(cfg), "login"], env=_env(tmp_path), input="\n"
        )
        assert result.exit_code == 0, result.output
        assert "currently credited" in result.output.lower()
        assert "You'll be credited" in result.output
        methods = [c[0] for c in fake_client_factory["instance"].calls]
        assert "get_attribution" in methods
        assert "set_attribution" not in methods  # unchanged -> no write -> no re-anonymize

    def test_opt_in_when_anonymous(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_client_factory
    ) -> None:
        """A first-time 'yes' from the anonymous default records the opt-in."""
        cfg = self._setup(tmp_path, monkeypatch)  # default state: public_attribution=False
        result = CliRunner().invoke(
            cli, ["--config", str(cfg), "login"], env=_env(tmp_path), input="y\n"
        )
        assert result.exit_code == 0, result.output
        assert "You'll be credited" in result.output
        assert (
            "set_attribution",
            {"account_id": "acct-fake", "public_attribution": True, "attribution_name": None},
        ) in fake_client_factory["instance"].calls

    def test_opt_out_when_credited(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_client_factory
    ) -> None:
        """An explicit 'no' while currently credited is the ONLY path that writes False."""
        fake_client_factory["instance"].attribution_state = {
            "account_id": "acct-fake",
            "public_attribution": True,
            "attribution_name": None,
        }
        cfg = self._setup(tmp_path, monkeypatch)
        result = CliRunner().invoke(
            cli, ["--config", str(cfg), "login"], env=_env(tmp_path), input="n\n"
        )
        assert result.exit_code == 0, result.output
        assert "anonymous" in result.output.lower()
        assert (
            "set_attribution",
            {"account_id": "acct-fake", "public_attribution": False, "attribution_name": None},
        ) in fake_client_factory["instance"].calls

    def test_read_failure_never_overwrites_opt_in(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_client_factory
    ) -> None:
        """If the standing choice can't be read, a default 'no' must NOT write False — the
        existing opt-in is left intact rather than silently overwritten."""
        inst = fake_client_factory["instance"]
        inst.attribution_state = {
            "account_id": "acct-fake",
            "public_attribution": True,  # actually opted in...
            "attribution_name": None,
        }
        inst.attribution_exc = CoordinatorError("coordinator unreachable")  # ...but read fails
        cfg = self._setup(tmp_path, monkeypatch)
        result = CliRunner().invoke(
            cli, ["--config", str(cfg), "login"], env=_env(tmp_path), input="\n"
        )
        assert result.exit_code == 0, result.output
        methods = [c[0] for c in inst.calls]
        assert "set_attribution" not in methods  # no write -> existing opt-in preserved


class TestWithdrawCommand:
    def test_not_enrolled_says_nothing_to_do(self, tmp_path: Path) -> None:
        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(cfg), "withdraw"], env=_env(tmp_path))
        assert result.exit_code == 0
        assert "nothing to withdraw" in result.output

    def test_yes_flag_purges_local_state(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_client_factory,
    ) -> None:
        db = _bootstrap_t0_identity(tmp_path)
        db.close()
        _generate_real_keystore(tmp_path)

        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(cfg), "withdraw", "--yes"], env=_env(tmp_path))
        assert result.exit_code == 0, result.output
        assert "worker withdrawn" in result.output

        # Coordinator was called.
        fake = fake_client_factory["instance"]
        assert fake.calls == [("retire_worker", {"worker_id": "wkr-test"})]

        # Local DB file is gone.
        db_path = (tmp_path / "state") / "worker.db"
        assert not db_path.exists()

    def test_confirmation_prompt_wrong_input_aborts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_client_factory,
    ) -> None:
        db = _bootstrap_t0_identity(tmp_path)
        db.close()
        _generate_real_keystore(tmp_path)

        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--config", str(cfg), "withdraw"], env=_env(tmp_path), input="no\n"
        )
        assert result.exit_code == 1
        assert "aborted" in result.output

        # DB still exists.
        db_path = (tmp_path / "state") / "worker.db"
        assert db_path.exists()

    def test_confirmation_prompt_correct_input_proceeds(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_client_factory,
    ) -> None:
        db = _bootstrap_t0_identity(tmp_path)
        db.close()
        _generate_real_keystore(tmp_path)

        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--config", str(cfg), "withdraw"],
            env=_env(tmp_path),
            input="withdraw\n",
        )
        assert result.exit_code == 0, result.output
        assert "worker withdrawn" in result.output

        db_path = (tmp_path / "state") / "worker.db"
        assert not db_path.exists()

    def test_coord_unreachable_still_purges_locally(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_client_factory,
    ) -> None:
        db = _bootstrap_t0_identity(tmp_path)
        db.close()
        _generate_real_keystore(tmp_path)

        from auspexai_worker.coordinator import CoordinatorError

        def factory(*args, **kwargs) -> _FakeCoordinatorClient:
            inst = _FakeCoordinatorClient(*args, **kwargs)
            inst.retire_exc = CoordinatorError("HTTP transport error")
            fake_client_factory["instance"] = inst
            return inst

        monkeypatch.setattr(cli_module, "CoordinatorClient", factory)

        cfg = _write_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(cfg), "withdraw", "--yes"], env=_env(tmp_path))
        # Coordinator failed but local purge still happens; CLI prints both
        # the warning and the final "worker withdrawn" message. Exit code 0
        # because withdrawal is volunteer-initiated and local state matters.
        assert result.exit_code == 0
        assert "Continuing with local purge" in result.output
        assert "worker withdrawn" in result.output

        db_path = (tmp_path / "state") / "worker.db"
        assert not db_path.exists()
