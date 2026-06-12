"""#40a executor-package auto-fetch (worker leg).

A dispatched unit whose package digest is NOT in the local package store is
fetched from the coordinator (GET /api/v0/packages/{digest}, worker-signed),
extracted traversal-safe, verified (manifest hash against the assignment pin +
`compute_package_digest` against the manifest's executor.package_sha256), and
installed content-addressed — then dispatch proceeds exactly as if pre-staged.
Verification failures refuse with `auto_fetch_digest_mismatch`; fetch failures
(network/404) refuse with `package_unavailable`. Pre-staged packages
short-circuit (no network), and the cache is permanent (immutable digests).
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from auspexai_worker.cli import CoordinatorPackageFetcher
from auspexai_worker.config import WorkerConfig
from auspexai_worker.coordinator import (
    CoordinatorClient,
    CoordinatorError,
    PackageNotFoundError,
)
from auspexai_worker.daemon.dispatch import DispatchOutcomeKind, RunnerDispatcher
from auspexai_worker.provisioning import (
    AutoFetchResolver,
    ExecutePolicy,
    ExecutionMode,
    ProvisioningResolver,
    decide_execution,
    hash_manifest,
)
from auspexai_worker.signing import Rfc9421Signer
from auspexai_worker.state import (
    Database,
    MigrationRunner,
    PendingSubmissionRepository,
    SubmittedResultRepository,
)
from auspexai_worker.workspace import WorkspaceManager
from tests.test_executor_dispatch import EXECUTOR_DOUBLER, _envelope, _make_key, _runner_bin

# ---- fixtures / builders ----------------------------------------------------

EXECUTOR_FILES = {"executor.py": b"# the tenant's executor\n"}


def _package_digest(files: dict[str, bytes]) -> str:
    """The shared digest contract (mirror of compute_package_digest)."""
    lines = [f"{rel}\x00{hashlib.sha256(c).hexdigest()}" for rel, c in sorted(files.items())]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _manifest(files: dict[str, bytes] = EXECUTOR_FILES, *, pin: str | None = "auto") -> dict:
    """A model-free manifest; `pin="auto"` pins the real package digest."""
    executor: dict = {"command": ["python", "executor.py"]}
    if pin == "auto":
        executor["package_sha256"] = _package_digest(files)
    elif pin is not None:
        executor["package_sha256"] = pin
    return {"tenant_id": "synth-geometry", "experiment_id": "synth-v1", "executor": executor}


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _package_archive(manifest: dict, files: dict[str, bytes] = EXECUTOR_FILES) -> bytes:
    return _tar_bytes({"manifest.json": json.dumps(manifest).encode("utf-8"), **files})


class FakeFetcher:
    """provisioning.PackageFetcher returning canned bytes (or raising)."""

    def __init__(self, blob: bytes | None = None, error: Exception | None = None) -> None:
        self.blob = blob
        self.error = error
        self.calls: list[str] = []

    def fetch(self, manifest_sha256: str) -> bytes:
        self.calls.append(manifest_sha256)
        if self.error is not None:
            raise self.error
        assert self.blob is not None
        return self.blob


def _decide(store: Path, resolver) -> object:
    return decide_execution(
        policy=ExecutePolicy.PROVISIONED,
        tenant_id="synth-geometry",
        manifest_sha256=hash_manifest(_manifest()),
        resolver=resolver,
        model_store_dir=store / "models",
    )


def _no_leftover_staging(provisioning_dir: Path) -> bool:
    return not list(provisioning_dir.glob(".fetch-*"))


# ---- happy path: fetch + verify + install, then run as if pre-staged --------


def test_auto_fetch_happy_path_installs_and_runs(tmp_path: Path) -> None:
    manifest = _manifest()
    sha = hash_manifest(manifest)
    fetcher = FakeFetcher(_package_archive(manifest))
    resolver = AutoFetchResolver(tmp_path / "tenants", fetcher)

    decision = _decide(tmp_path, resolver)

    assert decision.mode is ExecutionMode.REAL
    assert decision.executor is not None
    assert decision.executor.command == ["python", "executor.py"]
    assert decision.executor.package_dir == tmp_path / "tenants" / sha
    # Installed content-addressed, exactly like an operator-staged package.
    assert (tmp_path / "tenants" / sha / "manifest.json").is_file()
    assert (tmp_path / "tenants" / sha / "executor.py").read_bytes() == EXECUTOR_FILES[
        "executor.py"
    ]
    assert fetcher.calls == [sha]
    assert _no_leftover_staging(tmp_path / "tenants")


def test_auto_fetch_cache_is_permanent_no_refetch(tmp_path: Path) -> None:
    manifest = _manifest()
    fetcher = FakeFetcher(_package_archive(manifest))
    resolver = AutoFetchResolver(tmp_path / "tenants", fetcher)

    first = _decide(tmp_path, resolver)
    second = _decide(tmp_path, resolver)

    assert first.mode is ExecutionMode.REAL
    assert second.mode is ExecutionMode.REAL
    assert len(fetcher.calls) == 1  # content-addressed + immutable: one fetch, ever


def test_prestaged_package_short_circuits_no_fetch(tmp_path: Path) -> None:
    manifest = _manifest()
    sha = hash_manifest(manifest)
    pkg = tmp_path / "tenants" / sha
    pkg.mkdir(parents=True)
    (pkg / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (pkg / "executor.py").write_bytes(EXECUTOR_FILES["executor.py"])

    fetcher = FakeFetcher(error=AssertionError("must not fetch a pre-staged package"))
    decision = _decide(tmp_path, AutoFetchResolver(tmp_path / "tenants", fetcher))

    assert decision.mode is ExecutionMode.REAL
    assert fetcher.calls == []


# ---- verification failures → EXECUTOR_REFUSED auto_fetch_digest_mismatch ----


def test_package_digest_mismatch_refuses_and_does_not_install(tmp_path: Path) -> None:
    manifest = _manifest()  # pins the digest of EXECUTOR_FILES...
    sha = hash_manifest(manifest)
    tampered = _package_archive(manifest, files={"executor.py": b"# TAMPERED\n"})
    resolver = AutoFetchResolver(tmp_path / "tenants", FakeFetcher(tampered))

    decision = _decide(tmp_path, resolver)

    assert decision.mode is ExecutionMode.REFUSE
    assert "auto_fetch_digest_mismatch" in decision.reason
    assert not (tmp_path / "tenants" / sha).exists()  # nothing installed
    assert _no_leftover_staging(tmp_path / "tenants")


def test_manifest_hash_mismatch_refuses(tmp_path: Path) -> None:
    # The archive's manifest does not hash to the digest the assignment pins.
    other = {**_manifest(), "experiment_id": "a-different-experiment"}
    resolver = AutoFetchResolver(tmp_path / "tenants", FakeFetcher(_package_archive(other)))

    decision = _decide(tmp_path, resolver)

    assert decision.mode is ExecutionMode.REFUSE
    assert "auto_fetch_digest_mismatch" in decision.reason


def test_missing_package_pin_refuses_unverifiable(tmp_path: Path) -> None:
    # No executor.package_sha256: an operator-staged package may omit it
    # (Phase-1 trust root), but network-fetched code without it is unverifiable.
    manifest = _manifest(pin=None)
    sha = hash_manifest(manifest)
    resolver = AutoFetchResolver(tmp_path / "tenants", FakeFetcher(_package_archive(manifest)))

    decision = decide_execution(
        policy=ExecutePolicy.PROVISIONED,
        tenant_id="synth-geometry",
        manifest_sha256=sha,
        resolver=resolver,
        model_store_dir=tmp_path / "models",
    )

    assert decision.mode is ExecutionMode.REFUSE
    assert "auto_fetch_digest_mismatch" in decision.reason
    assert not (tmp_path / "tenants" / sha).exists()


# ---- hostile archives → refused, nothing escapes ----------------------------


def _hostile_traversal() -> bytes:
    manifest = _manifest()
    return _tar_bytes(
        {
            "manifest.json": json.dumps(manifest).encode("utf-8"),
            "../evil.py": b"# escapes the staging dir",
        }
    )


def _hostile_absolute() -> bytes:
    manifest = _manifest()
    return _tar_bytes(
        {
            "manifest.json": json.dumps(manifest).encode("utf-8"),
            "/abs-evil.py": b"# absolute path",
        }
    )


def _hostile_symlink() -> bytes:
    manifest = _manifest()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        payload = json.dumps(manifest).encode("utf-8")
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
        link = tarfile.TarInfo(name="executor.py")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tf.addfile(link)
    return buf.getvalue()


@pytest.mark.parametrize(
    "build_archive",
    [_hostile_traversal, _hostile_absolute, _hostile_symlink],
    ids=["dotdot-member", "absolute-member", "symlink-member"],
)
def test_traversal_tar_refused_nothing_extracted(tmp_path: Path, build_archive) -> None:
    sha = hash_manifest(_manifest())
    store = tmp_path / "tenants"
    resolver = AutoFetchResolver(store, FakeFetcher(build_archive()))

    decision = _decide(tmp_path, resolver)

    assert decision.mode is ExecutionMode.REFUSE
    assert not (store / sha).exists()
    assert _no_leftover_staging(store)
    # The traversal member must not have landed anywhere above the staging dir.
    assert not (store / "evil.py").exists()
    assert not (tmp_path / "evil.py").exists()


def test_garbage_archive_refuses_package_unavailable(tmp_path: Path) -> None:
    # Not-even-a-tarball (e.g. a truncated download) is availability, not
    # integrity — a retry may fetch intact bytes.
    resolver = AutoFetchResolver(tmp_path / "tenants", FakeFetcher(b"\x00not a tarball"))
    decision = _decide(tmp_path, resolver)
    assert decision.mode is ExecutionMode.REFUSE
    assert "package_unavailable" in decision.reason


# ---- fetch failures → EXECUTOR_REFUSED package_unavailable ------------------


def test_404_refuses_package_unavailable(tmp_path: Path) -> None:
    fetcher = FakeFetcher(error=PackageNotFoundError("no package for this digest"))
    decision = _decide(tmp_path, AutoFetchResolver(tmp_path / "tenants", fetcher))
    assert decision.mode is ExecutionMode.REFUSE
    assert "package_unavailable" in decision.reason


def test_network_error_refuses_package_unavailable(tmp_path: Path) -> None:
    fetcher = FakeFetcher(error=CoordinatorError("HTTP transport error: connection refused"))
    decision = _decide(tmp_path, AutoFetchResolver(tmp_path / "tenants", fetcher))
    assert decision.mode is ExecutionMode.REFUSE
    assert "package_unavailable" in decision.reason
    assert isinstance(decision.reason, str)


# ---- the signed client leg (mock transport) ---------------------------------


def _make_signer() -> Rfc9421Signer:
    privkey = Ed25519PrivateKey.generate()
    pub = privkey.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    return Rfc9421Signer(privkey, pub)


def _make_client(handler) -> CoordinatorClient:
    return CoordinatorClient(
        base_url="http://test-coordinator.invalid",
        signer=_make_signer(),
        transport=httpx.MockTransport(handler),
    )


def test_fetch_package_200_returns_bytes_signed() -> None:
    blob = _package_archive(_manifest())
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["signed"] = "signature" in request.headers and "signature-input" in request.headers
        return httpx.Response(200, content=blob, headers={"Content-Type": "application/gzip"})

    with _make_client(handler) as client:
        result = client.fetch_package(digest="A" * 64)  # case-normalized on the wire

    assert result == blob
    assert seen["method"] == "GET"
    assert seen["path"] == f"/api/v0/packages/{'a' * 64}"
    assert seen["signed"] is True


def test_fetch_package_404_raises_package_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"detail": {"error": {"code": "package_not_found", "message": "unknown digest"}}},
        )

    with _make_client(handler) as client, pytest.raises(PackageNotFoundError):
        client.fetch_package(digest="b" * 64)


def test_fetch_package_requires_signer() -> None:
    client = CoordinatorClient(
        base_url="http://test-coordinator.invalid",
        transport=httpx.MockTransport(lambda request: httpx.Response(200)),
    )
    with client, pytest.raises(CoordinatorError, match="requires a signer"):
        client.fetch_package(digest="c" * 64)


def test_fetch_package_unexpected_status_raises_coordinator_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with _make_client(handler) as client, pytest.raises(CoordinatorError, match="unexpected"):
        client.fetch_package(digest="d" * 64)


def test_happy_path_end_to_end_through_signed_client(tmp_path: Path) -> None:
    """The full worker leg: signed GET via CoordinatorClient (mock transport) →
    CoordinatorPackageFetcher → AutoFetchResolver → decide_execution REAL."""
    manifest = _manifest()
    sha = hash_manifest(manifest)
    blob = _package_archive(manifest)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/api/v0/packages/{sha}"
        return httpx.Response(200, content=blob)

    with _make_client(handler) as client:
        resolver = AutoFetchResolver(tmp_path / "tenants", CoordinatorPackageFetcher(client))
        decision = _decide(tmp_path, resolver)

    assert decision.mode is ExecutionMode.REAL
    assert (tmp_path / "tenants" / sha / "executor.py").is_file()


def test_client_transport_error_surfaces_as_package_unavailable(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with _make_client(handler) as client:
        resolver = AutoFetchResolver(tmp_path / "tenants", CoordinatorPackageFetcher(client))
        decision = _decide(tmp_path, resolver)

    assert decision.mode is ExecutionMode.REFUSE
    assert "package_unavailable" in decision.reason


# ---- dispatch level: outcome kinds + runs-as-if-pre-staged ------------------


def _make_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "worker.db")
    MigrationRunner(db).apply_all()
    return db


def _autofetch_dispatcher(client, db, tmp_path: Path, privkey, pub) -> RunnerDispatcher:
    return RunnerDispatcher(
        coordinator=client,
        worker_id="wkr-a",
        worker_pubkey=pub,
        privkey=privkey,
        workspace_manager=WorkspaceManager(tmp_path / "runs"),
        submitted_repo=SubmittedResultRepository(db),
        pending_repo=PendingSubmissionRepository(db),
        use_bubblewrap=False,
        runner_bin=_runner_bin(),
        execute_policy=ExecutePolicy.PROVISIONED,
        executor_resolver=AutoFetchResolver(
            tmp_path / "tenants", CoordinatorPackageFetcher(client)
        ),
    )


def test_dispatch_auto_fetched_unit_runs_and_submits(tmp_path: Path) -> None:
    """Full dispatch through the real runner: the package arrives by fetch
    instead of staging and the unit then runs EXACTLY as if pre-staged —
    real executor output, signed + submitted."""
    privkey, pub = _make_key()
    files = {"executor.py": EXECUTOR_DOUBLER.encode("utf-8")}
    manifest = {
        "tenant_id": "synth-doubler",
        "experiment_id": "synth-doubler-v1",
        "executor": {
            "command": [sys.executable, "executor.py"],
            "package_sha256": _package_digest(files),
        },
        "models": [],
    }
    sha = hash_manifest(manifest)
    blob = _tar_bytes({"manifest.json": json.dumps(manifest).encode("utf-8"), **files})
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and req.url.path == f"/api/v0/packages/{sha}":
            return httpx.Response(200, content=blob)
        captured["body"] = json.loads(req.content)
        return httpx.Response(
            201,
            json={
                "result_id": "res-001",
                "unit_id": captured["body"]["unit_id"],
                "unit_status_after": "in_progress",
                "completions_so_far": 1,
                "replication_target": 3,
            },
        )

    db = _make_db(tmp_path)
    client = CoordinatorClient(
        base_url="http://test-coord.invalid",
        signer=Rfc9421Signer(privkey, pub),
        transport=httpx.MockTransport(handler),
    )
    with client:
        outcome = _autofetch_dispatcher(client, db, tmp_path, privkey, pub).run_unit(
            _envelope(sha, value=21)
        )

    assert outcome.kind == DispatchOutcomeKind.SUBMITTED
    assert captured["body"]["payload"] == {"doubled": 42}
    assert (tmp_path / "tenants" / sha / "executor.py").is_file()  # cached for next time


def test_dispatch_404_is_executor_refused_package_unavailable(tmp_path: Path) -> None:
    privkey, pub = _make_key()

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"detail": {"error": {"code": "package_not_found", "message": "unknown digest"}}},
        )

    db = _make_db(tmp_path)
    client = CoordinatorClient(
        base_url="http://test-coord.invalid",
        signer=Rfc9421Signer(privkey, pub),
        transport=httpx.MockTransport(handler),
    )
    with client:
        outcome = _autofetch_dispatcher(client, db, tmp_path, privkey, pub).run_unit(
            _envelope(hash_manifest(_manifest()))
        )

    assert outcome.kind == DispatchOutcomeKind.EXECUTOR_REFUSED
    assert "package_unavailable" in outcome.reason


def test_dispatch_tampered_package_is_executor_refused_digest_mismatch(tmp_path: Path) -> None:
    privkey, pub = _make_key()
    manifest = _manifest()  # pins EXECUTOR_FILES' digest...
    tampered = _package_archive(manifest, files={"executor.py": b"# TAMPERED\n"})

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=tampered)

    db = _make_db(tmp_path)
    client = CoordinatorClient(
        base_url="http://test-coord.invalid",
        signer=Rfc9421Signer(privkey, pub),
        transport=httpx.MockTransport(handler),
    )
    with client:
        outcome = _autofetch_dispatcher(client, db, tmp_path, privkey, pub).run_unit(
            _envelope(hash_manifest(manifest))
        )

    assert outcome.kind == DispatchOutcomeKind.EXECUTOR_REFUSED
    assert "auto_fetch_digest_mismatch" in outcome.reason


# ---- config: [provisioning] auto_fetch, default ON --------------------------


def test_config_auto_fetch_defaults_on(tmp_path: Path) -> None:
    cfg = WorkerConfig.load(config_path=tmp_path / "missing.toml", env={})
    assert cfg.auto_fetch is True


def test_config_provisioning_block_can_disable(tmp_path: Path) -> None:
    toml = tmp_path / "worker.toml"
    toml.write_text("[provisioning]\nauto_fetch = false\n", encoding="utf-8")
    cfg = WorkerConfig.load(config_path=toml, env={})
    assert cfg.auto_fetch is False


def test_config_env_overrides_auto_fetch(tmp_path: Path) -> None:
    toml = tmp_path / "worker.toml"
    toml.write_text("[provisioning]\nauto_fetch = true\n", encoding="utf-8")
    cfg = WorkerConfig.load(config_path=toml, env={"AUSPEXAI_WORKER_AUTO_FETCH": "false"})
    assert cfg.auto_fetch is False


def test_auto_fetch_off_keeps_staged_only_behavior(tmp_path: Path) -> None:
    """`auto_fetch = false` wires the plain ProvisioningResolver: an unstaged
    digest refuses (refuse-don't-echo) with NO fetch attempt — the pre-#40a
    behavior, byte-for-byte."""
    decision = _decide(tmp_path, ProvisioningResolver(tmp_path / "tenants"))
    assert decision.mode is ExecutionMode.REFUSE
    assert "no provisioned executor" in decision.reason
    assert not (tmp_path / "tenants").exists()  # never even created
