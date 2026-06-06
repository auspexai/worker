"""Real tenant-executor dispatch through the worker (§9 #37).

End-to-end through the actual dispatcher + the real `auspexai-worker-runner`
binary (passthrough mode) + a fixture executor that implements the SDK
ExecutorHarness CLI contract WITHOUT importing the SDK (workers don't depend on
tenant-sdk). Proves: provisioned + hash-matched -> runs the real executor and
submits its output; unresolved -> refuse-don't-echo; executor failure -> refuse.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from auspexai_worker.coordinator import (
    AssignmentResponse,
    CoordinatorClient,
    WorkUnitEnvelope,
)
from auspexai_worker.daemon.dispatch import DispatchOutcomeKind, RunnerDispatcher
from auspexai_worker.provisioning import ExecutePolicy, ProvisioningResolver, hash_manifest
from auspexai_worker.signing import Rfc9421Signer
from auspexai_worker.state import (
    Database,
    MigrationRunner,
    PendingSubmissionRepository,
    SubmittedResultRepository,
)
from auspexai_worker.workspace import WorkspaceManager

# A minimal executor honoring `--input/--output/--models/--timeout`; doubles
# payload["input"] into payload["doubled"].
EXECUTOR_DOUBLER = """
import argparse, json
from datetime import datetime, timezone
p = argparse.ArgumentParser()
for f in ("input", "output", "models"):
    p.add_argument(f"--{f}", required=True)
p.add_argument("--timeout", type=int, default=600)
a = p.parse_args()
unit = json.loads(open(a.input).read())
out = {
    "schema_version": "0.1",
    "unit_id": unit["unit_id"],
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "exit_code": 0,
    "payload": {"doubled": unit["payload"]["input"] * 2},
}
open(a.output, "w").write(json.dumps(out))
"""

EXECUTOR_FAIL = """
import sys
sys.stderr.write("boom: tenant code raised\\n")
sys.exit(1)
"""

# Loads a weight file from the --models dir (the BYOM store) to prove the store
# is what --models resolves to.
EXECUTOR_USES_MODEL = """
import argparse, json, pathlib
from datetime import datetime, timezone
p = argparse.ArgumentParser()
for f in ("input", "output", "models"):
    p.add_argument(f"--{f}", required=True)
p.add_argument("--timeout", type=int, default=600)
a = p.parse_args()
unit = json.loads(open(a.input).read())
weight = (pathlib.Path(a.models) / "weights.txt").read_text().strip()
out = {
    "schema_version": "0.1",
    "unit_id": unit["unit_id"],
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "exit_code": 0,
    "payload": {"loaded_weight": weight},
}
open(a.output, "w").write(json.dumps(out))
"""


def _make_key() -> tuple[Ed25519PrivateKey, str]:
    pk = Ed25519PrivateKey.generate()
    pub = pk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    return pk, pub


def _runner_bin() -> str:
    return str(Path(sys.executable).parent / "auspexai-worker-runner")


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "worker.db")
    MigrationRunner(d).apply_all()
    return d


def _provision(
    root: Path, executor_src: str, *, tenant: str = "synth-doubler", models: list | None = None
) -> str:
    """Stage a tenant package and return its manifest_sha256."""
    manifest = {
        "tenant_id": tenant,
        "experiment_id": f"{tenant}-v1",
        "executor": {"command": [sys.executable, "executor.py"]},
        "models": models if models is not None else [],
    }
    sha = hash_manifest(manifest)
    pkg = root / sha
    pkg.mkdir(parents=True)
    (pkg / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (pkg / "executor.py").write_text(executor_src, encoding="utf-8")
    return sha


def _envelope(manifest_sha256: str, *, value: int = 21) -> AssignmentResponse:
    return AssignmentResponse(
        assignment_id="asg-1",
        assigned_at=datetime(2026, 6, 6, 12, 0, 0, tzinfo=UTC),
        coordinator_experiment_id="exp-coord-1",
        work_unit=WorkUnitEnvelope(
            schema_version="0.1",
            unit_id="u-1",
            tenant_id="synth-doubler",
            experiment_id="exp-label",
            manifest_sha256=manifest_sha256,
            created_at=datetime(2026, 6, 6, 11, 0, 0, tzinfo=UTC),
            payload={"input": value},
        ),
    )


def _client(captured: dict, privkey, pub) -> CoordinatorClient:
    def handler(req: httpx.Request) -> httpx.Response:
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

    return CoordinatorClient(
        base_url="http://test-coord.invalid",
        signer=Rfc9421Signer(privkey, pub),
        transport=httpx.MockTransport(handler),
    )


def _dispatcher(
    client,
    db,
    runs_dir,
    *,
    policy,
    provisioning_dir,
    privkey,
    pub,
    model_store_dir=None,
    thermal_monitor=None,
):
    return RunnerDispatcher(
        coordinator=client,
        worker_id="wkr-a",
        worker_pubkey=pub,
        privkey=privkey,
        workspace_manager=WorkspaceManager(runs_dir),
        submitted_repo=SubmittedResultRepository(db),
        pending_repo=PendingSubmissionRepository(db),
        use_bubblewrap=False,
        runner_bin=_runner_bin(),
        execute_policy=policy,
        executor_resolver=ProvisioningResolver(provisioning_dir),
        model_store_dir=model_store_dir,
        thermal_monitor=thermal_monitor,
    )


def test_provisioned_real_executor_submits_its_output(tmp_path: Path, db: Database):
    privkey, pub = _make_key()
    prov = tmp_path / "tenants"
    sha = _provision(prov, EXECUTOR_DOUBLER)
    captured: dict = {}
    with _client(captured, privkey, pub) as client:
        disp = _dispatcher(
            client,
            db,
            tmp_path / "runs",
            policy=ExecutePolicy.PROVISIONED,
            provisioning_dir=prov,
            privkey=privkey,
            pub=pub,
        )
        outcome = disp.run_unit(_envelope(sha, value=21))

    assert outcome.kind == DispatchOutcomeKind.SUBMITTED
    # The REAL executor ran (doubled), not the synthetic echo.
    assert captured["body"]["payload"] == {"doubled": 42}
    assert "echo" not in captured["body"]["payload"]


def test_provisioned_unresolved_refuses_does_not_echo(tmp_path: Path, db: Database):
    privkey, pub = _make_key()
    captured: dict = {}
    with _client(captured, privkey, pub) as client:
        disp = _dispatcher(
            client,
            db,
            tmp_path / "runs",
            policy=ExecutePolicy.PROVISIONED,
            provisioning_dir=tmp_path / "empty",  # nothing staged
            privkey=privkey,
            pub=pub,
        )
        outcome = disp.run_unit(_envelope("a" * 64))

    assert outcome.kind == DispatchOutcomeKind.EXECUTOR_REFUSED
    assert "no provisioned executor" in outcome.reason
    assert "body" not in captured  # never submitted a result


def test_provisioned_executor_failure_refuses(tmp_path: Path, db: Database):
    privkey, pub = _make_key()
    prov = tmp_path / "tenants"
    sha = _provision(prov, EXECUTOR_FAIL)
    captured: dict = {}
    with _client(captured, privkey, pub) as client:
        disp = _dispatcher(
            client,
            db,
            tmp_path / "runs",
            policy=ExecutePolicy.PROVISIONED,
            provisioning_dir=prov,
            privkey=privkey,
            pub=pub,
        )
        outcome = disp.run_unit(_envelope(sha))

    # Executor exited non-zero -> runner failed -> daemon will refuse + re-offer.
    assert outcome.kind == DispatchOutcomeKind.RUNNER_CRASH
    assert "body" not in captured


def test_provisioned_resolves_model_from_store(tmp_path: Path, db: Database):
    privkey, pub = _make_key()
    prov = tmp_path / "tenants"
    store = tmp_path / "models"
    # The volunteer's BYOM store holds the model, keyed by id.
    (store / "test-model").mkdir(parents=True)
    (store / "test-model" / "weights.txt").write_text("W42", encoding="utf-8")
    sha = _provision(
        prov,
        EXECUTOR_USES_MODEL,
        models=[{"id": "test-model", "local_weights_required": True}],
    )
    captured: dict = {}
    with _client(captured, privkey, pub) as client:
        disp = _dispatcher(
            client,
            db,
            tmp_path / "runs",
            policy=ExecutePolicy.PROVISIONED,
            provisioning_dir=prov,
            privkey=privkey,
            pub=pub,
            model_store_dir=store,
        )
        outcome = disp.run_unit(_envelope(sha))

    assert outcome.kind == DispatchOutcomeKind.SUBMITTED
    # --models resolved to the store dir; the executor loaded the staged weight.
    assert captured["body"]["payload"]["loaded_weight"] == "W42"


def test_provisioned_refuses_when_required_model_absent(tmp_path: Path, db: Database):
    privkey, pub = _make_key()
    prov = tmp_path / "tenants"
    sha = _provision(
        prov,
        EXECUTOR_USES_MODEL,
        models=[{"id": "missing-model", "local_weights_required": True}],
    )
    captured: dict = {}
    with _client(captured, privkey, pub) as client:
        disp = _dispatcher(
            client,
            db,
            tmp_path / "runs",
            policy=ExecutePolicy.PROVISIONED,
            provisioning_dir=prov,
            privkey=privkey,
            pub=pub,
            model_store_dir=tmp_path / "empty-store",
        )
        outcome = disp.run_unit(_envelope(sha))

    assert outcome.kind == DispatchOutcomeKind.EXECUTOR_REFUSED
    assert "not in the worker model store" in outcome.reason
    assert "body" not in captured


def test_thermal_critical_refuses_before_running(tmp_path: Path, db: Database):
    from auspexai_worker.health import ThermalSnapshot, ThermalState

    class _HotMonitor:
        def state(self):
            return ThermalState.CRITICAL

        def snapshot(self):
            return ThermalSnapshot(ThermalState.CRITICAL, 88.0, 88.0, 1)

    privkey, pub = _make_key()
    prov = tmp_path / "tenants"
    sha = _provision(prov, EXECUTOR_DOUBLER)  # staged + would run if not hot
    captured: dict = {}
    with _client(captured, privkey, pub) as client:
        disp = _dispatcher(
            client,
            db,
            tmp_path / "runs",
            policy=ExecutePolicy.PROVISIONED,
            provisioning_dir=prov,
            privkey=privkey,
            pub=pub,
            thermal_monitor=_HotMonitor(),
        )
        outcome = disp.run_unit(_envelope(sha))

    assert outcome.kind == DispatchOutcomeKind.THERMAL_CRITICAL
    assert "thermal critical" in outcome.reason
    assert "body" not in captured  # never ran the executor / submitted


def test_synthetic_policy_echoes_not_real(tmp_path: Path, db: Database):
    privkey, pub = _make_key()
    prov = tmp_path / "tenants"
    sha = _provision(prov, EXECUTOR_DOUBLER)  # staged, but policy is synthetic
    captured: dict = {}
    with _client(captured, privkey, pub) as client:
        disp = _dispatcher(
            client,
            db,
            tmp_path / "runs",
            policy=ExecutePolicy.SYNTHETIC,
            provisioning_dir=prov,
            privkey=privkey,
            pub=pub,
        )
        outcome = disp.run_unit(_envelope(sha, value=21))

    assert outcome.kind == DispatchOutcomeKind.SUBMITTED
    # synthetic policy ignores the staged executor and echoes.
    assert captured["body"]["payload"]["echo"] == {"input": 21}
