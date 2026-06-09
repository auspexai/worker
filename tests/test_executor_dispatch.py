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

# Enforces the SDK `auspexai_tenant.workunits.WorkUnit` contract (extra=forbid +
# required manifest_sha256/created_at) WITHOUT importing the SDK, so the
# runner->harness seam is actually exercised. The other fixtures read
# unit["payload"] directly and would MASK a dropped required field — the exact
# regression this guards: the worker once materialized --input without
# manifest_sha256/created_at, which the real ExecutorHarness rejects -> every
# unit refused.
EXECUTOR_VALIDATES_CONTRACT = """
import argparse, json, re, sys
from datetime import datetime
p = argparse.ArgumentParser()
for f in ("input", "output", "models"):
    p.add_argument(f"--{f}", required=True)
p.add_argument("--timeout", type=int, default=600)
a = p.parse_args()
unit = json.loads(open(a.input).read())
required = {"schema_version", "unit_id", "tenant_id", "experiment_id", "manifest_sha256", "created_at", "payload"}
missing = required - set(unit)
extra = set(unit) - required
if missing or extra:
    sys.stderr.write("workunit contract violation: missing=%s extra=%s\\n" % (sorted(missing), sorted(extra)))
    sys.exit(1)
if not re.fullmatch(r"[a-f0-9]{64}", unit["manifest_sha256"] or ""):
    sys.stderr.write("manifest_sha256 not 64-hex\\n")
    sys.exit(1)
datetime.fromisoformat(unit["created_at"])  # raises if missing/unparseable
out = {
    "schema_version": "0.1",
    "unit_id": unit["unit_id"],
    "completed_at": datetime.now().isoformat(),
    "exit_code": 0,
    "payload": {"contract": "ok"},
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
    live_executor=None,
    open_inference_session=None,
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
        live_executor=live_executor,
        open_inference_session=open_inference_session,
    )


def test_live_executor_hot_reload_overrides_static_policy(tmp_path: Path, db: Database):
    """Hot-reload: a live_executor is re-read PER UNIT and overrides the daemon-start
    static policy — so an owner's policy change applies without a restart. Static
    SYNTHETIC + a live OFF → the unit is refused (not echoed)."""
    privkey, pub = _make_key()
    prov = tmp_path / "tenants"
    sha = _provision(prov, EXECUTOR_DOUBLER)
    captured: dict = {}
    with _client(captured, privkey, pub) as client:
        disp = _dispatcher(
            client,
            db,
            tmp_path / "runs",
            policy=ExecutePolicy.SYNTHETIC,  # daemon-start snapshot
            provisioning_dir=prov,
            privkey=privkey,
            pub=pub,
            live_executor=lambda: (ExecutePolicy.OFF, False),  # live owner consent
        )
        outcome = disp.run_unit(_envelope(sha, value=21))
    assert outcome.kind == DispatchOutcomeKind.EXECUTOR_REFUSED


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


def test_real_executor_input_satisfies_sdk_workunit_contract(tmp_path: Path, db: Database):
    """Regression guard for the worker->SDK ExecutorHarness seam: the
    runner-materialized --input MUST satisfy the SDK WorkUnit contract
    (extra=forbid + required manifest_sha256/created_at). A fixture executor that
    enforces that exact contract (mirroring tenant-sdk) submits only if the input
    is valid; a dropped field would make it exit non-zero -> refuse. The other
    fixtures read payload directly and can't catch a missing required field."""
    privkey, pub = _make_key()
    prov = tmp_path / "tenants"
    sha = _provision(prov, EXECUTOR_VALIDATES_CONTRACT)
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

    assert outcome.kind == DispatchOutcomeKind.SUBMITTED, getattr(outcome, "reason", outcome)
    assert captured["body"]["payload"] == {"contract": "ok"}


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


# ---- W-S (§9 #43): per-unit inference broker through dispatch ---------------

# An executor that talks to the worker's inference broker over the per-unit
# unix socket using ONLY the stdlib — this doubles as the prototype of the
# vendorable stdlib InferenceClient (W-S build step 4).
EXECUTOR_USES_INFERENCE = """
import argparse, json, os, socket
from datetime import datetime, timezone
p = argparse.ArgumentParser()
for f in ("input", "output", "models"):
    p.add_argument(f"--{f}", required=True)
p.add_argument("--timeout", type=int, default=600)
a = p.parse_args()
unit = json.loads(open(a.input).read())

def ask(body):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(10.0)
    s.connect(os.environ["AUSPEXAI_INFERENCE_SOCKET"])
    s.sendall(json.dumps(body).encode() + b"\\n")
    buf = b""
    while b"\\n" not in buf:
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    s.close()
    return json.loads(buf.split(b"\\n", 1)[0])

model = os.environ["AUSPEXAI_INFERENCE_MODEL"]
gen = ask({"op": "generate", "model": model,
           "messages": [{"role": "user", "content": "hello"}],
           "options": {"seed": 0}})
info = ask({"op": "info"})
out = {
    "schema_version": "0.1",
    "unit_id": unit["unit_id"],
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "exit_code": 0,
    "payload": {
        "generation": gen["message"]["content"],
        "gen_ok": gen["ok"],
        "model_digest": info["gguf_sha256"],
    },
}
open(a.output, "w").write(json.dumps(out))
"""


class _FakeChatBackend:
    """Inference backend double for dispatch-level tests."""

    def is_healthy(self):
        return True

    def has_model(self, handle):
        return True

    def create_model(self, handle, modelfile):
        pass

    def chat(self, handle, messages, options):
        return {
            "message": {"role": "assistant", "content": f"reply-from-{handle}"},
            "eval_count": 2,
        }


def test_inference_broker_e2e_through_dispatch(tmp_path: Path, db: Database):
    """Full W-S slice in passthrough mode: dispatch serves the unit's model,
    opens the broker socket in the workspace, the (subprocess) executor reaches
    it via env + a stdlib socket client, the generation lands in the submitted
    payload, and the session is closed (socket gone) when the unit ends."""
    from auspexai_worker.inference import ServedModel, open_unit_session

    privkey, pub = _make_key()
    prov = tmp_path / "tenants"
    store = tmp_path / "models"
    (store / "tiny-q4").mkdir(parents=True)
    (store / "tiny-q4" / "weights.gguf").write_bytes(b"fake gguf")
    sha = _provision(
        prov,
        EXECUTOR_USES_INFERENCE,
        models=[{"id": "tiny-q4", "local_weights_required": True}],
    )

    opened: list = []

    def provider(model_id: str, socket_dir):
        served = ServedModel(
            model_id=model_id,
            handle=f"auspex-{model_id}",
            gguf_sha256="cd" * 32,
            gguf_path=store / model_id / "weights.gguf",
        )
        session = open_unit_session(
            served=served, backend=_FakeChatBackend(), socket_dir=socket_dir
        )
        opened.append(session)
        return session

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
            open_inference_session=provider,
        )
        outcome = disp.run_unit(_envelope(sha))

    assert outcome.kind == DispatchOutcomeKind.SUBMITTED, outcome.reason
    payload = captured["body"]["payload"]
    assert payload["gen_ok"] is True
    assert payload["generation"] == "reply-from-auspex-tiny-q4"
    # op:"info" provenance: the served GGUF digest reached the result payload.
    assert payload["model_digest"] == "cd" * 32
    # The session was opened in the unit workspace and closed after the unit.
    assert len(opened) == 1
    assert not opened[0].socket_path.exists()


def test_inference_serving_failure_refuses_not_echoes(tmp_path: Path, db: Database):
    privkey, pub = _make_key()
    prov = tmp_path / "tenants"
    store = tmp_path / "models"
    (store / "tiny-q4").mkdir(parents=True)
    (store / "tiny-q4" / "weights.gguf").write_bytes(b"fake gguf")
    sha = _provision(
        prov,
        EXECUTOR_USES_INFERENCE,
        models=[{"id": "tiny-q4", "local_weights_required": True}],
    )

    def provider(model_id: str, socket_dir):
        raise RuntimeError("ollama is down")

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
            open_inference_session=provider,
        )
        outcome = disp.run_unit(_envelope(sha))

    assert outcome.kind == DispatchOutcomeKind.EXECUTOR_REFUSED
    assert "inference serving unavailable" in outcome.reason
    assert "body" not in captured  # nothing submitted, nothing echoed
