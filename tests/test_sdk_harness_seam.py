"""Worker ↔ SDK ExecutorHarness seam — the REAL harness through real dispatch.

§9 #47 follow-up (external-review recommendation). The 2026-06-08 audit's
critical finding lived exactly here: worker tests exercised dispatch with
hand-rolled fixture executors that encode our BELIEFS about the SDK contract
(`EXECUTOR_VALIDATES_CONTRACT` mirrors it field-by-field), while the SDK's own
tests build inputs that already satisfy it — the seam between the two was
never exercised, so the worker shipped `--input` WorkUnits missing
`manifest_sha256`/`created_at` and the official harness refused every unit.

Here the dispatched executor imports the real `auspexai_tenant.executor
.ExecutorHarness` (dev-dep pinned to the released SDK tag), so any drift
between the worker's materialized input and the SDK's actual schema fails THIS
suite instead of a tenant's first live run. A second test pins the harness's
strictness in-process (required fields + extra=forbid), so a future SDK
relaxation also surfaces as a deliberate diff.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from auspexai_worker.daemon.dispatch import DispatchOutcomeKind
from auspexai_worker.provisioning import ExecutePolicy
from auspexai_worker.state import Database, MigrationRunner
from tests.test_executor_dispatch import (
    _client,
    _dispatcher,
    _envelope,
    _make_key,
    _provision,
)

pytest.importorskip("auspexai_tenant", reason="seam test needs the real tenant-sdk dev dep")


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "worker.db")
    MigrationRunner(d).apply_all()
    return d


# A REAL tenant executor: imports the official SDK harness, no contract
# mirroring — whatever the harness requires, the worker's input must provide.
EXECUTOR_REAL_SDK_HARNESS = """
import sys
from auspexai_tenant.executor import ExecutorHarness

def run_one(unit, models_dir):
    return {"sdk_doubled": unit.payload["input"] * 2}

if __name__ == "__main__":
    sys.exit(ExecutorHarness(run_one).main())
"""


def test_dispatch_through_real_sdk_harness_submits(tmp_path: Path, db: Database):
    """The runner-materialized --input must satisfy the REAL ExecutorHarness —
    accepted unit, real output submitted (not an echo, not a refusal)."""
    privkey, pub = _make_key()
    prov = tmp_path / "tenants"
    sha = _provision(prov, EXECUTOR_REAL_SDK_HARNESS)
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

    assert outcome.kind == DispatchOutcomeKind.SUBMITTED, getattr(outcome, "detail", outcome)
    assert captured["body"]["payload"] == {"sdk_doubled": 42}


def test_real_harness_strictness_matches_worker_materialization(tmp_path: Path):
    """Pin the REAL harness's input contract in-process: the exact field set
    the worker's runner materializes is accepted; dropping a required field or
    adding an unknown one is rejected (exit 2). If the SDK ever relaxes or
    widens the contract, this diff surfaces deliberately."""
    from auspexai_tenant.executor import ExecutorHarness

    def run_one(unit, models_dir):
        return {"ok": True}

    harness = ExecutorHarness(run_one)
    models = tmp_path / "models"
    models.mkdir()

    # The runner's materialization (runner/main.py): exactly these fields.
    worker_shape = {
        "schema_version": "0.1",
        "unit_id": "u-1",
        "tenant_id": "synth-doubler",
        "experiment_id": "exp-label",
        "manifest_sha256": "ab" * 32,
        "created_at": "2026-06-11T00:00:00+00:00",
        "payload": {"input": 1},
    }

    def run(unit_dict: dict) -> int:
        inp = tmp_path / "in.json"
        out = tmp_path / "out.json"
        inp.write_text(json.dumps(unit_dict))
        return harness.main(
            ["--input", str(inp), "--output", str(out), "--models", str(models), "--timeout", "60"]
        )

    assert run(worker_shape) == 0

    missing = dict(worker_shape)
    del missing["manifest_sha256"]  # the audit's exact dropped field
    assert run(missing) == 2

    extra = dict(worker_shape, surprise="field")  # extra=forbid
    assert run(extra) == 2
