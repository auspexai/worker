"""Executor resolution + the §9 #37 code-execution consent gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from auspexai_worker.provisioning import (
    ExecutePolicy,
    ExecutionMode,
    ProvisioningIntegrityError,
    ProvisioningResolver,
    ResolvedExecutor,
    compute_package_digest,
    decide_execution,
    hash_manifest,
    resolve_model_dir,
)

MANIFEST = {
    "tenant_id": "synth-geometry",
    "experiment_id": "synth-geometry-v1",
    "executor": {"command": ["python", "executor.py"]},
    "models": [{"id": "m", "version": "1.0", "local_weights_required": True}],
}


def _stage(root: Path, manifest: dict, *, sha: str | None = None) -> str:
    """Stage a tenant package under root/<sha>/ and return the sha used."""
    sha = sha or hash_manifest(manifest)
    pkg = root / sha
    pkg.mkdir(parents=True)
    (pkg / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (pkg / "executor.py").write_text("# executor", encoding="utf-8")
    return sha


# ---- manifest hashing must match the coordinator exactly -------------------


def test_hash_manifest_matches_coordinator_canonicalization():
    # Coordinator: sha256(json.dumps(m, sort_keys=True, separators=(",",":")))
    expected = hashlib.sha256(
        json.dumps(MANIFEST, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert hash_manifest(MANIFEST) == expected


def test_hash_manifest_is_key_order_independent():
    reordered = dict(reversed(list(MANIFEST.items())))
    assert hash_manifest(reordered) == hash_manifest(MANIFEST)


# ---- ProvisioningResolver --------------------------------------------------


def test_resolve_returns_none_when_not_provisioned(tmp_path: Path):
    resolver = ProvisioningResolver(tmp_path)
    assert resolver.resolve("a" * 64) is None


def test_resolve_returns_executor_when_hash_matches(tmp_path: Path):
    sha = _stage(tmp_path, MANIFEST)
    resolved = ProvisioningResolver(tmp_path).resolve(sha)
    assert isinstance(resolved, ResolvedExecutor)
    assert resolved.command == ["python", "executor.py"]
    assert resolved.package_dir == tmp_path / sha
    assert resolved.manifest["tenant_id"] == "synth-geometry"


def test_resolve_rejects_hash_mismatch(tmp_path: Path):
    # Stage the manifest under a sha that doesn't match its content.
    wrong_sha = "b" * 64
    _stage(tmp_path, MANIFEST, sha=wrong_sha)
    with pytest.raises(ProvisioningIntegrityError, match="content-addressing"):
        ProvisioningResolver(tmp_path).resolve(wrong_sha)


def test_resolve_rejects_missing_executor_command(tmp_path: Path):
    bad = {k: v for k, v in MANIFEST.items() if k != "executor"}
    sha = _stage(tmp_path, bad)
    with pytest.raises(ProvisioningIntegrityError, match=r"executor\.command"):
        ProvisioningResolver(tmp_path).resolve(sha)


def test_resolve_is_case_insensitive_on_sha(tmp_path: Path):
    sha = _stage(tmp_path, MANIFEST)
    # An uppercase assignment hash must still resolve + verify.
    resolved = ProvisioningResolver(tmp_path).resolve(sha.upper())
    assert resolved is not None
    assert resolved.manifest_sha256 == sha  # normalized to lowercase


# ---- executor package digest (§9 #37 code content-addressing) --------------


def _package_digest(*files: tuple[str, bytes]) -> str:
    """The shared contract (mirror of auspexai_tenant.manifest.compute_package_digest)."""
    lines = [f"{rel}\x00{hashlib.sha256(c).hexdigest()}" for rel, c in sorted(files)]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def test_compute_package_digest_matches_shared_contract(tmp_path: Path):
    (tmp_path / "executor.py").write_bytes(b"# executor")
    (tmp_path / "manifest.json").write_bytes(b'{"x":1}')  # excluded
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "x.pyc").write_bytes(b"\x00")  # excluded
    assert compute_package_digest(tmp_path) == _package_digest(("executor.py", b"# executor"))


def test_resolve_accepts_matching_package_digest(tmp_path: Path):
    # _stage writes executor.py = b"# executor"; pin its digest in the manifest.
    manifest = {
        **MANIFEST,
        "executor": {
            "command": ["python", "executor.py"],
            "package_sha256": _package_digest(("executor.py", b"# executor")),
        },
    }
    sha = _stage(tmp_path, manifest)
    resolved = ProvisioningResolver(tmp_path).resolve(sha)
    assert resolved is not None


def test_resolve_refuses_package_digest_mismatch(tmp_path: Path):
    manifest = {
        **MANIFEST,
        "executor": {"command": ["python", "executor.py"], "package_sha256": "ab" * 32},
    }
    sha = _stage(tmp_path, manifest)
    with pytest.raises(ProvisioningIntegrityError, match="content-addressing"):
        ProvisioningResolver(tmp_path).resolve(sha)


def test_resolve_ignores_absent_package_digest(tmp_path: Path):
    # No package_sha256 → Phase-1 backward-compatible (operator trust), still resolves.
    sha = _stage(tmp_path, MANIFEST)
    assert ProvisioningResolver(tmp_path).resolve(sha) is not None


# ---- decide_execution: the consent + resolution gate -----------------------


class _StubResolver:
    def __init__(self, result=None, raises: Exception | None = None):
        self._result = result
        self._raises = raises

    def resolve(self, manifest_sha256):
        if self._raises is not None:
            raise self._raises
        return self._result


def test_policy_off_refuses():
    d = decide_execution(
        policy=ExecutePolicy.OFF,
        tenant_id="t",
        manifest_sha256="a" * 64,
        resolver=_StubResolver(),
    )
    assert d.mode is ExecutionMode.REFUSE
    assert "off" in d.reason


def test_policy_synthetic_runs_synthetic():
    d = decide_execution(
        policy=ExecutePolicy.SYNTHETIC,
        tenant_id="t",
        manifest_sha256="a" * 64,
        resolver=_StubResolver(),
    )
    assert d.mode is ExecutionMode.SYNTHETIC


def test_deny_list_refuses_even_when_provisioned():
    resolved = ResolvedExecutor("a" * 64, ["python", "x.py"], Path("/p"), {})
    d = decide_execution(
        policy=ExecutePolicy.PROVISIONED,
        tenant_id="bad-tenant",
        manifest_sha256="a" * 64,
        resolver=_StubResolver(resolved),
        deny_list=("bad-tenant",),
    )
    assert d.mode is ExecutionMode.REFUSE
    assert "deny-list" in d.reason


def test_allow_list_refuses_tenant_not_listed():
    d = decide_execution(
        policy=ExecutePolicy.PROVISIONED,
        tenant_id="stranger",
        manifest_sha256="a" * 64,
        resolver=_StubResolver(),
        allow_list=("known",),
    )
    assert d.mode is ExecutionMode.REFUSE
    assert "allow-list" in d.reason


def test_provisioned_runs_real_when_resolved():
    resolved = ResolvedExecutor("a" * 64, ["python", "x.py"], Path("/p"), {})
    d = decide_execution(
        policy=ExecutePolicy.PROVISIONED,
        tenant_id="known",
        manifest_sha256="a" * 64,
        resolver=_StubResolver(resolved),
        allow_list=("known",),
    )
    assert d.mode is ExecutionMode.REAL
    assert d.executor is resolved


def test_provisioned_refuses_when_unresolved_not_echo():
    d = decide_execution(
        policy=ExecutePolicy.PROVISIONED,
        tenant_id="known",
        manifest_sha256="a" * 64,
        resolver=_StubResolver(None),
    )
    assert d.mode is ExecutionMode.REFUSE  # refuse-don't-echo
    assert "no provisioned executor" in d.reason


def test_provisioned_refuses_on_integrity_error():
    d = decide_execution(
        policy=ExecutePolicy.PROVISIONED,
        tenant_id="known",
        manifest_sha256="a" * 64,
        resolver=_StubResolver(raises=ProvisioningIntegrityError("tampered")),
    )
    assert d.mode is ExecutionMode.REFUSE
    assert "tampered" in d.reason


# ---- model store resolution (BYOM, §5.8) -----------------------------------


def test_resolve_model_dir_no_models(tmp_path: Path):
    assert resolve_model_dir({"models": []}, tmp_path) == (None, None)
    assert resolve_model_dir({}, tmp_path) == (None, None)


def test_resolve_model_dir_present_in_store(tmp_path: Path):
    (tmp_path / "llama-3-70b").mkdir()
    manifest = {"models": [{"id": "llama-3-70b", "version": "1.0"}]}
    models_dir, reason = resolve_model_dir(manifest, tmp_path)
    assert models_dir == tmp_path / "llama-3-70b"
    assert reason is None


def test_resolve_model_dir_missing_required_refuses(tmp_path: Path):
    manifest = {"models": [{"id": "llama-3-70b", "local_weights_required": True}]}
    models_dir, reason = resolve_model_dir(manifest, tmp_path)
    assert models_dir is None
    assert "not in the worker model store" in reason


def test_resolve_model_dir_missing_optional_is_empty(tmp_path: Path):
    manifest = {"models": [{"id": "opt-model", "local_weights_required": False}]}
    assert resolve_model_dir(manifest, tmp_path) == (None, None)


def test_resolve_model_dir_multi_model_unsupported(tmp_path: Path):
    manifest = {"models": [{"id": "a"}, {"id": "b"}]}
    models_dir, reason = resolve_model_dir(manifest, tmp_path)
    assert models_dir is None
    assert "multi-model" in reason


def test_decide_execution_refuses_missing_required_model(tmp_path: Path):
    manifest = {
        "tenant_id": "t",
        "executor": {"command": ["python", "x.py"]},
        "models": [{"id": "big-llm", "local_weights_required": True}],
    }
    resolved = ResolvedExecutor("a" * 64, ["python", "x.py"], Path("/p"), manifest)
    d = decide_execution(
        policy=ExecutePolicy.PROVISIONED,
        tenant_id="t",
        manifest_sha256="a" * 64,
        resolver=_StubResolver(resolved),
        model_store_dir=tmp_path,  # empty store
    )
    assert d.mode is ExecutionMode.REFUSE
    assert "big-llm" in d.reason


# ---- M3 lazy auto-acquire --------------------------------------------------


def test_model_acquisition_coords_present_and_absent():
    from auspexai_worker.provisioning import model_acquisition_coords

    have = {
        "models": [
            {
                "id": "m-x",
                "local_weights_required": True,
                "hf_repo": "Org/M-GGUF",
                "hf_filename": "M-Q4.gguf",
            }
        ]
    }
    assert model_acquisition_coords(have) == ("m-x", "Org/M-GGUF", "M-Q4.gguf")
    # no coords -> not acquirable
    assert model_acquisition_coords({"models": [{"id": "m-x"}]}) is None
    # multi-model thin-slice unsupported
    assert model_acquisition_coords({"models": [{"id": "a"}, {"id": "b"}]}) is None


class _FakeAcquirer:
    """Records the requested pull and lays down a non-empty model dir so the
    re-resolve sees the model as present (or raises to simulate a failed pull)."""

    def __init__(self, store_dir: Path, *, raises: Exception | None = None):
        self._store_dir = store_dir
        self._raises = raises
        self.calls: list[tuple[str, str, str]] = []

    def acquire(self, *, model_id: str, hf_repo: str, hf_filename: str) -> Path:
        self.calls.append((model_id, hf_repo, hf_filename))
        if self._raises is not None:
            raise self._raises
        dest = self._store_dir / model_id
        dest.mkdir(parents=True, exist_ok=True)
        (dest / hf_filename).write_text("weights")
        return dest


def _gated_manifest() -> dict:
    return {
        "tenant_id": "t",
        "executor": {"command": ["python", "x.py"]},
        "models": [
            {
                "id": "m-x",
                "local_weights_required": True,
                "hf_repo": "Org/M-GGUF",
                "hf_filename": "M-Q4.gguf",
            }
        ],
    }


def test_auto_acquire_pulls_then_runs(tmp_path: Path):
    resolved = ResolvedExecutor("a" * 64, ["python", "x.py"], Path("/p"), _gated_manifest())
    acquirer = _FakeAcquirer(tmp_path)
    d = decide_execution(
        policy=ExecutePolicy.PROVISIONED,
        tenant_id="t",
        manifest_sha256="a" * 64,
        resolver=_StubResolver(resolved),
        model_store_dir=tmp_path,  # empty store -> would refuse without auto-acquire
        auto_acquire=True,
        acquirer=acquirer,
    )
    assert d.mode is ExecutionMode.REAL
    assert acquirer.calls == [("m-x", "Org/M-GGUF", "M-Q4.gguf")]
    assert d.models_dir == tmp_path / "m-x"


def test_auto_acquire_off_still_refuses(tmp_path: Path):
    resolved = ResolvedExecutor("a" * 64, ["python", "x.py"], Path("/p"), _gated_manifest())
    acquirer = _FakeAcquirer(tmp_path)
    d = decide_execution(
        policy=ExecutePolicy.PROVISIONED,
        tenant_id="t",
        manifest_sha256="a" * 64,
        resolver=_StubResolver(resolved),
        model_store_dir=tmp_path,
        auto_acquire=False,  # opt-in off
        acquirer=acquirer,
    )
    assert d.mode is ExecutionMode.REFUSE
    assert acquirer.calls == []


def test_auto_acquire_no_coords_refuses_not_acquirable(tmp_path: Path):
    manifest = {
        "tenant_id": "t",
        "executor": {"command": ["python", "x.py"]},
        "models": [{"id": "m-x", "local_weights_required": True}],  # no hf_repo/filename
    }
    resolved = ResolvedExecutor("a" * 64, ["python", "x.py"], Path("/p"), manifest)
    acquirer = _FakeAcquirer(tmp_path)
    d = decide_execution(
        policy=ExecutePolicy.PROVISIONED,
        tenant_id="t",
        manifest_sha256="a" * 64,
        resolver=_StubResolver(resolved),
        model_store_dir=tmp_path,
        auto_acquire=True,
        acquirer=acquirer,
    )
    assert d.mode is ExecutionMode.REFUSE
    assert "model_not_acquirable" in d.reason
    assert acquirer.calls == []


def test_auto_acquire_pull_failure_refuses(tmp_path: Path):
    resolved = ResolvedExecutor("a" * 64, ["python", "x.py"], Path("/p"), _gated_manifest())
    acquirer = _FakeAcquirer(tmp_path, raises=RuntimeError("connection died"))
    d = decide_execution(
        policy=ExecutePolicy.PROVISIONED,
        tenant_id="t",
        manifest_sha256="a" * 64,
        resolver=_StubResolver(resolved),
        model_store_dir=tmp_path,
        auto_acquire=True,
        acquirer=acquirer,
    )
    assert d.mode is ExecutionMode.REFUSE
    assert "model_pull_failed" in d.reason
    assert acquirer.calls == [("m-x", "Org/M-GGUF", "M-Q4.gguf")]


# ---- config wiring ---------------------------------------------------------


def test_config_defaults_to_synthetic_policy(tmp_path: Path):
    from auspexai_worker.config import WorkerConfig

    cfg = WorkerConfig.load(config_path=tmp_path / "missing.toml", env={})
    assert cfg.execute_tenant_code == "synthetic"
    # provisioning_path defaults under data_dir
    assert cfg.provisioning_path == cfg.data_dir / "tenants"


def test_config_parses_executor_block(tmp_path: Path):
    from auspexai_worker.config import WorkerConfig

    cfg_file = tmp_path / "worker.toml"
    cfg_file.write_text(
        "\n".join(
            [
                "[executor]",
                'execute_tenant_code = "provisioned"',
                f'provisioning_dir = "{tmp_path / "pkgs"}"',
            ]
        ),
        encoding="utf-8",
    )
    cfg = WorkerConfig.load(config_path=cfg_file, env={})
    assert cfg.execute_tenant_code == "provisioned"
    assert cfg.provisioning_path == tmp_path / "pkgs"


def test_config_env_overrides_policy(tmp_path: Path):
    from auspexai_worker.config import WorkerConfig

    cfg = WorkerConfig.load(
        config_path=tmp_path / "missing.toml",
        env={"AUSPEXAI_WORKER_EXECUTE_TENANT_CODE": "off"},
    )
    assert cfg.execute_tenant_code == "off"


def test_config_rejects_unknown_policy(tmp_path: Path):
    from auspexai_worker.config import WorkerConfig

    cfg_file = tmp_path / "worker.toml"
    cfg_file.write_text('[executor]\nexecute_tenant_code = "yolo"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="execute_tenant_code"):
        WorkerConfig.load(config_path=cfg_file, env={})


def test_package_digest_excludes_manifest_sig(tmp_path):
    """Lockstep with the SDK: a manifest.json.sig staged alongside the package
    (the SDK `manifest sign` default drops it into the package dir) must not
    change the digest the worker re-derives against the signed manifest."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "executor.py").write_text("print('hi')")
    before = compute_package_digest(pkg)
    (pkg / "manifest.json").write_text("{}")
    (pkg / "manifest.json.sig").write_text('{"sig": "..."}')
    assert compute_package_digest(pkg) == before
