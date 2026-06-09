"""Tenant-executor resolution + the volunteer's code-execution consent gate (§9 #37).

The worker daemon receives only `{unit_id, tenant_id, experiment_id,
manifest_sha256, payload}` for an assigned unit — never the executor command or
the executor code. Before it can run a *real* tenant executor it must (a) decide
whether the resource owner consents to running third-party code at all, and (b)
resolve the actual executor for this experiment.

**Consent — the §5.14 opt-in ladder, with the new code-execution axis.**
Running tenant-supplied code is a trust step-change from running the worker's own
built-in synthetic executor, so it is opt-in. The `execute_tenant_code` policy is
the resource owner's say:

  - ``synthetic``  — default. Only the built-in echo executor runs. **No
    third-party code.** Dev/test/CI; produces meaningless echo payloads, never a
    real tenant's results in practice.
  - ``provisioned`` — run a tenant executor **only** if it has been locally
    provisioned (operator-staged) and its staged manifest hash-matches the
    coordinator's `manifest_sha256`. Unresolved / denied → **refuse**, never echo.
  - ``off``         — refuse all work.

This composes with the ratified §5.14 tenant allow/deny lists. The two-way trust
matrix (network-trusts-worker T0-T3 by worker-trusts-code provenance) collapses
under local provisioning because the operator is the trust root on both sides.

**Resolution — local provisioning (Phase 1).** A provisioned tenant package lives
at ``<provisioning_dir>/<manifest_sha256>/`` and contains the as-submitted
``manifest.json`` (executor.command + models declaration), the executor files, and
an optional ``models/`` directory. The resolver re-derives the manifest hash the
**coordinator's** way (canonical JSON) and refuses to run anything whose staged
manifest doesn't match the pin in the assignment — the content-addressing §5.14
mandates. The `ExecutorResolver` protocol keeps a coordinator-fetch resolver a
drop-in for Phase 2.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol


class ExecutePolicy(StrEnum):
    """The resource owner's code-execution consent setting."""

    SYNTHETIC = "synthetic"
    PROVISIONED = "provisioned"
    OFF = "off"


class ProvisioningError(Exception):
    """Base for resolution failures."""


class ProvisioningIntegrityError(ProvisioningError):
    """A staged manifest does not hash to the coordinator's `manifest_sha256`.

    This is a hard failure, not a 'not provisioned' miss: a package IS staged for
    this hash but its `manifest.json` content doesn't match the pinned experiment.
    Refuse loudly rather than run mismatched code."""


def hash_manifest(manifest: dict[str, Any]) -> str:
    """Re-derive `manifest_sha256` exactly as the coordinator does
    (`db/repositories/manifests.py: hash_manifest`): SHA-256 over the canonical
    JSON serialization (sorted keys, compact separators). Must stay byte-for-byte
    identical to the coordinator or every verification fails."""
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Files never part of the executor package digest (mirror of the SDK's helper).
# manifest.json.sig: the SDK's `manifest sign` drops the signature file into
# the package dir by default, so it must not contribute to the digest — kept
# in lockstep with auspexai_tenant.manifest._PACKAGE_DIGEST_EXCLUDE.
_PACKAGE_DIGEST_EXCLUDE = ("manifest.json", "manifest.json.sig")


def compute_package_digest(package_dir: Path) -> str:
    """Digest over the executor *files* in `package_dir`, byte-for-byte identical
    to `auspexai_tenant.manifest.compute_package_digest` (the tenant computes it;
    we re-derive it to verify the staged code matches the signed manifest's
    `executor.package_sha256`). Standalone replica, not an import — the shared
    contract is the format, not shared code (worker AGPL, SDK Apache).

    Each regular file except `manifest.json` / `__pycache__` / `*.pyc` contributes
    ``<posix-relpath>\\x00<sha256-hex>``; lines sorted by relpath, joined by ``\\n``,
    SHA-256'd."""
    lines: list[str] = []
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(package_dir).as_posix()
        parts = rel.split("/")
        if rel in _PACKAGE_DIGEST_EXCLUDE or rel.endswith(".pyc") or "__pycache__" in parts:
            continue
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{rel}\x00{file_hash}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResolvedExecutor:
    """A tenant executor the worker is cleared to run for one unit. Models are
    resolved separately (from the worker-local BYOM store, not the package)."""

    manifest_sha256: str
    command: list[str]  # executor.command from the staged manifest
    package_dir: Path  # cwd for the executor; where the executor files live
    manifest: dict[str, Any]  # the full staged manifest (for §5.14 consent display)


class ExecutorResolver(Protocol):
    """Resolve a tenant executor by `manifest_sha256`, or return None if this
    resolver has no package for it. Raises `ProvisioningIntegrityError` on a
    hash mismatch. Phase 1 = `ProvisioningResolver`; a coordinator-fetch
    resolver implements the same shape in Phase 2."""

    def resolve(self, manifest_sha256: str) -> ResolvedExecutor | None: ...


class ProvisioningResolver:
    """Resolve from a local provisioning directory (operator pre-staged)."""

    def __init__(self, provisioning_dir: Path) -> None:
        self._dir = provisioning_dir

    def resolve(self, manifest_sha256: str) -> ResolvedExecutor | None:
        # The coordinator stores manifest hashes lowercased; normalize so the
        # staged-dir lookup is case-insensitive regardless of the caller.
        manifest_sha256 = manifest_sha256.lower()
        pkg = self._dir / manifest_sha256
        manifest_path = pkg / "manifest.json"
        if not manifest_path.is_file():
            return None  # not provisioned for this experiment
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProvisioningIntegrityError(
                f"staged manifest at {manifest_path} is unreadable/invalid: {exc}"
            ) from exc
        computed = hash_manifest(manifest)
        if computed != manifest_sha256.lower():
            raise ProvisioningIntegrityError(
                f"staged manifest hash {computed} != assigned manifest_sha256 "
                f"{manifest_sha256.lower()} (content-addressing violation; refusing)"
            )
        executor = manifest.get("executor") or {}
        command = executor.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(c, str) for c in command)
        ):
            raise ProvisioningIntegrityError(
                f"staged manifest {manifest_sha256} has no valid executor.command"
            )
        # Code content-addressing (§9 #37 hardening): when the signed manifest pins
        # the executor *files'* digest, verify the staged files match it — so we run
        # the code the tenant signed, not just whatever was staged. Absent pin =
        # Phase-1 'operator is the trust root' behavior (backward compatible).
        package_sha256 = executor.get("package_sha256")
        if package_sha256 is not None:
            computed_pkg = compute_package_digest(pkg)
            if computed_pkg != str(package_sha256).lower():
                raise ProvisioningIntegrityError(
                    f"staged executor package digest {computed_pkg} != manifest "
                    f"executor.package_sha256 {str(package_sha256).lower()} for "
                    f"{manifest_sha256} (code content-addressing violation; refusing)"
                )
        return ResolvedExecutor(
            manifest_sha256=manifest_sha256.lower(),
            command=list(command),
            package_dir=pkg,
            manifest=manifest,
        )


def resolve_model_dir(
    manifest: dict[str, Any], model_store_dir: Path
) -> tuple[Path | None, str | None]:
    """Resolve `--models` from the worker's local BYOM model store, keyed by the
    manifest's declared model id (`<store>/<model_id>/`). Returns
    `(models_dir, refusal_reason)`:

      - no models declared  -> (None, None): the runner uses an empty dir.
      - one model, present   -> (<store>/<id>, None).
      - one model, missing + local_weights_required -> (None, <reason>): refuse.
      - one model, missing + optional -> (None, None): empty dir; executor copes.
      - multiple models      -> (None, <reason>): not supported in the thin slice.

    The platform never distributes weights (§5.8); the volunteer fills the store
    (BYOM). This is the supply side a model-acquisition onramp will populate.
    """
    models = manifest.get("models") or []
    if not models:
        return None, None
    if len(models) > 1:
        return None, "worker model resolution does not yet support multi-model experiments"
    model = models[0]
    model_id = model.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None, "manifest model declaration is missing a valid `id`"
    model_dir = model_store_dir / model_id
    if model_dir.is_dir():
        return model_dir, None
    if model.get("local_weights_required"):
        return None, (
            f"locally-required model {model_id!r} is not in the worker model store "
            f"({model_store_dir}); install it (BYOM) before this experiment can run here"
        )
    return None, None


def model_acquisition_coords(manifest: dict[str, Any]) -> tuple[str, str, str] | None:
    """Return `(model_id, hf_repo, hf_filename)` for the single locally-required
    model IF the manifest carries M3 acquisition coords, else None (the model
    can't be auto-acquired — it must be staged out-of-band). Thin-slice: a single
    model only, matching `resolve_model_dir`."""
    models = manifest.get("models") or []
    if len(models) != 1:
        return None
    m = models[0]
    model_id = m.get("id")
    hf_repo = m.get("hf_repo")
    hf_filename = m.get("hf_filename")
    if (
        isinstance(model_id, str)
        and model_id
        and isinstance(hf_repo, str)
        and hf_repo
        and isinstance(hf_filename, str)
        and hf_filename
    ):
        return model_id, hf_repo, hf_filename
    return None


class ModelAcquirer(Protocol):
    """Pulls a missing model into the worker's store on demand (M3 lazy
    auto-acquire). The dispatch layer supplies a concrete one wrapping
    `models.fetch.pull_from_coords` + the store + a disk-headroom check; keeping
    it a protocol lets `decide_execution` stay free of the fetch dependency and
    be unit-tested with a fake."""

    def acquire(self, *, model_id: str, hf_repo: str, hf_filename: str) -> Path:
        """Pull the model into the store and return its dir. Raises on failure."""
        ...


class ExecutionMode(StrEnum):
    REAL = "real"  # run the resolved tenant executor
    SYNTHETIC = "synthetic"  # run the built-in echo executor
    REFUSE = "refuse"  # decline the unit (with a reason)


@dataclass(frozen=True)
class ExecutionDecision:
    """Outcome of the consent + resolution gate for one unit."""

    mode: ExecutionMode
    reason: str | None = None  # human-readable; surfaced on refuse + in logs
    executor: ResolvedExecutor | None = None  # set iff mode == REAL
    models_dir: Path | None = None  # resolved BYOM store dir; None => empty dir


def _tenant_allowed(
    tenant_id: str,
    *,
    allow_list: tuple[str, ...],
    deny_list: tuple[str, ...],
) -> str | None:
    """Return a refusal reason if the tenant is gated out, else None (§5.14)."""
    if tenant_id in deny_list:
        return f"tenant {tenant_id!r} is on the worker's tenant deny-list"
    if allow_list and tenant_id not in allow_list:
        return f"tenant {tenant_id!r} is not on the worker's tenant allow-list"
    return None


def decide_execution(
    *,
    policy: ExecutePolicy,
    tenant_id: str,
    manifest_sha256: str,
    resolver: ExecutorResolver | None,
    model_store_dir: Path | None = None,
    allow_list: tuple[str, ...] = (),
    deny_list: tuple[str, ...] = (),
    auto_acquire: bool = False,
    acquirer: ModelAcquirer | None = None,
) -> ExecutionDecision:
    """The consent + resolution gate. Composes the code-execution policy with the
    §5.14 tenant allow/deny lists. Refuse-don't-echo: a `provisioned` worker that
    can't resolve a unit refuses it rather than submitting a synthetic echo under
    a real tenant's experiment.

    M3 lazy auto-acquire: when `auto_acquire` is set and an `acquirer` is supplied,
    a missing locally-required model is *pulled* (from the manifest's
    hf_repo/hf_filename) and the unit then runs, instead of refusing. A model
    with no acquisition coords, or a pull that fails, still refuses (the worker
    won't run a real tenant's experiment without the pinned weights)."""
    if policy is ExecutePolicy.OFF:
        return ExecutionDecision(ExecutionMode.REFUSE, "worker policy execute_tenant_code=off")

    gate = _tenant_allowed(tenant_id, allow_list=allow_list, deny_list=deny_list)
    if gate is not None:
        return ExecutionDecision(ExecutionMode.REFUSE, gate)

    if policy is ExecutePolicy.SYNTHETIC:
        return ExecutionDecision(ExecutionMode.SYNTHETIC)

    # policy is PROVISIONED: only run hash-verified, operator-staged executors.
    if resolver is None:
        return ExecutionDecision(
            ExecutionMode.REFUSE,
            "execute_tenant_code=provisioned but no executor resolver is configured",
        )
    try:
        resolved = resolver.resolve(manifest_sha256)
    except ProvisioningIntegrityError as exc:
        return ExecutionDecision(ExecutionMode.REFUSE, str(exc))
    if resolved is None:
        return ExecutionDecision(
            ExecutionMode.REFUSE,
            f"no provisioned executor for manifest {manifest_sha256} "
            "(execute_tenant_code=provisioned; refusing rather than echoing)",
        )

    # Resolve --models from the worker-local BYOM store (§5.8). A missing
    # locally-required model is a refuse, not an echo.
    store_dir = model_store_dir if model_store_dir is not None else Path()
    models_dir, model_reason = resolve_model_dir(resolved.manifest, store_dir)
    if model_reason is not None:
        # M3 lazy auto-acquire: try to pull the missing model, then re-resolve.
        if auto_acquire and acquirer is not None:
            coords = model_acquisition_coords(resolved.manifest)
            if coords is None:
                return ExecutionDecision(
                    ExecutionMode.REFUSE,
                    f"{model_reason}; auto_acquire is on but the manifest carries no "
                    "hf_repo/hf_filename to pull from (model_not_acquirable)",
                )
            model_id, hf_repo, hf_filename = coords
            try:
                acquirer.acquire(model_id=model_id, hf_repo=hf_repo, hf_filename=hf_filename)
            except Exception as exc:
                return ExecutionDecision(
                    ExecutionMode.REFUSE,
                    f"auto-acquire of model {model_id!r} from {hf_repo}/{hf_filename} "
                    f"failed: {exc} (model_pull_failed)",
                )
            models_dir, model_reason = resolve_model_dir(resolved.manifest, store_dir)
            if model_reason is not None:  # pragma: no cover — pull succeeded but still unresolved
                return ExecutionDecision(ExecutionMode.REFUSE, model_reason)
        else:
            return ExecutionDecision(ExecutionMode.REFUSE, model_reason)
    return ExecutionDecision(ExecutionMode.REAL, executor=resolved, models_dir=models_dir)
