"""Runner dispatch — the orchestration that turns an accepted assignment
into an executed work unit and a submitted result.

Lives between the AssignmentPoller (which makes the accept/refuse
decision) and the runner subprocess (which actually executes the
payload). One dispatch:

  1. Create the per-unit workspace
  2. Build the sandbox argv (bubblewrap or passthrough per config)
  3. Spawn the subprocess, write the envelope to stdin, close
  4. Wait for exit
  5. Read output.json
  6. Sign the Result body with the worker key
  7. POST it to the coordinator
  8. Record locally in submitted_results
  9. Clean up the workspace

Any of those steps failing turns the dispatch into a refusal (the
coordinator gets a runner_failed refuse) rather than letting the
assignment dangle. Per Q-W4, refusals are explicit.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from auspexai_worker.coordinator import (
    AssignmentResponse,
    CoordinatorClient,
    CoordinatorError,
    ResultSubmissionResponse,
)
from auspexai_worker.sandbox import SandboxConfig, SandboxNotAvailableError, build_argv
from auspexai_worker.signing import sign_result
from auspexai_worker.state import SubmittedResultRepository
from auspexai_worker.workspace import RunnerWorkspace, WorkspaceManager

logger = logging.getLogger(__name__)


class DispatchOutcomeKind:
    SUBMITTED = "submitted"
    RUNNER_CRASH = "runner_failed"
    SUBMIT_FAILED = "submit_failed"
    SANDBOX_UNAVAILABLE = "sandbox_unavailable"


@dataclass(frozen=True)
class DispatchOutcome:
    """Result of a single dispatch."""

    kind: str  # one of DispatchOutcomeKind constants
    reason: str | None
    result_response: ResultSubmissionResponse | None = None


class RunnerDispatcher:
    """Owns the full per-unit execute + submit flow."""

    def __init__(
        self,
        *,
        coordinator: CoordinatorClient,
        worker_id: str,
        worker_pubkey: str,
        privkey: Ed25519PrivateKey,
        workspace_manager: WorkspaceManager,
        submitted_repo: SubmittedResultRepository,
        use_bubblewrap: bool,
        runner_bin: str = "auspexai-worker-runner",
        runner_timeout_seconds: float | None = None,
        on_runner_spawned=None,  # Callable[[int], None] — receives PID
    ) -> None:
        self._coordinator = coordinator
        self._worker_id = worker_id
        self._worker_pubkey = worker_pubkey
        self._privkey = privkey
        self._workspaces = workspace_manager
        self._submitted = submitted_repo
        self._use_bubblewrap = use_bubblewrap
        self._runner_bin = runner_bin
        self._runner_timeout_seconds = runner_timeout_seconds
        self._on_runner_spawned = on_runner_spawned

    def run_unit(self, response: AssignmentResponse) -> DispatchOutcome:
        """Execute the assigned unit and submit the result. Always cleans
        up the workspace before returning, even on error paths."""
        assert response.work_unit is not None
        unit = response.work_unit
        workspace = self._workspaces.create(unit.unit_id)
        try:
            return self._run_unit_inner(response, workspace)
        finally:
            workspace.cleanup()

    # ---- internals ------------------------------------------------------

    def _run_unit_inner(
        self,
        response: AssignmentResponse,
        workspace: RunnerWorkspace,
    ) -> DispatchOutcome:
        assert response.work_unit is not None
        unit = response.work_unit

        sandbox_config = SandboxConfig(
            use_bubblewrap=self._use_bubblewrap,
            runner_bin=self._runner_bin,
            workspace_path=str(workspace.workspace_dir),
            output_path=str(workspace.output_path),
            unit_id=unit.unit_id,
            manifest_sha256=unit.manifest_sha256,
        )
        try:
            argv = build_argv(sandbox_config)
        except SandboxNotAvailableError as exc:
            logger.error("sandbox unavailable for unit %s: %s", unit.unit_id, exc)
            return DispatchOutcome(
                kind=DispatchOutcomeKind.SANDBOX_UNAVAILABLE,
                reason=str(exc),
            )

        envelope_bytes = json.dumps(
            {
                "unit_id": unit.unit_id,
                "tenant_id": unit.tenant_id,
                "experiment_id": unit.experiment_id,
                "manifest_sha256": unit.manifest_sha256,
                "payload": unit.payload,
            }
        ).encode("utf-8")

        # Passthrough mode skips bwrap; we need to set env on the
        # subprocess ourselves. Bubblewrap mode passes env via --setenv
        # (built into argv), so the parent env is irrelevant.
        env = dict(os.environ)
        if not self._use_bubblewrap:
            env.update(
                {
                    "AUSPEXAI_UNIT_ID": unit.unit_id,
                    "AUSPEXAI_MANIFEST_SHA256": unit.manifest_sha256,
                    "AUSPEXAI_OUTPUT_PATH": str(workspace.output_path),
                }
            )

        logger.info("spawning runner for unit %s (argv head: %s)", unit.unit_id, argv[0])
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError as exc:
            return DispatchOutcome(
                kind=DispatchOutcomeKind.SANDBOX_UNAVAILABLE,
                reason=f"runner binary not found: {exc}",
            )

        workspace.write_pid(proc.pid)
        if self._on_runner_spawned is not None:
            try:
                self._on_runner_spawned(proc.pid)
            except Exception:
                logger.exception("on_runner_spawned callback raised; continuing")

        try:
            _stdout, stderr = proc.communicate(
                input=envelope_bytes,
                timeout=self._runner_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            logger.warning("runner timed out for unit %s", unit.unit_id)
            return DispatchOutcome(
                kind=DispatchOutcomeKind.RUNNER_CRASH,
                reason=f"runner timed out after {self._runner_timeout_seconds}s",
            )

        exit_code = proc.returncode
        if exit_code != 0:
            stderr_tail = stderr.decode("utf-8", errors="replace")[-500:]
            logger.warning(
                "runner exit_code=%d for unit %s; stderr tail: %s",
                exit_code,
                unit.unit_id,
                stderr_tail,
            )
            return DispatchOutcome(
                kind=DispatchOutcomeKind.RUNNER_CRASH,
                reason=f"runner exit_code={exit_code}; stderr: {stderr_tail}",
            )

        try:
            output_body = workspace.read_output()
        except FileNotFoundError:
            return DispatchOutcome(
                kind=DispatchOutcomeKind.RUNNER_CRASH,
                reason="runner exited 0 but did not write output.json",
            )
        except (OSError, json.JSONDecodeError) as exc:
            return DispatchOutcome(
                kind=DispatchOutcomeKind.RUNNER_CRASH,
                reason=f"failed to read runner output.json: {exc}",
            )

        try:
            completed_at = str(output_body["completed_at"])
            runner_exit_code = int(output_body["exit_code"])
            result_payload = output_body["payload"]
            if not isinstance(result_payload, dict):
                raise TypeError("payload must be a dict")
        except (KeyError, TypeError, ValueError) as exc:
            return DispatchOutcome(
                kind=DispatchOutcomeKind.RUNNER_CRASH,
                reason=f"runner output.json malformed: {exc}",
            )

        worker_signature = sign_result(
            privkey=self._privkey,
            pubkey_hex=self._worker_pubkey,
            unit_id=unit.unit_id,
            completed_at=completed_at,
            exit_code=runner_exit_code,
            payload=result_payload,
        )

        try:
            submission = self._coordinator.submit_result(
                worker_id=self._worker_id,
                unit_id=unit.unit_id,
                worker_pubkey=self._worker_pubkey,
                completed_at=completed_at,
                exit_code=runner_exit_code,
                payload=result_payload,
                worker_signature=worker_signature,
            )
        except CoordinatorError as exc:
            logger.warning("submit_result failed for unit %s: %s", unit.unit_id, exc)
            return DispatchOutcome(
                kind=DispatchOutcomeKind.SUBMIT_FAILED,
                reason=str(exc),
            )

        self._submitted.record(
            unit_id=unit.unit_id,
            assignment_id=response.assignment_id,
            result_id=submission.result_id,
            exit_code=runner_exit_code,
            completed_at=completed_at,
            coord_unit_status_after=submission.unit_status_after,
            coord_completions_so_far=submission.completions_so_far,
            coord_replication_target=submission.replication_target,
            payload_json=json.dumps(result_payload, separators=(",", ":"), sort_keys=True),
        )

        logger.info(
            "submitted result for unit %s (result_id=%s, unit_status=%s, completions=%d/%d)",
            unit.unit_id,
            submission.result_id,
            submission.unit_status_after,
            submission.completions_so_far,
            submission.replication_target,
        )
        return DispatchOutcome(
            kind=DispatchOutcomeKind.SUBMITTED,
            reason=None,
            result_response=submission,
        )


def datetime_iso_now() -> str:
    """Convenience: ISO 8601 UTC timestamp for now() — used by tests."""
    return datetime.now(UTC).isoformat()
