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
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from auspexai_worker.coordinator import (
    AssignmentNotFoundError,
    AssignmentResponse,
    CoordinatorClient,
    CoordinatorError,
    ResultAlreadySubmittedError,
    ResultSubmissionResponse,
    UnauthorizedError,
    UnitIdMismatchError,
    WorkerIdMismatchError,
    WorkerPubkeyMismatchError,
)
from auspexai_worker.health import ThermalMonitor, ThermalState
from auspexai_worker.provisioning import (
    ExecutePolicy,
    ExecutionMode,
    ExecutorResolver,
    decide_execution,
)
from auspexai_worker.sandbox import (
    SandboxConfig,
    SandboxNotAvailableError,
    SandboxPolicy,
    build_argv,
)
from auspexai_worker.sandbox.seccomp import SeccompUnavailableError, open_seccomp_fd
from auspexai_worker.signing import RESULT_SCHEMA_VERSION, sign_result
from auspexai_worker.state import (
    PendingSubmissionRepository,
    SubmittedResultRepository,
)
from auspexai_worker.workspace import RunnerWorkspace, WorkspaceManager

logger = logging.getLogger(__name__)


class DispatchOutcomeKind:
    SUBMITTED = "submitted"
    RUNNER_CRASH = "runner_failed"
    SUBMIT_FAILED_TRANSIENT = "submit_failed_transient"  # queued for retry
    SUBMIT_FAILED_TERMINAL = "submit_failed_terminal"  # surfaced to operator
    SANDBOX_UNAVAILABLE = "sandbox_unavailable"
    # §9 #37: the worker declined to run this unit on consent/resolution
    # grounds (policy off, tenant denied, not provisioned, hash mismatch).
    EXECUTOR_REFUSED = "executor_refused"
    # W-H: host is thermally critical — refuse to protect the hardware AND
    # result integrity (a throttled host produces divergent results).
    THERMAL_CRITICAL = "thermal_critical"
    # Back-compat: previous code used a single SUBMIT_FAILED constant. Tests
    # and external callers can still check that value as a substring match
    # if needed; the more specific variants above are preferred.
    SUBMIT_FAILED = "submit_failed_transient"


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
        pending_repo: PendingSubmissionRepository,
        use_bubblewrap: bool,
        sandbox_policy: SandboxPolicy = SandboxPolicy.PERMISSIVE,
        runner_bin: str = "auspexai-worker-runner",
        runner_timeout_seconds: float | None = None,
        on_runner_spawned=None,  # Callable[[int], None] — receives PID
        max_pending_attempts: int = 100,
        # §9 #37 tenant code-execution consent. `execute_policy` is the
        # resource owner's setting; `executor_resolver` resolves provisioned
        # packages (only consulted in `provisioned` mode). Defaults keep the
        # synthetic-only behavior for callers that don't wire #37.
        execute_policy: ExecutePolicy = ExecutePolicy.SYNTHETIC,
        executor_resolver: ExecutorResolver | None = None,
        model_store_dir=None,  # Path | None — worker-local BYOM model store
        tenant_allow_list: tuple[str, ...] = (),
        tenant_deny_list: tuple[str, ...] = (),
        thermal_monitor: ThermalMonitor | None = None,  # W-H health governor
        # W-H increment 2 (M5): how often the mid-run thermal watchdog polls the
        # monitor while a runner subprocess is executing. On CRITICAL it kills
        # the runner (a unit that spikes hot mid-run no longer relies solely on
        # the hard runner_timeout). Pre-dispatch gate is increment 1.
        thermal_poll_interval_seconds: float = 5.0,
        # M3 lazy auto-acquire: when `auto_acquire` is on, a missing
        # locally-required model is pulled (via `model_acquirer`) instead of
        # refused. Default off keeps refuse-don't-echo.
        auto_acquire: bool = False,
        model_acquirer=None,  # provisioning.ModelAcquirer | None
        # Hot-reload of the consent gate (no daemon restart): when provided, the
        # dispatcher calls this PER UNIT to get the live (ExecutePolicy, auto_acquire)
        # from disk, so an owner's policy change applies to the next unit without a
        # restart. The provider folds in its own fail-safe (a read error → refuse).
        # When None, the static `execute_policy`/`auto_acquire` above are used
        # (back-compat for tests + callers that don't wire hot-reload).
        live_executor: Callable[[], tuple[ExecutePolicy, bool]] | None = None,
        # W-S (§9 #43): opens the per-unit inference-broker session for a
        # real-executor unit — `(model_id, socket_dir) -> session` where the
        # session exposes `.socket_path` and `.close()`. None (default — and
        # whenever `[inference] backend = "none"`) means no broker socket and
        # no serving: the entire W-S surface stays dormant.
        open_inference_session: Callable[[str, object], object] | None = None,
        # M1 (v0_2): this worker's serving version (e.g. "ollama/0.17.7"), so a
        # unit pinning a different serving_version_pin is refused. None ⇒ the
        # pin gate fails closed (a pinned unit is refused).
        serving_version: str | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._worker_id = worker_id
        self._worker_pubkey = worker_pubkey
        self._privkey = privkey
        self._workspaces = workspace_manager
        self._submitted = submitted_repo
        self._pending = pending_repo
        self._use_bubblewrap = use_bubblewrap
        self._sandbox_policy = sandbox_policy
        self._runner_bin = runner_bin
        self._runner_timeout_seconds = runner_timeout_seconds
        self._on_runner_spawned = on_runner_spawned
        self._max_pending_attempts = max_pending_attempts
        self._execute_policy = execute_policy
        self._executor_resolver = executor_resolver
        self._model_store_dir = model_store_dir
        self._tenant_allow_list = tenant_allow_list
        self._tenant_deny_list = tenant_deny_list
        self._thermal_monitor = thermal_monitor
        self._serving_version = serving_version
        self._thermal_poll_interval_seconds = thermal_poll_interval_seconds
        self._auto_acquire = auto_acquire
        self._live_executor = live_executor
        self._model_acquirer = model_acquirer
        self._open_inference_session = open_inference_session
        # The live per-unit broker session. Dispatch is sequential (one unit
        # per tick), so a single slot closed in run_unit's finally covers
        # every return path without re-indenting the whole inner method.
        self._inference_session = None

    def run_unit(self, response: AssignmentResponse) -> DispatchOutcome:
        """Execute the assigned unit and submit the result. Always cleans
        up the workspace before returning, even on error paths."""
        assert response.work_unit is not None
        unit = response.work_unit
        workspace = self._workspaces.create(unit.unit_id)
        try:
            return self._run_unit_inner(response, workspace)
        finally:
            if self._inference_session is not None:
                try:
                    self._inference_session.close()
                except Exception:
                    logger.exception("failed to close inference session; continuing")
                self._inference_session = None
            workspace.cleanup()

    # ---- internals ------------------------------------------------------

    def _run_unit_inner(
        self,
        response: AssignmentResponse,
        workspace: RunnerWorkspace,
    ) -> DispatchOutcome:
        assert response.work_unit is not None
        unit = response.work_unit

        # W-H thermal gate — physical safety BEFORE authorization. A host at the
        # critical threshold refuses new work (re-offered to a cooler worker; the
        # box cools by not running the heavy executor). Protects the volunteer's
        # hardware AND result integrity (a throttled host diverges from quorum).
        if self._thermal_monitor is not None and self._thermal_monitor.state() is (
            ThermalState.CRITICAL
        ):
            snap = self._thermal_monitor.snapshot()
            reason = (
                f"host thermal critical ({snap.current_temp_c}°C) — refusing to "
                "protect hardware + result integrity"
            )
            logger.warning("thermal refuse for unit %s: %s", unit.unit_id, reason)
            return DispatchOutcome(kind=DispatchOutcomeKind.THERMAL_CRITICAL, reason=reason)

        # §9 #37 consent + resolution gate. Decide BEFORE spawning anything:
        # refuse (decline the unit, re-offer), synthetic (built-in echo), or
        # real (a hash-verified, consented tenant executor). Hot-reload: when a
        # live_executor is wired, re-read the owner's policy from disk PER UNIT so a
        # config change applies without a daemon restart (the provider fails safe to
        # refuse on a read error); otherwise use the daemon-start snapshot.
        if self._live_executor is not None:
            policy, auto_acquire = self._live_executor()
        else:
            policy, auto_acquire = self._execute_policy, self._auto_acquire
        decision = decide_execution(
            policy=policy,
            tenant_id=unit.tenant_id,
            manifest_sha256=unit.manifest_sha256,
            resolver=self._executor_resolver,
            model_store_dir=self._model_store_dir,
            allow_list=self._tenant_allow_list,
            deny_list=self._tenant_deny_list,
            auto_acquire=auto_acquire,
            acquirer=self._model_acquirer,
            serving_version=self._serving_version,
        )
        if decision.mode is ExecutionMode.REFUSE:
            logger.info("declining unit %s: %s", unit.unit_id, decision.reason)
            return DispatchOutcome(
                kind=DispatchOutcomeKind.EXECUTOR_REFUSED,
                reason=decision.reason,
            )
        resolved = decision.executor if decision.mode is ExecutionMode.REAL else None
        models_dir = decision.models_dir
        has_models = models_dir is not None and models_dir.is_dir()

        # W-S (§9 #43): on an inference-enabled worker, a real-executor unit
        # with a model requirement gets the model served (loaded + warm in the
        # backend) and a per-unit broker socket in its workspace BEFORE the
        # runner spawns. Serving failure is a refusal (refuse-don't-echo) so
        # the coordinator re-offers — same posture as provisioning.
        inference_socket: str | None = None
        inference_model: str | None = None
        # §9 #13a: the worker-attested served-weights digest for THIS unit —
        # captured from the daemon's own ModelServer/broker view, never the
        # executor's self-report. {} when no model is served (non-inference
        # units still sign as v1 with an empty map). Signed into the v1 result.
        served_weights: dict[str, str] = {}
        if (
            self._open_inference_session is not None
            and decision.mode is ExecutionMode.REAL
            and has_models
        ):
            model_id = models_dir.name
            try:
                self._inference_session = self._open_inference_session(
                    model_id, workspace.workspace_dir
                )
            except Exception as exc:
                logger.warning(
                    "declining unit %s: inference serving failed for %s: %s",
                    unit.unit_id,
                    model_id,
                    exc,
                )
                return DispatchOutcome(
                    kind=DispatchOutcomeKind.EXECUTOR_REFUSED,
                    reason=f"inference serving unavailable for {model_id}: {exc}",
                )
            inference_socket = str(self._inference_session.socket_path)
            inference_model = model_id
            served_weights = {model_id: self._inference_session.served_gguf_sha256}

        sandbox_config = SandboxConfig(
            use_bubblewrap=self._use_bubblewrap,
            policy=self._sandbox_policy,
            runner_bin=self._runner_bin,
            workspace_path=str(workspace.workspace_dir),
            output_path=str(workspace.output_path),
            unit_id=unit.unit_id,
            manifest_sha256=unit.manifest_sha256,
            executor_command=resolved.command if resolved else None,
            executor_package_dir=str(resolved.package_dir) if resolved else None,
            models_dir=str(models_dir) if has_models else None,
            executor_timeout_seconds=self._runner_timeout_seconds,
            inference_socket=inference_socket,
            inference_model=inference_model,
        )
        # §41(a): STRICT requires a seccomp filter (the "escape via syscall"
        # gate). Build it fail-closed — if libseccomp/pyseccomp can't produce
        # it, refuse the unit rather than run STRICT without the syscall gate.
        seccomp_fd: int | None = None
        if sandbox_config.use_bubblewrap and sandbox_config.policy is SandboxPolicy.STRICT:
            try:
                seccomp_fd = open_seccomp_fd()
            except SeccompUnavailableError as exc:
                logger.error("unit %s: STRICT sandbox seccomp unavailable: %s", unit.unit_id, exc)
                return DispatchOutcome(
                    kind=DispatchOutcomeKind.SANDBOX_UNAVAILABLE,
                    reason=f"STRICT sandbox requires seccomp (libseccomp/pyseccomp): {exc}",
                )
        try:
            argv = build_argv(sandbox_config, seccomp_fd=seccomp_fd)
        except SandboxNotAvailableError as exc:
            if seccomp_fd is not None:
                os.close(seccomp_fd)
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
                # created_at is REQUIRED by the SDK WorkUnit (extra=forbid) that
                # the real-executor harness validates; the runner forwards it
                # into the executor --input. Omitting it made the official
                # ExecutorHarness refuse every unit.
                "created_at": unit.created_at.isoformat(),
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
            # Passthrough mode (tests / no-bwrap hosts): the executor env that
            # bwrap mode injects via --setenv must be set on the subprocess.
            if resolved is not None:
                env["AUSPEXAI_EXECUTOR_COMMAND"] = json.dumps(resolved.command)
                env["AUSPEXAI_EXECUTOR_DIR"] = str(resolved.package_dir)
                if has_models:
                    env["AUSPEXAI_MODELS_DIR"] = str(models_dir)
                if self._runner_timeout_seconds is not None:
                    env["AUSPEXAI_EXECUTOR_TIMEOUT"] = str(int(self._runner_timeout_seconds))
                if inference_socket is not None:
                    env["AUSPEXAI_INFERENCE_SOCKET"] = inference_socket
                if inference_model is not None:
                    env["AUSPEXAI_INFERENCE_MODEL"] = inference_model
        else:
            # bwrap mode: env is injected via --setenv (in argv); strip any
            # inherited executor vars so a real-executor unit never leaks into
            # a later synthetic unit's runner.
            for k in (
                "AUSPEXAI_EXECUTOR_COMMAND",
                "AUSPEXAI_EXECUTOR_DIR",
                "AUSPEXAI_MODELS_DIR",
                "AUSPEXAI_EXECUTOR_TIMEOUT",
                "AUSPEXAI_INFERENCE_SOCKET",
                "AUSPEXAI_INFERENCE_MODEL",
            ):
                env.pop(k, None)

        logger.info("spawning runner for unit %s (argv head: %s)", unit.unit_id, argv[0])
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                # §41(a): the seccomp BPF fd must be inherited by bwrap so it can
                # read the program for --seccomp <fd>; pass_fds keeps it open +
                # inheritable across exec without changing its number.
                pass_fds=(seccomp_fd,) if seccomp_fd is not None else (),
            )
        except FileNotFoundError as exc:
            return DispatchOutcome(
                kind=DispatchOutcomeKind.SANDBOX_UNAVAILABLE,
                reason=f"runner binary not found: {exc}",
            )
        finally:
            # The child holds its own inherited copy; the parent doesn't need it.
            if seccomp_fd is not None:
                os.close(seccomp_fd)

        workspace.write_pid(proc.pid)
        if self._on_runner_spawned is not None:
            try:
                self._on_runner_spawned(proc.pid)
            except Exception:
                logger.exception("on_runner_spawned callback raised; continuing")

        # W-H increment 2 (M5): watch the thermal monitor WHILE the runner runs.
        # On CRITICAL mid-run, kill the runner — protects the volunteer's hardware
        # and result integrity (a throttled host diverges from quorum) without
        # waiting out the hard runner_timeout. THERMAL_CRITICAL is a retryable
        # refusal (§2.1 #8) so the unit is re-offered (to this worker once it
        # cools, or another worker the coordinator routes to meanwhile).
        thermal_abort = threading.Event()
        stop_watch = threading.Event()
        monitor: threading.Thread | None = None
        if self._thermal_monitor is not None and self._thermal_monitor.enabled:

            def _thermal_watch() -> None:
                while not stop_watch.wait(self._thermal_poll_interval_seconds):
                    try:
                        if self._thermal_monitor.state() is ThermalState.CRITICAL:
                            thermal_abort.set()
                            proc.kill()
                            return
                    except Exception:  # a sensor read must never crash the watchdog
                        logger.exception("thermal watchdog read failed; continuing")

            monitor = threading.Thread(
                target=_thermal_watch, name=f"thermal-watch-{unit.unit_id}", daemon=True
            )
            monitor.start()

        timed_out = False
        try:
            _stdout, stderr = proc.communicate(
                input=envelope_bytes,
                timeout=self._runner_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            _stdout, stderr = proc.communicate()
            timed_out = True
        finally:
            stop_watch.set()
            if monitor is not None:
                monitor.join(timeout=2.0)

        if thermal_abort.is_set():
            snap = self._thermal_monitor.snapshot()
            logger.warning(
                "thermal CRITICAL mid-run for unit %s (%s°C) — killed runner",
                unit.unit_id,
                snap.current_temp_c,
            )
            return DispatchOutcome(
                kind=DispatchOutcomeKind.THERMAL_CRITICAL,
                reason=(
                    f"host went thermal-critical mid-run ({snap.current_temp_c}°C); "
                    "killed runner to protect hardware + result integrity"
                ),
            )
        if timed_out:
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
            # §9 #13a: sign every result as v1, binding the served-weights
            # digest the daemon captured for this unit (empty for non-inference).
            schema_version=RESULT_SCHEMA_VERSION,
            served_weights=served_weights,
        )

        payload_json = json.dumps(result_payload, separators=(",", ":"), sort_keys=True)
        served_weights_json = json.dumps(served_weights, separators=(",", ":"), sort_keys=True)

        # Write-before-submit (M6-tail): persist the signed Result to the
        # pending queue BEFORE the network call. If the submit fails or the
        # process exits between submit and local record, the next dispatch
        # tick will retry from the queue. The row is removed atomically
        # alongside the submitted_results insert on success.
        self._pending.queue(
            unit_id=unit.unit_id,
            assignment_id=response.assignment_id,
            completed_at=completed_at,
            exit_code=runner_exit_code,
            payload_json=payload_json,
            worker_signature=worker_signature,
            worker_pubkey=self._worker_pubkey,
            result_schema_version=RESULT_SCHEMA_VERSION,
            served_weights_json=served_weights_json,
        )

        return self._attempt_submit_pending(
            unit_id=unit.unit_id,
            assignment_id=response.assignment_id,
            completed_at=completed_at,
            exit_code=runner_exit_code,
            payload=result_payload,
            payload_json=payload_json,
            worker_signature=worker_signature,
            result_schema_version=RESULT_SCHEMA_VERSION,
            served_weights=served_weights,
        )

    def retry_pending(self, *, max_per_tick: int = 5) -> list[DispatchOutcome]:
        """Attempt to submit any queued pending results.

        Called at the top of the dispatch tick, before pulling a new
        assignment, so a coord that was unreachable earlier has a chance to
        accept the result before more work is taken on.

        Returns one DispatchOutcome per attempted row (in queued order).
        Terminal-marked rows are skipped — they sit until the operator
        intervenes via `auspexai-worker pending` (TBD CLI verb) or until
        the local DB is reset.
        """
        outcomes: list[DispatchOutcome] = []
        for pending in self._pending.list_retryable(limit=max_per_tick):
            if pending.attempt_count >= self._max_pending_attempts:
                # Cap exceeded — promote to terminal so the operator can
                # see and decide. Don't drop the row; the payload is the
                # volunteer's contribution and should be preserved until
                # explicitly cleared.
                self._pending.mark_attempt(
                    assignment_id=pending.assignment_id,
                    failure_kind="terminal",
                    failure_reason=(
                        f"exceeded max_pending_attempts={self._max_pending_attempts}; "
                        "operator must reconcile manually"
                    ),
                    attempted_at=datetime.now(UTC),
                )
                outcomes.append(
                    DispatchOutcome(
                        kind=DispatchOutcomeKind.SUBMIT_FAILED_TERMINAL,
                        reason=(
                            f"unit {pending.unit_id}: exceeded retry cap "
                            f"({self._max_pending_attempts}); marked terminal"
                        ),
                    )
                )
                continue
            try:
                payload = json.loads(pending.payload_json)
            except json.JSONDecodeError as exc:
                # Shouldn't happen — pending_submissions.payload_json comes
                # from json.dumps. Treat as terminal corruption.
                self._pending.mark_attempt(
                    assignment_id=pending.assignment_id,
                    failure_kind="terminal",
                    failure_reason=f"payload_json corrupted: {exc}",
                    attempted_at=datetime.now(UTC),
                )
                outcomes.append(
                    DispatchOutcome(
                        kind=DispatchOutcomeKind.SUBMIT_FAILED_TERMINAL,
                        reason=f"unit {pending.unit_id}: payload corrupted: {exc}",
                    )
                )
                continue
            # §9 #13a: re-submit the exact signed fields. served_weights_json is
            # NULL on legacy (v0) rows queued before migration 0009.
            served_weights = (
                json.loads(pending.served_weights_json) if pending.served_weights_json else None
            )
            outcomes.append(
                self._attempt_submit_pending(
                    unit_id=pending.unit_id,
                    assignment_id=pending.assignment_id,
                    completed_at=pending.completed_at,
                    exit_code=pending.exit_code,
                    payload=payload,
                    payload_json=pending.payload_json,
                    worker_signature=pending.worker_signature,
                    result_schema_version=pending.result_schema_version,
                    served_weights=served_weights,
                )
            )
        return outcomes

    def _attempt_submit_pending(
        self,
        *,
        unit_id: str,
        assignment_id: str,
        completed_at: str,
        exit_code: int,
        payload: dict,
        payload_json: str,
        worker_signature: str,
        result_schema_version: int = 0,
        served_weights: dict[str, str] | None = None,
    ) -> DispatchOutcome:
        """Single submit attempt for an already-queued pending row.

        On success: write submitted_results + delete pending row, atomically
        from the application's point of view (sequential calls on the same
        sqlite connection inside the dispatcher's process).
        On 409 result_already_submitted: idempotent success path — uses the
        existing_result_id from the coord 409 to write submitted_results,
        delete pending. Same observable outcome.
        On transient failure (network / 5xx / generic CoordinatorError):
        mark_attempt(transient); return SUBMIT_FAILED_TRANSIENT.
        On 4xx terminal: mark_attempt(terminal); return SUBMIT_FAILED_TERMINAL.
        """
        try:
            submission = self._coordinator.submit_result(
                worker_id=self._worker_id,
                unit_id=unit_id,
                worker_pubkey=self._worker_pubkey,
                completed_at=completed_at,
                exit_code=exit_code,
                payload=payload,
                worker_signature=worker_signature,
                # §9 #46 D6 fix: exact assignment disambiguation — unit_ids
                # are tenant-chosen and can collide across experiments.
                assignment_id=assignment_id,
                # §9 #13a: the worker-attested served-weights digest + the
                # canonical-schema version the coordinator reconstructs against.
                result_schema_version=result_schema_version,
                served_weights=served_weights,
            )
        except ResultAlreadySubmittedError as exc:
            # The coord already has this result. Use the existing_result_id
            # from the 409 to write a submitted_results row, then remove
            # from pending. Net effect: full local reconciliation.
            if exc.existing_result_id is None:
                # Pre-existing coord build that doesn't include the detail?
                # Be safe: mark terminal so the operator can investigate.
                logger.warning(
                    "unit %s: 409 result_already_submitted with no existing_result_id; "
                    "marking pending row terminal for operator review",
                    unit_id,
                )
                self._pending.mark_attempt(
                    assignment_id=assignment_id,
                    failure_kind="terminal",
                    failure_reason=(
                        "409 result_already_submitted but coord did not return "
                        "existing_result_id; cannot reconcile locally"
                    ),
                    attempted_at=datetime.now(UTC),
                )
                return DispatchOutcome(
                    kind=DispatchOutcomeKind.SUBMIT_FAILED_TERMINAL,
                    reason=str(exc),
                )
            logger.info(
                "unit %s: 409 result_already_submitted; reconciling with existing_result_id=%s",
                unit_id,
                exc.existing_result_id,
            )
            self._submitted.record(
                unit_id=unit_id,
                assignment_id=exc.existing_assignment_id or assignment_id,
                result_id=exc.existing_result_id,
                exit_code=exit_code,
                completed_at=completed_at,
                coord_unit_status_after=None,  # Unknown; M7-tail will backfill
                coord_completions_so_far=None,
                coord_replication_target=None,
                payload_json=payload_json,
            )
            self._pending.remove(assignment_id)
            return DispatchOutcome(
                kind=DispatchOutcomeKind.SUBMITTED,
                reason=f"reconciled-via-409 (existing_result_id={exc.existing_result_id})",
            )
        except (
            UnitIdMismatchError,
            WorkerPubkeyMismatchError,
            WorkerIdMismatchError,
            UnauthorizedError,
            AssignmentNotFoundError,
        ) as exc:
            # Terminal — these can't be fixed by retrying. Keep the pending
            # row so the operator can see it; don't repeatedly hammer the
            # coord with a request guaranteed to fail.
            logger.error(
                "unit %s: submit failed terminally (%s); marking pending row for operator review",
                unit_id,
                type(exc).__name__,
            )
            self._pending.mark_attempt(
                assignment_id=assignment_id,
                failure_kind="terminal",
                failure_reason=f"{type(exc).__name__}: {exc}",
                attempted_at=datetime.now(UTC),
            )
            return DispatchOutcome(
                kind=DispatchOutcomeKind.SUBMIT_FAILED_TERMINAL,
                reason=str(exc),
            )
        except CoordinatorError as exc:
            # Generic CoordinatorError covers transport errors (DNS, TCP,
            # TLS, timeout) and any unexpected status codes. Assume transient
            # until proven otherwise — the next dispatch tick will retry.
            logger.warning(
                "unit %s: submit failed transiently (%s); will retry next tick",
                unit_id,
                exc,
            )
            self._pending.mark_attempt(
                assignment_id=assignment_id,
                failure_kind="transient",
                failure_reason=str(exc),
                attempted_at=datetime.now(UTC),
            )
            return DispatchOutcome(
                kind=DispatchOutcomeKind.SUBMIT_FAILED_TRANSIENT,
                reason=str(exc),
            )

        # Success path: write submitted_results + remove from pending.
        self._submitted.record(
            unit_id=unit_id,
            assignment_id=assignment_id,
            result_id=submission.result_id,
            exit_code=exit_code,
            completed_at=completed_at,
            coord_unit_status_after=submission.unit_status_after,
            coord_completions_so_far=submission.completions_so_far,
            coord_replication_target=submission.replication_target,
            payload_json=payload_json,
        )
        self._pending.remove(assignment_id)

        logger.info(
            "submitted result for unit %s (result_id=%s, unit_status=%s, completions=%d/%d)",
            unit_id,
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

    def fetch_pending_canonical(self, *, max_per_tick: int = 5) -> int:
        """M7-tail: pull canonical-receipt blobs for submitted results that
        still have placeholder receipts.

        For each `receipt_status='placeholder'` row, ask the coord for the
        canonical bytes via `get_canonical_receipt`. On 200 → `set_canonical`
        promotes the row to `canonical`. On 404 → leave as placeholder (the
        unit's quorum may have disagreed, or M7c issuance hasn't fired yet;
        we'll try again next tick). Transport errors are logged but don't
        propagate — the fetch loop is best-effort, never blocking.

        Returns the count of rows promoted to canonical this tick.
        """
        promoted = 0
        try:
            pending = self._submitted.list_pending_canonical(limit=max_per_tick)
        except Exception:
            logger.exception("fetch_pending_canonical: failed to list pending rows")
            return 0

        for row in pending:
            try:
                resp = self._coordinator.get_canonical_receipt(
                    worker_id=self._worker_id,
                    result_id=row.result_id,
                )
            except Exception:
                logger.debug(
                    "fetch_pending_canonical: coord call failed for %s; leaving as placeholder",
                    row.result_id,
                )
                continue
            if resp is None:
                # 404 — receipt not (yet) issued. Leave as placeholder; try
                # again on a future tick.
                continue
            try:
                updated = self._submitted.set_canonical(
                    result_id=row.result_id,
                    canonical_blob=resp.cose_signed_blob,
                    canonical_format="cose-sign1-cbor-receipt-v0.1",
                    fetched_at=datetime.now(UTC),
                )
            except Exception:
                logger.exception(
                    "fetch_pending_canonical: set_canonical failed for %s",
                    row.result_id,
                )
                continue
            if updated:
                promoted += 1
                logger.info(
                    "promoted local receipt for result %s to canonical (receipt_id=%s)",
                    row.result_id,
                    resp.receipt_id,
                )
        return promoted


def datetime_iso_now() -> str:
    """Convenience: ISO 8601 UTC timestamp for now() — used by tests."""
    return datetime.now(UTC).isoformat()
