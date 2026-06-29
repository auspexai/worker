"""`auspexai-worker-runner` entry point.

Reads a WorkUnit envelope from stdin and produces a Result body at
$AUSPEXAI_OUTPUT_PATH. Two modes, selected by the daemon via env:

  - **Synthetic** (no $AUSPEXAI_EXECUTOR_COMMAND): runs the built-in echo
    executor. Dev/test/CI; no tenant code.
  - **Real tenant executor** ($AUSPEXAI_EXECUTOR_COMMAND set — §9 #37): the
    daemon has already resolved + consented to a hash-verified tenant package.
    The runner materializes the SDK WorkUnit JSON, invokes the tenant's
    `executor.command --input/--output/--models/--timeout` (the SDK
    ExecutorHarness contract) as a child inside this sandbox, reads the
    `ExecutorOutput`, and maps it to the Result body. An executor that exits
    non-zero (tenant-code or harness failure) makes the runner exit non-zero
    so the daemon refuses + re-offers rather than submitting a partial result.

Exits 0 unless the envelope is malformed (1) or the output path is unwritable
(2). Synthetic-executor exceptions become exit_code=3 with an `error` field.

Wire shape on the way in (from the daemon, via stdin):

    {
      "unit_id": "u-...",
      "tenant_id": "...",
      "experiment_id": "...",       # tenant's experiment_label
      "manifest_sha256": "...",
      "created_at": "ISO 8601",     # both required by the SDK WorkUnit harness
      "payload": {...}              # opaque to the runner
    }

Wire shape on the way out (file at $AUSPEXAI_OUTPUT_PATH):

    {
      "completed_at": "ISO 8601 UTC timestamp",
      "exit_code": 0,                # the executor's own exit code
      "payload": {...}               # executor's output
    }

The daemon adds unit_id, worker_pubkey, and worker_signature to the
body before submitting it to the coordinator — the runner never sees
the worker's identity by design.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .executor import SyntheticExecutor

# Default advisory timeout (seconds) handed to the executor via --timeout. The
# daemon still enforces the hard wall-clock kill on the runner subprocess.
DEFAULT_EXECUTOR_TIMEOUT = 600


def _augmented_path() -> str:
    """A PATH that resolves bare commands under a minimal launchd environment
    (macOS hands the daemon /usr/bin:/bin:/usr/sbin:/sbin): the running
    interpreter's own dir + the standard bin dirs, ahead of the inherited PATH."""
    extra = [
        os.path.dirname(sys.executable),
        "/opt/homebrew/bin",  # macOS Apple-Silicon Homebrew
        "/usr/local/bin",  # macOS Intel Homebrew / Linux
        "/usr/bin",
        "/bin",
    ]
    return os.pathsep.join([*extra, os.environ.get("PATH", "")])


def _resolve_program(program: str) -> str:
    """Resolve the executor command's program so it runs under a minimal PATH.

    A bare `python`/`python3` becomes the runner's OWN interpreter
    (`sys.executable`) — it is always present, the right version, and carries the
    SDK the executor imports; macOS has no bare `python`, only `python3` off the
    launchd PATH (the symptom: runner FileNotFoundError 'python'). On Linux the
    daemon's python and a bare `python` are the same interpreter, so this is a
    no-op there. Other bare names resolve via the augmented PATH; absolute paths
    pass through untouched."""
    if os.path.isabs(program):
        return program
    if program in ("python", "python3"):
        return sys.executable
    return shutil.which(program, path=_augmented_path()) or program


def main() -> int:
    try:
        raw = sys.stdin.read()
        envelope = json.loads(raw)
    except (json.JSONDecodeError, OSError) as exc:
        _emit_error(f"failed to read work-unit envelope from stdin: {exc}", exit_code=1)
        return 1

    output_path = os.environ.get("AUSPEXAI_OUTPUT_PATH")
    if not output_path:
        _emit_error("AUSPEXAI_OUTPUT_PATH env var is required", exit_code=2)
        return 2

    if not isinstance(envelope, dict):
        _emit_error("envelope was not a JSON object", exit_code=1)
        return 1

    # §9 #37: the daemon sets AUSPEXAI_EXECUTOR_COMMAND only after it has
    # resolved + consented to a hash-verified tenant package. Its presence
    # selects the real-executor path over the synthetic echo.
    if os.environ.get("AUSPEXAI_EXECUTOR_COMMAND"):
        return _run_real_executor(envelope, output_path)

    return _run_synthetic(envelope, output_path)


def _run_synthetic(envelope: dict[str, Any], output_path: str) -> int:
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        result_payload: dict[str, Any] = {
            "error": "envelope.payload was missing or not a dict",
        }
        executor_exit = 1
    else:
        try:
            result_payload = SyntheticExecutor().run(payload)
            executor_exit = 0
        except Exception as exc:
            result_payload = {"error": f"{type(exc).__name__}: {exc}"}
            executor_exit = 3

    body = {
        "completed_at": datetime.now(UTC).isoformat(),
        "exit_code": executor_exit,
        "payload": result_payload,
    }
    return _write_body(body, output_path)


def _run_real_executor(envelope: dict[str, Any], output_path: str) -> int:
    """Invoke the resolved tenant executor per the SDK ExecutorHarness CLI and
    translate its ExecutorOutput into the daemon Result body.

    Returns non-zero (so the daemon refuses + re-offers) if the executor fails
    or doesn't produce a valid ExecutorOutput — we never submit a partial
    result under a real tenant's experiment."""
    try:
        command = json.loads(os.environ["AUSPEXAI_EXECUTOR_COMMAND"])
    except (KeyError, json.JSONDecodeError) as exc:
        _emit_error(f"AUSPEXAI_EXECUTOR_COMMAND is missing/invalid: {exc}", exit_code=2)
        return 2
    if not isinstance(command, list) or not command:
        _emit_error("AUSPEXAI_EXECUTOR_COMMAND must be a non-empty JSON array", exit_code=2)
        return 2

    package_dir = os.environ.get("AUSPEXAI_EXECUTOR_DIR") or None
    workspace = Path(output_path).parent

    # Materialize the SDK WorkUnit JSON the executor reads via --input. This
    # MUST carry every field the SDK `WorkUnit` model requires — it is
    # `extra="forbid"` AND requires `manifest_sha256` + `created_at`, so the
    # official `ExecutorHarness` rejects any unit missing them (which silently
    # refused every unit for SDK-harness tenants until this was fixed).
    workunit = {
        "schema_version": "0.1",
        "unit_id": envelope.get("unit_id"),
        "tenant_id": envelope.get("tenant_id"),
        "experiment_id": envelope.get("experiment_id"),
        "manifest_sha256": envelope.get("manifest_sha256"),
        "created_at": envelope.get("created_at"),
        "payload": envelope.get("payload"),
    }
    input_path = workspace / "exec_input.json"
    exec_output_path = workspace / "exec_output.json"
    try:
        input_path.write_text(json.dumps(workunit), encoding="utf-8")
    except OSError as exc:
        _emit_error(f"failed to write executor input: {exc}", exit_code=2)
        return 2

    # The harness requires --models to be a directory even when the tenant
    # declares no weights. Use the staged models/ dir if present, else an empty
    # one in the workspace.
    models_dir = os.environ.get("AUSPEXAI_MODELS_DIR")
    models_path = Path(models_dir) if models_dir else None
    if models_path is None or not models_path.is_dir():
        models_path = workspace / "models"
        models_path.mkdir(exist_ok=True)

    timeout = os.environ.get("AUSPEXAI_EXECUTOR_TIMEOUT")
    # Resolve the program so it runs under macOS launchd's minimal PATH (a bare
    # `python` -> the runner's own interpreter; see _resolve_program).
    argv = [
        _resolve_program(command[0]),
        *command[1:],
        "--input",
        str(input_path),
        "--output",
        str(exec_output_path),
        "--models",
        str(models_path),
        "--timeout",
        timeout or str(DEFAULT_EXECUTOR_TIMEOUT),
    ]

    # Hand the executor an augmented PATH too, so any further bare-command lookups
    # it makes resolve under launchd's minimal environment.
    exec_env = dict(os.environ)
    exec_env["PATH"] = _augmented_path()
    proc = subprocess.run(
        argv,
        cwd=package_dir,
        capture_output=True,
        text=True,
        check=False,
        env=exec_env,
    )
    if proc.returncode != 0:
        # Tenant-code (1) / harness-IO (2) failure — surface stderr + fail so
        # the daemon refuses + re-offers rather than submitting a partial.
        _emit_error(
            f"executor exited {proc.returncode}; stderr: {proc.stderr.strip()[-800:]}",
            exit_code=3,
        )
        return 3

    try:
        exec_output = json.loads(exec_output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _emit_error(f"executor exited 0 but produced no valid ExecutorOutput: {exc}", exit_code=3)
        return 3
    if not isinstance(exec_output, dict) or "payload" not in exec_output:
        _emit_error("ExecutorOutput missing required `payload`", exit_code=3)
        return 3

    body = {
        "completed_at": exec_output.get("completed_at") or datetime.now(UTC).isoformat(),
        "exit_code": int(exec_output.get("exit_code", 0)),
        "payload": exec_output["payload"],
    }
    return _write_body(body, output_path)


def _write_body(body: dict[str, Any], output_path: str) -> int:
    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(body, fh)
    except OSError as exc:
        _emit_error(f"failed to write result to {output_path}: {exc}", exit_code=2)
        return 2
    return 0


def _emit_error(message: str, *, exit_code: int) -> None:
    """Best-effort emit an error line. Some failure modes (no stdout) make
    even this impossible — exit code is the canonical signal."""
    try:
        print(f"auspexai-worker-runner: {message}", file=sys.stderr)
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
