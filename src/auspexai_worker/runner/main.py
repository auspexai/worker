"""`auspexai-worker-runner` entry point.

Reads a WorkUnit envelope from stdin, runs the synthetic executor, writes
a Result body to $AUSPEXAI_OUTPUT_PATH. Always exits with 0 unless the
envelope is malformed (1) or the output path is unwritable (2). Executor
exceptions become exit_code=3 with an `error` field in the result.

Wire shape on the way in (from the daemon, via stdin):

    {
      "unit_id": "u-...",
      "tenant_id": "...",
      "experiment_id": "...",       # tenant's experiment_label
      "manifest_sha256": "...",
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
import sys
from datetime import UTC, datetime
from typing import Any

from .executor import SyntheticExecutor


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

    payload = envelope.get("payload") if isinstance(envelope, dict) else None
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
