"""Synthetic executor — M4 placeholder.

Echoes the input payload back as the output, with a small annotation so a
test can tell echoed output from a no-op. Real tenant executors (eventually
including the first real tenant Vigiles per §5.3) replace this in a later
milestone by loading executor code from the tenant package — the runner
subprocess will then dispatch to that code via a published-contract entrypoint.

For M4 the synthetic executor lives in this module so:
  - The runner can be exercised end-to-end without any tenant code present
  - CI doesn't need GPU or model weights to validate the full pipeline
  - The coordinator's M4 verification path completes a real work-unit loop
"""

from __future__ import annotations

from typing import Any


class SyntheticExecutor:
    """Trivial echo executor. `run(payload)` returns a dict that includes
    the input verbatim plus an `echo` marker so tests can verify the round
    trip didn't lose anything."""

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "echo": payload,
            "executor": "auspexai_worker.runner.executor.SyntheticExecutor",
        }
