"""Tests for the runner subprocess entrypoint."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _invoke_runner(envelope: dict, *, output_path: Path, env_extras: dict | None = None):
    """Invoke the runner as a subprocess via the installed entry point."""
    env = {"AUSPEXAI_OUTPUT_PATH": str(output_path)}
    if env_extras:
        env.update(env_extras)
    proc = subprocess.run(
        ["auspexai-worker-runner"],
        input=json.dumps(envelope),
        text=True,
        capture_output=True,
        env={**env, "PATH": _venv_bin_path()},
    )
    return proc


def _venv_bin_path() -> str:
    """Return the directory containing the python venv that owns the runner
    entry point — needed so the subprocess can find auspexai-worker-runner."""
    return str(Path(sys.executable).parent)


class TestRunnerHappyPath:
    def test_synthetic_executor_echoes_payload(self, tmp_path: Path) -> None:
        envelope = {
            "unit_id": "u-1",
            "tenant_id": "t-1",
            "experiment_id": "exp-label",
            "manifest_sha256": "a" * 64,
            "payload": {"input": 7, "label": "test"},
        }
        output_path = tmp_path / "output.json"
        proc = _invoke_runner(envelope, output_path=output_path)
        assert proc.returncode == 0, proc.stderr
        body = json.loads(output_path.read_text())
        assert body["exit_code"] == 0
        assert "completed_at" in body
        assert body["payload"]["echo"] == envelope["payload"]


class TestRunnerErrors:
    def test_missing_output_path_env_var_exits_2(self, tmp_path: Path) -> None:
        # Drop AUSPEXAI_OUTPUT_PATH from env entirely.
        proc = subprocess.run(
            ["auspexai-worker-runner"],
            input='{"payload":{}}',
            text=True,
            capture_output=True,
            env={"PATH": _venv_bin_path()},
        )
        assert proc.returncode == 2
        assert "AUSPEXAI_OUTPUT_PATH" in proc.stderr

    def test_malformed_envelope_exits_1(self, tmp_path: Path) -> None:
        output_path = tmp_path / "output.json"
        proc = subprocess.run(
            ["auspexai-worker-runner"],
            input="not json at all",
            text=True,
            capture_output=True,
            env={"AUSPEXAI_OUTPUT_PATH": str(output_path), "PATH": _venv_bin_path()},
        )
        assert proc.returncode == 1
        assert not output_path.exists()

    def test_non_dict_payload_writes_error_result(self, tmp_path: Path) -> None:
        output_path = tmp_path / "output.json"
        envelope = {"payload": "not-a-dict"}
        proc = _invoke_runner(envelope, output_path=output_path)
        assert proc.returncode == 0  # runner itself succeeded
        body = json.loads(output_path.read_text())
        assert body["exit_code"] != 0
        assert "error" in body["payload"]


class TestExecutorProgramResolution:
    """The runner resolves the executor's program so it runs under macOS
    launchd's minimal PATH (the `python` FileNotFoundError class of bug)."""

    def test_bare_python_resolves_to_runner_interpreter(self) -> None:
        from auspexai_worker.runner.main import _resolve_program

        # macOS has no bare `python`; map both bare names to the runner's own
        # interpreter, which is always present and carries the SDK.
        assert _resolve_program("python") == sys.executable
        assert _resolve_program("python3") == sys.executable

    def test_absolute_path_passes_through(self) -> None:
        from auspexai_worker.runner.main import _resolve_program

        assert _resolve_program("/opt/custom/python") == "/opt/custom/python"

    def test_other_bare_name_resolved_via_augmented_path(self, monkeypatch) -> None:
        from auspexai_worker.runner.main import _resolve_program

        monkeypatch.setattr("shutil.which", lambda name, path=None: "/opt/homebrew/bin/deno")
        assert _resolve_program("deno") == "/opt/homebrew/bin/deno"

    def test_unresolvable_name_falls_back_unchanged(self, monkeypatch) -> None:
        from auspexai_worker.runner.main import _resolve_program

        monkeypatch.setattr("shutil.which", lambda name, path=None: None)
        assert _resolve_program("nope") == "nope"

    def test_augmented_path_includes_interpreter_dir(self) -> None:
        from auspexai_worker.runner.main import _augmented_path

        parts = _augmented_path().split(":")
        assert str(Path(sys.executable).parent) in parts
        assert "/opt/homebrew/bin" in parts
