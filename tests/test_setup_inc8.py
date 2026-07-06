"""Onboarding inc 8 — `auspexai-worker setup` + `service` ("installer
provisions, product onboards"): the guided flow the curl installer delegates
to, reachable identically from a plain pip install."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from click.testing import CliRunner

import auspexai_worker.service as svc
from auspexai_worker.cli import cli
from auspexai_worker.state import Database, MigrationRunner, WorkerSelfRepository


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "AUSPEXAI_WORKER_STATE_DIR": str(tmp_path / "state"),
        "AUSPEXAI_WORKER_DATA_DIR": str(tmp_path / "data"),
    }


def _config(tmp_path: Path, body: str = "") -> Path:
    p = tmp_path / "worker.toml"
    # GUARD (the 2026-07-03 lesson): the default coordinator_url is the PUBLIC
    # network — an un-mocked enroll from a test reaches production (it happened;
    # 12 phantom T0 workers, retired same day). Every setup test pins an
    # unroutable coordinator so any accidental network call fails fast instead.
    p.write_text('[coordinator]\nurl = "http://127.0.0.1:9"\n' + body, encoding="utf-8")
    return p


def _pre_enroll(tmp_path: Path) -> None:
    """Seed an enrollment row so `setup` takes the already-enrolled branch —
    the bootstrap flow itself is covered by test_bootstrap.py with a mocked
    transport; these tests must never enroll for real."""
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    db = Database(state / "worker.db")
    MigrationRunner(db).apply_all()
    WorkerSelfRepository(db).insert(
        worker_id="wkr-setup-test",
        trust_tier=0,
        pubkey_hex="a" * 64,
        enrolled_at=datetime(2026, 7, 3, tzinfo=UTC),
    )
    db.close()


# ── service unit rendering ────────────────────────────────────────────────────


def test_launchd_plist_renders_current_binary():
    plist = svc.render_launchd_plist(binary="/x/bin/auspexai-worker")
    assert "<string>/x/bin/auspexai-worker</string>" in plist
    assert "<string>daemon</string>" in plist
    assert svc.LAUNCHD_LABEL in plist


def test_systemd_unit_renders_the_proven_fleet_set():
    unit = svc.render_systemd_unit(binary="/x/bin/auspexai-worker")
    assert "ExecStart=/x/bin/auspexai-worker daemon" in unit
    # The PROVEN minimal set (2026-07-03 incident: the full §5.17 hardening
    # 218/CAPABILITIES'd on production user managers — see render docstring).
    assert "Delegate=yes" in unit
    assert "Nice=19" in unit
    assert "PrivateTmp=true" in unit
    for forbidden in ("SystemCallFilter", "ProtectHome", "ProtectSystem", "ProtectKernel"):
        assert forbidden not in unit, forbidden


def test_service_install_linux_writes_user_unit(tmp_path: Path, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(svc.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        svc,
        "_run",
        lambda argv: (
            calls.append(argv),
            type("R", (), {"returncode": 0, "stdout": "yes", "stderr": ""})(),
        )[1],
    )
    messages = svc.install(start=True, platform="linux")
    unit = tmp_path / ".config" / "systemd" / "user" / svc.SYSTEMD_UNIT
    assert unit.exists()
    assert any("enable" in c and "--now" in c for c in calls)
    assert any("written" in m for m in messages)


def test_service_uninstall_linux_removes_unit(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(svc.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        svc,
        "_run",
        lambda argv: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )
    unit = tmp_path / ".config" / "systemd" / "user" / svc.SYSTEMD_UNIT
    unit.parent.mkdir(parents=True)
    unit.write_text("x")
    messages = svc.uninstall(platform="linux")
    assert not unit.exists()
    assert any("removed" in m for m in messages)


def test_service_install_darwin_writes_plist(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(svc.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        svc,
        "_run",
        lambda argv: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )
    svc.install(start=False, platform="darwin")
    assert (tmp_path / "Library" / "LaunchAgents" / f"{svc.LAUNCHD_LABEL}.plist").exists()


# ── the setup command (non-interactive path — what the no-tty installer runs) ─


def test_setup_lean_noninteractive_records_and_installs_service(tmp_path: Path, monkeypatch):
    installed: dict = {}
    monkeypatch.setattr(
        svc,
        "install",
        lambda start=True, platform=None: installed.update(start=start) or ["service ok"],
    )
    cfg = _config(tmp_path)
    _pre_enroll(tmp_path)
    r = CliRunner().invoke(
        cli,
        [
            "--config",
            str(cfg),
            "setup",
            "--flavor",
            "lean",
            "--sandbox",
            "permissive",
            "--yes",
            "--skip-models",
        ],
        env=_env(tmp_path),
    )
    assert r.exit_code == 0, r.output
    # Flavor + backend + sandbox recorded in worker.toml (declarative flavors).
    body = cfg.read_text()
    assert 'flavor = "lean"' in body
    assert 'backend = "none"' in body
    assert 'policy = "permissive"' in body
    # --yes drove the whole chain: the pre-seeded enrollment is detected, then
    # the service install (only-after-enroll ordering), then the footer.
    assert "Already enrolled" in r.output
    assert installed.get("start") is True
    assert "Setup complete" in r.output


def test_setup_inference_flags_skip_prompts_and_warn_on_missing_ollama(tmp_path: Path, monkeypatch):
    import shutil as shutil_mod
    import subprocess as subprocess_mod

    monkeypatch.setattr(svc, "install", lambda start=True, platform=None: ["service ok"])
    monkeypatch.setattr(shutil_mod, "which", lambda name: None)  # no ollama
    monkeypatch.setattr(
        subprocess_mod,
        "run",
        lambda *a, **k: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )
    cfg = _config(tmp_path)
    _pre_enroll(tmp_path)
    r = CliRunner().invoke(
        cli,
        [
            "--config",
            str(cfg),
            "setup",
            "--flavor",
            "inference",
            "--sandbox",
            "permissive",
            "--auto-acquire",
            "on",
            "--yes",
            "--skip-models",
        ],
        env=_env(tmp_path),
    )
    assert r.exit_code == 0, r.output
    body = cfg.read_text()
    assert 'flavor = "inference"' in body
    assert 'backend = "ollama"' in body
    assert "auto_acquire = true" in body
    assert "Ollama" in r.output  # the missing-system-dep advisory (never installs)


def test_logprobs_whitelisted_determinism_safe():
    # D2: logprobs are output diagnostics — always requestable, determinism-safe.
    from auspexai_worker.inference.broker import sanitize_options

    out = sanitize_options({"logprobs": True, "top_logprobs": 3})
    assert out["logprobs"] is True and out["top_logprobs"] == 3
    assert out["temperature"] == 0  # greedy unaffected
