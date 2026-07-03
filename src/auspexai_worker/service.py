"""Worker service management — launchd (macOS) / systemd user unit (Linux).

Onboarding inc 8 ("installer provisions, product onboards"): the persistent
service is a PRODUCT surface, one command the installer merely calls — so a
plain `pip install auspexai-worker` reaches the identical persistent setup via
`auspexai-worker service install`, mirroring the researcher dashboard's
`auspexai-dashboard service install` exactly.

Layout notes:
- The rendered unit points at THIS environment's `auspexai-worker` binary
  (resolved beside the running interpreter), so it works for the curl-installed
  venv, a deb install, and any pip venv alike.
- Linux writes the USER-local unit (`~/.config/systemd/user/`) — no sudo, and
  systemd gives it precedence over the installer's `/etc/systemd/user/` copy,
  so `service install` also cleanly supersedes an installer-provisioned unit.
- The unit body is the hardened Phase-1 template (low-priority scheduling,
  §5.17 hardening, the §41(a) cgroup delegation the STRICT caps need).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

LAUNCHD_LABEL = "network.auspexai.worker"
SYSTEMD_UNIT = "auspexai-worker.service"


def worker_bin() -> str:
    """The absolute `auspexai-worker` entry point of THIS environment."""
    candidate = Path(sys.executable).with_name("auspexai-worker")
    return str(candidate) if candidate.exists() else "auspexai-worker"


def launchd_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def systemd_unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / SYSTEMD_UNIT


def render_launchd_plist(binary: str | None = None) -> str:
    log = Path.home() / "Library" / "Logs" / "auspexai-worker.log"
    exe = binary or worker_bin()
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{LAUNCHD_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{exe}</string>
    <string>daemon</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>{log}</string>
  <key>StandardErrorPath</key><string>{log}</string>
</dict>
</plist>
"""


def render_systemd_unit(binary: str | None = None) -> str:
    """The systemd user unit — the PROVEN fleet directive set, deliberately
    minimal. LIVE INCIDENT 2026-07-03: the first render used the packaging
    template's full §5.17 hardening (ProtectHome/ProtectSystem/ProtectKernel*/
    SystemCallFilter/…) and the daemon died with 218/CAPABILITIES on both
    production Ubuntu hosts' user managers — those directives need privileges/
    user namespaces a user manager cannot always grant. The set below is what
    has run the fleet for weeks (PrivateTmp + NoNewPrivileges + the §41(a)
    Delegate) plus the unprivileged low-priority scheduling; daemon-tier
    hardening beyond it belongs to the system-level unit, not here. The
    TENANT-CODE sandbox (bwrap + seccomp) is unaffected — it wraps the runner,
    not the daemon."""
    exe = binary or worker_bin()
    return f"""; AuspexAI worker — systemd user unit (written by `auspexai-worker service install`).
[Unit]
Description=AuspexAI volunteer worker
Documentation=https://github.com/auspexai/worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart={exe} daemon
Restart=on-failure
RestartSec=10
; Low-priority (unprivileged): the worker runs in the volunteer's SPARE capacity.
Nice=19
IOSchedulingClass=idle
PrivateTmp=true
NoNewPrivileges=true
; §41(a): delegate a cgroup-v2 subtree so the daemon can cap the runner.
Delegate=yes

[Install]
WantedBy=default.target
"""


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=30)


def install(*, start: bool = True, platform: str | None = None) -> list[str]:
    """Write the unit for this OS and (optionally) start it. Returns
    human-readable progress lines (the CLI prints them). Idempotent —
    re-install rewrites the unit and restarts."""
    plat = platform or sys.platform
    messages: list[str] = []
    if plat == "darwin":
        path = launchd_plist_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        (Path.home() / "Library" / "Logs").mkdir(parents=True, exist_ok=True)
        path.write_text(render_launchd_plist())
        messages.append(f"launchd agent written: {path}")
        if start:
            _run(["launchctl", "unload", str(path)])
            loaded = _run(["launchctl", "load", str(path)])
            if loaded.returncode == 0:
                messages.append("worker loaded via launchd (auto-starts on login)")
            else:
                messages.append(
                    f"launchctl load failed ({loaded.stderr.strip() or 'unknown'}); "
                    "start manually: auspexai-worker daemon"
                )
        return messages

    path = systemd_unit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_systemd_unit())
    messages.append(f"systemd user unit written: {path}")
    _run(["systemctl", "--user", "daemon-reload"])
    if start:
        started = _run(["systemctl", "--user", "enable", "--now", SYSTEMD_UNIT])
        if started.returncode == 0:
            messages.append("worker enabled + started (systemd --user)")
        else:
            messages.append(
                f"systemd --user unavailable ({started.stderr.strip() or 'unknown'}); "
                "start manually: auspexai-worker daemon"
            )
    # Linger keeps the user service alive after logout; advise, never sudo here.
    linger = _run(["sh", "-c", 'loginctl show-user "$(id -un)" -p Linger --value'])
    if linger.returncode == 0 and linger.stdout.strip() != "yes":
        messages.append(
            "note: enable linger so the worker survives logout: "
            "sudo loginctl enable-linger $(id -un)"
        )
    return messages


def uninstall(*, platform: str | None = None) -> list[str]:
    """Stop the service and remove the unit this module manages."""
    plat = platform or sys.platform
    messages: list[str] = []
    if plat == "darwin":
        path = launchd_plist_path()
        _run(["launchctl", "unload", str(path)])
        if path.exists():
            path.unlink()
            messages.append(f"removed {path}")
        else:
            messages.append("no launchd agent installed")
        return messages
    _run(["systemctl", "--user", "disable", "--now", SYSTEMD_UNIT])
    path = systemd_unit_path()
    if path.exists():
        path.unlink()
        _run(["systemctl", "--user", "daemon-reload"])
        messages.append(f"removed {path}")
    else:
        messages.append(
            f"no user unit at {path} (an installer-provisioned /etc unit, if any, stays)"
        )
    return messages


def restart(*, platform: str | None = None) -> list[str]:
    plat = platform or sys.platform
    if plat == "darwin":
        path = launchd_plist_path()
        _run(["launchctl", "unload", str(path)])
        loaded = _run(["launchctl", "load", str(path)])
        return ["restarted via launchd" if loaded.returncode == 0 else "launchctl load failed"]
    r = _run(["systemctl", "--user", "restart", SYSTEMD_UNIT])
    return [
        "restarted (systemd --user)" if r.returncode == 0 else f"restart failed: {r.stderr.strip()}"
    ]


def status(*, platform: str | None = None) -> str:
    plat = platform or sys.platform
    if plat == "darwin":
        r = _run(["launchctl", "list", LAUNCHD_LABEL])
        return "running (launchd)" if r.returncode == 0 else "not loaded (launchd)"
    r = _run(["systemctl", "--user", "is-active", SYSTEMD_UNIT])
    return f"{r.stdout.strip() or 'unknown'} (systemd --user)"
