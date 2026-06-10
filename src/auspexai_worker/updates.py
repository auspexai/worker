"""Update-availability helpers (§9 #46 — release announcements).

Pure functions, deliberately import-free of coordinator/state: the dashboard,
CLI, and heartbeat loop all consume them. The coordinator ANNOUNCES the latest
release in the heartbeat response; whether to upgrade is the volunteer's
election — these helpers only compare versions and print the command, they
never execute anything.
"""

from __future__ import annotations

import re

_NUMERIC_PREFIX = re.compile(r"^(\d+(?:\.\d+)*)")

ONRAMP_URL = "https://getworker.auspexai.network"


def _split(version: str) -> tuple[tuple[int, ...] | None, str]:
    """(numeric parts, trailing suffix) — e.g. '0.2.0.dev3' → ((0,2,0), '.dev3')."""
    s = version.strip().lstrip("v")
    m = _NUMERIC_PREFIX.match(s)
    if not m:
        return None, ""
    return tuple(int(p) for p in m.group(1).split(".")), s[m.end() :]


def is_newer_version(latest: str, current: str) -> bool:
    """True when `latest` (the announced release) is newer than `current`
    (this worker's __version__).

    Compares leading dotted-numeric components; unparsable input is never
    "newer" (don't nag on garbage). Tie-break: equal numerics where `current`
    carries a dev/local suffix (a hatch-vcs between-tags build, e.g.
    `0.2.0.dev3+g1234567`) count the clean `latest` as newer — the dev build
    predates the tagged release, and this also lets a dev-build worker
    demo the banner.
    """
    latest_parts, latest_rest = _split(latest)
    current_parts, current_rest = _split(current)
    if latest_parts is None or current_parts is None:
        return False
    # Zero-pad to equal length: 0.2 == 0.2.0, not older.
    width = max(len(latest_parts), len(current_parts))
    latest_parts = latest_parts + (0,) * (width - len(latest_parts))
    current_parts = current_parts + (0,) * (width - len(current_parts))
    if latest_parts != current_parts:
        return latest_parts > current_parts
    return bool(current_rest) and not latest_rest


def upgrade_command(flavor: str | None) -> str:
    """The onramp command the volunteer runs to upgrade — printed, NEVER
    executed by the worker (updates are always the volunteer's election)."""
    return f"curl -sSL {ONRAMP_URL} | bash -s -- --flavor {flavor or 'lean'}"
