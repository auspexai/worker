"""§9 #46 — update-availability helpers (pure version compare + printed command)."""

from __future__ import annotations

import pytest

from auspexai_worker.updates import is_newer_version, upgrade_command


class TestIsNewerVersion:
    @pytest.mark.parametrize(
        ("latest", "current", "expected"),
        [
            ("0.2.0", "0.1.31", True),
            ("0.1.31", "0.2.0", False),
            ("0.2.0", "0.2.0", False),
            ("1.0.0", "0.99.99", True),
            ("0.2.1", "0.2.0", True),
            # 2-vs-3 component: tuple compare handles ragged lengths
            ("0.2", "0.1.31", True),
            ("0.2.0", "0.2", False),
            # leading v tolerated on either side
            ("v0.2.0", "0.1.31", True),
            ("0.2.0", "v0.1.31", True),
        ],
    )
    def test_numeric_compare(self, latest: str, current: str, expected: bool) -> None:
        assert is_newer_version(latest, current) is expected

    def test_dev_suffix_tiebreak(self) -> None:
        # A dev build between tags predates the clean tagged release — the
        # announcement counts as newer (also enables the banner dogfood demo).
        assert is_newer_version("0.2.0", "0.2.0.dev3+g1234567") is True
        # ...but a clean build is NOT older than its own dev successor's base.
        assert is_newer_version("0.2.0.dev3", "0.2.0") is False

    @pytest.mark.parametrize(
        ("latest", "current"),
        [
            ("garbage", "0.2.0"),
            ("0.2.0", "garbage"),
            ("", "0.2.0"),
            ("0.2.0", ""),
        ],
    )
    def test_unparsable_is_never_newer(self, latest: str, current: str) -> None:
        # Never nag on garbage — a malformed announcement or weird local
        # version silently shows no banner.
        assert is_newer_version(latest, current) is False


class TestUpgradeCommand:
    def test_no_flavor_flag(self) -> None:
        # The installer menu defaults to the recorded flavor; an explicit
        # flag in the copied command would bypass the volunteer's chance
        # to switch at update time.
        cmd = upgrade_command("inference")
        assert "getworker.auspexai.network" in cmd
        assert "--flavor" not in cmd

    def test_none_flavor_same_command(self) -> None:
        assert upgrade_command(None) == upgrade_command("inference")
