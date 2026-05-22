"""Tests for the RFC 8628 Device Authorization Flow client.

`httpx.MockTransport` stands in for GitHub's Device Flow endpoints; injected
`clock` and `sleep` callables remove real-time waits.
"""

from __future__ import annotations

import httpx
import pytest

from auspexai_worker.oauth import (
    AccessDeniedError,
    DeviceCode,
    DeviceFlowError,
    ExpiredTokenError,
    start_device_flow,
)
from auspexai_worker.oauth.device_flow import (
    GITHUB_DEVICE_CODE_URL,
    GITHUB_TOKEN_URL,
    run_device_flow,
)


def _device_code_response() -> dict[str, object]:
    return {
        "device_code": "DEV-abc",
        "user_code": "WXYZ-1234",
        "verification_uri": "https://github.com/login/device",
        "expires_in": 900,
        "interval": 5,
    }


class _FakeClock:
    """Monotonic clock that advances only on `tick(seconds)`."""

    def __init__(self) -> None:
        self._now = 0.0

    def now(self) -> float:
        return self._now

    def tick(self, seconds: float) -> None:
        self._now += seconds


def _make_handler(steps: list[tuple[str, httpx.Response]]):
    """Build a MockTransport handler that returns each scripted response in order.

    `steps` is a list of (expected_url, response) pairs. Any extra calls fail
    the test loudly.
    """
    counter = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        i = counter["i"]
        if i >= len(steps):
            raise AssertionError(f"unexpected extra request: {request.method} {request.url}")
        expected_url, response = steps[i]
        assert str(request.url) == expected_url, (
            f"step {i}: expected {expected_url!r}, got {request.url!s}"
        )
        counter["i"] += 1
        return response

    return handler, counter


class TestStartDeviceFlow:
    def test_returns_parsed_device_code(self) -> None:
        handler, _ = _make_handler(
            [(GITHUB_DEVICE_CODE_URL, httpx.Response(200, json=_device_code_response()))]
        )
        code = start_device_flow(transport=httpx.MockTransport(handler))
        assert code.device_code == "DEV-abc"
        assert code.user_code == "WXYZ-1234"
        assert code.verification_uri == "https://github.com/login/device"
        assert code.expires_in == 900
        assert code.interval == 5

    def test_http_error_status_raises_device_flow_error(self) -> None:
        handler, _ = _make_handler([(GITHUB_DEVICE_CODE_URL, httpx.Response(500, text="oops"))])
        with pytest.raises(DeviceFlowError, match="HTTP 500"):
            start_device_flow(transport=httpx.MockTransport(handler))

    def test_missing_fields_raise_device_flow_error(self) -> None:
        handler, _ = _make_handler(
            [(GITHUB_DEVICE_CODE_URL, httpx.Response(200, json={"foo": "bar"}))]
        )
        with pytest.raises(DeviceFlowError, match="missing/invalid"):
            start_device_flow(transport=httpx.MockTransport(handler))


class TestRunDeviceFlow:
    def test_happy_path_one_poll(self) -> None:
        clock = _FakeClock()
        sleep_log: list[float] = []
        handler, _ = _make_handler(
            [
                (GITHUB_DEVICE_CODE_URL, httpx.Response(200, json=_device_code_response())),
                (
                    GITHUB_TOKEN_URL,
                    httpx.Response(200, json={"access_token": "gho_abc", "token_type": "bearer"}),
                ),
            ]
        )
        codes_seen: list[DeviceCode] = []
        token = run_device_flow(
            on_code=codes_seen.append,
            transport=httpx.MockTransport(handler),
            clock=clock.now,
            sleep=sleep_log.append,
        )
        assert token == "gho_abc"
        assert len(codes_seen) == 1
        assert codes_seen[0].user_code == "WXYZ-1234"
        # Slept once at the polling interval.
        assert sleep_log == [5]

    def test_authorization_pending_then_success(self) -> None:
        clock = _FakeClock()
        sleep_log: list[float] = []
        # First poll says pending; second succeeds.
        handler, _ = _make_handler(
            [
                (GITHUB_DEVICE_CODE_URL, httpx.Response(200, json=_device_code_response())),
                (GITHUB_TOKEN_URL, httpx.Response(400, json={"error": "authorization_pending"})),
                (GITHUB_TOKEN_URL, httpx.Response(200, json={"access_token": "gho_xyz"})),
            ]
        )
        token = run_device_flow(
            on_code=lambda c: None,
            transport=httpx.MockTransport(handler),
            clock=clock.now,
            sleep=sleep_log.append,
        )
        assert token == "gho_xyz"
        assert sleep_log == [5, 5]

    def test_slow_down_bumps_interval(self) -> None:
        clock = _FakeClock()
        sleep_log: list[float] = []
        handler, _ = _make_handler(
            [
                (GITHUB_DEVICE_CODE_URL, httpx.Response(200, json=_device_code_response())),
                (GITHUB_TOKEN_URL, httpx.Response(400, json={"error": "slow_down"})),
                (GITHUB_TOKEN_URL, httpx.Response(200, json={"access_token": "gho_slow"})),
            ]
        )
        token = run_device_flow(
            on_code=lambda c: None,
            transport=httpx.MockTransport(handler),
            clock=clock.now,
            sleep=sleep_log.append,
        )
        assert token == "gho_slow"
        # Interval started at 5; slow_down bumps by 5; so first sleep is 5, second is 10.
        assert sleep_log == [5, 10]

    def test_expired_token_raises(self) -> None:
        clock = _FakeClock()
        handler, _ = _make_handler(
            [
                (GITHUB_DEVICE_CODE_URL, httpx.Response(200, json=_device_code_response())),
                (GITHUB_TOKEN_URL, httpx.Response(400, json={"error": "expired_token"})),
            ]
        )
        with pytest.raises(ExpiredTokenError):
            run_device_flow(
                on_code=lambda c: None,
                transport=httpx.MockTransport(handler),
                clock=clock.now,
                sleep=lambda s: None,
            )

    def test_access_denied_raises(self) -> None:
        clock = _FakeClock()
        handler, _ = _make_handler(
            [
                (GITHUB_DEVICE_CODE_URL, httpx.Response(200, json=_device_code_response())),
                (GITHUB_TOKEN_URL, httpx.Response(400, json={"error": "access_denied"})),
            ]
        )
        with pytest.raises(AccessDeniedError):
            run_device_flow(
                on_code=lambda c: None,
                transport=httpx.MockTransport(handler),
                clock=clock.now,
                sleep=lambda s: None,
            )

    def test_deadline_exceeded_raises_expired(self) -> None:
        """If the device_code's expires_in passes without success, raise ExpiredTokenError."""
        clock = _FakeClock()
        handler, _ = _make_handler(
            [
                (GITHUB_DEVICE_CODE_URL, httpx.Response(200, json=_device_code_response())),
                (GITHUB_TOKEN_URL, httpx.Response(400, json={"error": "authorization_pending"})),
            ]
        )

        def advance_past_deadline(seconds: float) -> None:
            # First sleep — push the clock past the 900s deadline so the loop terminates.
            clock.tick(seconds + 1000)

        with pytest.raises(ExpiredTokenError, match="expired before authorization"):
            run_device_flow(
                on_code=lambda c: None,
                transport=httpx.MockTransport(handler),
                clock=clock.now,
                sleep=advance_past_deadline,
            )

    def test_unknown_error_code_raises_device_flow_error(self) -> None:
        clock = _FakeClock()
        handler, _ = _make_handler(
            [
                (GITHUB_DEVICE_CODE_URL, httpx.Response(200, json=_device_code_response())),
                (GITHUB_TOKEN_URL, httpx.Response(400, json={"error": "weird_error"})),
            ]
        )
        with pytest.raises(DeviceFlowError, match="weird_error"):
            run_device_flow(
                on_code=lambda c: None,
                transport=httpx.MockTransport(handler),
                clock=clock.now,
                sleep=lambda s: None,
            )
