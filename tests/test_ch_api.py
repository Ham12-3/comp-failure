"""Tests for the Companies House API client: rate limiting, retries, 429 handling, caching.

No network: every test uses `httpx.MockTransport` and a fake clock, so waits are recorded
instead of slept.
"""

import base64
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from cfp.ingest.ch_api import (
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    AuthenticationError,
    CacheMissError,
    CompaniesHouseClient,
    CompaniesHouseError,
    RetriesExhaustedError,
    TokenBucket,
    cache_key,
    save_fixtures,
)

Handler = Callable[[httpx.Request], httpx.Response]


class FakeClock:
    """Time that only moves when someone sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class Recorder:
    """Mock transport that replays a scripted list of responses (or exceptions)."""

    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def make_client(
    handler: Handler, tmp_path: Path, clock: FakeClock, **kwargs: float
) -> CompaniesHouseClient:
    bucket = TokenBucket.for_limit(clock=clock.time, sleep=clock.sleep)
    return CompaniesHouseClient(
        "test-key",
        tmp_path / "cache.sqlite",
        bucket=bucket,
        transport=httpx.MockTransport(handler),
        sleep=clock.sleep,
        **kwargs,  # type: ignore[arg-type]
    )


PROFILE = {"company_number": "00000006", "company_name": "EXAMPLE LIMITED"}


# ------------------------------------------------------------------ token bucket
def test_bucket_allows_burst_then_waits(clock: FakeClock) -> None:
    bucket = TokenBucket(rate=2.0, capacity=3, clock=clock.time, sleep=clock.sleep)
    for _ in range(3):
        bucket.acquire()
    assert clock.sleeps == []

    bucket.acquire()
    assert clock.sleeps == [pytest.approx(0.5)]  # one token at 2 tokens/s


def test_bucket_never_exceeds_api_limit_in_any_window(clock: FakeClock) -> None:
    bucket = TokenBucket.for_limit(clock=clock.time, sleep=clock.sleep)
    times = []
    for _ in range(3 * RATE_LIMIT_REQUESTS):
        bucket.acquire()
        times.append(clock.now)

    # Slide a 5-minute window over every request and count what falls inside it.
    worst = 0
    j = 0
    for i, start in enumerate(times):
        while j < len(times) and times[j] < start + RATE_LIMIT_WINDOW_SECONDS:
            j += 1
        worst = max(worst, j - i)
    assert worst <= RATE_LIMIT_REQUESTS * 0.95


def test_bucket_drain_forces_wait(clock: FakeClock) -> None:
    bucket = TokenBucket(rate=1.0, capacity=5, clock=clock.time, sleep=clock.sleep)
    bucket.drain()
    bucket.acquire()
    assert clock.sleeps == [pytest.approx(1.0)]


# ------------------------------------------------------------------ requests
def test_uses_basic_auth_with_key_as_username(tmp_path: Path, clock: FakeClock) -> None:
    transport = Recorder(httpx.Response(200, json=PROFILE))
    with make_client(transport, tmp_path, clock) as client:
        client.company_profile("6")

    request = transport.requests[0]
    expected = "Basic " + base64.b64encode(b"test-key:").decode()
    assert request.headers["Authorization"] == expected
    assert request.url.path == "/company/00000006"  # number normalised to 8 characters


def test_successful_response_is_cached_on_disk(tmp_path: Path, clock: FakeClock) -> None:
    transport = Recorder(httpx.Response(200, json=PROFILE))
    with make_client(transport, tmp_path, clock) as client:
        first = client.get_response("/company/00000006")
        second = client.get_response("/company/00000006")
    assert first.data == second.data == PROFILE
    assert (first.from_cache, second.from_cache) == (False, True)
    assert len(transport.requests) == 1

    # A brand-new client reading the same cache file also makes no request.
    transport_2 = Recorder()
    with make_client(transport_2, tmp_path, clock) as client:
        assert client.get("/company/00000006") == PROFILE
    assert transport_2.requests == []


def test_404_returns_none_and_is_cached(tmp_path: Path, clock: FakeClock) -> None:
    transport = Recorder(httpx.Response(404, json={"errors": [{"error": "not found"}]}))
    with make_client(transport, tmp_path, clock) as client:
        assert client.insolvency("00000006") is None
        assert client.insolvency("00000006") is None
    assert len(transport.requests) == 1


# ------------------------------------------------------------------ 429 handling
def test_429_with_retry_after_waits_that_long_then_succeeds(
    tmp_path: Path, clock: FakeClock
) -> None:
    transport = Recorder(
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.Response(200, json=PROFILE),
    )
    with make_client(transport, tmp_path, clock) as client:
        assert client.get("/company/00000006") == PROFILE
    assert 7.0 in clock.sleeps
    assert len(transport.requests) == 2


def test_429_without_retry_after_backs_off_exponentially(tmp_path: Path, clock: FakeClock) -> None:
    transport = Recorder(
        httpx.Response(429),
        httpx.Response(429),
        httpx.Response(429),
        httpx.Response(200, json=PROFILE),
    )
    with make_client(transport, tmp_path, clock, rate_limited_wait_seconds=60.0) as client:
        assert client.get("/company/00000006") == PROFILE
    # Backoff sleeps are 60, 120, 240 (capped at the 300s window); bucket sleeps are tiny.
    long_waits = [s for s in clock.sleeps if s >= 10]
    assert long_waits == [60.0, 120.0, 240.0]


def test_429_backoff_is_capped_at_the_rate_limit_window(tmp_path: Path, clock: FakeClock) -> None:
    transport = Recorder(*[httpx.Response(429)] * 5, httpx.Response(200, json=PROFILE))
    with make_client(transport, tmp_path, clock, rate_limited_wait_seconds=100.0) as client:
        client.get("/company/00000006")
    assert max(clock.sleeps) == RATE_LIMIT_WINDOW_SECONDS


def test_429_drains_the_token_bucket(tmp_path: Path, clock: FakeClock) -> None:
    transport = Recorder(
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(200, json=PROFILE),
    )
    with make_client(transport, tmp_path, clock) as client:
        client.get("/company/00000006")
    # After the 429 the bucket is empty, so the retry had to wait for a fresh token.
    assert any(0 < s < 1 for s in clock.sleeps)


def test_429_is_not_cached(tmp_path: Path, clock: FakeClock) -> None:
    transport = Recorder(*[httpx.Response(429)] * 2)
    with (
        make_client(transport, tmp_path, clock, max_retries=1) as client,
        pytest.raises(RetriesExhaustedError),
    ):
        client.get("/company/00000006")

    transport_2 = Recorder(httpx.Response(200, json=PROFILE))
    with make_client(transport_2, tmp_path, clock) as client:
        assert client.get("/company/00000006") == PROFILE


# ------------------------------------------------------------------ other retries
def test_server_errors_are_retried_with_exponential_backoff(
    tmp_path: Path, clock: FakeClock
) -> None:
    transport = Recorder(
        httpx.Response(502),
        httpx.Response(503),
        httpx.Response(200, json=PROFILE),
    )
    with make_client(transport, tmp_path, clock, backoff_base_seconds=2.0) as client:
        assert client.get("/company/00000006") == PROFILE
    assert [s for s in clock.sleeps if s >= 1] == [2.0, 4.0]


def test_network_errors_are_retried(tmp_path: Path, clock: FakeClock) -> None:
    transport = Recorder(
        httpx.ConnectError("connection refused"),
        httpx.ReadTimeout("timed out"),
        httpx.Response(200, json=PROFILE),
    )
    with make_client(transport, tmp_path, clock) as client:
        assert client.get("/company/00000006") == PROFILE
    assert len(transport.requests) == 3


def test_gives_up_after_max_retries(tmp_path: Path, clock: FakeClock) -> None:
    transport = Recorder(*[httpx.Response(500)] * 4)
    with (
        make_client(transport, tmp_path, clock, max_retries=3) as client,
        pytest.raises(RetriesExhaustedError),
    ):
        client.get("/company/00000006")
    assert len(transport.requests) == 4


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_fail_fast_without_retry(tmp_path: Path, clock: FakeClock, status: int) -> None:
    transport = Recorder(httpx.Response(status, json={"error": "Invalid Authorization"}))
    with make_client(transport, tmp_path, clock) as client, pytest.raises(AuthenticationError):
        client.get("/company/00000006")
    assert len(transport.requests) == 1


def test_unexpected_client_error_raises(tmp_path: Path, clock: FakeClock) -> None:
    transport = Recorder(httpx.Response(400, text="bad request"))
    with make_client(transport, tmp_path, clock) as client, pytest.raises(CompaniesHouseError):
        client.get("/company/00000006")


def test_api_key_never_appears_in_error_messages(tmp_path: Path, clock: FakeClock) -> None:
    transport = Recorder(httpx.Response(401, json={"error": "Invalid Authorization"}))
    with (
        make_client(transport, tmp_path, clock) as client,
        pytest.raises(AuthenticationError) as excinfo,
    ):
        client.get("/company/00000006")
    assert "test-key" not in str(excinfo.value)


# ------------------------------------------------------------------ pagination
def test_paginate_follows_start_index_until_total(tmp_path: Path, clock: FakeClock) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params["start_index"])
        size = int(request.url.params["items_per_page"])
        items = [{"n": i} for i in range(start, min(start + size, 5))]
        return httpx.Response(200, json={"items": items, "total_count": 5})

    with make_client(handler, tmp_path, clock) as client:
        items = client.paginate("/company/00000006/filing-history", items_per_page=2)
    assert [item["n"] for item in items] == [0, 1, 2, 3, 4]


def test_paginate_returns_empty_list_for_404(tmp_path: Path, clock: FakeClock) -> None:
    transport = Recorder(httpx.Response(404))
    with make_client(transport, tmp_path, clock) as client:
        assert client.charges("00000006") == []


def test_cache_key_is_order_independent() -> None:
    a = cache_key("/x", {"b": 2, "a": 1})
    b = cache_key("/x", {"a": 1, "b": 2})
    assert a == b == "/x?a=1&b=2"


# ------------------------------------------------------------------ fixtures
def test_save_fixtures_writes_one_file_per_endpoint(tmp_path: Path, clock: FakeClock) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/insolvency"):
            return httpx.Response(404)
        return httpx.Response(200, json={"items": [], "total_count": 0})

    with make_client(handler, tmp_path, clock) as client:
        written = save_fixtures(client, "6", tmp_path / "fixtures")

    names = sorted(p.name for p in written)
    assert names == [
        "charges_00000006.json",
        "company_profile_00000006.json",
        "filing_history_00000006.json",
        "insolvency_00000006.json",
        "officers_00000006.json",
        "psc_00000006.json",
    ]
    insolvency = json.loads((tmp_path / "fixtures" / "insolvency_00000006.json").read_text())
    assert insolvency == {"status": 404, "path": "/company/00000006/insolvency", "body": None}


def test_429_waits_until_rate_limit_reset_header(tmp_path: Path, clock: FakeClock) -> None:
    # Real responses carry X-Ratelimit-Reset as epoch seconds.
    transport = Recorder(
        httpx.Response(429, headers={"X-Ratelimit-Reset": "1000042"}),
        httpx.Response(200, json=PROFILE),
    )
    bucket = TokenBucket.for_limit(clock=clock.time, sleep=clock.sleep)
    with CompaniesHouseClient(
        "test-key",
        tmp_path / "cache.sqlite",
        bucket=bucket,
        transport=httpx.MockTransport(transport),
        sleep=clock.sleep,
        wall_clock=lambda: 1_000_000.0,
    ) as client:
        assert client.get("/company/00000006") == PROFILE
    assert 43.0 in clock.sleeps  # 42 s to the reset, plus 1


def test_cache_only_client_never_calls_the_network(tmp_path: Path, clock: FakeClock) -> None:
    transport = Recorder(httpx.Response(200, json=PROFILE))
    with make_client(transport, tmp_path, clock) as client:
        client.get("/company/00000006")

    offline = CompaniesHouseClient("", tmp_path / "cache.sqlite", cache_only=True)
    assert offline.get("/company/00000006") == PROFILE
    with pytest.raises(CacheMissError):
        offline.get("/company/00000007")
    offline.close()
