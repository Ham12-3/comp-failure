"""Tests for the per-company fetch logic: which endpoints are called, and when."""

import json
from pathlib import Path

import httpx

from cfp.ingest.ch_api import CompaniesHouseClient, TokenBucket
from cfp.ingest.fetch_companies import ENDPOINTS, fetch_company, run, sample_order

FIXTURES = Path(__file__).parent / "fixtures" / "ch_api"


def fixture_body(name: str) -> object:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))["body"]


def make_client(tmp_path: Path, handler: object) -> CompaniesHouseClient:
    return CompaniesHouseClient(
        "test-key",
        tmp_path / "cache.sqlite",
        bucket=TokenBucket(rate=1e9, capacity=1e9),
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
        sleep=lambda _: None,
    )


class FakeApi:
    """Serves the real Carillion / Tesco profiles; empty lists for list endpoints."""

    def __init__(self) -> None:
        self.paths: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.paths.append(path)
        if path == "/company/03782379":
            return httpx.Response(200, json=fixture_body("company_profile_03782379.json"))
        if path == "/company/00445790":
            return httpx.Response(200, json=fixture_body("company_profile_00445790.json"))
        if path.endswith("/insolvency"):
            return httpx.Response(200, json=fixture_body("insolvency_03782379.json"))
        if path.count("/") > 2:
            return httpx.Response(200, json={"items": [], "total_count": 0})
        return httpx.Response(404)


def test_insolvency_only_fetched_when_profile_links_to_it(tmp_path: Path) -> None:
    api = FakeApi()
    with make_client(tmp_path, api) as client:
        assert fetch_company(client, "03782379", ENDPOINTS)  # Carillion: has insolvency link
        assert fetch_company(client, "00445790", ENDPOINTS)  # Tesco: no insolvency link
    assert "/company/03782379/insolvency" in api.paths
    assert "/company/00445790/insolvency" not in api.paths
    assert "/company/00445790/charges" in api.paths  # Tesco links to charges


def test_missing_company_stops_after_profile(tmp_path: Path) -> None:
    api = FakeApi()
    with make_client(tmp_path, api) as client:
        assert not fetch_company(client, "99999999", ENDPOINTS)
    assert api.paths == ["/company/99999999"]


def test_endpoint_selection(tmp_path: Path) -> None:
    api = FakeApi()
    with make_client(tmp_path, api) as client:
        fetch_company(client, "03782379", ["profile", "insolvency"])
    assert api.paths == ["/company/03782379", "/company/03782379/insolvency"]


def test_run_continues_past_failures(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/company/00000001":
            return httpx.Response(400, text="bad")
        return httpx.Response(404)

    with make_client(tmp_path, handler) as client:
        failed = run(client, ["00000001", "00000002"], ENDPOINTS)
    assert failed == ["00000001"]


def test_sample_order_is_stable() -> None:
    numbers = ["00000001", "SC123456", "12345678"]
    assert sorted(numbers, key=sample_order) == sorted(numbers, key=sample_order)
