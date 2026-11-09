"""Tests for the API: the full live-scoring path against a mocked Companies House.

Mocked responses reuse the real fixtures (document metadata, accounts iXBRL, search, Carillion
profile). The model is a small LightGBM trained on synthetic data, saved like the real one.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from cfp.features.build_features import FEATURES
from cfp.ingest.ch_api import CompaniesHouseClient, TokenBucket
from cfp.serve.api import create_app, risk_band
from cfp.serve.live import LiveScorer
from cfp.serve.model import RiskModel
from cfp.train.calibrate import PlattCalibrator
from cfp.train.data import learn_categories, target, to_matrix
from cfp.train.train import fit_lightgbm
from tests.test_train import synthetic

FIXTURES = Path(__file__).parent / "fixtures"
API = "https://api.company-information.service.gov.uk"
DOC_META = "https://document-api.company-information.service.gov.uk/document/ymrvLyQ2XfWg2LEtgORS4CTjhiVjXzMP25Vv-ATaxCI"
S3_URL = "https://s3.example.test/accounts.xhtml?signature=abc"
IXBRL = (FIXTURES / "accounts" / "Prod224_2510_10851932_20250731.html").read_bytes()


def fixture_body(name: str) -> Any:
    """Response body of a saved fixture (most are wrapped as {status, path, body})."""
    saved = json.loads((FIXTURES / "ch_api" / name).read_text(encoding="utf-8"))
    return saved.get("body", saved)


def profile(number: str, status: str = "active", **extra: Any) -> dict[str, Any]:
    """Shape of real profile responses (see company_profile_*.json)."""
    return {
        "company_number": number,
        "company_name": f"TEST COMPANY {number} LIMITED",
        "company_status": status,
        "date_of_creation": "2017-06-29",
        "type": "ltd",
        "jurisdiction": "england-wales",
        "sic_codes": ["86900"],
        "registered_office_address": {"postal_code": "LE1 1AA"},
        "links": {"self": f"/company/{number}"},
        **extra,
    }


def accounts_item(media: str = "ixbrl") -> dict[str, Any]:
    """The real latest accounts filing of 10851932 (shape from filing_history_*.json)."""
    return {
        "category": "accounts",
        "type": "AA",
        "date": "2025-10-03",
        "description": "accounts-with-accounts-type-micro-entity",
        "description_values": {"made_up_date": "2025-07-31"},
        "links": {"document_metadata": DOC_META if media == "ixbrl" else DOC_META + "-pdf"},
    }


class FakeCompaniesHouse:
    def __init__(self) -> None:
        self.profiles = {
            "10851932": profile("10851932"),
            "00000001": profile("00000001"),  # no accounts filed
            "00000002": profile("00000002"),  # PDF-only accounts
            "03782379": fixture_body("company_profile_03782379.json"),  # Carillion, liquidation
        }
        self.history = {
            "10851932": [accounts_item()],
            "00000001": [],
            "00000002": [accounts_item(media="pdf")],
        }
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url).split("?")[0]
        path = request.url.path
        if url == DOC_META:
            return httpx.Response(200, json=fixture_body("document_metadata_10851932.json"))
        if url == DOC_META + "-pdf":
            body = fixture_body("document_metadata_10851932.json")
            body["resources"] = {"application/pdf": {"content_length": 1000}}
            return httpx.Response(200, json=body)
        if url == f"{DOC_META}/content":
            assert request.headers["Accept"] == "application/xhtml+xml"
            return httpx.Response(302, headers={"location": S3_URL})
        if path == "/search/companies":
            return httpx.Response(200, json=fixture_body("search_companies.json"))
        parts = path.strip("/").split("/")
        number = parts[1] if len(parts) > 1 else ""
        if len(parts) == 2:
            return (
                httpx.Response(200, json=self.profiles[number])
                if number in self.profiles
                else httpx.Response(404)
            )
        if parts[2] == "filing-history":
            items = self.history.get(number, [])
            return httpx.Response(200, json={"items": items, "total_count": len(items)})
        if parts[2] in ("officers", "charges"):
            return httpx.Response(200, json={"items": [], "total_results": 0, "total_count": 0})
        return httpx.Response(404)


class FakeS3:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, content=IXBRL)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small model saved exactly like `cfp.train.train` saves the real one."""
    data = synthetic(n=1500)
    categories = learn_categories(data)
    x = to_matrix(data, categories)
    model = fit_lightgbm(
        x, target(data), {"num_leaves": 4, "min_child_samples": 20, "n_estimators": 40}
    )
    raw = np.asarray(model.predict(x, raw_score=True))
    platt = PlattCalibrator().fit(raw, target(data))
    out = tmp_path_factory.mktemp("models")
    model.booster_.save_model(str(out / "lightgbm.txt"))
    meta = {
        "features": list(FEATURES),
        "categories": categories,
        "calibrator": platt.to_dict(),
        "reference_percentiles": np.quantile(platt.predict(raw), np.linspace(0, 1, 101)).tolist(),
        "train_base_rate": float(target(data).mean()),
        "trained_at": "2026-11-09T00:00:00+00:00",
    }
    (out / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return out


@pytest.fixture
def fakes() -> tuple[FakeCompaniesHouse, FakeS3]:
    return FakeCompaniesHouse(), FakeS3()


@pytest.fixture
def client(
    model_dir: Path, tmp_path: Path, fakes: tuple[FakeCompaniesHouse, FakeS3]
) -> Iterator[TestClient]:
    ch, s3 = fakes

    def factory() -> LiveScorer:
        api = CompaniesHouseClient(
            "secret-test-key",
            tmp_path / "cache.sqlite",
            base_url=API,
            bucket=TokenBucket(rate=1e9, capacity=1e9),
            transport=httpx.MockTransport(ch),
            sleep=lambda _: None,
        )
        documents = httpx.Client(transport=httpx.MockTransport(s3))
        return LiveScorer(api, RiskModel(model_dir), documents, document_dir=tmp_path / "docs")

    app = create_app(factory, frontend_dir=Path(__file__).parents[1] / "app")
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------- /predict
def test_predict_scores_a_real_filing(client: TestClient) -> None:
    response = client.post("/predict", json={"company_number": "10851932"})
    assert response.status_code == 200
    body = response.json()
    assert body["scored"] is True and body["message"] is None
    assert 0 <= body["probability"] <= 1
    assert body["score_percent"] == pytest.approx(100 * body["probability"], abs=0.01)
    assert 0 <= body["percentile"] <= 100
    assert body["risk_band"] in ("typical", "elevated", "high")
    assert body["accounts_filed_on"] == "2025-10-03"
    assert body["balance_sheet_date"] == "2025-07-31"
    assert body["horizon_end"] == "2026-10-03"
    assert body["accounts_type"] == "micro-entity"
    assert len(body["reasons"]) == 5
    assert all(r["effect"] in ("raises", "lowers") and r["text"] for r in body["reasons"])
    assert body["links"][0]["url"].endswith("/company/10851932")
    assert "Not financial or credit advice" in body["disclaimer"]


def test_api_key_is_never_sent_to_the_document_store(
    client: TestClient, fakes: tuple[FakeCompaniesHouse, FakeS3]
) -> None:
    client.post("/predict", json={"company_number": "10851932"})
    ch, s3 = fakes
    assert s3.requests, "the accounts document should have been downloaded"
    assert all("authorization" not in r.headers for r in s3.requests)
    assert all("authorization" in r.headers for r in ch.requests)


def test_downloaded_document_is_cached(
    client: TestClient, fakes: tuple[FakeCompaniesHouse, FakeS3]
) -> None:
    client.post("/predict", json={"company_number": "10851932"})
    client.post("/predict", json={"company_number": "10851932"})
    assert len(fakes[1].requests) == 1


@pytest.mark.parametrize(
    ("number", "message"),
    [
        ("03782379", "in liquidation"),
        ("00000001", "No accounts on file"),
        ("00000002", "not in machine-readable"),
    ],
)
def test_unscorable_companies_get_a_clear_message(
    client: TestClient, number: str, message: str
) -> None:
    body = client.post("/predict", json={"company_number": number}).json()
    assert body["scored"] is False
    assert message in body["message"]
    assert body["probability"] is None and body["reasons"] == []
    assert body["links"]  # the Companies House pages are still linked


def test_unknown_and_invalid_company_numbers(client: TestClient) -> None:
    assert client.post("/predict", json={"company_number": "99999999"}).status_code == 404
    assert client.post("/predict", json={"company_number": "not a number!"}).status_code == 422


# ---------------------------------------------------------------- other endpoints
def test_search_returns_real_fields(client: TestClient) -> None:
    body = client.get("/search", params={"q": "carillion"}).json()
    first = body["items"][0]
    assert first["company_number"] == "03782379"
    assert first["title"] == "CARILLION PLC"
    assert first["company_status"] == "liquidation"
    assert client.get("/search", params={"q": "x"}).status_code == 422  # too short


def test_health_and_metrics(client: TestClient) -> None:
    health = client.get("/health").json()
    assert health == {
        "status": "ok",
        "model_loaded": True,
        "model_trained_at": "2026-11-09T00:00:00+00:00",
        "api_key_configured": True,
    }
    client.post("/predict", json={"company_number": "99999999"})  # a 404
    metrics = client.get("/metrics").json()
    assert metrics["requests_by_path"]["/predict"] == 1
    assert metrics["requests_by_path"]["/health"] == 1
    assert metrics["client_errors_total"] == 1
    assert metrics["latency_ms"]["p95"] >= metrics["latency_ms"]["p50"] >= 0


def test_front_end_is_served(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "UK Company Failure Predictor" in page.text
    assert "not financial or credit advice" in page.text.lower()


def test_degraded_when_model_missing(tmp_path: Path) -> None:
    app = create_app(lambda: None, frontend_dir=tmp_path)
    with TestClient(app) as test_client:
        assert test_client.get("/health").json()["status"] == "degraded"
        assert test_client.post("/predict", json={"company_number": "10851932"}).status_code == 503


def test_risk_bands() -> None:
    assert [risk_band(p) for p in (10, 74.9, 75, 94.9, 95, 100)] == [
        "typical",
        "typical",
        "elevated",
        "elevated",
        "high",
        "high",
    ]


def test_full_history_is_only_fetched_for_scorable_companies(
    client: TestClient, fakes: tuple[FakeCompaniesHouse, FakeS3]
) -> None:
    """Large groups can have thousands of filings: do not page through them for nothing."""
    client.post("/predict", json={"company_number": "00000002"})  # PDF-only accounts
    calls = [r.url for r in fakes[0].requests if r.url.path.endswith("/filing-history")]
    assert calls and all(u.params.get("category") == "accounts" for u in calls)

    client.post("/predict", json={"company_number": "10851932"})  # scorable
    full = [
        r.url
        for r in fakes[0].requests
        if r.url.path == "/company/10851932/filing-history" and "category" not in r.url.params
    ]
    assert full
