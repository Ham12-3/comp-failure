"""FastAPI service: POST /predict, GET /search, GET /health, GET /metrics, and the web page.

Run locally:  uv run uvicorn cfp.serve.api:app --reload
In Docker:    docker compose up   (see Dockerfile / docker-compose.yml)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter, deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.staticfiles import StaticFiles

from cfp.ingest.ch_api import AuthenticationError, CompaniesHouseError
from cfp.serve.live import CompanyNotFoundError, LiveResult, LiveScorer
from cfp.serve.schemas import (
    HealthResponse,
    LatencySummary,
    LinkOut,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
    ReasonOut,
    SearchItem,
    SearchResponse,
)

logger = logging.getLogger(__name__)
TRACKED_PATHS = ("/predict", "/search", "/health", "/metrics")


class RequestMetrics:
    """In-process counters and recent latencies for GET /metrics."""

    def __init__(self, window: int = 1000) -> None:
        self.started = time.monotonic()
        self.by_path: Counter[str] = Counter()
        self.client_errors = 0
        self.server_errors = 0
        self.latencies_ms: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()

    def record(self, path: str, status: int, elapsed_ms: float) -> None:
        with self._lock:
            self.by_path[path if path in TRACKED_PATHS else "other"] += 1
            self.client_errors += 400 <= status < 500
            self.server_errors += status >= 500
            self.latencies_ms.append(elapsed_ms)

    def snapshot(self) -> MetricsResponse:
        with self._lock:
            latencies = np.asarray(self.latencies_ms) if self.latencies_ms else np.zeros(1)
            return MetricsResponse(
                uptime_seconds=round(time.monotonic() - self.started, 1),
                requests_total=sum(self.by_path.values()),
                client_errors_total=self.client_errors,
                server_errors_total=self.server_errors,
                requests_by_path=dict(self.by_path),
                latency_ms=LatencySummary(
                    mean=round(float(latencies.mean()), 1),
                    p50=round(float(np.percentile(latencies, 50)), 1),
                    p95=round(float(np.percentile(latencies, 95)), 1),
                ),
            )


def risk_band(percentile: float) -> str:
    if percentile >= 95:
        return "high"
    if percentile >= 75:
        return "elevated"
    return "typical"


def to_response(result: LiveResult, model_trained_at: str | None) -> PredictResponse:
    response = PredictResponse(
        company_number=result.company_number,
        company_name=result.company_name,
        company_status=result.company_status,
        scored=result.scored,
        message=result.message,
        accounts_filed_on=result.accounts_filed_on,
        balance_sheet_date=result.balance_sheet_date,
        accounts_type=result.accounts_type,
        links=[LinkOut(label=label, url=url) for label, url in result.links],
        warnings=result.warnings,
        model_trained_at=model_trained_at,
    )
    score = result.score
    if score is not None and result.accounts_filed_on is not None:
        filed = result.accounts_filed_on
        response.probability = round(score.probability, 5)
        response.score_percent = round(100 * score.probability, 2)
        response.percentile = round(score.percentile, 1)
        response.relative_risk = round(score.relative_risk, 2)
        response.risk_band = risk_band(score.percentile)  # type: ignore[assignment]
        response.horizon_end = (
            filed.replace(year=filed.year + 1)
            if not (filed.month == 2 and filed.day == 29)
            else filed.replace(year=filed.year + 1, day=28)
        )
        response.reasons = [
            ReasonOut(
                feature=r.feature,
                value=r.value,
                effect="raises" if r.contribution > 0 else "lowers",
                contribution=round(r.contribution, 4),
                text=r.text,
            )
            for r in score.reasons
        ]
    return response


def default_scorer() -> LiveScorer | None:
    """Scorer from settings and models/, or None (with a log line) if something is missing."""
    from cfp.config import get_settings
    from cfp.ingest.ch_api import default_client
    from cfp.serve.model import RiskModel

    settings = get_settings()
    model_dir = Path(os.environ.get("CFP_MODEL_DIR", "models"))
    try:
        model = RiskModel(model_dir)
    except FileNotFoundError:
        logger.error("No trained model in %s: run `python -m cfp.train.train` first", model_dir)
        return None
    except ValueError as exc:  # model and code disagree on the feature list
        logger.error("Model in %s cannot be used: %s", model_dir, exc)
        return None
    if settings.companies_house_api_key is None:
        logger.error("COMPANIES_HOUSE_API_KEY is not set: /predict and /search are disabled")
        return None
    return LiveScorer(
        default_client(),
        model,
        httpx.Client(timeout=60, follow_redirects=True),
        document_dir=settings.api_cache_dir / "documents",
    )


def create_app(
    scorer_factory: Callable[[], LiveScorer | None] = default_scorer,
    frontend_dir: Path | None = None,
) -> FastAPI:
    state: dict[str, LiveScorer | None] = {"scorer": None}
    metrics = RequestMetrics()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        state["scorer"] = scorer_factory()
        yield

    app = FastAPI(
        title="UK Company Failure Predictor",
        description="12-month insolvency risk from Companies House data. Not financial advice.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def record_metrics(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            metrics.record(request.url.path, status, 1000 * (time.perf_counter() - started))

    def scorer() -> LiveScorer:
        current = state["scorer"]
        if current is None:
            raise HTTPException(503, "Service not ready: model or API key missing (see /health)")
        return current

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        current = state["scorer"]
        return HealthResponse(
            status="ok" if current else "degraded",
            model_loaded=current is not None,
            model_trained_at=current.model.trained_at if current else None,
            api_key_configured=bool(os.environ.get("COMPANIES_HOUSE_API_KEY"))
            or current is not None,
        )

    @app.get("/metrics", response_model=MetricsResponse)
    def get_metrics() -> MetricsResponse:
        return metrics.snapshot()

    @app.get("/search", response_model=SearchResponse)
    def search(q: str = Query(min_length=2, max_length=100)) -> SearchResponse:
        try:
            items = scorer().search(q)
        except CompaniesHouseError as exc:
            raise HTTPException(502, "Companies House search failed") from exc
        return SearchResponse(
            items=[
                SearchItem(**{k: item.get(k) for k in SearchItem.model_fields})
                for item in items
                if item.get("company_number")
            ]
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        current = scorer()
        try:
            result = current.score_company(request.company_number)
        except ValueError as exc:
            raise HTTPException(
                422, f"Not a valid company number: {request.company_number}"
            ) from exc
        except CompanyNotFoundError as exc:
            raise HTTPException(404, f"No company with number {exc}") from exc
        except AuthenticationError as exc:
            raise HTTPException(503, "Companies House rejected the API key") from exc
        except (CompaniesHouseError, httpx.HTTPError) as exc:
            logger.exception("Upstream failure scoring %s", request.company_number)
            raise HTTPException(502, "Companies House did not respond as expected") from exc
        logger.info(
            "Scored %s at %s (scored=%s)",
            result.company_number,
            datetime.now(UTC).isoformat(timespec="seconds"),
            result.scored,
        )
        return to_response(result, current.model.trained_at)

    frontend = frontend_dir or Path(os.environ.get("CFP_FRONTEND_DIR", "app"))
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


app = create_app()
