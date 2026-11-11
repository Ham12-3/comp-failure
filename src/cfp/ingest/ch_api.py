"""Companies House Public Data API client: rate-limited, retrying, and cached on disk.

* **Rate limit.** The documented limit is 600 requests per 5-minute window per API key
  (https://developer-specs.company-information.service.gov.uk/guides/rateLimiting). A token
  bucket keeps us below it with a safety margin.
* **Retries.** HTTP 429 drains the bucket and waits: `Retry-After` if sent, else until
  `X-Ratelimit-Reset` (epoch seconds; seen on real responses alongside `X-Ratelimit-Limit`,
  `X-Ratelimit-Remain` and `X-Ratelimit-Window`, though not in the docs), else exponential
  backoff. The docs say a block lasts until the 5-minute window ends. 5xx responses and
  network errors are retried with exponential backoff. 401/403 fail immediately: retrying a
  bad key is pointless.
* **Cache.** Every 200 and 404 response is stored in SQLite, keyed by path + query string, with
  the time it was fetched. Nothing is fetched twice. 404 is cached because it is a real answer:
  e.g. a company with no insolvency history has no insolvency resource.

The API returns the company's *current* state. Callers that build features must filter events
by their own dates (`event_date <= prediction_date`); the `fetched_at` timestamp records when
each snapshot was taken.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self
from urllib.parse import urlencode

import httpx

from cfp.company_number import normalise_company_number

logger = logging.getLogger(__name__)

BASE_URL = "https://api.company-information.service.gov.uk"
RATE_LIMIT_REQUESTS = 600
RATE_LIMIT_WINDOW_SECONDS = 300.0
USER_AGENT = "cfp-company-failure-predictor/0.1"

Json = dict[str, Any]
Clock = Callable[[], float]
Sleep = Callable[[float], None]


class CompaniesHouseError(RuntimeError):
    """Base class for API client errors."""


class AuthenticationError(CompaniesHouseError):
    """The API rejected the key (HTTP 401/403)."""


class RetriesExhaustedError(CompaniesHouseError):
    """A request still failed after the maximum number of retries."""


class CacheMissError(CompaniesHouseError):
    """A cache-only client was asked for something that was never fetched."""


class TokenBucket:
    """Token bucket: holds at most `capacity` tokens and refills at `rate` tokens per second.

    Over any window of W seconds it allows at most `capacity + rate * W` requests. Use
    `for_limit` to pick values that keep that below an API limit.
    """

    def __init__(
        self,
        rate: float,
        capacity: float,
        *,
        clock: Clock = time.monotonic,
        sleep: Sleep = time.sleep,
    ) -> None:
        if rate <= 0 or capacity < 1:
            raise ValueError("rate must be > 0 and capacity >= 1")
        self.rate = rate
        self.capacity = capacity
        self._clock = clock
        self._sleep = sleep
        self._tokens = capacity
        self._last = clock()

    @classmethod
    def for_limit(
        cls,
        requests: int = RATE_LIMIT_REQUESTS,
        window_seconds: float = RATE_LIMIT_WINDOW_SECONDS,
        *,
        safety: float = 0.95,
        capacity: float = 10,
        clock: Clock = time.monotonic,
        sleep: Sleep = time.sleep,
    ) -> TokenBucket:
        """Bucket that never exceeds `safety * requests` in any `window_seconds` window.

        With the defaults: rate = (570 - 10) / 300 = 1.87 requests/s, burst of 10.
        """
        rate = (requests * safety - capacity) / window_seconds
        return cls(rate, capacity, clock=clock, sleep=sleep)

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    def acquire(self) -> None:
        """Take one token, sleeping until one is available."""
        self._refill()
        while self._tokens < 1:
            # The 1 ms floor avoids spinning on floating-point leftovers.
            self._sleep(max((1 - self._tokens) / self.rate, 0.001))
            self._refill()
        self._tokens -= 1

    def drain(self) -> None:
        """Empty the bucket, e.g. after the server said we are over the limit."""
        self._tokens = 0
        self._last = self._clock()


@dataclass(frozen=True)
class CachedResponse:
    status: int
    data: Json | None
    fetched_at: datetime
    from_cache: bool


class ResponseCache:
    """SQLite store of API responses. One file, safe to interrupt, easy to inspect."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # The API server shares one client across worker threads, serialised by a lock
        # (cfp.serve.live), so the connection may be used from a thread other than its creator.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            " key TEXT PRIMARY KEY,"
            " status INTEGER NOT NULL,"
            " body TEXT,"
            " fetched_at TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> CachedResponse | None:
        row = self._conn.execute(
            "SELECT status, body, fetched_at FROM responses WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        status, body, fetched_at = row
        data = json.loads(body) if body is not None else None
        return CachedResponse(status, data, datetime.fromisoformat(fetched_at), from_cache=True)

    def put(self, key: str, status: int, data: Json | None, fetched_at: datetime) -> None:
        body = json.dumps(data) if data is not None else None
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (key, status, body, fetched_at) VALUES (?, ?, ?, ?)",
            (key, status, body, fetched_at.isoformat()),
        )
        self._conn.commit()

    def __len__(self) -> int:
        (count,) = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()
        return int(count)

    def close(self) -> None:
        self._conn.close()


def cache_key(path: str, params: Mapping[str, str | int] | None = None) -> str:
    """Stable key for a request: the path plus its sorted query string."""
    if not params:
        return path
    return f"{path}?{urlencode(sorted((k, str(v)) for k, v in params.items()))}"


class CompaniesHouseClient:
    """Thin client over the endpoints this project needs. Not thread-safe; use one per thread."""

    def __init__(
        self,
        api_key: str,
        cache_path: Path,
        *,
        base_url: str = BASE_URL,
        bucket: TokenBucket | None = None,
        max_retries: int = 6,
        backoff_base_seconds: float = 2.0,
        backoff_max_seconds: float = 120.0,
        rate_limited_wait_seconds: float = 60.0,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Sleep = time.sleep,
        wall_clock: Clock = time.time,
        cache_only: bool = False,
    ) -> None:
        if not api_key and not cache_only:
            raise ValueError("api_key is empty")
        self._http = httpx.Client(
            base_url=base_url,
            auth=httpx.BasicAuth(api_key, ""),  # key as username, blank password (per docs)
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=timeout_seconds,
            transport=transport,
        )
        self._cache = ResponseCache(cache_path)
        self._bucket = bucket or TokenBucket.for_limit(sleep=sleep)
        self._sleep = sleep
        self._wall_clock = wall_clock
        self.cache_only = cache_only
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self.rate_limited_wait_seconds = rate_limited_wait_seconds
        self.requests_made = 0

    # ------------------------------------------------------------------ core
    def get_response(
        self, path: str, params: Mapping[str, str | int] | None = None
    ) -> CachedResponse:
        """GET `path`, from the cache if possible. Returns status 200 or 404; raises otherwise."""
        key = cache_key(path, params)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if self.cache_only:
            raise CacheMissError(key)

        for attempt in range(self.max_retries + 1):
            self._bucket.acquire()
            self.requests_made += 1
            try:
                response = self._http.get(path, params=dict(params) if params else None)
            except httpx.TransportError as exc:
                wait = self._backoff(attempt)
                logger.warning("Network error on %s (%s); retrying in %.0fs", path, exc, wait)
                self._sleep(wait)
                continue

            status = response.status_code
            if status in (200, 404):
                data: Json | None = response.json() if status == 200 else None
                fetched_at = datetime.now(UTC)
                self._cache.put(key, status, data, fetched_at)
                return CachedResponse(status, data, fetched_at, from_cache=False)
            if status in (401, 403):
                raise AuthenticationError(
                    f"HTTP {status} from Companies House for {path}. Check COMPANIES_HOUSE_API_KEY "
                    "is a REST key for the Live environment."
                )
            if status == 429:
                self._bucket.drain()
                retry_after = self._retry_after(response)
                if retry_after is None:
                    retry_after = self._until_reset(response)
                if retry_after is not None:
                    wait = retry_after
                else:
                    wait = min(
                        self.rate_limited_wait_seconds * 2**attempt, RATE_LIMIT_WINDOW_SECONDS
                    )
                logger.warning("Rate limited (429) on %s; waiting %.0fs", path, wait)
                self._sleep(wait)
                continue
            if status >= 500:
                wait = self._backoff(attempt)
                logger.warning("HTTP %d on %s; retrying in %.0fs", status, path, wait)
                self._sleep(wait)
                continue
            raise CompaniesHouseError(f"Unexpected HTTP {status} for {path}: {response.text[:200]}")

        raise RetriesExhaustedError(f"Gave up on {path} after {self.max_retries + 1} attempts")

    def get(self, path: str, params: Mapping[str, str | int] | None = None) -> Json | None:
        """GET `path` and return the JSON body, or None for 404."""
        return self.get_response(path, params).data

    def paginate(self, path: str, *, items_per_page: int = 100, max_pages: int = 100) -> list[Json]:
        """Collect `items` across all pages of a list endpoint. Returns [] for 404."""
        items: list[Json] = []
        for _ in range(max_pages):
            page = self.get(path, {"items_per_page": items_per_page, "start_index": len(items)})
            if page is None:
                break
            page_items = page.get("items") or []
            items.extend(page_items)
            total = page.get("total_count", page.get("total_results", 0))
            if not page_items or len(items) >= int(total or 0):
                break
        else:
            logger.warning("Stopped paginating %s after %d pages", path, max_pages)
        return items

    def _backoff(self, attempt: int) -> float:
        return float(min(self.backoff_base_seconds * 2**attempt, self.backoff_max_seconds))

    def _until_reset(self, response: httpx.Response) -> float | None:
        """Seconds until `X-Ratelimit-Reset` (epoch seconds), plus 1, capped at the window."""
        value = response.headers.get("X-Ratelimit-Reset")
        if value is None:
            return None
        try:
            wait = float(value) - self._wall_clock() + 1
        except ValueError:
            return None
        return min(max(wait, 1.0), RATE_LIMIT_WINDOW_SECONDS)

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        value = response.headers.get("Retry-After")
        if value is None:
            return None
        try:
            return max(float(value), 0.0)
        except ValueError:
            return None  # HTTP-date form: fall back to our own backoff

    # ------------------------------------------------------------- endpoints
    def company_profile(self, company_number: str) -> Json | None:
        return self.get(f"/company/{normalise_company_number(company_number)}")

    def insolvency(self, company_number: str) -> Json | None:
        return self.get(f"/company/{normalise_company_number(company_number)}/insolvency")

    def officers(self, company_number: str) -> list[Json]:
        return self.paginate(f"/company/{normalise_company_number(company_number)}/officers")

    def filing_history(self, company_number: str) -> list[Json]:
        return self.paginate(f"/company/{normalise_company_number(company_number)}/filing-history")

    def recent_filings(self, company_number: str, category: str, count: int = 25) -> list[Json]:
        """The newest `count` filings in one category: a single request, unlike filing_history."""
        number = normalise_company_number(company_number)
        page = self.get(
            f"/company/{number}/filing-history", {"category": category, "items_per_page": count}
        )
        return list((page or {}).get("items") or [])

    def charges(self, company_number: str) -> list[Json]:
        return self.paginate(f"/company/{normalise_company_number(company_number)}/charges")

    def persons_with_significant_control(self, company_number: str) -> list[Json]:
        number = normalise_company_number(company_number)
        return self.paginate(f"/company/{number}/persons-with-significant-control")

    def search_companies(self, query: str, items_per_page: int = 20) -> Json | None:
        return self.get("/search/companies", {"q": query, "items_per_page": items_per_page})

    # ------------------------------------------------------------- documents
    def document_metadata(self, url: str) -> Json | None:
        """Document API metadata (absolute URL from a filing's `links.document_metadata`)."""
        return self.get(url)

    def document_download_url(self, content_url: str, media_type: str) -> str | None:
        """Where to download a document in `media_type` (a short-lived S3 address), or None if
        that format does not exist. The API answers with a redirect; it is *not* followed here,
        so the API key is never sent to the storage host."""
        if self.cache_only:
            raise CacheMissError(content_url)
        self._bucket.acquire()
        self.requests_made += 1
        response = self._http.get(
            content_url, headers={"Accept": media_type}, follow_redirects=False
        )
        if response.is_redirect:
            location = response.headers.get("location")
            return str(location) if location else None
        if response.status_code in (404, 406):
            return None
        if response.status_code in (401, 403):
            raise AuthenticationError(f"HTTP {response.status_code} for document {content_url}")
        raise CompaniesHouseError(f"Unexpected HTTP {response.status_code} for {content_url}")

    # ------------------------------------------------------------- lifecycle
    def close(self) -> None:
        self._http.close()
        self._cache.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def default_client(*, cache_only: bool = False) -> CompaniesHouseClient:
    """Client configured from `cfp.config` (API key from env, cache under data/raw).

    `cache_only=True` never touches the network (no key needed): use it to build tables from
    responses that were already fetched.
    """
    from cfp.config import get_settings

    settings = get_settings()
    key = "" if cache_only else settings.require_api_key()
    return CompaniesHouseClient(
        key, settings.api_cache_dir / "responses.sqlite", cache_only=cache_only
    )


# ------------------------------------------------------------------ fixtures
FIXTURE_DIR = Path("tests/fixtures/ch_api")


def save_fixtures(client: CompaniesHouseClient, company_number: str, out_dir: Path) -> list[Path]:
    """Save one real response per endpoint, so tests and code use real field names.

    List endpoints are saved as a single page (5 items) to keep fixtures small.
    """
    number = normalise_company_number(company_number)
    requests: dict[str, tuple[str, dict[str, str | int] | None]] = {
        "company_profile": (f"/company/{number}", None),
        "insolvency": (f"/company/{number}/insolvency", None),
        "officers": (f"/company/{number}/officers", {"items_per_page": 5}),
        "filing_history": (f"/company/{number}/filing-history", {"items_per_page": 5}),
        "charges": (f"/company/{number}/charges", {"items_per_page": 5}),
        "psc": (f"/company/{number}/persons-with-significant-control", {"items_per_page": 5}),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (path, params) in requests.items():
        response = client.get_response(path, params)
        target = out_dir / f"{name}_{number}.json"
        payload = {
            "status": response.status,
            "path": cache_key(path, params),
            "body": response.data,
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Companies House API utilities.")
    sub = parser.add_subparsers(dest="command", required=True)
    fx = sub.add_parser("save-fixtures", help="Save one real response per endpoint.")
    fx.add_argument("company_numbers", nargs="+")
    fx.add_argument("--search", default=None, help="Also save a /search/companies response.")
    fx.add_argument("--out", type=Path, default=FIXTURE_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with default_client() as client:
        for number in args.company_numbers:
            for path in save_fixtures(client, number, args.out):
                print(f"wrote {path}")
        if args.search:
            path = args.out / "search_companies.json"
            body = client.search_companies(args.search, items_per_page=5)
            path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
