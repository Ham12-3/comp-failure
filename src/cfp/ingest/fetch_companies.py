"""Fetch Companies House API data for every sampled company into the on-disk cache.

Per company: the profile, then (only when the profile says they exist) insolvency and
charges, plus the full filing history and officer list. Everything goes through
`CompaniesHouseClient`, so it is rate-limited, retried and cached: re-running skips whatever
was already fetched, and an interrupted run loses at most one company.

Companies are processed in the order of their sample hash, so the companies done so far are
always a random subsample of the full sample. That makes partial runs usable.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from cfp.ingest.accounts_download import SAMPLE_SALT
from cfp.ingest.ch_api import (
    AuthenticationError,
    CompaniesHouseClient,
    CompaniesHouseError,
    default_client,
)
from cfp.labels.build_labels import has_insolvency_resource

logger = logging.getLogger(__name__)

ENDPOINTS = ("profile", "insolvency", "filing_history", "officers", "charges")


def sample_order(company_number: str) -> int:
    digest = hashlib.sha256(f"{SAMPLE_SALT}:{company_number}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def sampled_companies(accounts_path: Path) -> list[str]:
    numbers = pl.read_parquet(accounts_path, columns=["company_number"])["company_number"].unique()
    return sorted(numbers.to_list(), key=sample_order)


def has_charges(profile: dict[str, object]) -> bool:
    """`has_charges` is deprecated in the spec in favour of `links.charges`; accept either."""
    links = profile.get("links")
    return bool(profile.get("has_charges")) or (isinstance(links, dict) and "charges" in links)


def fetch_company(client: CompaniesHouseClient, number: str, endpoints: Sequence[str]) -> bool:
    """Fetch one company. Returns False if the company does not exist (profile 404)."""
    profile = client.company_profile(number)
    if profile is None:
        return False
    if "insolvency" in endpoints and has_insolvency_resource(profile):
        client.insolvency(number)
    if "filing_history" in endpoints:
        client.filing_history(number)
    if "officers" in endpoints:
        client.officers(number)
    if "charges" in endpoints and has_charges(profile):
        client.charges(number)
    return True


def run(
    client: CompaniesHouseClient,
    companies: Sequence[str],
    endpoints: Sequence[str],
    progress_every: int = 250,
) -> list[str]:
    """Fetch all companies; returns the ones that failed (to retry later)."""
    failed: list[str] = []
    started = last_time = time.monotonic()
    requests_at_start = last_requests = client.requests_made
    last_i = 0
    for i, number in enumerate(companies, start=1):
        try:
            fetch_company(client, number, endpoints)
        except AuthenticationError:
            raise  # a bad key will not get better
        except CompaniesHouseError as exc:
            logger.error("%s failed: %s", number, exc)
            failed.append(number)
        if i % progress_every == 0 or i == len(companies):
            # Rates over the latest interval only, so time asleep does not skew the estimate.
            now = time.monotonic()
            interval_requests = client.requests_made - last_requests
            rate = interval_requests / (now - last_time) if now > last_time else 0.0
            per_company = interval_requests / (i - last_i)
            remaining = (len(companies) - i) * per_company / rate / 3600 if rate else 0.0
            logger.info(
                "%d/%d companies, %d requests in %.1f h (now %.2f/s, %.1f per company), "
                "~%.1f h left, %d failed",
                i,
                len(companies),
                client.requests_made - requests_at_start,
                (now - started) / 3600,
                rate,
                per_company,
                remaining,
                len(failed),
            )
            last_time, last_requests, last_i = now, client.requests_made, i
    return failed


def drop_active_in_snapshot(companies: Sequence[str], snapshot_path: Path) -> list[str]:
    """Companies *not* active in the Free Company Data Product snapshot, in the same order.

    An active company is not in liquidation, administration or receivership and is not
    dissolved, so its labels need no API call (checked on 1,548 fetched companies: none had a
    failure case that could affect a 2018+ label). See docs/DECISIONS.md 2.10.
    """
    from cfp.ingest.company_data import SNAPSHOT_ACTIVE_STATUSES

    active = set(
        pl.read_parquet(snapshot_path, columns=["company_number", "company_status"])
        .filter(pl.col("company_status").is_in(list(SNAPSHOT_ACTIVE_STATUSES)))["company_number"]
        .to_list()
    )
    return [n for n in companies if n not in active]


def keep_system_awake() -> bool:
    """Ask Windows not to idle-sleep while this process runs (ES_CONTINUOUS |
    ES_SYSTEM_REQUIRED). The request ends when the process exits; no setting is changed.
    No-op elsewhere. A closed lid or a manual sleep still sleeps."""
    if sys.platform == "win32":
        es_continuous, es_system_required = 0x80000000, 0x00000001
        result = ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
        return bool(result)
    else:  # keep the else: mypy only skips platform-dependent code inside if/else branches
        return False


def main() -> None:
    from cfp.config import get_settings

    parser = argparse.ArgumentParser(description="Fetch API data for the sampled companies.")
    parser.add_argument("--endpoints", nargs="+", choices=ENDPOINTS, default=list(ENDPOINTS))
    parser.add_argument("--limit", type=int, default=None, help="Only the first N companies.")
    parser.add_argument(
        "--skip-active",
        action="store_true",
        help="Skip companies active in the company data snapshot (enough for labels).",
    )
    parser.add_argument(
        "--keep-awake", action="store_true", help="Stop Windows idle-sleeping during the run."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.keep_awake:
        logger.info("Keep-awake request %s", "active" if keep_system_awake() else "not available")
    settings = get_settings()
    companies = sampled_companies(settings.interim_dir / "accounts.parquet")
    if args.skip_active:
        companies = drop_active_in_snapshot(
            companies, settings.interim_dir / "company_snapshot.parquet"
        )
    if args.limit:
        companies = companies[: args.limit]
    logger.info("Fetching %s for %d companies", ", ".join(args.endpoints), len(companies))
    with default_client() as client:
        failed = run(client, companies, args.endpoints)
    if failed:
        out = settings.interim_dir / "fetch_failed.txt"
        out.write_text("\n".join(failed) + "\n", encoding="utf-8")
        logger.warning("%d companies failed; listed in %s (re-run to retry)", len(failed), out)


if __name__ == "__main__":
    main()
