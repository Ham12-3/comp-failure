"""Score any company live from Companies House.

1. Profile: does the company exist, and is it still trading? A company already in
   liquidation, administration or dissolved gets a clear message instead of a score.
2. Latest accounts filing from filing history, and its iXBRL from the Document API. Accounts
   available only as a scanned/PDF document cannot be read, so they get a message too.
3. The same parser and feature code as training, with the prediction date set to the filing
   date of those accounts: the score is "risk of insolvency within 12 months of that filing".
4. The trained model: calibrated probability, percentile and top 5 reasons.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from cfp.company_number import normalise_company_number
from cfp.features.build_features import api_data_from_payloads, build_feature_table
from cfp.ingest.accounts_parse import SCHEMA, parse_live_document
from cfp.ingest.ch_api import CompaniesHouseClient
from cfp.ingest.fetch_companies import has_charges
from cfp.labels.build_labels import has_insolvency_resource
from cfp.serve.model import RiskModel, Score

FIND_AND_UPDATE = "https://find-and-update.company-information.service.gov.uk/company"
IXBRL = "application/xhtml+xml"
SCORABLE_STATUSES = frozenset({"active"})
_ALREADY = "the insolvency the model predicts has already happened."
STATUS_MESSAGES = {
    "liquidation": f"This company is in liquidation: {_ALREADY}",
    "administration": f"This company is in administration: {_ALREADY}",
    "receivership": f"A receiver has been appointed: {_ALREADY}",
    "insolvency-proceedings": "This company is in insolvency proceedings already.",
    "voluntary-arrangement": (
        "This company is in a company voluntary arrangement (CVA) with its creditors, "
        "a situation the model was not trained to score."
    ),
    "dissolved": "This company has been dissolved and no longer exists.",
    "converted-closed": "This company has been converted or closed.",
}


class CompanyNotFoundError(LookupError):
    pass


@dataclass
class LiveResult:
    company_number: str
    company_name: str | None
    company_status: str | None
    scored: bool
    message: str | None = None
    score: Score | None = None
    accounts_filed_on: date | None = None
    balance_sheet_date: date | None = None
    accounts_type: str | None = None
    links: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def latest_accounts_filing(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    accounts = [
        item
        for item in history
        if item.get("category") == "accounts"
        and item.get("type") == "AA"
        and _date(item.get("date"))
    ]
    return max(accounts, key=lambda item: str(item["date"])) if accounts else None


def company_links(number: str, profile: dict[str, Any]) -> list[tuple[str, str]]:
    base = f"{FIND_AND_UPDATE}/{number}"
    links = [
        ("Company overview", base),
        ("Filing history (accounts)", f"{base}/filing-history"),
        ("People (directors)", f"{base}/officers"),
    ]
    if has_charges(profile):
        links.append(("Charges", f"{base}/charges"))
    if has_insolvency_resource(profile):
        links.append(("Insolvency", f"{base}/insolvency"))
    return links


class LiveScorer:
    """Not safe for parallel use on its own: calls are serialised with a lock, which also
    keeps the API client (and its SQLite cache) on one thread at a time."""

    def __init__(
        self,
        client: CompaniesHouseClient,
        model: RiskModel,
        documents: httpx.Client,
        document_dir: Path | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.documents = documents  # plain client, no API key: downloads from S3
        self.document_dir = document_dir
        self.lock = threading.Lock()

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        with self.lock:
            body = self.client.search_companies(query, items_per_page=limit) or {}
        return list(body.get("items") or [])

    def score_company(self, company_number: str) -> LiveResult:
        number = normalise_company_number(company_number)
        with self.lock:
            return self._score(number)

    def _score(self, number: str) -> LiveResult:
        profile = self.client.company_profile(number)
        if profile is None:
            raise CompanyNotFoundError(number)
        status = profile.get("company_status")
        result = LiveResult(
            company_number=number,
            company_name=profile.get("company_name"),
            company_status=status,
            scored=False,
            links=company_links(number, profile),
        )
        if status not in SCORABLE_STATUSES:
            result.message = STATUS_MESSAGES.get(
                str(status),
                f"Company status is '{status}': the model only scores active companies.",
            )
            return result

        # One request for the latest accounts; the full history (which can run to thousands
        # of items for large groups) is fetched only once we know the company can be scored.
        latest = latest_accounts_filing(self.client.recent_filings(number, "accounts"))
        if latest is None:
            result.message = "No accounts on file yet, so there is nothing to score."
            return result
        filed = _date(latest["date"])
        made_up = _date((latest.get("description_values") or {}).get("made_up_date"))
        assert filed is not None
        result.accounts_filed_on, result.balance_sheet_date = filed, made_up
        result.accounts_type = (
            str(latest.get("description") or "").replace("accounts-with-accounts-type-", "") or None
        )
        if made_up is None:
            result.message = (
                "The latest accounts filing has no period end date, so it cannot be scored."
            )
            return result

        document = self._download_ixbrl(latest)
        if document is None:
            result.message = (
                "The latest accounts are only available as a PDF or scanned document, not in "
                "machine-readable (iXBRL) form, so they cannot be scored. Large groups and "
                "paper filers often file this way."
            )
            return result
        row = parse_live_document(document, number, made_up, filed, source="document-api")
        if row["parse_error"]:
            result.message = "The latest accounts could not be read."
            return result

        payload = {
            "profile": profile,
            "filing_history": self.client.filing_history(number),
            "officers": self.client.officers(number),
            "charges": self.client.charges(number) if has_charges(profile) else [],
        }
        accounts = pl.DataFrame([row], schema=SCHEMA).with_columns(
            pl.lit(filed).alias("prediction_date")
        )
        features = build_feature_table(accounts, api_data_from_payloads({number: payload}))
        result.score = self.model.score(features)
        result.scored = True

        age_days = (datetime.now(UTC).date() - filed).days
        if age_days > 400:
            result.warnings.append(
                f"The latest accounts were filed {age_days // 30} months ago; the score describes "
                "the company as it was then."
            )
        next_accounts = (profile.get("accounts") or {}).get("next_accounts") or {}
        if next_accounts.get("overdue"):
            result.warnings.append("Companies House shows this company's next accounts as overdue.")
        return result

    def _download_ixbrl(self, filing: dict[str, Any]) -> bytes | None:
        meta_url = (filing.get("links") or {}).get("document_metadata")
        if not meta_url:
            return None
        doc_id = str(meta_url).rstrip("/").rsplit("/", 1)[-1]
        cached = self.document_dir / f"{doc_id}.xhtml" if self.document_dir else None
        if cached is not None and cached.exists():
            return cached.read_bytes()
        meta = self.client.document_metadata(str(meta_url))
        if not meta or IXBRL not in (meta.get("resources") or {}):
            return None
        location = self.client.document_download_url(meta["links"]["document"], IXBRL)
        if location is None:
            return None
        response = self.documents.get(location)
        response.raise_for_status()
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(response.content)
        return response.content
