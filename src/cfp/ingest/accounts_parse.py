"""Parse sampled accounts files (iXBRL `.html` and XBRL `.xml`) into one Parquet row per filing.

For every filing we keep:

* identifiers: company number and balance sheet date (from the file name), the month the
  accounts were filed (from the archive name) and where the file came from;
* report metadata tagged in the accounts: accounting standard (micro-entity, small, FRS 102...),
  accounts type, audit status, dormant flag, trading status, legal form, authorisation date;
* key financial values for the current period **and the prior-year comparatives** that UK
  accounts include, so year-on-year change is available from a single filing.

Values are extracted as tagged, without deriving anything. A concept is read only when its
context carries exactly an allowed set of dimension members: none for most concepts, and for
amounts split by maturity (creditors, bank borrowings) one of the two conventions filers use:

* `MaturitiesOrExpirationPeriodsDimension` = `WithinOneYear` / `AfterOneYear`, or
* `FinancialInstrumentCurrentNon-currentDimension` = `CurrentFinancialInstruments` /
  `Non-currentFinancialInstruments`, or both together.

A few filers tag balance-sheet creditors with no dimension at all; that case is kept in its
own column (`creditors_no_maturity`) instead of being guessed. Derivations (current
liabilities, ratios, sign repair...) belong in the features step.

Concept names and dimension conventions were chosen from tag counts over real 2019 and 2025
filings (FRS 102 / FRS 105 taxonomies), not from memory.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import zipfile
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
from ixbrlparse import IXBRL

from cfp.ingest.accounts_download import filing_month_from_name, parse_member_name

logger = logging.getLogger(__name__)


Members = frozenset[str]
NO_DIMENSIONS: Members = frozenset()
WITHIN_ONE_YEAR: tuple[Members, ...] = (
    frozenset({"WithinOneYear"}),
    frozenset({"CurrentFinancialInstruments"}),
    frozenset({"CurrentFinancialInstruments", "WithinOneYear"}),
)
AFTER_ONE_YEAR: tuple[Members, ...] = (
    frozenset({"AfterOneYear"}),
    frozenset({"Non-currentFinancialInstruments"}),
    frozenset({"Non-currentFinancialInstruments", "AfterOneYear"}),
)


@dataclass(frozen=True)
class Concept:
    """An output column and the (concept name, allowed dimension members) pairs that feed it,
    in priority order. `duration` concepts (P&L, employees) are matched on period end date."""

    column: str
    sources: tuple[tuple[str, Members], ...]
    duration: bool = False


def plain(*names: str) -> tuple[tuple[str, Members], ...]:
    """Sources for concepts read only from contexts without dimensions."""
    return tuple((name, NO_DIMENSIONS) for name in names)


def split(name: str, member_sets: tuple[Members, ...]) -> tuple[tuple[str, Members], ...]:
    return tuple((name, members) for members in member_sets)


# Second names come from the older UK GAAP taxonomy seen in XBRL `.xml` filings.
CONCEPTS: tuple[Concept, ...] = (
    Concept("equity", plain("Equity", "ShareholderFunds")),
    Concept(
        "net_assets",
        plain("NetAssetsLiabilities", "NetAssetsLiabilitiesIncludingPensionAssetLiability"),
    ),
    Concept("net_current_assets", plain("NetCurrentAssetsLiabilities")),
    Concept("total_assets_less_current_liabilities", plain("TotalAssetsLessCurrentLiabilities")),
    Concept("fixed_assets", plain("FixedAssets")),
    Concept("current_assets", plain("CurrentAssets")),
    Concept("cash", plain("CashBankOnHand", "CashBankInHand")),
    Concept("debtors", plain("Debtors")),
    Concept("inventories", plain("TotalInventories")),
    Concept("creditors_within_one_year", split("Creditors", WITHIN_ONE_YEAR)),
    Concept("creditors_after_one_year", split("Creditors", AFTER_ONE_YEAR)),
    Concept("creditors_no_maturity", plain("Creditors")),
    Concept("provisions", plain("ProvisionsForLiabilitiesBalanceSheetSubtotal")),
    Concept("bank_borrowings_within_one_year", split("BankBorrowingsOverdrafts", WITHIN_ONE_YEAR)),
    Concept("bank_borrowings_after_one_year", split("BankBorrowingsOverdrafts", AFTER_ONE_YEAR)),
    Concept("turnover", plain("TurnoverRevenue"), duration=True),
    Concept("gross_profit", plain("GrossProfitLoss"), duration=True),
    Concept("operating_profit", plain("OperatingProfitLoss"), duration=True),
    Concept("profit_before_tax", plain("ProfitLossOnOrdinaryActivitiesBeforeTax"), duration=True),
    Concept("profit_loss", plain("ProfitLoss"), duration=True),
    Concept("employees", plain("AverageNumberEmployeesDuringPeriod"), duration=True),
)

# Report-level facts expressed as a dimension member on the context of a tagged fact,
# e.g. AccountingStandardsDimension = Micro-entities.
DIMENSION_COLUMNS: dict[str, str] = {
    "AccountingStandardsDimension": "accounting_standard",
    "AccountsTypeDimension": "accounts_type",
    "AccountsStatusDimension": "accounts_status",
    "EntityTradingStatusDimension": "trading_status",
    "LegalFormEntityDimension": "legal_form",
}

# Dimensions that say something about the whole document, not about the value. Some filing
# software puts `OriginalRevisedDataDimension = Original` on *every* context; treating it as a
# real split hid all values in about 5% of 2023 filings. If a document holds both Original and
# Revised facts for the same item, the first one tagged wins (rare; not handled specially).
IGNORED_DIMENSIONS = frozenset({"OriginalRevisedDataDimension"})

# Report-level non-numeric facts. The second name of each pair is the older XBRL equivalent.
NON_NUMERIC: dict[str, str] = {
    "EntityDormantTruefalse": "is_dormant",
    "CompanyDormant": "is_dormant",
    "DateAuthorisationFinancialStatementsForIssue": "date_authorised",
    "DateApproval": "date_authorised",
    "DateAccountsReceived": "date_received",  # only in Companies House-generated XBRL
    "UKCompaniesHouseRegisteredNumber": "company_number_in_doc",
    "CompaniesHouseRegisteredNumber": "company_number_in_doc",
}

SCHEMA: dict[str, Any] = {
    "company_number": pl.String,
    "balance_sheet_date": pl.Date,
    "filing_month": pl.Date,
    "source_archive": pl.String,
    "source_member": pl.String,
    "file_type": pl.String,
    "parse_error": pl.String,
    "company_number_in_doc": pl.String,
    "period_end": pl.Date,
    "prior_period_end": pl.Date,
    "date_authorised": pl.Date,
    "date_received": pl.Date,
    "is_dormant": pl.Boolean,
    **dict.fromkeys(DIMENSION_COLUMNS.values(), pl.String),
    "n_numeric_facts": pl.Int32,
    **{c.column: pl.Float64 for c in CONCEPTS},
    **{f"{c.column}_prev": pl.Float64 for c in CONCEPTS},
}


def _local(name: str | None) -> str:
    """`frs-core:WithinOneYear` -> `WithinOneYear`."""
    return (name or "").rsplit(":", 1)[-1]


def _members(context: Any) -> frozenset[str]:
    """Dimension members on a context, as local names, minus dimensions that describe the
    document rather than split a value (see IGNORED_DIMENSIONS)."""
    segments = getattr(context, "segments", None) or []
    return frozenset(
        _local(s.get("value"))
        for s in segments
        if s.get("dimension") and _local(s.get("dimension")) not in IGNORED_DIMENSIONS
    )


def _fact_date(context: Any, duration: bool) -> date | None:
    value = getattr(context, "enddate" if duration else "instant", None)
    return value if isinstance(value, date) else None


def _as_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        return None


def _as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _empty_row() -> dict[str, Any]:
    return dict.fromkeys(SCHEMA)


def extract(doc: IXBRL, balance_sheet_date: date) -> dict[str, Any]:
    """Pull the report metadata and financial values out of a parsed document."""
    row: dict[str, Any] = {}

    # (concept, members, date) -> first value seen. Duplicates (balance sheet + notes) agree.
    facts: dict[tuple[str, frozenset[str], date], float] = {}
    instants: set[date] = set()
    for fact in doc.numeric:
        if fact.value is None or fact.name is None:
            continue
        context = fact.context
        for duration in (False, True):
            when = _fact_date(context, duration)
            if when is not None:
                facts.setdefault((fact.name, _members(context), when), float(fact.value))
                if not duration:
                    instants.add(when)
    row["n_numeric_facts"] = len(doc.numeric)

    # Current period = the balance sheet date in the file name. Fall back to the latest date
    # tagged if no fact matches it exactly (rare: e.g. a mistyped date).
    current = balance_sheet_date
    if instants and current not in instants:
        current = max(instants)
    earlier = [d for d in instants if d < current]
    prior = max(earlier) if earlier else None
    row["period_end"] = current
    row["prior_period_end"] = prior

    for concept in CONCEPTS:
        # Each period takes the first source (in priority order) that has a value for it, so a
        # filing that tags the two years with different conventions still yields both.
        # For duration concepts the prior period ends on the prior balance sheet date.
        current_value: float | None = None
        prior_value: float | None = None
        for name, wanted in concept.sources:
            dated = {d: v for (n, m, d), v in facts.items() if n == name and m == wanted}
            if current_value is None:
                current_value = dated.get(current)
            if prior_value is None and prior is not None:
                prior_value = dated.get(prior)
        row[concept.column] = current_value
        row[f"{concept.column}_prev"] = prior_value

    for context in doc.contexts.values():
        for segment in getattr(context, "segments", None) or []:
            column = DIMENSION_COLUMNS.get(_local(segment.get("dimension")))
            if column and row.get(column) is None:
                row[column] = _local(segment.get("value"))

    for fact in doc.nonnumeric:
        column = NON_NUMERIC.get(fact.name or "")
        if column is None or row.get(column) is not None:
            continue
        if column == "is_dormant":
            row[column] = _as_bool(fact.value)
        elif column == "company_number_in_doc":
            text = str(fact.value).strip() if fact.value is not None else ""
            row[column] = text or None
        else:
            row[column] = _as_date(fact.value)
    return row


def parse_document(data: bytes, member: str, archive: str) -> dict[str, Any]:
    """Parse one accounts file. Never raises: failures are recorded in `parse_error`."""
    row = _empty_row()
    row["source_member"] = member
    row["source_archive"] = archive
    row["filing_month"] = filing_month_from_name(archive)
    parsed = parse_member_name(member)
    if parsed is None:
        row["parse_error"] = "unrecognised file name"
        return row
    row["company_number"], row["balance_sheet_date"], row["file_type"] = parsed
    try:
        doc = IXBRL(io.BytesIO(data), raise_on_error=False)
        row.update(extract(doc, row["balance_sheet_date"]))
    except Exception as exc:  # a malformed filing must not stop a 100k-file run
        row["parse_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return row


def _parse_chunk(zip_path: str, members: list[str]) -> list[dict[str, Any]]:
    """Worker: open the archive's local zip itself (cheaper than shipping bytes between
    processes) and parse a chunk of its members."""
    archive = Path(zip_path).stem
    with zipfile.ZipFile(zip_path) as zf:
        return [parse_document(zf.read(m), m, archive) for m in members]


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def parse_archive_zip(
    zip_path: Path, pool: ProcessPoolExecutor, chunk_size: int = 200
) -> pl.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
    futures = [
        pool.submit(_parse_chunk, str(zip_path), chunk) for chunk in _chunks(members, chunk_size)
    ]
    rows = [row for future in futures for row in future.result()]
    return pl.DataFrame(rows, schema=SCHEMA)


def parse_all(raw_dir: Path, out_dir: Path, workers: int, force: bool = False) -> pl.DataFrame:
    """Parse every sampled archive and combine them. Archives whose Parquet is newer than the
    zip are skipped unless `force` (use it after changing the parser)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    zips = sorted(raw_dir.glob("Accounts_Monthly_Data-*.zip"))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, zip_path in enumerate(zips, start=1):
            target = out_dir / f"{zip_path.stem}.parquet"
            fresh = target.exists() and target.stat().st_mtime >= zip_path.stat().st_mtime
            if fresh and not force:
                continue
            frame = parse_archive_zip(zip_path, pool)
            frame.write_parquet(target)
            errors = frame["parse_error"].is_not_null().sum()
            logger.info(
                "[%d/%d] %s: %d rows, %d errors", i, len(zips), zip_path.stem, frame.height, errors
            )
    parts = sorted(out_dir.glob("Accounts_Monthly_Data-*.parquet"))
    return pl.concat([pl.read_parquet(p) for p in parts]) if parts else pl.DataFrame(schema=SCHEMA)


def summarise(frame: pl.DataFrame) -> pl.DataFrame:
    """Share of filings with each financial value present (current period)."""
    return frame.select(
        pl.len().alias("rows"),
        *[pl.col(c.column).is_not_null().mean().round(3).alias(c.column) for c in CONCEPTS],
    )


def main() -> None:
    from cfp.config import get_settings

    parser = argparse.ArgumentParser(description="Parse sampled accounts into Parquet.")
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    parser.add_argument("--force", action="store_true", help="Re-parse every archive.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = get_settings()
    frame = parse_all(
        settings.raw_dir / "accounts_sample",
        settings.interim_dir / "accounts_parsed",
        args.workers,
        force=args.force,
    )
    combined = settings.interim_dir / "accounts.parquet"
    frame.write_parquet(combined)
    logger.info(
        "Wrote %s: %d filings, %d companies",
        combined,
        frame.height,
        frame["company_number"].n_unique(),
    )
    with pl.Config(tbl_rows=-1, tbl_formatting="ASCII_MARKDOWN", tbl_hide_dataframe_shape=True):
        print(summarise(frame).transpose(include_header=True, column_names=["share_present"]))


if __name__ == "__main__":
    main()
