"""Assemble the feature table: one row per accounts filing, keyed like labels.parquet.

Inputs: the label table (parsed accounts + prediction date) and the API responses for each
company, read from the cache only. Output: data/processed/features.parquet with the key
columns and the features listed in FEATURES (the list is checked against the output by a
test, and docs/DATA.md is generated from it).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import polars as pl

from cfp.features.charges import CHARGE_SCHEMA, charge_features, charge_rows
from cfp.features.filings import (
    FILING_SCHEMA,
    filing_features,
    filing_rows,
    office_postcode_area,
)
from cfp.features.financial import financial_features
from cfp.features.officers import OFFICER_SCHEMA, officer_features, officer_rows
from cfp.features.profile import PROFILE_SCHEMA, profile_features, profile_static

logger = logging.getLogger(__name__)

KEYS = ("company_number", "balance_sheet_date", "source_member", "prediction_date")

# name -> (source, description). Sources: accounts = the filing itself; filing history,
# officers, charges and profile = Companies House API, as of the prediction date.
FEATURES: dict[str, tuple[str, str]] = {
    # --- amounts (accounts, signs repaired)
    "net_assets": ("accounts", "Net assets (falls back to equity)."),
    "total_assets": ("accounts", "Fixed assets + current assets."),
    "current_assets": ("accounts", "Current assets."),
    "current_liabilities": (
        "accounts",
        "Creditors due within one year (or current assets - net current assets).",
    ),
    "cash": ("accounts", "Cash at bank and in hand."),
    "total_liabilities": ("accounts", "Total assets - net assets."),
    "total_debt": ("accounts", "Creditors due within + after one year."),
    "bank_debt": (
        "accounts",
        "Bank loans and overdrafts, within + after one year (0 if not tagged).",
    ),
    "turnover": ("accounts", "Turnover, if filed (rare for small companies)."),
    "profit_loss": ("accounts", "Profit or loss for the year, if filed."),
    "employees": ("accounts", "Average number of employees."),
    # --- ratios
    "current_ratio": ("accounts", "Current assets / current liabilities."),
    "cash_to_current_liabilities": ("accounts", "Cash / current liabilities."),
    "debt_to_assets": ("accounts", "Total liabilities / total assets."),
    "equity_ratio": ("accounts", "Net assets / total assets."),
    "net_assets_negative": ("accounts", "Net assets below zero."),
    "net_current_liabilities": ("accounts", "Net current assets below zero."),
    "log_total_assets": ("accounts", "log(1 + |total assets|): company size."),
    # --- year on year (prior-year comparatives in the same filing)
    "net_assets_change": ("accounts", "Net assets minus prior-year net assets."),
    "total_assets_pct_change": ("accounts", "Relative change in total assets."),
    "current_assets_pct_change": ("accounts", "Relative change in current assets."),
    "cash_pct_change": ("accounts", "Relative change in cash."),
    "turnover_pct_change": ("accounts", "Relative change in turnover."),
    "net_assets_turned_negative": ("accounts", "Net assets negative now, not negative last year."),
    "has_prior_year": ("accounts", "Prior-year comparatives present."),
    # --- missingness and report metadata
    "missing_current_assets": ("accounts", "Current assets not tagged."),
    "missing_cash": ("accounts", "Cash not tagged."),
    "missing_debtors": ("accounts", "Debtors not tagged."),
    "missing_fixed_assets": ("accounts", "Fixed assets not tagged."),
    "missing_creditors_within_one_year": ("accounts", "Short-term creditors not tagged."),
    "missing_creditors_after_one_year": ("accounts", "Long-term creditors not tagged."),
    "missing_turnover": ("accounts", "Turnover not filed."),
    "missing_profit_loss": ("accounts", "Profit or loss not filed."),
    "missing_employees": ("accounts", "Employee count not tagged."),
    "n_numeric_facts": ("accounts", "Number of numeric facts tagged: how much was disclosed."),
    "sign_repaired": ("accounts", "A sign-tagging error was corrected in this filing."),
    "accounting_standard": ("accounts", "Micro-entities / SmallEntities / FRS102 / ..."),
    "accounts_type": ("accounts", "Full / filleted / abridged accounts (tagged)."),
    "accounts_status": ("accounts", "Audited or audit-exempt (tagged)."),
    "is_dormant": ("accounts", "Company declares itself dormant."),
    "prior_period_gap_days": ("accounts", "Days between this and the prior balance sheet date."),
    # --- filing history
    "days_late_this_filing": (
        "filing history",
        "Filing date - (balance sheet date + 9 months); negative = early.",
    ),
    "first_accounts": ("filing history", "No earlier accounts filing on record."),
    "days_since_previous_accounts": ("filing history", "Days since the previous accounts filing."),
    "previous_accounts_late_3y": (
        "filing history",
        "Earlier accounts filed late in the last 3 years.",
    ),
    "previous_accounts_max_days_late_3y": (
        "filing history",
        "Worst lateness of earlier accounts, last 3 years.",
    ),
    "confirmations_late_3y": (
        "filing history",
        "Confirmation statements / annual returns filed late, last 3 years.",
    ),
    "confirmation_max_days_late_3y": (
        "filing history",
        "Worst confirmation-statement lateness, last 3 years.",
    ),
    "days_since_confirmation_made_up": (
        "filing history",
        "Days since the last confirmation statement's made-up date.",
    ),
    "filings_12m": ("filing history", "Filings of any kind in the last 12 months."),
    "office_moves_12m": ("filing history", "Registered office changes in the last 12 months."),
    "default_address_applied_ever": (
        "filing history",
        "Companies House replaced an invalid registered office (RP05).",
    ),
    "strike_off_notices_12m": (
        "filing history",
        "Compulsory strike-off gazette notices (GAZ1), last 12 months.",
    ),
    "strike_off_notices_ever": ("filing history", "Compulsory strike-off gazette notices, ever."),
    "strike_off_discontinued_12m": (
        "filing history",
        "Strike-off action discontinued (DISS40), last 12 months.",
    ),
    "strike_off_applications_ever": (
        "filing history",
        "Voluntary strike-off applications (DS01), ever.",
    ),
    "reference_date_changes_12m": (
        "filing history",
        "Accounting reference date changes (AA01), last 12 months.",
    ),
    "psc_changes_12m": (
        "filing history",
        "Person-with-significant-control filings, last 12 months.",
    ),
    "director_appointments_12m": (
        "filing history",
        "Director appointments filed in the last 12 months.",
    ),
    "director_terminations_12m": (
        "filing history",
        "Director terminations filed in the last 12 months.",
    ),
    "accounts_type_filed": (
        "filing history",
        "Accounts type Companies House recorded for this filing.",
    ),
    "postcode_area": (
        "filing history + profile",
        "Postcode area of the registered office on the prediction date.",
    ),
    # --- officers
    "directors_in_office": ("officers", "Directors in office on the prediction date."),
    "director_median_tenure_days": ("officers", "Median tenure of those directors, in days."),
    # --- charges
    "charges_outstanding": (
        "charges",
        "Charges registered and not satisfied by the prediction date.",
    ),
    "charges_registered_12m": ("charges", "Charges registered in the last 12 months."),
    "charges_satisfied_12m": ("charges", "Charges satisfied in the last 12 months."),
    "charges_ever": ("charges", "Charges ever registered."),
    # --- profile (static)
    "company_age_years": ("profile", "Years since incorporation."),
    "company_type": ("profile", "Legal form (ltd, plc, llp, ...)."),
    "jurisdiction": ("profile", "England and Wales / Scotland / Northern Ireland."),
    "sic_section": (
        "profile",
        "Industry: UK SIC 2007 section (A-U) of the first SIC code (today's code).",
    ),
    "n_sic_codes": ("profile", "Number of SIC codes."),
}


@dataclass(frozen=True)
class ApiData:
    filings: pl.DataFrame
    officers: pl.DataFrame
    charges: pl.DataFrame
    profiles: pl.DataFrame
    current_postcodes: pl.DataFrame


def api_data_from_payloads(payloads: Mapping[str, Mapping[str, Any]]) -> ApiData:
    """Build event tables from raw API bodies: {company_number: {profile, filing_history,
    officers, charges}} where the last three are lists of items."""
    filings, officers, charges, profiles, postcodes = [], [], [], [], []
    for number, data in payloads.items():
        profile = data["profile"]
        profiles.append(profile_static(number, profile))
        office = profile.get("registered_office_address") or {}
        postcodes.append({"company_number": number, "postal_code": office.get("postal_code")})
        filings += filing_rows(number, data.get("filing_history") or [])
        officers += officer_rows(number, data.get("officers") or [])
        charges += charge_rows(number, data.get("charges") or [])
    return ApiData(
        filings=pl.DataFrame(filings, schema=FILING_SCHEMA),
        officers=pl.DataFrame(officers, schema=OFFICER_SCHEMA),
        charges=pl.DataFrame(charges, schema=CHARGE_SCHEMA),
        profiles=pl.DataFrame(profiles, schema=PROFILE_SCHEMA),
        current_postcodes=pl.DataFrame(
            postcodes, schema={"company_number": pl.String, "postal_code": pl.String}
        ),
    )


def build_feature_table(accounts_rows: pl.DataFrame, api: ApiData) -> pl.DataFrame:
    """Features for every row of `accounts_rows` (parsed accounts columns + prediction_date)."""
    rows = accounts_rows.with_row_index("row_id")
    keys = rows.select("row_id", "company_number", "prediction_date", "balance_sheet_date")
    parts = [
        financial_features(rows),
        filing_features(keys, api.filings),
        office_postcode_area(keys, api.filings, api.current_postcodes),
        officer_features(keys, api.officers),
        charge_features(keys, api.charges),
        profile_features(keys, api.profiles),
    ]
    out = rows.select("row_id", *KEYS)
    for part in parts:
        out = out.join(part, on="row_id", how="left")
    return out.drop("row_id").select(*KEYS, *FEATURES)


def payloads_from_cache(client: Any, companies: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Raw API bodies for every company whose fetch has completed. A company whose fetch is
    still in progress (some endpoints missing from the cache) is skipped."""
    from cfp.ingest.ch_api import CacheMissError
    from cfp.ingest.fetch_companies import has_charges

    payloads: dict[str, dict[str, Any]] = {}
    for number in companies:
        try:
            profile = client.company_profile(number)
            if profile is None:
                continue
            payloads[number] = {
                "profile": profile,
                "filing_history": client.filing_history(number),
                "officers": client.officers(number),
                "charges": client.charges(number) if has_charges(profile) else [],
            }
        except CacheMissError:
            continue
    return payloads


def feature_docs_markdown() -> str:
    lines = ["| Feature | Source | Meaning |", "|---|---|---|"]
    lines += [f"| `{name}` | {src} | {desc} |" for name, (src, desc) in FEATURES.items()]
    return "\n".join(lines)


def main() -> None:
    from cfp.config import get_settings
    from cfp.ingest.ch_api import default_client

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    labels = pl.read_parquet(settings.processed_dir / "labels.parquet")
    with default_client(cache_only=True) as client:
        payloads = payloads_from_cache(client, labels["company_number"].unique().to_list())
    rows = labels.filter(pl.col("company_number").is_in(list(payloads)))
    features = build_feature_table(rows, api_data_from_payloads(payloads))
    out = settings.processed_dir / "features.parquet"
    features.write_parquet(out)
    logger.info(
        "Wrote %s: %d rows x %d features, %d companies",
        out,
        features.height,
        len(FEATURES),
        len(payloads),
    )
    nulls = features.select(pl.col(list(FEATURES)).null_count() / features.height).transpose(
        include_header=True, column_names=["null_share"]
    )
    with pl.Config(tbl_rows=-1, tbl_formatting="ASCII_MARKDOWN", tbl_hide_dataframe_shape=True):
        print(nulls.sort("null_share", descending=True).head(25))


if __name__ == "__main__":
    main()
