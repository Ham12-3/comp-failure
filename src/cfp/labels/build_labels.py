"""Build failure labels: one row per (company, accounts filing).

Label definition (PROJECT_SPEC.md section 4, made precise here):

* **Prediction date** = the date the accounts were filed (public), taken from filing history.
  If no matching filing-history item is found, the last day of the archive's filing month is
  used instead and flagged (`filing_date_source = "month_end"`). That is never earlier than
  the true filing date, so it cannot let a feature see the future.
* **label_insolvency_12m = 1** if a *failure* case starts in (prediction_date, +12 months].
  Failure cases: compulsory liquidation, creditors' voluntary liquidation, administration,
  receivership. A members' voluntary liquidation is a *solvent* wind-down and is **not** a
  failure; voluntary arrangements, moratoria and restructuring plans are recorded but not
  counted (the spec names liquidation, administration and receivership only).
* **label_strike_off_12m = 1** if the company is dissolved in the window with no insolvency
  case at all (second target, not used by the main model).
* Rows are kept but marked with `excluded_reason` when they must not be used for training:
  `already_insolvent` (a failure case started on or before the prediction date),
  `outcome_not_observable` (the 12-month window ends after the outcome cutoff), or
  `no_profile` (the company could not be found in the API).

Field names and enumeration values come from the documented API: swagger spec
`insolvency.json` / `companyProfile.json` / `filingHistory.json` and the `constants.yml`
enumerations in https://github.com/companieshouse/api-enumerations. They are re-checked
against the real responses saved in tests/fixtures/ch_api/ once a Live API key is available.

`company_status` and dissolution dates are used here, for labels only. They must never reach
the feature table (enforced by the leakage tests in Phase 3).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- case types
FAILURE_CASE_TYPES = frozenset(
    {
        "compulsory-liquidation",
        "creditors-voluntary-liquidation",
        "in-administration",
        "administration-order",
        "administrative-receiver",
        "scottish-administrative-receiver",
        "receiver-manager",
        "receivership",
    }
)
SOLVENT_CASE_TYPES = frozenset({"members-voluntary-liquidation"})
# Everything else (CVA, moratoria, restructuring plans, overseas winding up) is "other".

# Date types that mark when a case *starts*, in the documented `caseDates.type` enumeration.
# `petitioned-on` is deliberately not one: a winding-up petition can be dismissed.
START_DATE_TYPES = frozenset(
    {
        "wound-up-on",  # commencement of winding up (liquidations)
        "ordered-to-wind-up-on",  # winding-up order (compulsory liquidation)
        "administration-started-on",
        "instrumented-on",  # instrument appointing a receiver
        "voluntary-arrangement-started-on",
        "moratorium-started-on",
        "declaration-solvent-on",  # MVL: declaration of solvency
    }
)

OUTCOME_LAG_DAYS = 30  # insolvency events can take a few weeks to appear on the register

CASE_SCHEMA: dict[str, Any] = {
    "company_number": pl.String,
    "case_number": pl.String,
    "case_type": pl.String,
    "case_category": pl.String,
    "start_date": pl.Date,
    "start_date_source": pl.String,
}


def case_category(case_type: str) -> str:
    if case_type in FAILURE_CASE_TYPES:
        return "failure"
    if case_type in SOLVENT_CASE_TYPES:
        return "solvent"
    return "other"


def _parse_date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def case_start_date(case: Mapping[str, Any]) -> tuple[date | None, str]:
    """Start of an insolvency case and where it came from.

    Prefer documented start-type dates; fall back to the earliest practitioner appointment,
    then to the earliest date of any type.
    """
    dates = [(d.get("type"), _parse_date(d.get("date"))) for d in case.get("dates") or []]
    starts = [d for t, d in dates if t in START_DATE_TYPES and d is not None]
    if starts:
        return min(starts), "case_date"
    appointed = [_parse_date(p.get("appointed_on")) for p in case.get("practitioners") or []]
    appointed_dates = [d for d in appointed if d is not None]
    if appointed_dates:
        return min(appointed_dates), "practitioner_appointed"
    any_dates = [d for _, d in dates if d is not None]
    if any_dates:
        return min(any_dates), "earliest_any_date"
    return None, "missing"


def flatten_insolvency(
    company_number: str, payload: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    """One row per case from a `GET /company/{n}/insolvency` body (None for 404)."""
    rows = []
    for case in (payload or {}).get("cases") or []:
        case_type = str(case.get("type") or "unknown")
        start, source = case_start_date(case)
        if start is None:
            logger.warning(
                "%s: insolvency case %s has no usable date", company_number, case.get("number")
            )
        rows.append(
            {
                "company_number": company_number,
                "case_number": str(case.get("number") or ""),
                "case_type": case_type,
                "case_category": case_category(case_type),
                "start_date": start,
                "start_date_source": source,
            }
        )
    return rows


def cases_frame(rows: Iterable[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=CASE_SCHEMA)


# ---------------------------------------------------------------- filing dates
FILING_SCHEMA: dict[str, Any] = {
    "company_number": pl.String,
    "filing_date": pl.Date,
    "made_up_date": pl.Date,
}


def flatten_accounts_filings(
    company_number: str, items: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Accounts filings from filing-history items: the processed `date` and, when present,
    the period the accounts were made up to (`description_values.made_up_date`)."""
    rows = []
    for item in items:
        if item.get("category") != "accounts":
            continue
        filed = _parse_date(item.get("date"))
        if filed is None:
            continue
        values = item.get("description_values") or {}
        rows.append(
            {
                "company_number": company_number,
                "filing_date": filed,
                "made_up_date": _parse_date(values.get("made_up_date")),
            }
        )
    return rows


def attach_filing_dates(accounts: pl.DataFrame, filings: pl.DataFrame) -> pl.DataFrame:
    """Add `prediction_date` (+ `filing_date_source`) to parsed accounts rows.

    A filing-history accounts item matches an accounts row when it was processed in the
    row's filing month. If several match, the one whose made-up date equals the balance sheet
    date wins, then the earliest. No match: last day of the filing month.
    """
    rows = accounts.select("company_number", "balance_sheet_date", "filing_month").with_row_index(
        "row_id"
    )
    candidates = (
        rows.join(filings, on="company_number", how="inner")
        .filter(pl.col("filing_date").dt.month_start() == pl.col("filing_month"))
        .with_columns(
            exact=(pl.col("made_up_date") == pl.col("balance_sheet_date")).fill_null(False)
        )
        .sort(["row_id", "exact", "filing_date"], descending=[False, True, False])
        .group_by("row_id", maintain_order=True)
        .agg(pl.col("filing_date").first(), pl.col("exact").first())
    )
    out = rows.join(candidates, on="row_id", how="left").with_columns(
        prediction_date=pl.coalesce("filing_date", pl.col("filing_month").dt.month_end()),
        filing_date_source=pl.when(pl.col("filing_date").is_null())
        .then(pl.lit("month_end"))
        .when(pl.col("exact"))
        .then(pl.lit("filing_history_exact"))
        .otherwise(pl.lit("filing_history_month")),
    )
    return (
        accounts.with_row_index("row_id")
        .join(out.select("row_id", "prediction_date", "filing_date_source"), on="row_id")
        .drop("row_id")
    )


# ---------------------------------------------------------------- labels
PROFILE_SCHEMA: dict[str, Any] = {
    "company_number": pl.String,
    "found": pl.Boolean,
    "company_status": pl.String,
    "date_of_cessation": pl.Date,
}


def profile_row(company_number: str, payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """The label-relevant bits of a company profile (None = 404)."""
    if payload is None:
        return {
            "company_number": company_number,
            "found": False,
            "company_status": None,
            "date_of_cessation": None,
        }
    return {
        "company_number": company_number,
        "found": True,
        "company_status": payload.get("company_status"),
        "date_of_cessation": _parse_date(payload.get("date_of_cessation")),
    }


def has_insolvency_resource(profile: Mapping[str, Any]) -> bool:
    """Whether to call the insolvency endpoint. `has_insolvency_history` is deprecated in the
    spec in favour of `links.insolvency`; accept either."""
    links = profile.get("links") or {}
    return bool(profile.get("has_insolvency_history")) or "insolvency" in links


def build_labels(
    filings: pl.DataFrame,
    cases: pl.DataFrame,
    profiles: pl.DataFrame,
    outcome_cutoff: date,
) -> pl.DataFrame:
    """Label every filing row. `filings` needs company_number, balance_sheet_date and
    prediction_date; extra columns are passed through."""
    rows = filings.with_row_index("row_id").with_columns(
        horizon_end=pl.col("prediction_date").dt.offset_by("12mo")
    )
    keys = rows.select("row_id", "company_number", "prediction_date", "horizon_end")

    # Failure cases relative to each row.
    failure = cases.filter(pl.col("case_category") == "failure").select(
        "company_number", "case_type", "start_date"
    )
    in_window = (pl.col("start_date") > pl.col("prediction_date")) & (
        pl.col("start_date") <= pl.col("horizon_end")
    )
    failure_stats = (
        keys.join(failure, on="company_number", how="inner")
        .sort("start_date")
        .group_by("row_id")
        .agg(
            already_insolvent=(pl.col("start_date") <= pl.col("prediction_date")).any(),
            insolvency_start_date=pl.col("start_date").filter(in_window).first(),
            insolvency_case_type=pl.col("case_type").filter(in_window).first(),
        )
    )

    solvent = cases.filter(pl.col("case_category") == "solvent").select(
        "company_number", "start_date"
    )
    solvent_stats = (
        keys.join(solvent, on="company_number", how="inner")
        .group_by("row_id")
        .agg(solvent_liquidation_12m=in_window.any())
    )

    companies_with_cases = (
        cases.select("company_number").unique().with_columns(has_any_case=pl.lit(True))
    )

    labelled = (
        rows.join(failure_stats, on="row_id", how="left")
        .join(solvent_stats, on="row_id", how="left")
        .join(
            profiles.select("company_number", "found", "company_status", "date_of_cessation"),
            on="company_number",
            how="left",
        )
        .join(companies_with_cases, on="company_number", how="left")
        # Separate step: expressions inside one with_columns all see the *input* columns, so the
        # labels below must not be computed in the same call that fills these nulls.
        .with_columns(
            pl.col("already_insolvent", "solvent_liquidation_12m", "has_any_case").fill_null(False)
        )
        .with_columns(
            label_insolvency_12m=pl.col("insolvency_start_date").is_not_null().cast(pl.Int8),
            label_strike_off_12m=(
                (pl.col("company_status") == "dissolved")
                & (pl.col("date_of_cessation") > pl.col("prediction_date"))
                & (pl.col("date_of_cessation") <= pl.col("horizon_end"))
                & pl.col("has_any_case").not_()
            )
            .fill_null(False)
            .cast(pl.Int8),
        )
    )
    observable_until = outcome_cutoff.toordinal() - OUTCOME_LAG_DAYS
    labelled = labelled.with_columns(
        excluded_reason=pl.when(pl.col("found").fill_null(False).not_())
        .then(pl.lit("no_profile"))
        .when(pl.col("already_insolvent"))
        .then(pl.lit("already_insolvent"))
        .when(pl.col("horizon_end") > pl.lit(date.fromordinal(observable_until)))
        .then(pl.lit("outcome_not_observable"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )
    # company_status / date_of_cessation describe the outcome: drop them from the output so
    # nothing downstream can pick them up as features.
    return labelled.drop("row_id", "found", "company_status", "date_of_cessation", "has_any_case")


def summarise_labels(labels: pl.DataFrame) -> pl.DataFrame:
    """Counts and positive rates for docs/DATA.md, usable rows only, by prediction year."""
    usable = labels.filter(pl.col("excluded_reason").is_null())
    return (
        usable.group_by(pl.col("prediction_date").dt.year().alias("year"))
        .agg(
            pl.len().alias("rows"),
            pl.col("company_number").n_unique().alias("companies"),
            pl.col("label_insolvency_12m").sum().alias("insolvencies"),
            pl.col("label_insolvency_12m").mean().round(4).alias("insolvency_rate"),
            pl.col("label_strike_off_12m").mean().round(4).alias("strike_off_rate"),
        )
        .sort("year")
    )


# ---------------------------------------------------------------- pipeline step
def collect_from_cache(
    client: Any, companies: Iterable[str]
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, date | None]:
    """Profiles, insolvency cases and accounts filings for every company already in the API
    cache, plus the earliest profile fetch date (the outcome cutoff). Companies not fetched
    yet are skipped, so labels can be built from a partial fetch."""
    from cfp.ingest.ch_api import CacheMissError

    profile_rows, case_rows, filing_rows = [], [], []
    earliest: date | None = None
    for number in companies:
        try:
            response = client.get_response(f"/company/{number}")
            profile = response.data
            profile_rows.append(profile_row(number, profile))
            fetched = response.fetched_at.date()
            earliest = fetched if earliest is None else min(earliest, fetched)
            if profile is None:
                continue
            if has_insolvency_resource(profile):
                case_rows += flatten_insolvency(number, client.insolvency(number))
            filing_rows += flatten_accounts_filings(number, client.filing_history(number))
        except CacheMissError:
            continue
    return (
        pl.DataFrame(profile_rows, schema=PROFILE_SCHEMA),
        cases_frame(case_rows),
        pl.DataFrame(filing_rows, schema=FILING_SCHEMA),
        earliest,
    )


def main() -> None:
    from cfp.config import get_settings
    from cfp.ingest.ch_api import default_client

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    accounts = pl.read_parquet(settings.interim_dir / "accounts.parquet").filter(
        pl.col("parse_error").is_null()
    )
    with default_client(cache_only=True) as client:
        profiles, cases, filings, cutoff = collect_from_cache(
            client, accounts["company_number"].unique().to_list()
        )
    if cutoff is None:
        raise SystemExit("No company profiles in the API cache yet: run cfp.ingest.fetch_companies")

    fetched = accounts.join(profiles.select("company_number"), on="company_number", how="semi")
    labels = build_labels(attach_filing_dates(fetched, filings), cases, profiles, cutoff)
    out = settings.processed_dir / "labels.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    labels.write_parquet(out)

    logger.info(
        "Wrote %s: %d rows for %d of %d companies; outcome cutoff %s",
        out,
        labels.height,
        profiles.height,
        accounts["company_number"].n_unique(),
        cutoff,
    )
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_hide_dataframe_shape=True, tbl_rows=30):
        print(labels.group_by("excluded_reason").len().sort("len", descending=True))
        print(labels.group_by("filing_date_source").len().sort("len", descending=True))
        print(cases.group_by("case_type", "case_category").len().sort("len", descending=True))
        print(summarise_labels(labels))


if __name__ == "__main__":
    main()
