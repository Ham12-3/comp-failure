"""Features from filing history: late filing, strike-off notices, address changes and more.

Event date = the filing-history `date`, i.e. when the filing became public. Filing types and
`description_values` keys were taken from real responses (see tests/fixtures/ch_api/).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any

import polars as pl

from cfp.features.events import as_of, in_last_days, per_row

FILING_SCHEMA: dict[str, Any] = {
    "company_number": pl.String,
    "date": pl.Date,
    "category": pl.String,
    "type": pl.String,
    "description": pl.String,
    "made_up_date": pl.Date,
    "old_address": pl.String,
    "new_address": pl.String,
}

ACCOUNTS_DUE_MONTHS = 9  # private companies; public companies have 6 (approximation, see DATA.md)
CONFIRMATION_DUE_DAYS = {"CS01": 14, "AR01": 28}  # confirmation statement / old annual return
OFFICE_MOVE_TYPES = ("AD01", "LLAD01", "RP05")  # RP05 = Companies House default address applied
DIRECTOR_APPOINTMENT_TYPES = ("AP01", "AP02", "288a")
DIRECTOR_TERMINATION_TYPES = ("TM01", "288b")

_POSTCODE_RE = re.compile(r"\b([A-Z]{1,2})[0-9][A-Z0-9]?\s*[0-9][A-Z]{2}\b")


def _date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def filing_rows(company_number: str, items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalise filing-history items into event rows."""
    rows = []
    for item in items:
        filed = _date(item.get("date"))
        if filed is None:
            continue
        values = item.get("description_values") or {}
        rows.append(
            {
                "company_number": company_number,
                "date": filed,
                "category": item.get("category"),
                "type": item.get("type"),
                "description": item.get("description"),
                "made_up_date": _date(values.get("made_up_date")),
                "old_address": values.get("old_address"),
                "new_address": values.get("new_address"),
            }
        )
    return rows


def postcode_area(address: str | None) -> str | None:
    """`... Wolverhampton, WV3 0SR, United Kingdom` -> `WV` (last postcode in the string)."""
    if not address:
        return None
    matches = _POSTCODE_RE.findall(address.upper())
    return matches[-1] if matches else None


def filing_features(rows: pl.DataFrame, filings: pl.DataFrame) -> pl.DataFrame:
    """Per-row features from filing history. `rows` needs row_id, company_number,
    prediction_date and balance_sheet_date."""
    ev = as_of(rows, filings, "date").join(rows.select("row_id", "balance_sheet_date"), on="row_id")
    typ, cat, filed = pl.col("type"), pl.col("category"), pl.col("date")
    pred = pl.col("prediction_date").first()
    last_12m, last_3y = in_last_days("date", 365), in_last_days("date", 3 * 365)

    previous_accounts = (typ == "AA") & (filed < pl.col("prediction_date"))
    accounts_days_late = (
        filed - pl.col("made_up_date").dt.offset_by(f"{ACCOUNTS_DUE_MONTHS}mo")
    ).dt.total_days()
    confirmation = typ.is_in(list(CONFIRMATION_DUE_DAYS))
    confirmation_days_late = (
        filed
        - pl.col("made_up_date")
        - pl.duration(days=pl.when(typ == "CS01").then(14).otherwise(28))
    ).dt.total_days()
    this_filing = (
        (typ == "AA")
        & (filed == pl.col("prediction_date"))
        & (pl.col("made_up_date") == pl.col("balance_sheet_date"))
    )

    counts = (
        "filings_12m",
        "previous_accounts_late_3y",
        "confirmations_late_3y",
        "office_moves_12m",
        "default_address_applied_ever",
        "strike_off_notices_12m",
        "strike_off_notices_ever",
        "strike_off_discontinued_12m",
        "strike_off_applications_ever",
        "reference_date_changes_12m",
        "psc_changes_12m",
        "director_appointments_12m",
        "director_terminations_12m",
    )
    agg = ev.group_by("row_id").agg(
        last_12m.sum().alias("filings_12m"),
        (previous_accounts & last_3y & (accounts_days_late > 0))
        .sum()
        .alias("previous_accounts_late_3y"),
        accounts_days_late.filter(previous_accounts & last_3y)
        .max()
        .alias("previous_accounts_max_days_late_3y"),
        (pred - filed.filter(previous_accounts).max())
        .dt.total_days()
        .alias("days_since_previous_accounts"),
        (confirmation & last_3y & (confirmation_days_late > 0))
        .sum()
        .alias("confirmations_late_3y"),
        confirmation_days_late.filter(confirmation & last_3y)
        .max()
        .alias("confirmation_max_days_late_3y"),
        (pred - pl.col("made_up_date").filter(confirmation).max())
        .dt.total_days()
        .alias("days_since_confirmation_made_up"),
        (typ.is_in(OFFICE_MOVE_TYPES) & last_12m).sum().alias("office_moves_12m"),
        (typ == "RP05").sum().alias("default_address_applied_ever"),
        ((typ == "GAZ1") & last_12m).sum().alias("strike_off_notices_12m"),
        (typ == "GAZ1").sum().alias("strike_off_notices_ever"),
        ((typ == "DISS40") & last_12m).sum().alias("strike_off_discontinued_12m"),
        (typ == "DS01").sum().alias("strike_off_applications_ever"),
        ((typ == "AA01") & last_12m).sum().alias("reference_date_changes_12m"),
        ((cat == "persons-with-significant-control") & last_12m).sum().alias("psc_changes_12m"),
        (typ.is_in(DIRECTOR_APPOINTMENT_TYPES) & last_12m).sum().alias("director_appointments_12m"),
        (typ.is_in(DIRECTOR_TERMINATION_TYPES) & last_12m).sum().alias("director_terminations_12m"),
        pl.col("description")
        .filter(this_filing)
        .first()
        .str.replace("accounts-with-accounts-type-", "")
        .alias("accounts_type_filed"),
    )
    out = per_row(rows, agg, fill_zero=counts)
    current_lateness = rows.select(
        "row_id",
        (
            pl.col("prediction_date")
            - pl.col("balance_sheet_date").dt.offset_by(f"{ACCOUNTS_DUE_MONTHS}mo")
        )
        .dt.total_days()
        .alias("days_late_this_filing"),
    )
    return out.join(current_lateness, on="row_id").with_columns(
        first_accounts=pl.col("days_since_previous_accounts").is_null()
    )


def office_postcode_area(
    rows: pl.DataFrame, filings: pl.DataFrame, current_postcodes: pl.DataFrame
) -> pl.DataFrame:
    """Postcode area of the registered office *on the prediction date*.

    1. `new_address` of the latest office move filed on or before the prediction date;
    2. else `old_address` of the first move filed after it (the address it moved away from);
    3. else today's address, but only if the company has not moved since the prediction date.

    Rule 2 reads a later event, but only to recover the address in force on the prediction
    date. It matters: liquidators often move the registered office to their own office, so
    today's address can give away a failure. `current_postcodes` has company_number and
    postal_code from the profile.
    """
    moves = filings.filter(pl.col("type").is_in(OFFICE_MOVE_TYPES)).select(
        "company_number", "date", "old_address", "new_address"
    )
    keys = rows.select("row_id", "company_number", "prediction_date")
    past = (
        as_of(rows, moves, "date")
        .filter(pl.col("new_address").is_not_null())
        .sort("date")
        .group_by("row_id")
        .agg(pl.col("new_address").last().alias("past_address"))
    )
    later = (
        keys.join(moves, on="company_number")
        .filter(pl.col("date") > pl.col("prediction_date"))
        .sort("date")
        .group_by("row_id")
        .agg(
            pl.col("old_address").first().alias("later_old_address"), pl.len().alias("later_moves")
        )
    )
    out = (
        keys.join(past, on="row_id", how="left")
        .join(later, on="row_id", how="left")
        .join(current_postcodes, on="company_number", how="left")
        .with_columns(
            pl.col("past_address").map_elements(postcode_area, return_dtype=pl.String).alias("a1"),
            pl.col("later_old_address")
            .map_elements(postcode_area, return_dtype=pl.String)
            .alias("a2"),
            pl.when(pl.col("later_moves").is_null())
            .then(pl.col("postal_code").map_elements(postcode_area, return_dtype=pl.String))
            .alias("a3"),
        )
    )
    return out.select("row_id", pl.coalesce("a1", "a2", "a3").alias("postcode_area"))
