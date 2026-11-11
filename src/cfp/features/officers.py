"""Directors in office on the prediction date, from the officers endpoint.

Appointments and resignations *in the last 12 months* are counted from filing history in
`filings.py` instead, because filing dates are when changes became public, while officer
records carry the effective dates.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any

import polars as pl

from cfp.features.events import as_of, per_row

OFFICER_SCHEMA: dict[str, Any] = {
    "company_number": pl.String,
    "officer_role": pl.String,
    "appointed_on": pl.Date,
    "resigned_on": pl.Date,
}
DIRECTOR_ROLES = (
    "director",
    "corporate-director",
    "nominee-director",
    "corporate-nominee-director",
    "llp-member",
    "llp-designated-member",
)
_LONG_AGO = date(1900, 1, 1)  # pre-1992 appointments have no `appointed_on`


def _date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def officer_rows(company_number: str, items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "company_number": company_number,
            "officer_role": item.get("officer_role"),
            "appointed_on": _date(item.get("appointed_on")) or _LONG_AGO,
            "resigned_on": _date(item.get("resigned_on")),
        }
        for item in items
    ]


def officer_features(rows: pl.DataFrame, officers: pl.DataFrame) -> pl.DataFrame:
    directors = officers.filter(pl.col("officer_role").is_in(DIRECTOR_ROLES))
    in_office = (
        as_of(rows, directors, "appointed_on")
        .filter(
            pl.col("resigned_on").is_null() | (pl.col("resigned_on") > pl.col("prediction_date"))
        )
        .group_by("row_id")
        .agg(
            pl.len().alias("directors_in_office"),
            (pl.col("prediction_date") - pl.col("appointed_on"))
            .dt.total_days()
            .median()
            .alias("director_median_tenure_days"),
        )
    )
    return per_row(rows, in_office, fill_zero=("directors_in_office",))
