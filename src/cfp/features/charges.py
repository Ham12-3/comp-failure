"""Charges (secured lending) from the charges endpoint.

Event date = `delivered_on`, the date the charge was registered at Companies House (public),
rather than `created_on`, which can be up to 21 days earlier.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any

import polars as pl

from cfp.features.events import as_of, in_last_days, per_row

CHARGE_SCHEMA: dict[str, Any] = {
    "company_number": pl.String,
    "delivered_on": pl.Date,
    "satisfied_on": pl.Date,
}


def _date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def charge_rows(company_number: str, items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        delivered = _date(item.get("delivered_on")) or _date(item.get("created_on"))
        if delivered is None:
            continue
        rows.append(
            {
                "company_number": company_number,
                "delivered_on": delivered,
                "satisfied_on": _date(item.get("satisfied_on")),
            }
        )
    return rows


def charge_features(rows: pl.DataFrame, charges: pl.DataFrame) -> pl.DataFrame:
    satisfied_by_then = pl.col("satisfied_on").is_not_null() & (
        pl.col("satisfied_on") <= pl.col("prediction_date")
    )
    agg = (
        as_of(rows, charges, "delivered_on")
        .group_by("row_id")
        .agg(
            satisfied_by_then.not_().sum().alias("charges_outstanding"),
            in_last_days("delivered_on", 365).sum().alias("charges_registered_12m"),
            (satisfied_by_then & in_last_days("satisfied_on", 365))
            .sum()
            .alias("charges_satisfied_12m"),
            pl.len().alias("charges_ever"),
        )
    )
    counts = (
        "charges_outstanding",
        "charges_registered_12m",
        "charges_satisfied_12m",
        "charges_ever",
    )
    return per_row(rows, agg, fill_zero=counts)
