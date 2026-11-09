"""Static company attributes from the profile: age, legal form, jurisdiction, industry.

Only fields fixed at incorporation or very rarely changed are read (`ALLOWED_FIELDS`). Status,
dissolution, overdue flags and next-due dates describe the company *today* and are never read
here. SIC codes can change over time, so using today's code is a documented approximation.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

import polars as pl

ALLOWED_FIELDS = frozenset({"date_of_creation", "type", "jurisdiction", "sic_codes"})

PROFILE_SCHEMA: dict[str, Any] = {
    "company_number": pl.String,
    "date_of_creation": pl.Date,
    "company_type": pl.String,
    "jurisdiction": pl.String,
    "sic_division": pl.String,
    "n_sic_codes": pl.Int32,
}


def profile_static(company_number: str, profile: Mapping[str, Any]) -> dict[str, Any]:
    """The allowed static fields of a profile, normalised."""
    allowed = {k: v for k, v in profile.items() if k in ALLOWED_FIELDS}
    created = allowed.get("date_of_creation")
    sic_codes = [str(s) for s in allowed.get("sic_codes") or []]
    return {
        "company_number": company_number,
        "date_of_creation": date.fromisoformat(created) if isinstance(created, str) else None,
        "company_type": allowed.get("type"),
        "jurisdiction": allowed.get("jurisdiction"),
        "sic_division": sic_codes[0][:2] if sic_codes else None,
        "n_sic_codes": len(sic_codes),
    }


def profile_features(rows: pl.DataFrame, profiles: pl.DataFrame) -> pl.DataFrame:
    return (
        rows.select("row_id", "company_number", "prediction_date")
        .join(profiles, on="company_number", how="left")
        .select(
            "row_id",
            (
                (pl.col("prediction_date") - pl.col("date_of_creation")).dt.total_days() / 365.25
            ).alias("company_age_years"),
            "company_type",
            "jurisdiction",
            "sic_division",
            "n_sic_codes",
        )
    )
