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

# UK SIC 2007 sections by 2-digit division. 21 levels instead of ~90 divisions: with tens of
# failures, finer industry codes are memorised rather than learned.
_SECTION_BY_DIVISION = {
    **dict.fromkeys(range(1, 4), "A"),
    **dict.fromkeys(range(5, 10), "B"),
    **dict.fromkeys(range(10, 34), "C"),
    35: "D",
    **dict.fromkeys(range(36, 40), "E"),
    **dict.fromkeys(range(41, 44), "F"),
    **dict.fromkeys(range(45, 48), "G"),
    **dict.fromkeys(range(49, 54), "H"),
    **dict.fromkeys(range(55, 57), "I"),
    **dict.fromkeys(range(58, 64), "J"),
    **dict.fromkeys(range(64, 67), "K"),
    68: "L",
    **dict.fromkeys(range(69, 76), "M"),
    **dict.fromkeys(range(77, 83), "N"),
    84: "O",
    85: "P",
    **dict.fromkeys(range(86, 89), "Q"),
    **dict.fromkeys(range(90, 94), "R"),
    **dict.fromkeys(range(94, 97), "S"),
    **dict.fromkeys(range(97, 99), "T"),
    99: "U",
}


def sic_section(sic_code: str) -> str | None:
    """`62020` -> `J`. Companies House uses 99999 for dormant companies (section U here)."""
    digits = sic_code.strip()[:2]
    return _SECTION_BY_DIVISION.get(int(digits)) if digits.isdigit() else None


PROFILE_SCHEMA: dict[str, Any] = {
    "company_number": pl.String,
    "date_of_creation": pl.Date,
    "company_type": pl.String,
    "jurisdiction": pl.String,
    "sic_section": pl.String,
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
        "sic_section": sic_section(sic_codes[0]) if sic_codes else None,
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
            "sic_section",
            "n_sic_codes",
        )
    )
