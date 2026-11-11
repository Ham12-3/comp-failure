"""The point-in-time gate for event-based features.

Every non-financial feature is computed from an events table (one row per event, keyed by
`company_number`) joined to the prediction rows with `as_of`, which keeps only events dated on
or before each row's prediction date. Feature modules do not join events to rows any other way.

A few features need to know whether something that existed at the prediction date had ended
by then (a director still in office, a charge still outstanding). They take entities that
*started* on or before the prediction date through `as_of`, then check the end date against
the prediction date. The result is the state of the register on the prediction date.

tests/test_leakage.py checks the whole feature table against a register snapshot taken on each
row's prediction date.
"""

from __future__ import annotations

import polars as pl

ROW_KEYS = ("row_id", "company_number", "prediction_date")


def as_of(rows: pl.DataFrame, events: pl.DataFrame, date_col: str) -> pl.DataFrame:
    """Events of each row's company dated on or before the row's prediction date."""
    return (
        rows.select(ROW_KEYS)
        .join(events, on="company_number", how="inner")
        .filter(pl.col(date_col) <= pl.col("prediction_date"))
    )


def in_last_days(date_col: str, days: int) -> pl.Expr:
    """Event within `days` before the prediction date (use on `as_of` output)."""
    return pl.col(date_col) > pl.col("prediction_date") - pl.duration(days=days)


def per_row(
    rows: pl.DataFrame, features: pl.DataFrame, fill_zero: tuple[str, ...] = ()
) -> pl.DataFrame:
    """Left-join per-row features back onto every row; counts with no events become 0."""
    out = rows.select("row_id").join(features, on="row_id", how="left")
    return out.with_columns(pl.col(list(fill_zero)).fill_null(0)) if fill_zero else out
