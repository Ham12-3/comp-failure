"""Tests for financial features, including sign repair on the real mis-signed filing."""

from pathlib import Path

import polars as pl
import pytest

from cfp.features.financial import financial_features, repair_signs
from cfp.ingest.accounts_parse import SCHEMA, parse_document

FIXTURES = Path(__file__).parent / "fixtures" / "accounts"


def parsed(name: str, archive: str) -> pl.DataFrame:
    row = parse_document((FIXTURES / name).read_bytes(), name, archive)
    return pl.DataFrame([row], schema=SCHEMA).with_row_index("row_id")


@pytest.fixture(scope="module")
def mis_signed() -> pl.DataFrame:
    # Real filing: shows (6,585) and (7,385) but tags both as positive.
    return parsed("Prod224_2510_10851932_20250731.html", "Accounts_Monthly_Data-October2025")


def test_sign_repair_fixes_the_real_mis_signed_filing(mis_signed: pl.DataFrame) -> None:
    fixed = repair_signs(mis_signed).row(0, named=True)
    assert fixed["net_current_assets"] == -6_585  # 1,630 - 8,215
    assert fixed["total_assets_less_current_liabilities"] == -6_585
    assert fixed["net_assets"] == -7_385
    assert fixed["equity"] == -7_385
    assert fixed["net_current_assets_prev"] == -2_580
    assert fixed["net_assets_prev"] == -3_830
    assert fixed["sign_repaired"] is True


def test_sign_repair_leaves_consistent_filings_alone() -> None:
    synthetic = parsed("Prod224_2412_01234567_20240331.html", "Accounts_Monthly_Data-December2024")
    before = synthetic.row(0, named=True)
    after = repair_signs(synthetic).row(0, named=True)
    for col in ("net_current_assets", "net_assets", "equity", "net_assets_prev"):
        assert after[col] == before[col]
    assert after["sign_repaired"] is False


def accounts_row(**values: float) -> pl.DataFrame:
    """One parsed-accounts row: the given values, everything else missing."""
    row = dict.fromkeys(SCHEMA)
    row.update(values)
    return pl.DataFrame([row], schema=SCHEMA).with_row_index("row_id")


def test_genuinely_negative_values_stay_negative() -> None:
    frame = accounts_row(
        current_assets=1_000.0,
        creditors_within_one_year=3_000.0,
        net_current_assets=-2_000.0,  # correctly tagged negative
        total_assets_less_current_liabilities=-2_000.0,
        net_assets=-2_000.0,
        equity=-2_000.0,
    )
    out = repair_signs(frame).row(0, named=True)
    assert out["net_assets"] == -2_000
    assert out["sign_repaired"] is False


def test_never_negative_balances_tagged_negative_are_made_positive() -> None:
    """The reverse error, in about 3% of real filings: creditors shown in brackets as a
    deduction and tagged with sign="-", even current assets and employee counts."""
    frame = accounts_row(
        current_assets=-1_000.0,
        creditors_within_one_year=-3_000.0,
        net_current_assets=-2_000.0,
        employees=-4.0,
        employees_prev=3.0,
    )
    out = repair_signs(frame).row(0, named=True)
    assert (out["current_assets"], out["creditors_within_one_year"]) == (1_000, 3_000)
    assert (out["employees"], out["employees_prev"]) == (4, 3)
    assert out["net_current_assets"] == -2_000  # consistent with the corrected inputs
    assert out["sign_repaired"] is True


def test_features_on_the_real_filing(mis_signed: pl.DataFrame) -> None:
    f = financial_features(mis_signed).row(0, named=True)
    assert f["net_assets"] == -7_385
    assert f["net_assets_negative"] is True
    assert f["current_liabilities"] == 8_215
    assert f["current_ratio"] == pytest.approx(1_630 / 8_215)
    # The prior year is mis-signed the same way (1,644 - 4,224 = -2,580, tagged +2,580), so
    # net assets were already negative: -3,830 -> -7,385. Not "turned negative".
    assert f["net_assets_change"] == -7_385 - -3_830
    assert f["net_assets_turned_negative"] is False
    assert f["accounting_standard"] == "Micro-entities"
    assert f["missing_turnover"] is True


def test_ratios_do_not_divide_by_zero() -> None:
    synthetic = parsed("Prod224_2412_01234567_20240331.html", "Accounts_Monthly_Data-December2024")
    zeroed = synthetic.with_columns(
        pl.lit(0.0).alias("creditors_within_one_year"),
        pl.lit(0.0).alias("cash_prev"),
    )
    f = financial_features(zeroed).row(0, named=True)
    assert f["cash_pct_change"] is None
    assert f["cash_to_current_liabilities"] is None or f["current_liabilities"] != 0
