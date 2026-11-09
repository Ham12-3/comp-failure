"""Tests for turning accounts files into rows.

Fixtures in tests/fixtures/accounts/:
* Prod224_2412_01234567_20240331.html: synthetic iXBRL covering sign, scale, zero dash,
  dimensions, duplicates and prior-year comparatives.
* Prod224_2510_10851932_20250731.html: a real 2025 micro-entity filing. It shows negative
  net assets in brackets but tags them as positive numbers (no sign="-"): a real data quality
  problem the features step has to handle.
* Prod224_0064_02675493_20181231.xml: a real 2019 dormant-company filing in the older XBRL
  format and taxonomy.
"""

import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from cfp.ingest.accounts_parse import SCHEMA, parse_archive_zip, parse_document

FIXTURES = Path(__file__).parent / "fixtures" / "accounts"
SYNTHETIC = "Prod224_2412_01234567_20240331.html"
REAL_IXBRL = "Prod224_2510_10851932_20250731.html"
REAL_XBRL = "Prod224_0064_02675493_20181231.xml"

Row = dict[str, Any]


def parse_fixture(name: str, archive: str = "Accounts_Monthly_Data-December2024") -> Row:
    return parse_document((FIXTURES / name).read_bytes(), name, archive)


# ---------------------------------------------------------------- synthetic
@pytest.fixture(scope="module")
def synthetic() -> Row:
    return parse_fixture(SYNTHETIC)


def test_identifiers_come_from_file_and_archive_names(synthetic: Row) -> None:
    assert synthetic["parse_error"] is None
    assert synthetic["company_number"] == "01234567"
    assert synthetic["balance_sheet_date"] == date(2024, 3, 31)
    assert synthetic["filing_month"] == date(2024, 12, 1)
    assert synthetic["file_type"] == "html"
    assert synthetic["company_number_in_doc"] == "01234567"


def test_current_and_prior_periods(synthetic: Row) -> None:
    assert synthetic["period_end"] == date(2024, 3, 31)
    assert synthetic["prior_period_end"] == date(2023, 3, 31)
    assert synthetic["fixed_assets"] == 20_000
    assert synthetic["fixed_assets_prev"] == 22_000


def test_number_formats(synthetic: Row) -> None:
    assert synthetic["current_assets"] == 4_500  # "4.5" with scale="3"
    assert synthetic["cash"] == 0  # zero dash "-"
    assert synthetic["cash_prev"] == 750
    assert synthetic["net_assets_prev"] == -3_000  # sign="-"
    assert synthetic["profit_loss"] == -1_200


def test_dimensions(synthetic: Row) -> None:
    assert synthetic["creditors_within_one_year"] == 2_000
    assert synthetic["creditors_within_one_year_prev"] == 1_500
    assert synthetic["creditors_after_one_year"] == 10_000
    # Prior year uses both conventions together: Non-current + AfterOneYear.
    assert synthetic["creditors_after_one_year_prev"] == 11_000
    assert synthetic["creditors_no_maturity"] is None
    # Bank borrowings use the FinancialInstrumentCurrentNon-current convention.
    assert synthetic["bank_borrowings_within_one_year"] == 900
    assert synthetic["bank_borrowings_after_one_year"] is None
    # Share capital (an Equity fact with a dimension) must not replace total equity.
    assert synthetic["equity"] == 12_500


def test_duration_concepts_use_period_end(synthetic: Row) -> None:
    assert synthetic["turnover"] == 100_000
    assert synthetic["turnover_prev"] == 80_000
    assert synthetic["employees"] == 3
    assert synthetic["employees_prev"] == 2


def test_report_metadata(synthetic: Row) -> None:
    assert synthetic["accounting_standard"] == "SmallEntities"
    assert synthetic["accounts_type"] == "FullAccounts"
    assert synthetic["is_dormant"] is False
    assert synthetic["date_authorised"] == date(2024, 12, 15)


def test_missing_concepts_are_null_not_zero(synthetic: Row) -> None:
    # Missingness is a signal, so absent values must stay null.
    assert synthetic["inventories"] is None
    assert synthetic["inventories_prev"] is None
    assert synthetic["profit_loss_prev"] is None


# ---------------------------------------------------------------- real files
def test_real_ixbrl_values_are_kept_as_tagged() -> None:
    row = parse_fixture(REAL_IXBRL, "Accounts_Monthly_Data-October2025")
    assert row["parse_error"] is None
    assert row["company_number"] == "10851932"
    assert row["accounting_standard"] == "Micro-entities"
    assert row["current_assets"] == 1_630
    assert row["creditors_within_one_year"] == 8_215
    # The filing displays "(6,585)" and "(7,385)" but tags both as positive. The parser keeps
    # what is tagged; the features step repairs signs using accounting identities
    # (here current assets - creditors = 1,630 - 8,215 = -6,585).
    assert row["net_current_assets"] == 6_585
    assert row["net_assets"] == row["equity"] == 7_385
    assert row["net_assets_prev"] == 3_830
    assert row["employees"] == 1


def test_real_xbrl_xml_with_older_taxonomy() -> None:
    row = parse_fixture(REAL_XBRL, "Accounts_Monthly_Data-March2019")
    assert row["parse_error"] is None
    assert row["file_type"] == "xml"
    assert row["company_number"] == "02675493"
    assert row["is_dormant"] is True  # CompanyDormant
    assert row["cash"] == 100  # CashBankInHand
    assert row["net_assets"] == 100  # NetAssetsLiabilitiesIncludingPensionAssetLiability
    assert row["equity"] == 100  # ShareholderFunds
    assert row["date_authorised"] == date(2019, 3, 15)  # DateApproval
    assert row["date_received"] == date(2019, 3, 28)  # DateAccountsReceived


# ---------------------------------------------------------------- robustness
def test_garbage_is_recorded_as_error_not_raised() -> None:
    row = parse_document(b"\x00\x01 not xml", "Prod224_2510_00000001_20250331.html", "x")
    assert row["company_number"] == "00000001"
    assert row["n_numeric_facts"] in (None, 0)


def test_unrecognised_member_name() -> None:
    row = parse_document(b"<html/>", "notes.txt", "Accounts_Monthly_Data-October2025")
    assert row["parse_error"] == "unrecognised file name"


def test_parse_archive_zip_builds_typed_frame(tmp_path: Path) -> None:
    zip_path = tmp_path / "Accounts_Monthly_Data-October2025.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name in (SYNTHETIC, REAL_IXBRL, REAL_XBRL):
            zf.write(FIXTURES / name, arcname=name)

    with ProcessPoolExecutor(max_workers=2) as pool:
        frame = parse_archive_zip(zip_path, pool, chunk_size=2)

    assert frame.height == 3
    assert frame.schema == pl.Schema(SCHEMA)
    assert frame["filing_month"].unique().to_list() == [date(2025, 10, 1)]
    assert frame["parse_error"].is_null().all()


def test_document_level_dimension_on_every_context_is_ignored() -> None:
    """Some software tags every context with OriginalRevisedDataDimension = Original.
    Values must come out exactly as if that dimension were absent."""
    original = (
        '<xbrldi:explicitMember dimension="bus:OriginalRevisedDataDimension">'
        "bus:Original</xbrldi:explicitMember>"
    )
    text = (FIXTURES / SYNTHETIC).read_text(encoding="utf-8")
    text = text.replace("<xbrli:segment>", "<xbrli:segment>" + original)
    text = text.replace(
        "</xbrli:identifier></xbrli:entity>",
        f"</xbrli:identifier><xbrli:segment>{original}</xbrli:segment></xbrli:entity>",
    )
    assert text.count("OriginalRevisedDataDimension") >= 12  # every context got one

    tagged = parse_document(text.encode(), SYNTHETIC, "Accounts_Monthly_Data-December2024")
    plain = parse_fixture(SYNTHETIC)
    assert tagged == plain
