"""Tests for the label definition (PROJECT_SPEC.md section 4).

Real API responses from tests/fixtures/ch_api/ are used where possible:
* Carillion plc (03782379): compulsory liquidation, wound up 2018-01-15; last accounts filed
  2017-05-12 for the year to 2016-12-31.
* Tesco plc (00445790): active, insolvency endpoint returns 404.
Other cases (solvent liquidation, strike-off, missing dates) use small hand-built payloads in
the same documented shape.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from cfp.labels.build_labels import (
    attach_filing_dates,
    build_labels,
    case_start_date,
    cases_frame,
    flatten_accounts_filings,
    flatten_insolvency,
    has_insolvency_resource,
    profile_row,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ch_api"
CARILLION = "03782379"
TESCO = "00445790"
CUTOFF = date(2026, 11, 1)


def body(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))["body"]


def filings(*rows: tuple[str, date]) -> pl.DataFrame:
    """Accounts rows with a known prediction date."""
    return pl.DataFrame(
        {
            "company_number": [r[0] for r in rows],
            "balance_sheet_date": [date(2000, 1, 1)] * len(rows),
            "prediction_date": [r[1] for r in rows],
        }
    )


def profiles(*payloads: tuple[str, Any]) -> pl.DataFrame:
    return pl.DataFrame([profile_row(n, p) for n, p in payloads])


@pytest.fixture
def real_cases() -> pl.DataFrame:
    return cases_frame(
        flatten_insolvency(CARILLION, body("insolvency_03782379.json"))
        + flatten_insolvency(TESCO, body("insolvency_00445790.json"))
    )


@pytest.fixture
def real_profiles() -> pl.DataFrame:
    return profiles(
        (CARILLION, body("company_profile_03782379.json")),
        (TESCO, body("company_profile_00445790.json")),
    )


# ---------------------------------------------------------------- parsing real responses
def test_flatten_real_insolvency_response() -> None:
    rows = flatten_insolvency(CARILLION, body("insolvency_03782379.json"))
    assert rows == [
        {
            "company_number": CARILLION,
            "case_number": "1",
            "case_type": "compulsory-liquidation",
            "case_category": "failure",
            "start_date": date(2018, 1, 15),  # wound-up-on, not petitioned-on
            "start_date_source": "case_date",
            "end_date": None,  # the liquidation is still open
        }
    ]
    assert flatten_insolvency(TESCO, body("insolvency_00445790.json")) == []  # 404


def test_insolvency_link_is_read_from_real_profiles() -> None:
    assert has_insolvency_resource(body("company_profile_03782379.json"))
    assert not has_insolvency_resource(body("company_profile_00445790.json"))


def test_flatten_real_accounts_filings() -> None:
    rows = flatten_accounts_filings(
        CARILLION, body("filing_history_accounts_03782379.json")["items"]
    )
    assert rows[0] == {
        "company_number": CARILLION,
        "filing_date": date(2017, 5, 12),
        "made_up_date": date(2016, 12, 31),
    }
    assert len(rows) == 5


def test_non_accounts_filings_are_ignored() -> None:
    items = body("filing_history_03782379.json")["items"]  # address changes, insolvency notices
    assert all(item["category"] != "accounts" for item in items)
    assert flatten_accounts_filings(CARILLION, items) == []


# ---------------------------------------------------------------- case start dates
def test_case_start_date_fallbacks() -> None:
    practitioner_only = {"dates": [], "practitioners": [{"appointed_on": "2020-03-02"}]}
    assert case_start_date(practitioner_only) == (date(2020, 3, 2), "practitioner_appointed")

    petition_only = {"dates": [{"type": "petitioned-on", "date": "2021-06-01"}]}
    assert case_start_date(petition_only) == (date(2021, 6, 1), "earliest_any_date")

    assert case_start_date({"dates": [], "practitioners": []}) == (None, "missing")


def test_case_categories() -> None:
    def category(case_type: str) -> str:
        payload = {"cases": [{"type": case_type, "number": "1", "dates": []}]}
        return str(flatten_insolvency("X", payload)[0]["case_category"])

    assert category("creditors-voluntary-liquidation") == "failure"
    assert category("in-administration") == "failure"
    assert category("receiver-manager") == "failure"
    assert category("members-voluntary-liquidation") == "solvent"
    assert category("corporate-voluntary-arrangement") == "other"


# ---------------------------------------------------------------- prediction dates
def test_attach_filing_dates_prefers_exact_made_up_date() -> None:
    accounts = pl.DataFrame(
        {
            "company_number": [CARILLION, CARILLION],
            "balance_sheet_date": [date(2016, 12, 31), date(2010, 12, 31)],
            "filing_month": [date(2017, 5, 1), date(2011, 5, 1)],
        }
    )
    history = pl.DataFrame(
        flatten_accounts_filings(CARILLION, body("filing_history_accounts_03782379.json")["items"])
    )
    out = attach_filing_dates(accounts, history)
    assert out["prediction_date"].to_list() == [date(2017, 5, 12), date(2011, 5, 31)]
    assert out["filing_date_source"].to_list() == ["filing_history_exact", "month_end"]


# ---------------------------------------------------------------- labels
def test_carillion_last_accounts_are_a_positive(
    real_cases: pl.DataFrame, real_profiles: pl.DataFrame
) -> None:
    rows = filings((CARILLION, date(2017, 5, 12)), (CARILLION, date(2016, 5, 11)))
    out = build_labels(rows, real_cases, real_profiles, CUTOFF)

    # Filed 2017-05-12, wound up 2018-01-15: inside 12 months.
    assert out.row(0, named=True)["label_insolvency_12m"] == 1
    assert out.row(0, named=True)["insolvency_case_type"] == "compulsory-liquidation"
    assert out.row(0, named=True)["insolvency_start_date"] == date(2018, 1, 15)
    # A year earlier the liquidation is outside the window.
    assert out.row(1, named=True)["label_insolvency_12m"] == 0
    assert out["excluded_reason"].to_list() == [None, None]


def test_window_boundaries(real_cases: pl.DataFrame, real_profiles: pl.DataFrame) -> None:
    rows = filings(
        (CARILLION, date(2017, 1, 15)),  # horizon ends exactly on the event: positive
        (CARILLION, date(2017, 1, 14)),  # horizon ends the day before: negative
        (CARILLION, date(2018, 1, 15)),  # event on the prediction date: already insolvent
    )
    out = build_labels(rows, real_cases, real_profiles, CUTOFF)
    assert out["label_insolvency_12m"].to_list()[:2] == [1, 0]
    assert out["excluded_reason"].to_list() == [None, None, "already_insolvent"]


def test_healthy_company_and_unobservable_window(
    real_cases: pl.DataFrame, real_profiles: pl.DataFrame
) -> None:
    rows = filings((TESCO, date(2024, 6, 1)), (TESCO, date(2026, 1, 1)))
    out = build_labels(rows, real_cases, real_profiles, CUTOFF)
    assert out["label_insolvency_12m"].to_list() == [0, 0]
    # 2026-01-01 + 12 months is after the cutoff (minus the reporting lag).
    assert out["excluded_reason"].to_list() == [None, "outcome_not_observable"]


def test_solvent_liquidation_is_not_a_failure() -> None:
    payload = {
        "cases": [
            {
                "type": "members-voluntary-liquidation",
                "number": "1",
                "dates": [
                    {"type": "declaration-solvent-on", "date": "2020-02-01"},
                    {"type": "wound-up-on", "date": "2020-02-10"},
                ],
            }
        ]
    }
    cases = cases_frame(flatten_insolvency("00000001", payload))
    profs = profiles(
        ("00000001", {"company_status": "dissolved", "date_of_cessation": "2021-06-01"})
    )
    out = build_labels(filings(("00000001", date(2019, 6, 1))), cases, profs, CUTOFF)
    row = out.row(0, named=True)
    assert row["label_insolvency_12m"] == 0
    assert row["solvent_liquidation_12m"] is True
    assert row["label_strike_off_12m"] == 0  # dissolved after a liquidation, not struck off


def test_strike_off_label_needs_no_insolvency_case() -> None:
    profs = profiles(
        ("00000002", {"company_status": "dissolved", "date_of_cessation": "2020-03-01"})
    )
    out = build_labels(filings(("00000002", date(2019, 6, 1))), cases_frame([]), profs, CUTOFF)
    row = out.row(0, named=True)
    assert (row["label_insolvency_12m"], row["label_strike_off_12m"]) == (0, 1)


def test_missing_profile_is_excluded() -> None:
    profs = profiles(("00000003", None))
    out = build_labels(filings(("00000003", date(2019, 6, 1))), cases_frame([]), profs, CUTOFF)
    assert out["excluded_reason"].to_list() == ["no_profile"]


def test_outcome_fields_do_not_leave_the_label_step(
    real_cases: pl.DataFrame, real_profiles: pl.DataFrame
) -> None:
    out = build_labels(filings((TESCO, date(2024, 6, 1))), real_cases, real_profiles, CUTOFF)
    assert not {"company_status", "date_of_cessation"} & set(out.columns)


def test_an_ended_case_does_not_make_later_filings_already_insolvent() -> None:
    """Seen in real data: companies rescued from administration in 2009-2010 trade on."""
    payload = {
        "cases": [
            {
                "type": "in-administration",
                "number": "1",
                "dates": [
                    {"type": "administration-started-on", "date": "2010-01-13"},
                    {"type": "administration-ended-on", "date": "2011-01-12"},
                ],
            }
        ]
    }
    cases = cases_frame(flatten_insolvency("00000004", payload))
    assert cases["end_date"].to_list() == [date(2011, 1, 12)]
    profs = profiles(("00000004", {"company_status": "active"}))
    rows = filings(("00000004", date(2010, 6, 1)), ("00000004", date(2019, 6, 1)))
    out = build_labels(rows, cases, profs, CUTOFF)
    # In administration on 2010-06-01: excluded. Rescued by 2019: a normal, usable row.
    assert out["excluded_reason"].to_list() == ["already_insolvent", None]
    assert out["label_insolvency_12m"].to_list() == [0, 0]
