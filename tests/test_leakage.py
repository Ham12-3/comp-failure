"""Leakage tests (PROJECT_SPEC.md section 6).

The rule: a feature may only use information public on or before the prediction date. The
test that enforces it: for every row, features computed from the *full* API history must be
identical to features computed from a *snapshot* of the register as it stood on that row's
prediction date (later filings removed, later resignations / charge satisfactions undone,
today's status and address replaced by what they were then). If any feature reads a later
event, the two disagree and the test fails.

Run on real data (Carillion: liquidated 2018, office moved to the liquidator in Leeds,
directors resigned, a charge registered after its last accounts) and on a synthetic company
that exercises every event type.
"""

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from cfp.features.build_features import (
    FEATURES,
    KEYS,
    api_data_from_payloads,
    build_feature_table,
    feature_docs_markdown,
)
from cfp.features.profile import ALLOWED_FIELDS, profile_static
from cfp.ingest.accounts_parse import SCHEMA

CARILLION = json.loads(
    (Path(__file__).parent / "fixtures" / "ch_api" / "carillion_full.json").read_text("utf-8")
)

Payload = dict[str, Any]


def accounts_rows(company_number: str, rows: list[tuple[date, date]]) -> pl.DataFrame:
    """Accounts rows (financial columns empty) with (balance sheet date, prediction date)."""
    records = []
    for i, (balance_sheet_date, _) in enumerate(rows):
        record = dict.fromkeys(SCHEMA)
        record.update(
            company_number=company_number,
            balance_sheet_date=balance_sheet_date,
            source_member=f"member_{i}",
        )
        records.append(record)
    frame = pl.DataFrame(records, schema=SCHEMA)
    return frame.with_columns(pl.Series("prediction_date", [p for _, p in rows], pl.Date))


def snapshot(data: Payload, as_of: date, postcode_then: str, status_then: str) -> Payload:
    """What the API would have shown on `as_of`."""
    day = as_of.isoformat()

    def on_or_before(value: str | None) -> bool:
        return value is not None and value[:10] <= day

    profile = {k: v for k, v in data["profile"].items() if k not in ("registered_office_address",)}
    profile.update(
        registered_office_address={"postal_code": postcode_then},
        company_status=status_then,
    )
    for later_only in ("date_of_cessation", "has_insolvency_history", "accounts", "links"):
        profile.pop(later_only, None)
    officers = [
        {**o, "resigned_on": o.get("resigned_on") if on_or_before(o.get("resigned_on")) else None}
        for o in data["officers"]
        if o.get("appointed_on") is None or on_or_before(o["appointed_on"])
    ]
    charges = [
        {
            **c,
            "satisfied_on": c.get("satisfied_on") if on_or_before(c.get("satisfied_on")) else None,
        }
        for c in data["charges"]
        if on_or_before(c.get("delivered_on") or c.get("created_on"))
    ]
    return {
        "profile": profile,
        "filing_history": [f for f in data["filing_history"] if on_or_before(f["date"])],
        "officers": officers,
        "charges": charges,
    }


def same(a: object, b: object) -> bool:
    """Equal, allowing float rounding noise (polars can round the last bit differently
    depending on how many rows it processes at once)."""
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def features_for(number: str, data: Payload, rows: list[tuple[date, date]]) -> list[dict[str, Any]]:
    table = build_feature_table(accounts_rows(number, rows), api_data_from_payloads({number: data}))
    return table.to_dicts()


def assert_same_as_snapshots(
    number: str, data: Payload, rows: list[tuple[date, date]], truth: dict[date, tuple[str, str]]
) -> list[dict[str, Any]]:
    """Full-history features == snapshot features, row by row. Returns the full features."""
    full = features_for(number, data, rows)
    for i, (balance_sheet_date, prediction_date) in enumerate(rows):
        postcode, status = truth[prediction_date]
        snap = snapshot(data, prediction_date, postcode, status)
        expected = features_for(number, snap, [(balance_sheet_date, prediction_date)])[0]
        differing = {
            k: (full[i][k], expected[k]) for k in FEATURES if not same(full[i][k], expected[k])
        }
        assert not differing, f"row {prediction_date}: features use later data: {differing}"
    return full


# ---------------------------------------------------------------- real data: Carillion
def carillion_rows() -> list[tuple[date, date]]:
    """(balance sheet date, filing date) of every accounts filing Carillion made."""
    rows = []
    for item in CARILLION["filing_history"]:
        made_up = (item.get("description_values") or {}).get("made_up_date")
        if item["type"] == "AA" and made_up:
            rows.append((date.fromisoformat(made_up), date.fromisoformat(item["date"])))
    return rows


def carillion_office_then(when: date) -> tuple[str, str]:
    """Registered office postcode from the AD01 records (independent of the feature code)."""
    postcode = "WV1 4HY" if when < date(2015, 3, 2) else "WV3 0SR"
    return postcode, "active"


def test_carillion_features_use_nothing_after_each_filing() -> None:
    rows = carillion_rows()
    assert len(rows) >= 15 and max(p for _, p in rows) == date(2017, 5, 12)
    truth = {p: carillion_office_then(p) for _, p in rows}
    full = assert_same_as_snapshots("03782379", CARILLION, rows, truth)

    last = next(f for f in full if f["prediction_date"] == date(2017, 5, 12))
    # Today's office is the liquidator's in Leeds (LS); in May 2017 it was Wolverhampton.
    assert CARILLION["profile"]["registered_office_address"]["postal_code"].startswith("LS")
    assert last["postcode_area"] == "WV"
    # The charge registered in October 2017 and the 2018 resignations are invisible.
    assert last["charges_registered_12m"] == 1  # only the December 2016 charge
    assert last["directors_in_office"] > 0


# ---------------------------------------------------------------- synthetic: every event type
N = "00000042"
MOVE_DESCRIPTION = "change-registered-office-address-company-with-date-old-address-new-address"
SYNTHETIC: Payload = {
    "profile": {
        "company_number": N,
        "date_of_creation": "2015-01-01",
        "type": "ltd",
        "jurisdiction": "england-wales",
        "sic_codes": ["62020"],
        # Today's state, all of it after both prediction dates:
        "company_status": "dissolved",
        "date_of_cessation": "2023-06-01",
        "registered_office_address": {"postal_code": "CF14 8LH"},  # Companies House default
        "links": {"charges": "/company/00000042/charges"},
    },
    "filing_history": [
        {
            "date": "2019-12-15",
            "category": "accounts",
            "type": "AA",
            "description": "accounts-with-accounts-type-micro-entity",
            "description_values": {"made_up_date": "2018-12-31"},
        },
        {
            "date": "2020-01-20",
            "category": "officers",
            "type": "AP01",
            "description": "appoint-person-director-company-with-name-date",
            "description_values": {},
        },
        {
            "date": "2020-06-30",
            "category": "accounts",
            "type": "AA",
            "description": "accounts-with-accounts-type-micro-entity",
            "description_values": {"made_up_date": "2019-12-31"},
        },
        {
            "date": "2021-02-01",
            "category": "confirmation-statement",
            "type": "CS01",
            "description": "confirmation-statement-with-no-updates",
            "description_values": {"made_up_date": "2021-01-01"},
        },
        {
            "date": "2021-03-05",
            "category": "officers",
            "type": "TM01",
            "description": "termination-director-company-with-name-termination-date",
            "description_values": {},
        },
        {
            "date": "2021-05-01",
            "category": "address",
            "type": "AD01",
            "description": MOVE_DESCRIPTION,
            "description_values": {
                "old_address": "1 High Street, Bristol, BS1 4DJ",
                "new_address": "2 Low Road, Leeds, LS1 4DL",
            },
        },
        {
            "date": "2022-06-30",
            "category": "accounts",
            "type": "AA",
            "description": "accounts-with-accounts-type-micro-entity",
            "description_values": {"made_up_date": "2021-12-31"},
        },
        # Everything below is after both prediction dates.
        {
            "date": "2022-09-01",
            "category": "gazette",
            "type": "GAZ1",
            "description": "gazette-notice-compulsory",
            "description_values": {},
        },
        {
            "date": "2022-10-01",
            "category": "address",
            "type": "AD01",
            "description": MOVE_DESCRIPTION,
            "description_values": {
                "old_address": "2 Low Road, Leeds, LS1 4DL",
                "new_address": "Suite 9, Manchester, M1 1AA",
            },
        },
        {
            "date": "2023-01-01",
            "category": "address",
            "type": "RP05",
            "description": "default-companies-house-registered-office-address-applied",
            "description_values": {},
        },
        {
            "date": "2023-02-01",
            "category": "dissolution",
            "type": "DS01",
            "description": "dissolution-application-strike-off-company",
            "description_values": {},
        },
        {
            "date": "2023-03-01",
            "category": "officers",
            "type": "TM01",
            "description": "termination-director-company-with-name-termination-date",
            "description_values": {},
        },
    ],
    "officers": [
        {"officer_role": "director", "appointed_on": "2015-01-01", "resigned_on": "2021-03-01"},
        {"officer_role": "director", "appointed_on": "2020-01-10", "resigned_on": "2023-03-01"},
        {"officer_role": "secretary", "appointed_on": "2015-01-01"},
        {"officer_role": "director", "appointed_on": "2022-08-01"},
    ],
    "charges": [
        {"delivered_on": "2019-05-01", "satisfied_on": "2021-01-01"},
        {"delivered_on": "2022-07-15", "satisfied_on": "2023-01-15"},
    ],
}
SYNTHETIC_ROWS = [(date(2019, 12, 31), date(2020, 6, 30)), (date(2021, 12, 31), date(2022, 6, 30))]
SYNTHETIC_TRUTH = {
    date(2020, 6, 30): ("BS1 4DJ", "active"),
    date(2022, 6, 30): ("LS1 4DL", "active"),
}


def test_synthetic_features_use_nothing_after_the_prediction_date() -> None:
    full = assert_same_as_snapshots(N, SYNTHETIC, SYNTHETIC_ROWS, SYNTHETIC_TRUTH)
    first, second = full

    assert (first["postcode_area"], second["postcode_area"]) == ("BS", "LS")
    assert (first["directors_in_office"], second["directors_in_office"]) == (2, 1)
    assert (first["charges_outstanding"], second["charges_outstanding"]) == (1, 0)
    assert first["director_appointments_12m"] == 1
    assert second["office_moves_12m"] == 0  # the 2021 move is more than 12 months earlier
    assert second["strike_off_notices_ever"] == 0
    assert second["default_address_applied_ever"] == 0
    assert second["strike_off_applications_ever"] == 0
    assert first["previous_accounts_late_3y"] == 1  # 2018 accounts filed 15 days late
    assert first["accounts_type_filed"] == "micro-entity"


def test_today_status_and_dissolution_do_not_change_features() -> None:
    altered = json.loads(json.dumps(SYNTHETIC))
    altered["profile"].update(company_status="active", date_of_cessation=None)
    altered["profile"]["links"] = {}
    assert features_for(N, altered, SYNTHETIC_ROWS) == features_for(N, SYNTHETIC, SYNTHETIC_ROWS)


# ---------------------------------------------------------------- column-level guards
OUTCOME_COLUMNS = {
    "company_status",
    "date_of_cessation",
    "dissolution_date",
    "excluded_reason",
    "insolvency_start_date",
    "insolvency_case_type",
    "solvent_liquidation_12m",
    "horizon_end",
    "already_insolvent",
}


def test_no_outcome_column_is_a_feature() -> None:
    bad = {f for f in FEATURES if f in OUTCOME_COLUMNS or f.startswith("label_")}
    assert not bad


def test_feature_table_has_exactly_the_documented_columns() -> None:
    table = build_feature_table(
        accounts_rows(N, SYNTHETIC_ROWS), api_data_from_payloads({N: SYNTHETIC})
    )
    assert table.columns == [*KEYS, *FEATURES]
    docs = feature_docs_markdown()
    assert all(f"`{name}`" in docs for name in FEATURES)


@pytest.mark.parametrize("leaky", ["company_status", "date_of_cessation", "accounts", "links"])
def test_profile_reader_ignores_current_state_fields(leaky: str) -> None:
    base = {"date_of_creation": "2015-01-01", "type": "ltd", "sic_codes": ["62020"]}
    assert leaky not in ALLOWED_FIELDS
    assert profile_static(N, {**base, leaky: "anything"}) == profile_static(N, base)
