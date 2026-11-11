"""Tests for loading the Free Company Data Product CSV.

tests/fixtures/company_data_header.csv is the real header line of
BasicCompanyDataAsOneFile-2026-11-01.csv (note the stray leading spaces in some names).
The rows below are made up.
"""

import csv
from datetime import date
from pathlib import Path

from cfp.ingest.company_data import COLUMNS, latest_snapshot_url, load_snapshot

HEADER = (Path(__file__).parent / "fixtures" / "company_data_header.csv").read_text(
    encoding="utf-8"
)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    names = next(csv.reader([HEADER.strip()]))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(names)
        for row in rows:
            writer.writerow([row.get(n.strip(), "") for n in names])


def test_real_header_has_every_column_we_use() -> None:
    names = {n.strip() for n in next(csv.reader([HEADER.strip()]))}
    assert set(COLUMNS) <= names


def test_load_snapshot_types_and_normalises(tmp_path: Path) -> None:
    path = tmp_path / "companies.csv"
    write_csv(
        path,
        [
            {
                "CompanyName": "EXAMPLE TRADING LTD",
                "CompanyNumber": "566433",
                "CompanyStatus": "Active",
                "IncorporationDate": "07/03/2019",
                "RegAddress.PostCode": "CF14 3UZ",
                "Mortgages.NumMortCharges": "2",
                "Mortgages.NumMortOutstanding": "1",
                "SICCode.SicText_1": "62020 - Information technology consultancy activities",
            },
            {
                "CompanyName": "SCOTTISH EXAMPLE LTD",
                "CompanyNumber": "SC123456",
                "CompanyStatus": "Liquidation",
                "IncorporationDate": "15/12/2001",
                "Mortgages.NumMortCharges": "0",
            },
        ],
    )
    frame = load_snapshot(path, "2026-11-01")

    assert frame["company_number"].to_list() == ["00566433", "SC123456"]
    assert frame["incorporation_date"].to_list() == [date(2019, 3, 7), date(2001, 12, 15)]
    assert frame["charges_outstanding"].to_list() == [1, None]
    assert frame["postcode"].to_list() == ["CF14 3UZ", None]  # empty string -> null
    assert frame["snapshot_date"].unique().to_list() == [date(2026, 11, 1)]


def test_latest_snapshot_url_picks_newest() -> None:
    html = """
      <a href="BasicCompanyDataAsOneFile-2026-10-01.zip">old</a>
      <a href="BasicCompanyDataAsOneFile-2026-11-01.zip">new</a>
      <a href="BasicCompanyData-2026-11-01-part1_7.zip">part</a>
    """
    url, snapshot = latest_snapshot_url(html)
    assert url.endswith("BasicCompanyDataAsOneFile-2026-11-01.zip")
    assert snapshot == "2026-11-01"
