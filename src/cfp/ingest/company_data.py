"""Download and load the Free Company Data Product (`BasicCompanyDataAsOneFile-*.zip`).

One CSV row per company currently on the register (live companies, plus companies in
liquidation, administration etc. that are not yet dissolved): status, incorporation date,
SIC codes, registered office postcode, accounts category, charge counts...

**Leakage warning.** This is a snapshot of *today*. Fields that describe the present state
(`company_status`, `dissolution_date`, `accounts_next_due_date`, charge counts...) must never be
used as features for a past prediction date. They are kept for labels, sanity checks and
filtering only. Fields that are fixed at incorporation (`incorporation_date`) are safe; SIC
code and postcode can change over time, so using today's value is an approximation that is
documented in docs/DATA.md.

Dissolved companies are *not* in this file, so it cannot build failure labels on its own.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl

from cfp.ingest.accounts_download import GB, USER_AGENT, DownloadBudget, Manifest

logger = logging.getLogger(__name__)

LISTING_PAGE = "https://download.companieshouse.gov.uk/en_output.html"
_LINK_RE = re.compile(
    r'href="(?P<href>BasicCompanyDataAsOneFile-(?P<date>\d{4}-\d{2}-\d{2})\.zip)"'
)

# CSV header (as published, after stripping spaces) -> our column name. Taken from the real
# file header; see tests/fixtures/company_data_header.csv.
COLUMNS: dict[str, str] = {
    "CompanyNumber": "company_number",
    "CompanyName": "company_name",
    "CompanyCategory": "company_category",
    "CompanyStatus": "company_status",
    "CountryOfOrigin": "country_of_origin",
    "DissolutionDate": "dissolution_date",
    "IncorporationDate": "incorporation_date",
    "RegAddress.PostTown": "post_town",
    "RegAddress.County": "county",
    "RegAddress.Country": "country",
    "RegAddress.PostCode": "postcode",
    "Accounts.AccountCategory": "accounts_category",
    "Accounts.LastMadeUpDate": "accounts_last_made_up_date",
    "Accounts.NextDueDate": "accounts_next_due_date",
    "Mortgages.NumMortCharges": "charges_total",
    "Mortgages.NumMortOutstanding": "charges_outstanding",
    "Mortgages.NumMortPartSatisfied": "charges_part_satisfied",
    "Mortgages.NumMortSatisfied": "charges_satisfied",
    "SICCode.SicText_1": "sic_text_1",
    "SICCode.SicText_2": "sic_text_2",
    "SICCode.SicText_3": "sic_text_3",
    "SICCode.SicText_4": "sic_text_4",
}
DATE_COLUMNS = (
    "dissolution_date",
    "incorporation_date",
    "accounts_last_made_up_date",
    "accounts_next_due_date",
)
# Snapshot statuses meaning "on the register and not in any insolvency procedure".
SNAPSHOT_ACTIVE_STATUSES = ("Active", "Active - Proposal to Strike off")
INT_COLUMNS = (
    "charges_total",
    "charges_outstanding",
    "charges_part_satisfied",
    "charges_satisfied",
)


def latest_snapshot_url(html: str) -> tuple[str, str]:
    """(url, snapshot date) of the newest single-file snapshot on the listing page."""
    matches = sorted(_LINK_RE.finditer(html), key=lambda m: m["date"])
    if not matches:
        raise RuntimeError("No BasicCompanyDataAsOneFile link found on the listing page")
    best = matches[-1]
    return f"https://download.companieshouse.gov.uk/{best['href']}", best["date"]


def _ledger_path(raw_dir: Path) -> Path:
    return raw_dir / "company_data" / "manifest.json"


def downloaded_bytes(raw_dir: Path) -> int:
    """Bytes downloaded so far by every step that counts towards the budget."""
    total = Manifest(raw_dir / "accounts_sample" / "manifest.json").total_bytes
    ledger = _ledger_path(raw_dir)
    if ledger.exists():
        total += sum(e["bytes"] for e in json.loads(ledger.read_text(encoding="utf-8")))
    return total


def download_snapshot(
    client: httpx.Client, url: str, raw_dir: Path, budget: DownloadBudget
) -> Path:
    """Stream the zip to data/raw/company_data/, refusing to start if it would bust the budget."""
    out_dir = raw_dir / "company_data"
    target = out_dir / url.rsplit("/", 1)[-1]
    if target.exists():
        return target
    size = int(client.head(url).headers["content-length"])
    budget.check(size)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".zip.tmp")
    with client.stream("GET", url) as response, tmp.open("wb") as fh:
        response.raise_for_status()
        for chunk in response.iter_bytes(1 << 20):
            fh.write(chunk)
    tmp.replace(target)
    budget.charge(size)

    ledger = _ledger_path(raw_dir)
    entries = json.loads(ledger.read_text(encoding="utf-8")) if ledger.exists() else []
    entries.append(
        {
            "file": target.name,
            "url": url,
            "bytes": size,
            "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    ledger.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return target


def load_snapshot(csv_path: Path, snapshot_date: str) -> pl.DataFrame:
    """Read the CSV into typed columns. Dates in the file are dd/mm/yyyy."""
    frame = pl.read_csv(csv_path, infer_schema=False, encoding="utf8-lossy")
    frame = frame.rename({c: c.strip() for c in frame.columns})
    missing = sorted(set(COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Columns missing from the company data CSV: {missing}")
    frame = frame.select([pl.col(src).str.strip_chars().alias(dst) for src, dst in COLUMNS.items()])
    frame = frame.with_columns(
        pl.col(DATE_COLUMNS).str.to_date("%d/%m/%Y", strict=False),
        pl.col(INT_COLUMNS).cast(pl.Int32, strict=False),
        pl.col("company_number").str.to_uppercase().str.zfill(8),
        pl.lit(snapshot_date).str.to_date().alias("snapshot_date"),
    )
    # Empty strings mean "not present".
    return frame.with_columns(pl.col(pl.String).replace("", None))


def main() -> None:
    from cfp.config import get_settings

    parser = argparse.ArgumentParser(description="Download and load the company data snapshot.")
    parser.add_argument("--max-gb", type=float, default=5.0, help="Total download budget.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = get_settings()
    budget = DownloadBudget(int(args.max_gb * GB), used=downloaded_bytes(settings.raw_dir))
    with httpx.Client(timeout=300, headers={"User-Agent": USER_AGENT}, follow_redirects=True) as c:
        page = c.get(LISTING_PAGE)
        page.raise_for_status()
        url, snapshot_date = latest_snapshot_url(page.text)
        logger.info(
            "Snapshot %s; %.2f GB of %.2f GB budget used so far",
            url,
            budget.used / GB,
            budget.max_bytes / GB,
        )
        zip_path = download_snapshot(c, url, settings.raw_dir, budget)

    extract_dir = settings.raw_dir / "company_data" / "extracted"
    with zipfile.ZipFile(zip_path) as zf:
        (member,) = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        csv_path = extract_dir / Path(member).name
        if not csv_path.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, csv_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    frame = load_snapshot(csv_path, snapshot_date)
    out = settings.interim_dir / "company_snapshot.parquet"
    frame.write_parquet(out)
    logger.info("Wrote %s: %d companies (snapshot %s)", out, frame.height, snapshot_date)
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_hide_dataframe_shape=True, tbl_rows=20):
        print(frame.group_by("company_status").len().sort("len", descending=True))


if __name__ == "__main__":
    main()
