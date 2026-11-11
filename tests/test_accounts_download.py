"""Tests for sampling accounts out of remote zip archives with HTTP range requests."""

import io
import zipfile
from datetime import date
from pathlib import Path

import httpx
import polars as pl
import pytest

from cfp.ingest.accounts_download import (
    Archive,
    DownloadBudget,
    DownloadBudgetExceededError,
    Manifest,
    download_archive_sample,
    fetch_member,
    filing_month_from_name,
    in_sample,
    parse_archive_links,
    parse_member_name,
    read_index,
)

URL = "https://download.example/Accounts_Monthly_Data-October2025.zip"
MEMBERS = {
    "Prod224_2510_00000086_20250331.html": b"<html>" + b"a" * 5000 + b"</html>",
    "Prod224_2510_SC307944_20250630.html": b"<html>scottish company</html>",
    "Prod224_2510_OC300001_20241231.xml": b"<xbrl>" + b"x" * 3000 + b"</xbrl>",
    "Prod224_2510_12345678_20250131.zip": b"nested zips are skipped",
}


def build_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for i, (name, data) in enumerate(MEMBERS.items()):
            # Mix compression methods: the real archives use deflate, but be robust.
            method = zipfile.ZIP_STORED if i == 1 else zipfile.ZIP_DEFLATED
            zf.writestr(name, data, compress_type=method)
    return buffer.getvalue()


class RangeServer:
    """Mock S3: answers HEAD with the size and GET with Range as 206 Partial Content."""

    def __init__(self, blob: bytes) -> None:
        self.blob = blob
        self.bytes_served = 0
        self.requests = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(self.blob))})
        start_s, end_s = request.headers["Range"].removeprefix("bytes=").split("-")
        start, end = int(start_s), min(int(end_s), len(self.blob) - 1)
        chunk = self.blob[start : end + 1]
        self.bytes_served += len(chunk)
        return httpx.Response(206, content=chunk)


@pytest.fixture
def server() -> RangeServer:
    return RangeServer(build_zip())


@pytest.fixture
def client(server: RangeServer) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(server))


ARCHIVE = Archive(date(2025, 10, 1), URL)


# ---------------------------------------------------------------- names and links
def test_parse_member_name() -> None:
    assert parse_member_name("Prod224_2510_SC307944_20250630.html") == (
        "SC307944",
        date(2025, 6, 30),
        "html",
    )
    assert parse_member_name("Prod224_0064_00484322_20180630.xml") == (
        "00484322",
        date(2018, 6, 30),
        "xml",
    )
    assert parse_member_name("Prod224_2510_12345678_20250131.zip") is None
    assert parse_member_name("Prod224_2510_12345678_20251399.html") is None  # bad date
    assert parse_member_name("readme.txt") is None


def test_filing_month_from_name() -> None:
    assert filing_month_from_name("Accounts_Monthly_Data-October2025") == date(2025, 10, 1)
    assert filing_month_from_name("archive/Accounts_Monthly_Data-March2019.zip") == date(2019, 3, 1)
    assert filing_month_from_name("Accounts_Monthly_Data-JanToDec2008.zip") is None


def test_parse_archive_links_resolves_relative_urls_and_skips_yearly_files() -> None:
    html = """
      <a href="archive/Accounts_Monthly_Data-JanToDec2008.zip">2008</a>
      <a href="archive/Accounts_Monthly_Data-March2019.zip">Mar 2019</a>
      <a href="Accounts_Monthly_Data-October2025.zip">Oct 2025 (2.2GiB)</a>
    """
    page = "https://download.companieshouse.gov.uk/historicmonthlyaccountsdata.html"
    archives = parse_archive_links(html, page)
    assert archives == [
        Archive(
            date(2019, 3, 1),
            "https://download.companieshouse.gov.uk/archive/Accounts_Monthly_Data-March2019.zip",
        ),
        Archive(
            date(2025, 10, 1),
            "https://download.companieshouse.gov.uk/Accounts_Monthly_Data-October2025.zip",
        ),
    ]
    assert archives[1].name == "Accounts_Monthly_Data-October2025"


# ---------------------------------------------------------------- sampling
def test_in_sample_is_deterministic_and_close_to_rate() -> None:
    numbers = [f"{i:08d}" for i in range(50_000)]
    first = [n for n in numbers if in_sample(n, 0.05)]
    second = [n for n in numbers if in_sample(n, 0.05)]
    assert first == second
    assert 0.045 < len(first) / len(numbers) < 0.055


def test_in_sample_is_nested_across_rates() -> None:
    # A bigger sample always contains the smaller one, so the sample can be grown later.
    numbers = [f"{i:08d}" for i in range(20_000)]
    small = {n for n in numbers if in_sample(n, 0.01)}
    large = {n for n in numbers if in_sample(n, 0.05)}
    assert small <= large


# ---------------------------------------------------------------- remote zip reads
def test_read_index_lists_accounts_files_from_central_directory(
    client: httpx.Client, server: RangeServer
) -> None:
    budget = DownloadBudget(10**9)
    index = read_index(client, ARCHIVE, budget)

    assert index.height == 3  # the nested .zip is skipped
    assert set(index["company_number"]) == {"00000086", "SC307944", "OC300001"}
    assert index.filter(pl.col("company_number") == "OC300001")["file_type"].item() == "xml"
    # Only the directory was read, not the whole archive, and the budget saw every byte.
    assert budget.used == server.bytes_served < len(server.blob)


def test_fetch_member_returns_exact_bytes(client: httpx.Client) -> None:
    index = read_index(client, ARCHIVE, DownloadBudget(10**9))
    for row in index.to_dicts():
        assert fetch_member(client, URL, row) == MEMBERS[row["member"]]


def test_fetch_member_detects_corruption(client: httpx.Client) -> None:
    row = read_index(client, ARCHIVE, DownloadBudget(10**9)).to_dicts()[0]
    row["crc"] = row["crc"] ^ 1
    with pytest.raises(ValueError, match="CRC"):
        fetch_member(client, URL, row)


# ---------------------------------------------------------------- full archive step
def test_download_archive_sample_writes_local_zip(client: httpx.Client, tmp_path: Path) -> None:
    budget = DownloadBudget(10**9)
    record = download_archive_sample(
        client,
        ARCHIVE,
        rate=1.0,  # take every company
        raw_dir=tmp_path / "raw",
        index_dir=tmp_path / "index",
        budget=budget,
        workers=2,
    )
    assert record.files_in_archive == record.files_sampled == 3
    with zipfile.ZipFile(tmp_path / "raw" / f"{ARCHIVE.name}.zip") as zf:
        assert {name: zf.read(name) for name in zf.namelist()} == {
            k: v for k, v in MEMBERS.items() if not k.endswith(".zip")
        }
    assert (tmp_path / "index" / f"{ARCHIVE.name}.parquet").exists()


def test_cached_index_is_not_downloaded_again(
    client: httpx.Client, server: RangeServer, tmp_path: Path
) -> None:
    kwargs = {"raw_dir": tmp_path / "raw", "index_dir": tmp_path / "index", "workers": 1}
    download_archive_sample(client, ARCHIVE, rate=0.0, budget=DownloadBudget(10**9), **kwargs)  # type: ignore[arg-type]
    requests_before = server.requests
    record = download_archive_sample(
        client,
        ARCHIVE,
        rate=0.0,
        budget=DownloadBudget(10**9),
        **kwargs,  # type: ignore[arg-type]
    )
    assert server.requests == requests_before
    assert record.index_bytes == 0


def test_budget_stops_download_before_exceeding_limit(
    client: httpx.Client, server: RangeServer, tmp_path: Path
) -> None:
    budget = DownloadBudget(max_bytes=100)  # room for the zip's end record, not its directory
    with pytest.raises(DownloadBudgetExceededError):
        download_archive_sample(
            client,
            ARCHIVE,
            rate=1.0,
            raw_dir=tmp_path / "raw",
            index_dir=tmp_path / "index",
            budget=budget,
        )
    # Every byte served was charged, and the refused read never went out.
    assert server.bytes_served == budget.used <= budget.max_bytes
    assert not (tmp_path / "raw").exists()


def test_manifest_round_trip_feeds_budget(client: httpx.Client, tmp_path: Path) -> None:
    record = download_archive_sample(
        client,
        ARCHIVE,
        rate=1.0,
        raw_dir=tmp_path / "raw",
        index_dir=tmp_path / "index",
        budget=DownloadBudget(10**9),
    )
    manifest = Manifest(tmp_path / "manifest.json")
    manifest.add(record)

    reloaded = Manifest(tmp_path / "manifest.json")
    assert reloaded.records[ARCHIVE.name] == record
    assert reloaded.total_bytes == record.index_bytes + record.sample_bytes > 0
