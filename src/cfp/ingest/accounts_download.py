"""Download a sample of company accounts from the Companies House Accounts Data Product.

Each monthly archive (`Accounts_Monthly_Data-October2025.zip`, ~2 GB) holds every
electronically filed set of accounts *filed in that month*, one file per filing, named
`Prod224_2510_<company number>_<balance sheet date>.html`.

We do not download whole archives. The files are served from S3, which supports HTTP range
requests, so for each archive we:

1. read only the zip's central directory (~1.4% of the archive) to get an index of every
   filing and its byte offset, cached as Parquet in `data/interim/accounts_index/`;
2. keep the filings of companies whose number hashes into the sample (`in_sample`). Hashing
   the company number, rather than picking random files, keeps the *same* companies in every
   month, so each sampled company contributes all of its yearly filings;
3. fetch just those members with range requests and store them, byte-for-byte, in a local zip
   per archive under `data/raw/accounts_sample/`.

This turns ~200 GB of archives into a few GB of downloads. A byte budget (default 5 GB,
persisted across runs in the manifest) stops the run before it can exceed the limit.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import io
import json
import logging
import re
import struct
import zipfile
import zlib
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
import polars as pl

logger = logging.getLogger(__name__)

LISTING_PAGES = (
    "https://download.companieshouse.gov.uk/historicmonthlyaccountsdata.html",
    "https://download.companieshouse.gov.uk/en_monthlyaccountsdata.html",
)
USER_AGENT = "cfp-company-failure-predictor/0.1"
SAMPLE_SALT = "cfp-v1"
# Measured on Oct 2025 (32 MB directory / 2,329 MB archive) and Mar 2019 archives.
CENTRAL_DIRECTORY_RATIO = 0.015
GB = 1_000_000_000

_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_ARCHIVE_HREF_RE = re.compile(
    r'href="(?P<href>[^"]*Accounts_Monthly_Data-(?P<month>[A-Za-z]+)(?P<year>\d{4})\.zip)"'
)
_ARCHIVE_NAME_RE = re.compile(r"Accounts_Monthly_Data-(?P<month>[A-Za-z]+)(?P<year>\d{4})")
_MEMBER_RE = re.compile(
    r"^Prod\d+_\d+_(?P<company>[A-Za-z0-9]{8})_(?P<date>\d{8})\.(?P<ext>html|xml)$"
)
_LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")  # zip local file header, 30 bytes
_LOCAL_HEADER_PAD = 1024  # room for the local "extra" field, whose length we learn on read


class DownloadBudgetExceededError(RuntimeError):
    """Raised before a download that would take total bytes over the budget."""


# ---------------------------------------------------------------- archives
@dataclass(frozen=True, order=True)
class Archive:
    filing_month: date  # first day of the month the accounts in it were filed
    url: str

    @property
    def name(self) -> str:
        return self.url.rsplit("/", 1)[-1].removesuffix(".zip")


def filing_month_from_name(name: str) -> date | None:
    """`Accounts_Monthly_Data-October2025` -> 2025-10-01. None for multi-month archives."""
    match = _ARCHIVE_NAME_RE.search(name)
    if not match or match["month"].lower() not in _MONTHS:
        return None
    return date(int(match["year"]), _MONTHS[match["month"].lower()], 1)


def parse_archive_links(html: str, page_url: str) -> list[Archive]:
    """Find monthly archive links on a listing page. Skips multi-month files (2008-2009)."""
    archives = []
    for match in _ARCHIVE_HREF_RE.finditer(html):
        month = filing_month_from_name(match["href"])
        if month is not None:
            archives.append(Archive(month, urljoin(page_url, match["href"])))
    return archives


def list_archives(client: httpx.Client) -> list[Archive]:
    """All monthly archives on the historic and current listing pages, one per month."""
    by_month: dict[date, Archive] = {}
    for page in LISTING_PAGES:
        response = client.get(page)
        response.raise_for_status()
        for archive in parse_archive_links(response.text, page):
            by_month.setdefault(archive.filing_month, archive)
    return sorted(by_month.values())


# ---------------------------------------------------------------- budget
class DownloadBudget:
    def __init__(self, max_bytes: int, used: int = 0) -> None:
        self.max_bytes = max_bytes
        self.used = used

    def check(self, planned: int) -> None:
        if self.used + planned > self.max_bytes:
            raise DownloadBudgetExceededError(
                f"Downloading {planned / GB:.2f} GB more would bring the total to "
                f"{(self.used + planned) / GB:.2f} GB, over the {self.max_bytes / GB:.2f} GB "
                "budget. Raise --max-gb only after agreeing it."
            )

    def charge(self, n: int) -> None:
        self.used += n


# ---------------------------------------------------------------- range reads
class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests, so `zipfile` can read a remote
    archive's central directory without downloading the archive."""

    def __init__(self, client: httpx.Client, url: str, budget: DownloadBudget) -> None:
        super().__init__()
        response = client.head(url)
        response.raise_for_status()
        self.size = int(response.headers["content-length"])
        self._client = client
        self._url = url
        self._budget = budget
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self.size + offset
        return self._pos

    def read(self, size: int | None = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self._pos
        end = min(self._pos + size, self.size)
        if end <= self._pos:
            return b""
        self._budget.check(end - self._pos)
        data = range_get(self._client, self._url, self._pos, end)
        self._budget.charge(len(data))
        self._pos = end
        return data

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def range_get(client: httpx.Client, url: str, start: int, end: int) -> bytes:
    """Bytes [start, end) of `url`. Fails loudly if the server ignores the Range header."""
    response = client.get(url, headers={"Range": f"bytes={start}-{end - 1}"})
    if response.status_code != 206:
        raise RuntimeError(f"Expected 206 Partial Content from {url}, got {response.status_code}")
    return response.content


# ---------------------------------------------------------------- index
INDEX_SCHEMA = {
    "member": pl.String,
    "company_number": pl.String,
    "balance_sheet_date": pl.Date,
    "file_type": pl.String,
    "header_offset": pl.Int64,
    "compress_size": pl.Int64,
    "file_size": pl.Int64,
    "crc": pl.Int64,
    "compress_type": pl.Int32,
}


def parse_member_name(name: str) -> tuple[str, date, str] | None:
    """`Prod224_2510_SC123456_20250331.html` -> ("SC123456", 2025-03-31, "html").

    None for anything else (e.g. the few nested .zip members, which we skip).
    """
    match = _MEMBER_RE.match(name)
    if not match:
        return None
    raw = match["date"]
    try:
        balance_sheet_date = date(int(raw[:4]), int(raw[4:6]), int(raw[6:]))
    except ValueError:
        return None
    return match["company"].upper(), balance_sheet_date, match["ext"].lower()


def read_index(client: httpx.Client, archive: Archive, budget: DownloadBudget) -> pl.DataFrame:
    """Index of every accounts file in a remote archive, read from its central directory."""
    with zipfile.ZipFile(HttpRangeFile(client, archive.url, budget)) as zf:
        infos = zf.infolist()
    rows = []
    skipped = 0
    for info in infos:
        parsed = parse_member_name(info.filename)
        if parsed is None:
            skipped += 1
            continue
        company_number, balance_sheet_date, file_type = parsed
        rows.append(
            {
                "member": info.filename,
                "company_number": company_number,
                "balance_sheet_date": balance_sheet_date,
                "file_type": file_type,
                "header_offset": info.header_offset,
                "compress_size": info.compress_size,
                "file_size": info.file_size,
                "crc": info.CRC,
                "compress_type": info.compress_type,
            }
        )
    logger.info("%s: %d accounts files indexed, %d skipped", archive.name, len(rows), skipped)
    return pl.DataFrame(rows, schema=INDEX_SCHEMA)


# ---------------------------------------------------------------- sampling
def in_sample(company_number: str, rate: float, salt: str = SAMPLE_SALT) -> bool:
    """Deterministic company-level sample: the same companies are chosen in every archive."""
    digest = hashlib.sha256(f"{salt}:{company_number}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate


def fetch_member(client: httpx.Client, url: str, row: dict[str, Any]) -> bytes:
    """Fetch one file out of a remote zip with a single range request, and verify its CRC."""
    start = int(row["header_offset"])
    name_length = len(str(row["member"]).encode())
    compressed_size = int(row["compress_size"])
    end = start + _LOCAL_HEADER.size + name_length + compressed_size + _LOCAL_HEADER_PAD
    blob = range_get(client, url, start, end)

    signature, *_, local_name_length, extra_length = _LOCAL_HEADER.unpack_from(blob)
    if signature != b"PK\x03\x04":
        raise ValueError(f"No local file header for {row['member']} at offset {start}")
    data_start = _LOCAL_HEADER.size + local_name_length + extra_length
    if data_start + compressed_size > len(blob):  # unusually large extra field: fetch the rest
        blob += range_get(client, url, start + len(blob), start + data_start + compressed_size)
    compressed = blob[data_start : data_start + compressed_size]

    if row["compress_type"] == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -zlib.MAX_WBITS)
    elif row["compress_type"] == zipfile.ZIP_STORED:
        data = compressed
    else:
        raise ValueError(f"Unsupported compression {row['compress_type']} for {row['member']}")
    if zlib.crc32(data) != int(row["crc"]):
        raise ValueError(f"CRC mismatch for {row['member']}")
    return data


def estimated_fetch_bytes(sample: pl.DataFrame) -> int:
    overhead = _LOCAL_HEADER.size + _LOCAL_HEADER_PAD + 40  # header + pad + file name
    return int(sample["compress_size"].sum()) + overhead * sample.height


# ---------------------------------------------------------------- manifest
@dataclass
class ArchiveRecord:
    archive: str
    url: str
    filing_month: str
    files_in_archive: int
    files_sampled: int
    index_bytes: int
    sample_bytes: int
    completed_at: str


class Manifest:
    """JSON record of completed archives and bytes downloaded (feeds the byte budget)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: dict[str, ArchiveRecord] = {}
        if path.exists():
            for item in json.loads(path.read_text(encoding="utf-8"))["archives"]:
                record = ArchiveRecord(**item)
                self.records[record.archive] = record

    @property
    def total_bytes(self) -> int:
        return sum(r.index_bytes + r.sample_bytes for r in self.records.values())

    def add(self, record: ArchiveRecord) -> None:
        self.records[record.archive] = record
        payload = {"archives": [asdict(r) for r in sorted(self.records.values(), key=_by_month)]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)


def _by_month(record: ArchiveRecord) -> str:
    return record.filing_month


# ---------------------------------------------------------------- pipeline
def download_archive_sample(
    client: httpx.Client,
    archive: Archive,
    *,
    rate: float,
    raw_dir: Path,
    index_dir: Path,
    budget: DownloadBudget,
    workers: int = 8,
) -> ArchiveRecord:
    """Index one archive (cached) and download its sampled members into a local zip."""
    index_path = index_dir / f"{archive.name}.parquet"
    used_before = budget.used
    if index_path.exists():
        index = pl.read_parquet(index_path)
    else:
        index = read_index(client, archive, budget)
        index_dir.mkdir(parents=True, exist_ok=True)
        index.write_parquet(index_path)
    index_bytes = budget.used - used_before

    sample = index.filter(
        pl.col("company_number").map_elements(lambda n: in_sample(n, rate), return_dtype=pl.Boolean)
    )
    budget.check(estimated_fetch_bytes(sample))

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"{archive.name}.zip"
    tmp_path = out_path.with_suffix(".zip.tmp")
    sample_bytes = 0
    rows = sample.to_dicts()
    with (
        zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as out,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        for row, data in zip(
            rows, pool.map(lambda r: fetch_member(client, archive.url, r), rows), strict=True
        ):
            out.writestr(row["member"], data)
            sample_bytes += int(row["compress_size"])
    budget.charge(sample_bytes)
    tmp_path.replace(out_path)

    return ArchiveRecord(
        archive=archive.name,
        url=archive.url,
        filing_month=archive.filing_month.isoformat(),
        files_in_archive=index.height,
        files_sampled=sample.height,
        index_bytes=index_bytes,
        sample_bytes=sample_bytes,
        completed_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def select_archives(archives: list[Archive], first: date, last: date) -> list[Archive]:
    return [a for a in archives if first <= a.filing_month <= last]


def estimate_plan(client: httpx.Client, archives: list[Archive], rate: float) -> int:
    """Rough bytes for indexes + samples, from archive sizes (HEAD requests only)."""
    total = 0
    for archive in archives:
        size = int(client.head(archive.url).headers["content-length"])
        total += int(size * (CENTRAL_DIRECTORY_RATIO + rate))
    return total


def iter_months(first: date, last: date) -> Iterator[date]:
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        yield date(year, month, 1)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def _parse_month(value: str) -> date:
    year, month = value.split("-")
    return date(int(year), int(month), 1)


def main() -> None:
    from cfp.config import get_settings

    parser = argparse.ArgumentParser(description="Download a company-level sample of accounts.")
    parser.add_argument("--from", dest="first", type=_parse_month, default=date(2018, 1, 1))
    parser.add_argument("--to", dest="last", type=_parse_month, default=date(2025, 10, 1))
    parser.add_argument("--rate", type=float, default=0.005, help="Share of companies sampled.")
    parser.add_argument("--max-gb", type=float, default=5.0, help="Total download budget.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Estimate size and stop.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = get_settings()
    raw_dir = settings.raw_dir / "accounts_sample"
    index_dir = settings.interim_dir / "accounts_index"
    manifest = Manifest(raw_dir / "manifest.json")
    budget = DownloadBudget(int(args.max_gb * GB), used=manifest.total_bytes)

    with httpx.Client(
        timeout=120,
        headers={"User-Agent": USER_AGENT},
        limits=httpx.Limits(max_connections=args.workers),
        follow_redirects=True,
    ) as client:
        archives = select_archives(list_archives(client), args.first, args.last)
        missing = sorted(
            set(iter_months(args.first, args.last)) - {a.filing_month for a in archives}
        )
        if missing:
            logger.warning("No archive listed for: %s", ", ".join(m.isoformat() for m in missing))
        todo = [a for a in archives if a.name not in manifest.records]
        estimate = estimate_plan(client, todo, args.rate)
        logger.info(
            "%d archives selected, %d to do. Estimated download %.2f GB; already used %.2f GB "
            "of %.2f GB budget.",
            len(archives),
            len(todo),
            estimate / GB,
            budget.used / GB,
            budget.max_bytes / GB,
        )
        if args.dry_run:
            return
        budget.check(estimate)
        for i, archive in enumerate(todo, start=1):
            record = download_archive_sample(
                client,
                archive,
                rate=args.rate,
                raw_dir=raw_dir,
                index_dir=index_dir,
                budget=budget,
                workers=args.workers,
            )
            manifest.add(record)
            logger.info(
                "[%d/%d] %s: %d of %d files sampled (%.1f MB index, %.1f MB files). Total %.2f GB",
                i,
                len(todo),
                archive.name,
                record.files_sampled,
                record.files_in_archive,
                record.index_bytes / 1e6,
                record.sample_bytes / 1e6,
                budget.used / GB,
            )


if __name__ == "__main__":
    main()
