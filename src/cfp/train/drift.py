"""Drift check: how far have feature distributions moved since the training years?

Population Stability Index (PSI) per feature, comparing a reference period (the training
years) with a recent period. Numeric features are binned on the reference deciles; missing
values and categories get their own bins, because a change in how often something is missing
is drift too.

Rule of thumb: PSI < 0.1 stable, 0.1-0.25 worth watching, > 0.25 a real shift.

    uv run python -m cfp.train.drift            # -> reports/drift.md
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import polars as pl

from cfp.features.build_features import FEATURES
from cfp.train.data import CATEGORICAL, load_dataset

logger = logging.getLogger(__name__)

EPSILON = 1e-4  # keeps empty bins from producing log(0)
WATCH, SHIFT = 0.1, 0.25


def psi(reference: np.ndarray, current: np.ndarray) -> float:
    """PSI between two vectors of bin shares (each summing to 1)."""
    ref = np.clip(reference, EPSILON, None)
    cur = np.clip(current, EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _shares(labels: np.ndarray, bins: list[str]) -> np.ndarray:
    counts = np.array([(labels == b).sum() for b in bins], dtype=float)
    shares: np.ndarray = counts / max(float(counts.sum()), 1.0)
    return shares


def numeric_psi(reference: pl.Series, current: pl.Series, n_bins: int = 10) -> float:
    ref = reference.cast(pl.Float64).to_numpy()
    cur = current.cast(pl.Float64).to_numpy()
    finite = ref[~np.isnan(ref)]
    if finite.size == 0:
        return 0.0
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1))[1:-1])

    def labels(x: np.ndarray) -> np.ndarray:
        out: np.ndarray = np.digitize(x, edges).astype(str)
        out[np.isnan(x)] = "missing"
        return out

    bins = [*map(str, range(len(edges) + 1)), "missing"]
    return psi(_shares(labels(ref), bins), _shares(labels(cur), bins))


def categorical_psi(reference: pl.Series, current: pl.Series) -> float:
    ref = reference.cast(pl.String).fill_null("missing").to_numpy()
    cur = current.cast(pl.String).fill_null("missing").to_numpy()
    bins = sorted(set(ref) | set(cur))
    return psi(_shares(ref, bins), _shares(cur, bins))


def drift_table(reference: pl.DataFrame, current: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for name in FEATURES:
        fn = categorical_psi if name in CATEGORICAL else numeric_psi
        value = fn(reference[name], current[name])
        status = "shift" if value > SHIFT else "watch" if value > WATCH else "stable"
        rows.append({"feature": name, "psi": round(value, 4), "status": status})
    return pl.DataFrame(rows).sort("psi", descending=True)


def drift_markdown(table: pl.DataFrame, reference: str, current: str, top: int = 15) -> str:
    counts = dict(table.group_by("status").len().iter_rows())
    lines = [
        "# Feature drift",
        "",
        f"Population Stability Index, reference = {reference}, current = {current}.",
        "PSI < 0.1 stable, 0.1-0.25 watch, > 0.25 shift.",
        "",
        f"Shift: {counts.get('shift', 0)} features, watch: {counts.get('watch', 0)}, "
        f"stable: {counts.get('stable', 0)}.",
        "",
        "| Feature | PSI | Status |",
        "|---|---|---|",
    ]
    lines += [
        f"| `{r['feature']}` | {r['psi']:.3f} | {r['status']} |"
        for r in table.head(top).iter_rows(named=True)
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    from cfp.config import get_settings

    parser = argparse.ArgumentParser(description="Feature drift between two periods.")
    parser.add_argument("--reference-years", default="2018-2022")
    parser.add_argument("--current-years", default="2024-2025")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    def years(spec: str) -> tuple[int, int]:
        first, last = spec.split("-")
        return int(first), int(last)

    settings = get_settings()
    data = load_dataset(settings.processed_dir)
    ref_years, cur_years = years(args.reference_years), years(args.current_years)
    table = drift_table(
        data.filter(pl.col("year").is_between(*ref_years)),
        data.filter(pl.col("year").is_between(*cur_years)),
    )
    out = settings.reports_dir / "drift.md"
    out.write_text(
        drift_markdown(table, f"filings {args.reference_years}", f"filings {args.current_years}"),
        encoding="utf-8",
    )
    logger.info("Wrote %s", out)
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_hide_dataframe_shape=True, tbl_rows=12):
        print(table.head(12))


if __name__ == "__main__":
    main()
