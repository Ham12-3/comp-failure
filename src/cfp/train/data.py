"""Training data: join features to labels, split by prediction date, build model matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from cfp.features.build_features import FEATURES

TARGET = "label_insolvency_12m"
CATEGORICAL = (
    "accounting_standard",
    "accounts_type",
    "accounts_status",
    "accounts_type_filed",
    "postcode_area",
    "company_type",
    "jurisdiction",
    "sic_section",
)
BOOLEAN = frozenset(
    {
        "net_assets_negative",
        "net_current_liabilities",
        "net_assets_turned_negative",
        "has_prior_year",
        "sign_repaired",
        "is_dormant",
        "first_accounts",
        *(f for f in FEATURES if f.startswith("missing_")),
    }
)
NUMERIC = tuple(f for f in FEATURES if f not in CATEGORICAL)  # includes BOOLEAN, as 0/1


@dataclass(frozen=True)
class Split:
    name: str
    first_year: int
    last_year: int


# Time-based split on the prediction (filing) year. No random splitting anywhere.
SPLITS = (Split("train", 2018, 2022), Split("valid", 2023, 2023), Split("test", 2024, 2025))


def load_dataset(processed_dir: Path) -> pl.DataFrame:
    """Usable labelled rows with their features and a `year` column."""
    labels = pl.read_parquet(processed_dir / "labels.parquet").filter(
        pl.col("excluded_reason").is_null()
    )
    features = pl.read_parquet(processed_dir / "features.parquet")
    data = features.join(
        labels.select("company_number", "source_member", TARGET),
        on=["company_number", "source_member"],
        how="inner",
    )
    return data.with_columns(year=pl.col("prediction_date").dt.year()).sort("prediction_date")


def split_by_year(
    data: pl.DataFrame, splits: tuple[Split, ...] = SPLITS
) -> dict[str, pl.DataFrame]:
    return {
        s.name: data.filter(pl.col("year").is_between(s.first_year, s.last_year)) for s in splits
    }


Categories = dict[str, list[str]]


def learn_categories(train: pl.DataFrame) -> Categories:
    """Category levels seen in training; anything else becomes missing at prediction time."""
    return {c: sorted(v for v in train[c].unique().to_list() if v is not None) for c in CATEGORICAL}


def to_matrix(data: pl.DataFrame, categories: Categories) -> pd.DataFrame:
    """pandas frame in FEATURES order: numbers and booleans as float, strings as categoricals
    with the training levels (LightGBM reads these natively)."""
    frame = pd.DataFrame({name: data[name].cast(pl.Float64).to_numpy() for name in NUMERIC})
    for name in CATEGORICAL:
        known = set(categories[name])
        values = [v if v in known else None for v in data[name].to_list()]  # unseen -> missing
        frame[name] = pd.Categorical(values, categories=categories[name])
    return frame[list(FEATURES)]


def target(data: pl.DataFrame) -> np.ndarray:
    return data[TARGET].to_numpy().astype(int)
