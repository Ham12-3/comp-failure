"""Tests for the PSI drift check."""

import numpy as np
import polars as pl
import pytest

from cfp.train.drift import categorical_psi, drift_table, numeric_psi, psi
from tests.test_train import synthetic


def test_psi_is_zero_for_identical_and_grows_with_shift() -> None:
    same = np.array([0.25, 0.25, 0.5])
    assert psi(same, same) == pytest.approx(0.0)
    assert psi(same, np.array([0.2, 0.3, 0.5])) < psi(same, np.array([0.05, 0.45, 0.5]))


def test_numeric_psi_detects_a_mean_shift() -> None:
    rng = np.random.default_rng(0)
    ref = pl.Series(rng.normal(0, 1, 5000))
    assert numeric_psi(ref, pl.Series(rng.normal(0, 1, 5000))) < 0.05
    assert numeric_psi(ref, pl.Series(rng.normal(1, 1, 5000))) > 0.25


def test_more_missing_values_count_as_drift() -> None:
    rng = np.random.default_rng(1)
    ref = pl.Series(rng.normal(0, 1, 2000))
    values = rng.normal(0, 1, 2000)
    values[:1000] = np.nan
    assert numeric_psi(ref, pl.Series(values)) > 0.25


def test_categorical_psi_handles_new_levels() -> None:
    ref = pl.Series(["a"] * 500 + ["b"] * 500)
    cur = pl.Series(["a"] * 500 + ["c"] * 500)
    assert categorical_psi(ref, ref) == pytest.approx(0.0)
    assert categorical_psi(ref, cur) > 0.25


def test_drift_table_covers_every_feature() -> None:
    data = synthetic(n=800)
    table = drift_table(data.head(400), data.tail(400))
    assert table.height > 70
    assert set(table["status"]) <= {"stable", "watch", "shift"}
    assert table["psi"].to_list() == sorted(table["psi"].to_list(), reverse=True)
