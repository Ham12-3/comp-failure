"""Tests for the training package, on a small synthetic dataset with a known signal."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from cfp.features.build_features import FEATURES
from cfp.train.calibrate import PlattCalibrator
from cfp.train.data import (
    BOOLEAN,
    CATEGORICAL,
    SPLITS,
    TARGET,
    learn_categories,
    split_by_year,
    target,
    to_matrix,
)
from cfp.train.evaluate import all_metrics, bootstrap_intervals, precision_recall_at
from cfp.train.explain import format_value, shap_values, top_reasons
from cfp.train.train import (
    fit_lightgbm,
    for_logistic,
    logistic_model,
    raw_margin,
    rolling_validation,
    rule_score,
    rule_scores,
)

LEVELS = {name: ["a", "b", "c"] for name in CATEGORICAL}


def synthetic(n: int = 2400, seed: int = 0) -> pl.DataFrame:
    """Rows spread over 2018-2025; failures are more likely when current_ratio is low."""
    rng = np.random.default_rng(seed)
    columns: dict[str, object] = {}
    for name in FEATURES:
        if name in CATEGORICAL:
            columns[name] = rng.choice(LEVELS[name], n).tolist()
        elif name in BOOLEAN:
            columns[name] = rng.random(n) < 0.3
        else:
            columns[name] = rng.normal(0, 1, n)
    years = rng.integers(2018, 2026, n)
    risk = 1 / (1 + np.exp(-(-4.0 - 2.5 * np.asarray(columns["current_ratio"]))))
    columns[TARGET] = (rng.random(n) < risk).astype(int)
    columns["year"] = years
    columns["prediction_date"] = [date(int(y), 6, 30) for y in years]
    columns["company_number"] = [f"{i:08d}" for i in range(n)]
    return pl.DataFrame(columns)


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    return synthetic()


def test_splits_are_by_year_and_disjoint(data: pl.DataFrame) -> None:
    parts = split_by_year(data)
    assert [s.name for s in SPLITS] == ["train", "valid", "test"]
    assert parts["train"]["year"].max() < parts["valid"]["year"].min()  # type: ignore[operator]
    assert parts["valid"]["year"].max() < parts["test"]["year"].min()  # type: ignore[operator]
    assert sum(p.height for p in parts.values()) == data.height


def test_matrix_columns_types_and_unseen_categories(data: pl.DataFrame) -> None:
    categories = learn_categories(data)
    other = data.head(3).with_columns(pl.lit("never-seen").alias("postcode_area"))
    x = to_matrix(other, categories)
    assert list(x.columns) == list(FEATURES)
    assert x["postcode_area"].isna().all()  # unseen level -> missing, not an error
    assert set(x["net_assets_negative"].unique()) <= {0.0, 1.0}


def test_precision_recall_at_top_fraction() -> None:
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert precision_recall_at(y, score, 0.2) == (0.5, 0.5)  # top 2 rows: one failure of two
    assert precision_recall_at(y, score, 0.3) == pytest.approx((2 / 3, 1.0))


def test_bootstrap_interval_contains_point_estimate() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.05).astype(int)
    score = y * 1.0 + rng.normal(0, 1, 2000)
    point = all_metrics(y, score, None)["pr_auc"]
    low, high = bootstrap_intervals(y, score, None, n_boot=200)["pr_auc"]
    assert low <= point <= high


def test_platt_calibration_matches_base_rate() -> None:
    rng = np.random.default_rng(2)
    raw = rng.normal(0, 2, 5000)
    y = (rng.random(5000) < 1 / (1 + np.exp(-(raw - 4)))).astype(int)  # true prob is shifted
    calibrated = PlattCalibrator().fit(raw, y)
    prob = calibrated.predict(raw)
    assert prob.mean() == pytest.approx(y.mean(), abs=0.005)
    assert np.all(np.diff(prob[np.argsort(raw)]) >= 0)  # monotone
    assert PlattCalibrator.from_dict(calibrated.to_dict()).predict(raw) == pytest.approx(prob)


def test_rule_score_flags_late_or_negative() -> None:
    rows = pl.DataFrame(
        {
            "days_late_this_filing": [10.0, -30.0, None, -5.0],
            "net_assets_negative": [False, True, None, False],
        }
    )
    assert rule_score(rows).tolist() == [1.0, 1.0, 0.0, 0.0]


def test_models_train_and_find_the_signal(data: pl.DataFrame) -> None:
    parts = split_by_year(data)
    categories = learn_categories(parts["train"])
    x_train, x_test = to_matrix(parts["train"], categories), to_matrix(parts["test"], categories)
    y_test = target(parts["test"])

    params = {"num_leaves": 4, "min_child_samples": 20, "n_estimators": 50}
    lgbm = fit_lightgbm(x_train, target(parts["train"]), params)
    lgbm_auc = all_metrics(y_test, raw_margin(lgbm, x_test), None)["roc_auc"]
    logit = logistic_model().fit(for_logistic(x_train), target(parts["train"]))
    logit_auc = all_metrics(y_test, logit.decision_function(for_logistic(x_test)), None)["roc_auc"]
    assert lgbm_auc > 0.7 and logit_auc > 0.7  # current_ratio drives the synthetic label


def test_shap_reasons_are_ranked_and_readable(data: pl.DataFrame) -> None:
    categories = learn_categories(data)
    x = to_matrix(data, categories)
    model = fit_lightgbm(
        x, target(data), {"num_leaves": 4, "min_child_samples": 20, "n_estimators": 50}
    )
    values = shap_values(model.booster_, x.head(20))
    assert values.shape == (20, len(FEATURES))

    reasons = top_reasons(values[0], x.iloc[0])
    assert len(reasons) == 5
    sizes = [abs(r.contribution) for r in reasons]
    assert sizes == sorted(sizes, reverse=True)
    assert all(("raises the risk" in r.text) or ("lowers the risk" in r.text) for r in reasons)


def test_format_value() -> None:
    assert format_value("net_assets", -7385.0) == "£-7,385"
    assert format_value("net_assets_negative", 1.0) == "yes"
    assert format_value("cash_pct_change", 0.25) == "+25%"
    assert format_value("cash", float("nan")) == "not available"


def test_rolling_validation_scores_each_year(data: pl.DataFrame) -> None:
    scores = rolling_validation(data.filter(pl.col("year") <= 2023), rule_scores)
    assert len(scores) == 3 and all(0 <= s <= 1 for s in scores)
