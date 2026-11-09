"""Train, calibrate, evaluate and explain: the Phase 4 pipeline.

1. Time split on prediction year: train 2018-2022, validation 2023, test 2024-2025.
2. Baselines: a rule ("filed late OR negative net assets") and logistic regression.
3. LightGBM with class weighting. Hyper-parameters are chosen on rolling validation years
   (2021, 2022, 2023, each predicted by a model trained on the years before it), so tuning
   never sees the test years and is not decided by one year's handful of failures.
4. Platt calibration on 2023; metrics with bootstrap intervals on 2024-2025.
5. SHAP summary and plain-English reasons; everything logged to MLflow and written to
   reports/ (committed) and models/ (gitignored, used by the API in Phase 5).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from cfp.features.build_features import FEATURES
from cfp.train.calibrate import PlattCalibrator
from cfp.train.data import (
    CATEGORICAL,
    NUMERIC,
    SPLITS,
    learn_categories,
    load_dataset,
    split_by_year,
    target,
    to_matrix,
)
from cfp.train.evaluate import (
    all_metrics,
    bootstrap_intervals,
    plot_pr_curves,
    plot_reliability,
    ranking_metrics,
    reliability_table,
)
from cfp.train.explain import plot_summary, shap_values, top_reasons

logger = logging.getLogger(__name__)

TUNING_YEARS = (2021, 2022, 2023)
LGBM_FIXED: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    # Categorical splits overfit easily with few positives: smooth them harder and allow
    # fewer categories per split than the defaults (10 / 32).
    "cat_smooth": 50.0,
    "max_cat_threshold": 8,
    "verbose": -1,
}
# Small, strongly regularised grid: there are only tens of failures to learn from.
LGBM_GRID: list[dict[str, Any]] = [
    {"num_leaves": leaves, "min_child_samples": min_child, "n_estimators": trees}
    for leaves in (4, 8, 16)
    for min_child in (20, 60)
    for trees in (100, 300)
]
LEAKAGE_ALARM_PR_AUC = 0.9  # CLAUDE.md: a result this good means "look for leakage first"


# ---------------------------------------------------------------- models
def rule_score(data: pl.DataFrame) -> np.ndarray:
    """Baseline 1: accounts filed late, or negative net assets."""
    flagged = (pl.col("days_late_this_filing") > 0).fill_null(False) | pl.col(
        "net_assets_negative"
    ).fill_null(False)
    return data.select(flagged.cast(pl.Float64))["days_late_this_filing"].to_numpy()


def _signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def for_logistic(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    for name in CATEGORICAL:
        out[name] = out[name].astype(object).where(out[name].notna(), "missing")
    return out


def logistic_model() -> Pipeline:
    """Baseline 2: logistic regression on signed-log, imputed, scaled numbers + one-hot."""
    numeric = make_pipeline(
        FunctionTransformer(_signed_log),
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )
    categorical = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)
    return make_pipeline(
        ColumnTransformer(
            [("num", numeric, list(NUMERIC)), ("cat", categorical, list(CATEGORICAL))]
        ),
        LogisticRegression(class_weight="balanced", C=0.1, max_iter=5000),
    )


def fit_lightgbm(
    x: pd.DataFrame, y: np.ndarray, params: dict[str, Any], seed: int = 0
) -> lgb.LGBMClassifier:
    positives = max(int(y.sum()), 1)
    model = lgb.LGBMClassifier(
        **LGBM_FIXED, **params, scale_pos_weight=(len(y) - positives) / positives, random_state=seed
    )
    model.fit(x, y)
    return model


def raw_margin(model: lgb.LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(x, raw_score=True), dtype=float)


def rolling_validation(data: pl.DataFrame, fit_and_score: Any) -> list[float]:
    """PR-AUC for each validation year, scoring a model trained on all earlier years."""
    scores = []
    for year in TUNING_YEARS:
        fit_part = data.filter(pl.col("year") < year)
        val_part = data.filter(pl.col("year") == year)
        scores.append(
            float(average_precision_score(target(val_part), fit_and_score(fit_part, val_part)))
        )
    return scores


def logistic_scores(fit_part: pl.DataFrame, val_part: pl.DataFrame) -> np.ndarray:
    categories = learn_categories(fit_part)
    model = logistic_model().fit(for_logistic(to_matrix(fit_part, categories)), target(fit_part))
    return np.asarray(model.decision_function(for_logistic(to_matrix(val_part, categories))))


def rule_scores(fit_part: pl.DataFrame, val_part: pl.DataFrame) -> np.ndarray:
    return rule_score(val_part)


def tune_lightgbm(data: pl.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pick LightGBM settings by mean PR-AUC over rolling validation years."""
    table = []
    for params in LGBM_GRID:

        def fit_and_score(
            fit_part: pl.DataFrame, val_part: pl.DataFrame, params: dict[str, Any] = params
        ) -> np.ndarray:
            categories = learn_categories(fit_part)
            model = fit_lightgbm(to_matrix(fit_part, categories), target(fit_part), params)
            return raw_margin(model, to_matrix(val_part, categories))

        scores = rolling_validation(data, fit_and_score)
        table.append(
            {
                **params,
                **{f"pr_auc_{y}": s for y, s in zip(TUNING_YEARS, scores, strict=True)},
                "pr_auc_mean": float(np.mean(scores)),
            }
        )
    best = max(table, key=lambda r: r["pr_auc_mean"])  # first (simplest) wins ties
    return {k: best[k] for k in LGBM_GRID[0]}, table


# ---------------------------------------------------------------- reporting
def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _fmt(value: float, interval: tuple[float, float] | None, pct: bool = False) -> str:
    if np.isnan(value):
        return "n/a"

    def show(v: float) -> str:
        return f"{v:.1%}" if pct else f"{v:.3f}"

    return f"{show(value)} [{show(interval[0])}, {show(interval[1])}]" if interval else show(value)


def metrics_markdown(
    split_sizes: dict[str, tuple[int, int]],
    test: dict[str, dict[str, float]],
    intervals: dict[str, dict[str, tuple[float, float]]],
    valid: dict[str, dict[str, float]],
    rolling: dict[str, float],
    best_params: dict[str, Any],
    tuning: list[dict[str, Any]],
    reliability: list[dict[str, float]],
) -> str:
    lines = ["# Model evaluation", "", "Generated by `python -m cfp.train.train`.", ""]
    lines += ["## Data", "", "| Split | Years | Rows | Failures |", "|---|---|---|---|"]
    for s in SPLITS:
        rows, pos = split_sizes[s.name]
        lines.append(f"| {s.name} | {s.first_year}-{s.last_year} | {rows:,} | {pos} |")
    lines += ["", "## Test set (2024-2025), with bootstrap 95% intervals", ""]
    columns = [
        ("PR-AUC", "pr_auc", False),
        ("ROC-AUC", "roc_auc", False),
        ("Precision top 1%", "precision_top1", True),
        ("Recall top 1%", "recall_top1", True),
        ("Precision top 5%", "precision_top5", True),
        ("Recall top 5%", "recall_top5", True),
        ("Brier", "brier", False),
    ]
    lines += [_md_row(["Model", *(c[0] for c in columns)]), _md_row(["---"] * (len(columns) + 1))]
    for name, m in test.items():
        ci = intervals[name]
        lines.append(_md_row([name, *(_fmt(m[key], ci.get(key), pct) for _, key, pct in columns)]))
    base = next(iter(test.values()))
    lines += [
        "",
        f"Base rate on the test set: {base['base_rate']:.2%} ({int(base['positives'])} failures). "
        "A random ranking has PR-AUC equal to the base rate.",
        "",
    ]
    lines += [
        "## Validation (used for choices; the test set was not)",
        "",
        "Rolling = mean PR-AUC over 2021, 2022 and 2023, each scored by a model trained on the",
        "years before it. 2023 = the final models (trained 2018-2022) scored on 2023.",
        "",
        "| Model | Rolling PR-AUC | 2023 PR-AUC | 2023 ROC-AUC |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {n} | {rolling[n]:.3f} | {m['pr_auc']:.3f} | {m['roc_auc']:.3f} |"
        for n, m in valid.items()
    ]
    lines += [
        "",
        "## Calibration (LightGBM + Platt, test set, 5 equal-count bins)",
        "",
        "| Mean predicted | Observed | Rows | Failures |",
        "|---|---|---|---|",
    ]
    lines += [
        _md_row(
            [
                f"{r['mean_predicted']:.2%}",
                f"{r['observed']:.2%}",
                f"{int(r['rows']):,}",
                str(int(r["positives"])),
            ]
        )
        for r in reliability
    ]
    lines += [
        "",
        "## LightGBM tuning (mean PR-AUC over validation years 2021-2023)",
        "",
        f"Chosen: `{json.dumps(best_params)}`",
        "",
        "| num_leaves | min_child_samples | n_estimators | mean PR-AUC |",
        "|---|---|---|---|",
    ]
    for r in sorted(tuning, key=lambda r: -r["pr_auc_mean"]):
        cells = [str(r["num_leaves"]), str(r["min_child_samples"]), str(r["n_estimators"])]
        lines.append(_md_row([*cells, f"{r['pr_auc_mean']:.3f}"]))
    return "\n".join(lines) + "\n"


def reasons_markdown(
    rows: pl.DataFrame, probs: np.ndarray, contributions: np.ndarray, x: pd.DataFrame, n: int = 3
) -> str:
    lines = [
        "# Example explanations (highest-risk test filings)",
        "",
        "Companies are not named: this repository is public, and a model score is not a",
        "judgement about a real business.",
        "",
    ]
    for rank, i in enumerate(np.argsort(-probs, kind="stable")[:n], start=1):
        row = rows.row(int(i), named=True)
        lines += [
            f"## Example {rank}: accounts filed {row['prediction_date']}",
            "",
            f"Calibrated 12-month insolvency probability: **{probs[i]:.1%}**. Actual outcome: "
            + ("insolvent" if row["label_insolvency_12m"] else "not insolvent")
            + " within 12 months.",
            "",
        ]
        lines += [
            f"{k}. {r.text}" for k, r in enumerate(top_reasons(contributions[i], x.iloc[int(i)]), 1)
        ]
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- pipeline
def main() -> None:
    import mlflow

    from cfp.config import get_settings

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    reports, models = settings.reports_dir, Path("models")
    reports.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)

    data = load_dataset(settings.processed_dir)
    parts = split_by_year(data)
    sizes = {
        name: (part.height, int(part["label_insolvency_12m"].sum())) for name, part in parts.items()
    }
    logger.info("Split sizes (rows, failures): %s", sizes)

    tuning_data = data.filter(pl.col("year") <= max(TUNING_YEARS))  # never the test years
    best_params, tuning = tune_lightgbm(tuning_data)
    logger.info("Best LightGBM params: %s", best_params)
    rolling = {
        "Rule: late or negative net assets": float(
            np.mean(rolling_validation(tuning_data, rule_scores))
        ),
        "Logistic regression": float(np.mean(rolling_validation(tuning_data, logistic_scores))),
        "LightGBM": max(r["pr_auc_mean"] for r in tuning),
    }
    logger.info("Rolling validation PR-AUC: %s", rolling)

    categories = learn_categories(parts["train"])
    x = {name: to_matrix(part, categories) for name, part in parts.items()}
    y = {name: target(part) for name, part in parts.items()}

    lgbm = fit_lightgbm(x["train"], y["train"], best_params)
    lgbm_raw = {name: raw_margin(lgbm, x[name]) for name in ("valid", "test")}
    platt = PlattCalibrator().fit(lgbm_raw["valid"], y["valid"])

    logit = logistic_model().fit(for_logistic(x["train"]), y["train"])
    logit_raw = {name: logit.decision_function(for_logistic(x[name])) for name in ("valid", "test")}
    logit_platt = PlattCalibrator().fit(logit_raw["valid"], y["valid"])

    rule = {name: rule_score(parts[name]) for name in ("valid", "test")}

    scores = {
        "Rule: late or negative net assets": rule,
        "Logistic regression": logit_raw,
        "LightGBM": lgbm_raw,
    }
    lgbm_prob = platt.predict(lgbm_raw["test"])
    logit_prob = logit_platt.predict(logit_raw["test"])
    probs_test: dict[str, np.ndarray | None] = {
        "Rule: late or negative net assets": None,  # a yes/no rule has no probability
        "Logistic regression": logit_prob,
        "LightGBM": lgbm_prob,
    }
    test_metrics, intervals = {}, {}
    for name, score in scores.items():
        test_metrics[name] = all_metrics(y["test"], score["test"], probs_test[name])
        intervals[name] = bootstrap_intervals(y["test"], score["test"], probs_test[name])
    valid_metrics = {
        name: ranking_metrics(y["valid"], score["valid"]) for name, score in scores.items()
    }

    if test_metrics["LightGBM"]["pr_auc"] > LEAKAGE_ALARM_PR_AUC:
        logger.error("Test PR-AUC above %.1f: assume leakage and investigate", LEAKAGE_ALARM_PR_AUC)

    uncalibrated = 1 / (1 + np.exp(-lgbm_raw["test"]))
    plot_pr_curves(
        y["test"], {name: s["test"] for name, s in scores.items()}, reports / "pr_curve.png"
    )
    plot_reliability(
        y["test"],
        {
            "LightGBM, uncalibrated (class-weighted)": uncalibrated,
            "LightGBM + Platt": lgbm_prob,
            "Logistic + Platt": logit_prob,
        },
        reports / "reliability.png",
    )
    contributions = shap_values(lgbm.booster_, x["test"])
    plot_summary(contributions, x["test"], reports / "shap_summary.png")

    reliability = reliability_table(y["test"], lgbm_prob)
    (reports / "metrics.md").write_text(
        metrics_markdown(
            sizes, test_metrics, intervals, valid_metrics, rolling, best_params, tuning, reliability
        ),
        encoding="utf-8",
    )
    (reports / "metrics.json").write_text(
        json.dumps(
            {
                "split_sizes": sizes,
                "test": test_metrics,
                "test_intervals": intervals,
                "valid": valid_metrics,
                "rolling_validation_pr_auc": rolling,
                "best_params": best_params,
                "tuning": tuning,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (reports / "example_reasons.md").write_text(
        reasons_markdown(parts["test"], lgbm_prob, contributions, x["test"]),
        encoding="utf-8",
    )

    lgbm.booster_.save_model(str(models / "lightgbm.txt"))
    (models / "model_meta.json").write_text(
        json.dumps(
            {
                "features": list(FEATURES),
                "categories": categories,
                "calibrator": platt.to_dict(),
                "params": {**LGBM_FIXED, **best_params},
                "split": [s.__dict__ for s in SPLITS],
                "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "train_rows": sizes["train"][0],
                "train_failures": sizes["train"][1],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("company-failure-predictor")
    with mlflow.start_run(run_name=f"lightgbm-{datetime.now(UTC):%Y%m%d-%H%M}"):
        mlflow.log_params(
            {
                **best_params,
                "train_rows": sizes["train"][0],
                "train_failures": sizes["train"][1],
                "test_rows": sizes["test"][0],
                "test_failures": sizes["test"][1],
            }
        )
        for name, m in test_metrics.items():
            key = name.split(":")[0].lower().replace(" ", "_")
            mlflow.log_metrics({f"test_{key}_{k}": v for k, v in m.items() if not np.isnan(v)})
        mlflow.log_artifacts(str(reports), artifact_path="reports")
        mlflow.log_artifact(str(models / "model_meta.json"))
    logger.info(
        "Done. Test metrics: %s",
        json.dumps({n: round(m["pr_auc"], 4) for n, m in test_metrics.items()}),
    )


if __name__ == "__main__":
    main()
