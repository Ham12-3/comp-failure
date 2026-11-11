"""Load the trained model once and score one row of features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import shap

from cfp.features.build_features import FEATURES
from cfp.train.calibrate import PlattCalibrator
from cfp.train.data import to_matrix
from cfp.train.explain import Reason, top_reasons


@dataclass(frozen=True)
class Score:
    probability: float  # calibrated 12-month insolvency probability
    raw_score: float  # model log-odds before calibration
    percentile: float  # share of reference filings with a lower or equal probability, 0-100
    relative_risk: float  # probability / average rate in the training years
    reasons: list[Reason]


class RiskModel:
    """LightGBM booster + Platt calibrator + SHAP explainer, built from `models/`."""

    def __init__(self, model_dir: Path) -> None:
        meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
        if meta["features"] != list(FEATURES):
            raise ValueError("Model was trained on a different feature list: retrain it")
        self.booster = lgb.Booster(model_file=str(model_dir / "lightgbm.txt"))
        self.categories: dict[str, list[str]] = meta["categories"]
        self.calibrator = PlattCalibrator.from_dict(meta["calibrator"])
        self.reference = np.asarray(meta["reference_percentiles"], dtype=float)
        self.base_rate = float(meta["train_base_rate"])
        self.trained_at = str(meta["trained_at"])
        self.explainer = shap.TreeExplainer(self.booster)

    def score(self, features: pl.DataFrame) -> Score:
        """Score the first row of a feature table (columns as in FEATURES)."""
        x = to_matrix(features.head(1), self.categories)
        raw = float(np.asarray(self.booster.predict(x, raw_score=True))[0])
        probability = float(self.calibrator.predict(np.array([raw]))[0])
        position = np.searchsorted(self.reference, probability, side="right")
        percentile = float(min(100.0, 100.0 * position / len(self.reference)))
        values = self.explainer.shap_values(x)
        if isinstance(values, list):  # older shap: one array per class
            values = values[1]
        return Score(
            probability=probability,
            raw_score=raw,
            percentile=percentile,
            relative_risk=probability / self.base_rate if self.base_rate else float("nan"),
            reasons=top_reasons(np.asarray(values)[0], x.iloc[0]),
        )
