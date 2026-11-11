"""Platt calibration: a logistic regression on the model's raw score, fitted on validation data.

Platt (2 parameters) rather than isotonic: the validation year has only a handful of
failures, and isotonic regression would fit a staircase to them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class PlattCalibrator:
    coef: float = 1.0
    intercept: float = 0.0

    def fit(self, raw_score: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        model = LogisticRegression(C=1e6, max_iter=1000)  # effectively unregularised
        model.fit(raw_score.reshape(-1, 1), y)
        self.coef = float(model.coef_[0, 0])
        self.intercept = float(model.intercept_[0])
        return self

    def predict(self, raw_score: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.coef * raw_score + self.intercept)))

    def to_dict(self) -> dict[str, float]:
        return {"coef": self.coef, "intercept": self.intercept}

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> PlattCalibrator:
        return cls(coef=values["coef"], intercept=values["intercept"])
