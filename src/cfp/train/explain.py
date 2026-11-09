"""SHAP explanations: a global summary plot and plain-English top reasons per company."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from cfp.features.build_features import FEATURES
from cfp.train.data import BOOLEAN

AMOUNTS = frozenset(
    {
        "net_assets",
        "total_assets",
        "current_assets",
        "current_liabilities",
        "cash",
        "total_liabilities",
        "total_debt",
        "bank_debt",
        "turnover",
        "profit_loss",
        "net_assets_change",
    }
)
PERCENT_CHANGES = frozenset(f for f in FEATURES if f.endswith("_pct_change"))


def shap_values(booster: lgb.Booster, x: pd.DataFrame) -> np.ndarray:
    """Per-row, per-feature contributions to the raw (log-odds) score."""
    values = shap.TreeExplainer(booster).shap_values(x)
    if isinstance(values, list):  # older shap returns one array per class
        values = values[1]
    return np.asarray(values)


def plot_summary(values: np.ndarray, x: pd.DataFrame, path: Path, max_display: int = 20) -> None:
    shap.summary_plot(values, x, max_display=max_display, show=False, plot_size=(8, 7))
    plt.title("SHAP: what drives the insolvency score (test set)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def format_value(name: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "not available"
    if isinstance(value, bool) or name in BOOLEAN:
        return "yes" if value else "no"
    if name in AMOUNTS:
        return f"£{value:,.0f}"
    if name in PERCENT_CHANGES:
        return f"{value:+.0%}"
    if isinstance(value, float):
        return f"{value:,.0f}" if value.is_integer() else f"{value:,.2f}"
    return str(value)


@dataclass(frozen=True)
class Reason:
    feature: str
    value: str
    contribution: float  # log-odds; > 0 raises the risk
    text: str


def top_reasons(contributions: np.ndarray, row: pd.Series, k: int = 5) -> list[Reason]:
    """The k features that moved this company's score most, in plain English."""
    order = np.argsort(-np.abs(contributions), kind="stable")[:k]
    reasons = []
    for i in order:
        name = list(FEATURES)[i]
        shown = format_value(name, row[name])
        direction = "raises" if contributions[i] > 0 else "lowers"
        meaning = FEATURES[name][1].rstrip(".")
        reasons.append(
            Reason(
                feature=name,
                value=shown,
                contribution=float(contributions[i]),
                text=f"{meaning}: {shown}. This {direction} the risk.",
            )
        )
    return reasons
