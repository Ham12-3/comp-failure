"""Evaluation: ranking and probability metrics, bootstrap confidence intervals, plots.

With a 0.4% base rate and few positives, point estimates alone mislead, so every test metric
is reported with a bootstrap 95% interval.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

Metric = Callable[[np.ndarray, np.ndarray], float]


def precision_recall_at(
    y: np.ndarray, score: np.ndarray, top_fraction: float
) -> tuple[float, float]:
    """Precision and recall among the `top_fraction` highest scores (ties keep input order)."""
    n_top = max(1, math.ceil(top_fraction * len(y)))
    top = np.argsort(-score, kind="stable")[:n_top]
    hits = int(y[top].sum())
    positives = int(y.sum())
    return hits / n_top, (hits / positives if positives else float("nan"))


def ranking_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    p1, r1 = precision_recall_at(y, score, 0.01)
    p5, r5 = precision_recall_at(y, score, 0.05)
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "precision_top1": p1,
        "recall_top1": r1,
        "precision_top5": p5,
        "recall_top5": r5,
    }


def all_metrics(y: np.ndarray, score: np.ndarray, prob: np.ndarray | None) -> dict[str, float]:
    out = ranking_metrics(y, score)
    out["brier"] = float(brier_score_loss(y, prob)) if prob is not None else float("nan")
    out["base_rate"] = float(y.mean())
    out["positives"] = float(y.sum())
    return out


def bootstrap_intervals(
    y: np.ndarray,
    score: np.ndarray,
    prob: np.ndarray | None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """95% percentile intervals for every metric. Resamples without a positive are skipped."""
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        values = all_metrics(y[idx], score[idx], prob[idx] if prob is not None else None)
        for k, v in values.items():
            samples.setdefault(k, []).append(v)
    return {
        k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
        for k, v in samples.items()
        if not all(math.isnan(x) for x in v)
    }


def plot_pr_curves(y: np.ndarray, scores: Mapping[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, score in scores.items():
        precision, recall, _ = precision_recall_curve(y, score)
        ax.plot(recall, precision, label=f"{name} (AP {average_precision_score(y, score):.3f})")
    ax.axhline(y.mean(), color="grey", linestyle=":", label=f"base rate {y.mean():.3%}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall, test set", ylim=(0, 1))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def reliability_table(y: np.ndarray, prob: np.ndarray, n_bins: int = 5) -> list[dict[str, float]]:
    """Mean predicted vs observed rate in equal-count bins (equal-width bins would put almost
    every row in the first bin at a 0.4% base rate)."""
    order = np.argsort(prob, kind="stable")
    rows = []
    for chunk in np.array_split(order, n_bins):
        rows.append(
            {
                "mean_predicted": float(prob[chunk].mean()),
                "observed": float(y[chunk].mean()),
                "rows": float(len(chunk)),
                "positives": float(y[chunk].sum()),
            }
        )
    return rows


def plot_reliability(
    y: np.ndarray, probs: Mapping[str, np.ndarray], path: Path, n_bins: int = 5
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    top = 0.0
    for name, prob in probs.items():
        table = reliability_table(y, prob, n_bins)
        xs = [r["mean_predicted"] for r in table]
        ys = [r["observed"] for r in table]
        top = max(top, *xs, *ys)
        ax.plot(xs, ys, marker="o", label=name)
    ax.plot([0, top], [0, top], color="grey", linestyle=":", label="perfect calibration")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed insolvency rate",
        title=f"Reliability, test set ({n_bins} equal-count bins)",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
