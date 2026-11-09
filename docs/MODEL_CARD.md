# Model card: UK company insolvency risk (12 months)

> **Not financial or credit advice.** This is a portfolio project showing an end-to-end,
> leakage-safe machine-learning pipeline on free public data. Its scores must not be used to
> make decisions about real companies or the people behind them.

## Model

| | |
|---|---|
| Task | Probability that a UK company enters compulsory liquidation, creditors' voluntary liquidation, administration or receivership within 12 months of filing its accounts. |
| Model | LightGBM (gradient-boosted trees), class-weighted (`scale_pos_weight` = negatives / positives), `num_leaves=8`, `min_child_samples=20`, 300 trees, learning rate 0.05, `cat_smooth=50`, `max_cat_threshold=8`. |
| Calibration | Platt scaling fitted on the 2023 validation year. |
| Explanations | SHAP (TreeExplainer). Each score comes with the 5 features that moved it most, in plain English. |
| Inputs | 73 point-in-time features: the accounts filing itself (with sign errors repaired), filing history, directors, charges and static company attributes (see `docs/DATA.md` section 5). |
| Baselines | Rule "filed late OR negative net assets"; logistic regression. |
| Trained | 9 November 2026, by `python -m cfp.train.train`. Runs are logged to MLflow (`sqlite:///mlflow.db`). |

## Intended use

* Demonstrating how to build an insolvency-risk model on Companies House data without leakage:
  point-in-time features, time-based evaluation, calibration and explanations.
* Exploring which public signals go with insolvency risk.

**Out of scope:** credit, lending, insurance, procurement or investment decisions; any
automated decision about a company; judging the directors or owners of a company.

## Data

* **Sample:** a random 0.5% of companies (hash of the company number) that filed accounts
  electronically between January 2018 and October 2025. Of those, **3,554 companies** had their
  Companies House API data fetched before the fetch was stopped. The fetch runs in random
  order, so these companies are a random subsample.
* **Rows:** one per accounts filing. The prediction date is the filing date.
* **Label:** an insolvency case of the types above starting within 12 months of the filing.
  Solvent (members' voluntary) liquidations, CVAs and strike-offs do not count. Full
  definition in `docs/DATA.md` section 4.
* **Split by filing year, never randomly:**

| Split | Filing years | Rows | Failures |
|---|---|---|---|
| Train | 2018-2022 | 8,980 | 30 |
| Validation | 2023 | 2,110 | 9 |
| Test | 2024-2025 | 3,859 | 15 |

## Performance

Test set (2024-2025), bootstrap 95% intervals. Base rate 0.39%: a random ranking scores a
PR-AUC of about 0.004.

| Model | PR-AUC | ROC-AUC | Precision in top 5% | Recall in top 5% | Brier |
|---|---|---|---|---|---|
| Rule: late or negative net assets | 0.006 [0.003, 0.010] | 0.64 [0.51, 0.75] | 0.0% [0.0, 2.1] | 0% [0, 26] | n/a |
| Logistic regression | 0.010 [0.005, 0.027] | 0.71 [0.61, 0.83] | 1.0% [0.0, 2.6] | 13% [0, 33] | 0.004 |
| **LightGBM (chosen)** | **0.016** [0.006, 0.060] | **0.74** [0.60, 0.86] | **2.1%** [0.5, 4.7] | **27%** [6, 50] | 0.004 |

Validation (the only basis for choosing; mean PR-AUC over 2021, 2022 and 2023, each predicted
by a model trained on earlier years): rule 0.006, logistic regression 0.012, **LightGBM 0.024**.

**How to read this honestly:**

* LightGBM ranks failures about **4 times better than chance** (PR-AUC 0.016 against a base
  rate of 0.004), and the riskiest 5% of companies hold about a quarter of the failures. That
  is a real but modest signal.
* With 15 failures in the test years, **every interval is wide** and they overlap: the test
  set cannot say for sure that LightGBM beats logistic regression. Validation, where it did,
  is what the choice rests on.
* **The test set is no longer untouched.** Its results were visible after each of the data
  fixes in `docs/DECISIONS.md` (4.6, 6.2-6.4). None was chosen for its test score (each fixed
  something demonstrably wrong in the data), but the test numbers are somewhat optimistic for
  that reason. A fresh year of outcomes would be the clean check.
* Full tables: `reports/metrics.md`. Plots: `reports/pr_curve.png`, `reports/reliability.png`,
  `reports/shap_summary.png`.

## Calibration

Platt scaling on 2023 puts probabilities on the right scale. Test set, five equal-size groups
from lowest to highest score:

| Predicted | 0.16% | 0.23% | 0.29% | 0.45% | 1.02% |
|---|---|---|---|---|---|
| **Observed** | 0.13% | 0.13% | 0.13% | 0.52% | 1.04% |

The top two groups match closely; the bottom three are all slightly over-predicted (one failure
each). With 15 failures, treat any single probability as a rough band.

## What drives the score

From the SHAP summary on the test set: current ratio (current assets / short-term debts),
directors' median tenure, current assets, equity ratio, registered-office postcode area,
current liabilities, cash relative to short-term debts, dormancy and how late earlier accounts
were filed. Most match credit-risk intuition: weak liquidity, thin equity and late filing raise
the score. Formal insolvency skews towards trading companies with creditors and staff, while
dormant shells tend to be struck off instead, so dormancy lowers the score.

## Limitations and known biases

* **Very few failures (30 to train on).** The single biggest limitation. Results will move as
  more companies are fetched (`cfp.ingest.fetch_companies` resumes where it stopped).
* **Postcode area ranks fifth in SHAP.** With about 120 areas and few failures this is likely
  partly overfitting, and it could encode regional bias. It is a candidate for removal or for
  coarser regions.
* **Small companies disclose little.** 45% of filings are micro-entity accounts with a handful
  of balance-sheet figures and no profit and loss. Turnover and profit are missing in over
  95% of filings, so scores lean on balance sheets, missingness and filing behaviour.
* **Messy tagging.** About 16% of filings have at least one sign error corrected (values shown
  in brackets and tagged the wrong way, in both directions). The repairs use accounting
  identities and cannot catch everything.
* **Reporting practice drifts** (`reports/drift.md`): employee numbers are disclosed far more
  often recently (missing in 39% of 2018-22 filings, 13% of 2024-25), and audit-status and
  accounting-standard tags have shifted. The model learned some patterns from years that
  reported differently.
* **Only electronic filings** (about 75% of accounts). Paper filers are not represented.
* **Approximations:** accounts are treated as due 9 months after the balance sheet date (6 for
  plcs, different for first accounts); SIC code and legal form are today's values.
* **Economic drift:** insolvencies were unusually low in 2020-2021 (COVID-era restrictions on
  winding-up petitions), so the training years differ from the test years.
* **What counts as failure:** most small companies that stop trading are struck off rather
  than entering formal insolvency. This model does not predict strike-off; there is a separate
  label for that, not yet modelled.
* **Repeated companies:** a company contributes one row per year, and the same company can
  appear in training and test years. This matches how the model would be used (scoring known
  companies) but makes rows less independent than the counts suggest.

## Ethical considerations

A high score can stigmatise a small business and its owners, and false positives far
outnumber true ones at a 0.4% base rate: even in the riskiest 5% of scores, only about 1
company in 50 actually failed. Scores are explanations of patterns in public data, not
judgements, and any real use would need human review, a much larger sample and fairness checks
across regions and sectors. The project's own screenshots and examples do not name companies.

## Reproduce

```bash
uv run python -m cfp.labels.build_labels
uv run python -m cfp.features.build_features
uv run python -m cfp.train.train      # reports/, models/, MLflow run
uv run python -m cfp.train.drift      # reports/drift.md
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse runs
```
