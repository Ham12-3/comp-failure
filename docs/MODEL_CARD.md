# Model card: UK company insolvency risk (12 months)

> **Not financial or credit advice.** This is a portfolio project showing an end-to-end,
> leakage-safe machine-learning pipeline on free public data. Its scores must not be used to
> make decisions about real companies or the people behind them.

## Model

| | |
|---|---|
| Task | Probability that a UK company enters compulsory liquidation, creditors' voluntary liquidation, administration or receivership within 12 months of filing its accounts. |
| Model | LightGBM (gradient-boosted trees), class-weighted (`scale_pos_weight` = negatives / positives), `num_leaves=16`, `min_child_samples=20`, 300 trees, learning rate 0.05, `cat_smooth=50`, `max_cat_threshold=8`. |
| Calibration | Platt scaling fitted on the 2023 validation year. |
| Explanations | SHAP (TreeExplainer). Each score comes with the 5 features that moved it most, in plain English. |
| Inputs | 73 point-in-time features: the accounts filing itself, filing history, directors, charges and static company attributes (see `docs/DATA.md` section 5). |
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

| Model | PR-AUC | ROC-AUC | Recall in top 5% | Brier |
|---|---|---|---|---|
| Rule: late or negative net assets | 0.006 [0.003, 0.010] | 0.64 [0.51, 0.75] | 0% [0%, 26%] | n/a |
| Logistic regression | 0.010 [0.004, 0.026] | 0.68 [0.55, 0.81] | 20% [0%, 43%] | 0.004 |
| **LightGBM (chosen)** | 0.007 [0.003, 0.017] | 0.56 [0.39, 0.74] | 13% [0%, 32%] | 0.004 |

Validation (the only basis for choosing; mean PR-AUC over 2021, 2022 and 2023, each predicted
by a model trained on earlier years): rule 0.006, logistic regression 0.012, **LightGBM 0.028**.

**How to read this honestly:**

* All models rank better than chance on validation, but **the signal is weak** and the test
  set is too small to tell them apart: with 15 failures, every interval overlaps.
* LightGBM was chosen on validation and kept even though logistic regression did better on
  the test years. Switching after looking at the test set would make the test score
  meaningless.
* LightGBM's validation-to-test drop (0.071 to 0.007 PR-AUC in 2023 vs 2024-2025) points to
  overfitting to a few dozen failures, and possibly drift between years.
* Full tables: `reports/metrics.md`. Plots: `reports/pr_curve.png`, `reports/reliability.png`,
  `reports/shap_summary.png`.

## Calibration

Platt scaling on 2023 brings predicted probabilities onto the right scale (test mean predicted
0.43% vs observed 0.39%). Bin by bin it is noisy: the highest-risk fifth is predicted 1.1% and
observed 0.8%. With so few failures, treat any single probability as a rough band, not a
precise number.

## What drives the score

From the SHAP summary on the test set: current ratio (current assets / short-term debts),
total debt, registered-office postcode area, directors' median tenure, current assets, total
assets and liabilities, equity ratio, and how late earlier accounts were filed. Most match the
credit-risk intuition: weaker liquidity, more debt and late filing raise the score. Formal
insolvency also skews towards trading companies with creditors and employees, while dormant
shells tend to be struck off instead.

## Limitations and known biases

* **Very few failures (30 to train on).** The single biggest limitation. Results will move as
  more companies are fetched (`cfp.ingest.fetch_companies` resumes where it stopped).
* **Postcode area ranks third in SHAP.** With about 120 areas and few failures this is likely
  partly overfitting, and it could encode regional bias. It is a candidate for removal or for
  coarser regions.
* **Small companies disclose little.** 45% of filings are micro-entity accounts with a handful
  of balance-sheet figures and no profit and loss. Turnover and profit are missing in over
  95% of filings, so scores lean on balance sheets, missingness and filing behaviour.
* **Only electronic filings** (about 75% of accounts). Paper filers are not represented.
* **Approximations:** accounts are treated as due 9 months after the balance sheet date (6 for
  plcs, different for first accounts); SIC code and legal form are today's values.
* **Drift:** insolvencies were unusually low in 2020-2021 (COVID-era restrictions on winding-up
  petitions), so the training years differ from the test years.
* **What counts as failure:** most small companies that stop trading are struck off rather
  than entering formal insolvency. This model does not predict strike-off; there is a separate
  label for that, not yet modelled.
* **Repeated companies:** a company contributes one row per year, and the same company can
  appear in training and test years. This matches how the model would be used (scoring known
  companies) but makes rows less independent than the counts suggest.

## Ethical considerations

A high score can stigmatise a small business and its owners, and false positives far
outnumber true ones at a 0.4% base rate. In the top 5% of scores, about 1 in 60 to 100
companies actually failed. Scores are explanations of patterns in public data, not
judgements, and any real use would need human review, a much larger sample and fairness
checks across regions and sectors.

## Reproduce

```bash
uv run python -m cfp.labels.build_labels
uv run python -m cfp.features.build_features
uv run python -m cfp.train.train      # reports/, models/, MLflow run
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse runs
```
