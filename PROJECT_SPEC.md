# Project Spec: UK Company Failure Predictor

## 1. Goal

Build an end-to-end ML application that predicts the chance a UK company will enter insolvency (liquidation or administration) within 12 months, using free Companies House data.

The user types a company name or number and gets:

* A risk score (0 to 100%), calibrated so "30%" means roughly 30 out of 100 similar companies fail
* The top 5 reasons behind the score (SHAP), in plain English
* Links to the Companies House records the features came from

This is a portfolio project. Code quality, honest evaluation and clear documentation matter more than squeezing out the last 1% of accuracy.

## 2. Tech stack

* Python 3.12, `uv` for dependency management
* Polars or pandas + DuckDB for data processing (data is large, prefer DuckDB / Parquet)
* `ixbrlparse` (or lxml) to read iXBRL accounts files
* `httpx` with retries for the Companies House API
* scikit-learn, LightGBM (or XGBoost), SHAP
* MLflow for experiment tracking (local)
* FastAPI for serving, Pydantic for schemas
* Streamlit or a small Next.js front end for the demo
* pytest, ruff, mypy, GitHub Actions CI
* Docker + docker compose

The dev machine is CPU only. Everything must run on CPU. Work on a sample first (for example 100k to 200k companies) and only scale up once the pipeline works.

## 3. Data sources

1. Companies House historic and monthly Accounts Data Product (free bulk iXBRL/XBRL accounts files)
   * https://download.companieshouse.gov.uk/historicmonthlyaccountsdata.html
   * https://download.companieshouse.gov.uk/en_monthlyaccountsdata.html
   * Main source of financial features. Each file has the company number and balance sheet date in its name.
2. Companies House Public Data API (free key, register at developer.company-information.service.gov.uk)
   * Endpoints: company profile, officers, filing history, charges, insolvency, persons with significant control
   * Used to get labels (current status, insolvency events and dates) and extra non-financial features
   * Respect the rate limit (check current docs, roughly 600 requests per 5 minutes). Use a token bucket, back off on HTTP 429, and cache every response on disk so nothing is fetched twice.
3. Free Company Data Product (monthly CSV of live companies)
   * Note: it only contains live companies, so it cannot be used alone to build failure labels.

API key goes in `.env` as `COMPANIES_HOUSE_API_KEY`. Never commit it.

## 4. Label definition (most important section)

* Each training row = one company at one prediction date.
* Prediction date = the date its accounts were filed (the date the information became public), not the balance sheet date.
* Label = 1 if the company has an insolvency event (liquidation, administration, receivership) within 12 months after the prediction date. Use the insolvency endpoint and case dates.
* Label = 0 if the company was still active, or not insolvent, 12 months after the prediction date.
* Voluntary strike-off (dissolved with no insolvency) is not counted as failure in the main model. Keep it as a separate label column so we can try it as a second target later.
* Drop rows where we cannot yet see 12 months of outcome (prediction date too recent).

Document the final label definition and class balance in `docs/DATA.md`.

## 5. Features

Only use information that was public on or before the prediction date.

Financial (from accounts):

* Net assets, current assets, current liabilities, cash, total debt, turnover (if filed), profit (if filed)
* Ratios: current ratio, debt to assets, cash to liabilities, net assets sign, year on year change where a previous filing exists
* Accounts type (micro, small, full, dormant), and whether fields are missing (missingness is a signal)

Non-financial (from API, filtered to before the prediction date):

* Company age, SIC code / sector, region
* Days late on previous accounts and confirmation statements
* Number of director appointments and resignations in the last 12 months
* Number of outstanding charges, charges created in the last 12 months
* Change of registered office in the last 12 months

## 6. Leakage rules (enforce with tests)

* Never use `company_status`, dissolution dates or any field that reflects the outcome.
* Every non-financial feature must be computed with a date filter `event_date <= prediction_date`.
* Write a pytest test that fails if any feature uses events after the prediction date.
* No random train/test split.

## 7. Training and evaluation

* Time-based split: for example train on prediction dates 2017 to 2021, validate on 2022, test on 2023. Adjust to the data actually available and document it.
* Baseline 1: simple rule (for example "accounts overdue OR negative net assets").
* Baseline 2: logistic regression.
* Main model: LightGBM with class weighting. Tune on the validation set only.
* Metrics: PR-AUC (main), ROC-AUC, precision and recall at the top 1% and 5% of risk scores, Brier score.
* Calibration: isotonic or Platt on validation data, plus a reliability plot.
* Explainability: global SHAP summary plot and per-company SHAP top 5.
* Log every run to MLflow. Save plots to `reports/`.
* Write an honest `docs/MODEL_CARD.md`: intended use, limits, known biases (for example micro-entity accounts have few fields), and "not financial or credit advice".

## 8. Application

* `POST /predict` with `{ "company_number": "..." }` returns score, calibrated probability, top 5 reasons, and source links.
* `GET /search?q=` returns matching companies (via API, cached).
* `GET /health` and `GET /metrics` (request count, latency, errors).
* If a company has no accounts on file, return a clear message instead of a score.
* Front end: search box, risk gauge, reasons list, links to Companies House pages.
* Show a disclaimer on the page.

## 9. Repo structure

```
company-failure-predictor/
  PROJECT_SPEC.md
  CLAUDE.md
  README.md
  docs/        DATA.md, MODEL_CARD.md, DECISIONS.md
  data/        raw/, interim/, processed/   (gitignored)
  src/cfp/
    ingest/    accounts_download.py, accounts_parse.py, ch_api.py
    labels/    build_labels.py
    features/  financial.py, filings.py, officers.py, charges.py
    train/     train.py, evaluate.py, calibrate.py
    serve/     api.py, schemas.py
  app/         front end
  tests/
  reports/     plots and metrics
  Dockerfile
  docker-compose.yml
  .github/workflows/ci.yml
```

## 10. Build phases

Work one phase at a time. At the end of each phase: run tests, show me a short summary of what was built and any decisions made, add them to `docs/DECISIONS.md`, then stop and wait for my go-ahead.

* **Phase 0: Setup.** Repo, uv, ruff, mypy, pytest, CI, `.env.example`, CLAUDE.md with project rules. Done when: CI passes on an empty test.
* **Phase 1: Data ingestion.** Download a small number of monthly accounts files, parse them to Parquet (company number, balance sheet date, filing date, key fields). Build the cached, rate-limited API client. Done when: a sample of at least 50k parsed accounts in Parquet, API client has tests for retries and 429 handling.
* **Phase 2: Labels.** Fetch profiles and insolvency data for the sampled companies, build labels as defined in section 4. Done when: `docs/DATA.md` shows row counts, date range, class balance.
* **Phase 3: Features + leakage tests.** Done when: feature table built, leakage test passes, feature list documented.
* **Phase 4: Training and evaluation.** Baselines, LightGBM, calibration, SHAP, MLflow, model card. Done when: `reports/` contains metrics table, PR curve, reliability plot, SHAP summary.
* **Phase 5: API + front end.** Done when: `docker compose up` starts everything and a real company number returns a score with reasons.
* **Phase 6: Polish.** README with problem, results table, architecture diagram, how to run, limits, and a "what went wrong" section. Scale to a bigger sample if time allows. Optional: monthly retrain script and drift check.

## 11. Rules for Claude Code

* Ask me before downloading more than 5 GB of data.
* Prefer simple, readable code over clever code. Type hints everywhere.
* Do not invent API fields. Check real responses and save one example of each in `tests/fixtures/`.
* Never hard-code secrets.
* If a result looks too good (for example PR-AUC above 0.9), assume leakage and investigate before moving on.
