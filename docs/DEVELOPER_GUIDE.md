# Developer guide

Everything you need to work on the UK Company Failure Predictor: how it fits together, how to
run and change it, the rules that keep it honest, and where the traps are. For *why* things are
the way they are, see [DECISIONS.md](DECISIONS.md); for the data, [DATA.md](DATA.md); for model
results and limits, [MODEL_CARD.md](MODEL_CARD.md).

## 1. What it is

A Python 3.12 project (`src/cfp`) that predicts whether a UK company will enter **compulsory or
creditors' voluntary liquidation, administration or receivership within 12 months of filing its
accounts**, from free Companies House data. A FastAPI service with a static web page serves the
scores, in Docker. The work runs in four stages:

1. **Sample and parse accounts** from the bulk Accounts Data Product.
2. **Fetch API data** (profile, filing history, officers, charges, insolvency) for the sampled
   companies.
3. **Build labels and point-in-time features.**
4. **Train and serve.**

Every stage is a resumable command, and each stage's output is the next one's input.

## 2. State as of 11 November 2026

| Thing | State |
|---|---|
| Branches | `phase-0-setup` → PR #1 into `master` (open). `phases-1-3-pipeline` → PR #2, stacked on #1 (open, CI green). Merge #1 first, then switch #2's base to `master`. |
| Data | 125,032 parsed filings for 29,584 sampled companies, but **API data for only 3,556 companies**: the fetch stopped and was deliberately not restarted. |
| Model | Trained on 3,554 companies: 54 failures (30 train / 9 validation / 15 test). Test PR-AUC 0.016 (about 4× chance), ROC-AUC 0.74. |
| App | Runs at http://localhost:8000 via Docker (`docker compose down` to stop). |
| On disk (gitignored) | `data/` about 1.7 GB, API cache 80 MB, `models/` 320 KB, `mlflow.db` 1 MB. |

## 3. Setup

You need:

* Python 3.12 (pinned in `.python-version`) and [uv](https://docs.astral.sh/uv/);
* Docker, for the app;
* a **Live** Companies House API key from the
  [developer hub](https://developer.company-information.service.gov.uk/). A "Test" key only
  reaches an empty sandbox: every real company returns 401 or 404.

```bash
cp .env.example .env   # put the key in COMPANIES_HOUSE_API_KEY
uv sync                # runtime + dev + train dependencies
```

`uv sync` installs three groups (`[tool.uv] default-groups`): runtime, `dev` (ruff, mypy,
pytest, stubs) and `train` (MLflow). The Docker image installs runtime only
(`--no-default-groups`).

**Settings** (`src/cfp/config.py`, read from the environment or `.env`):

| Variable | Default | Used for |
|---|---|---|
| `COMPANIES_HOUSE_API_KEY` | none | Everything that calls the API. Stored as a `SecretStr`; a blank value counts as unset. |
| `CFP_DATA_DIR` | `data` | Raw, interim and processed data, plus the API cache |
| `CFP_REPORTS_DIR` | `reports` | Metrics, plots, drift report |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow runs |
| `CFP_MODEL_DIR` | `models` | Serving: where the trained model is |
| `CFP_FRONTEND_DIR` | `app` | Serving: where the web page is |

## 4. The pipeline

All commands are `uv run python -m <module>`.

| # | Module | Reads | Writes | Notes |
|---|---|---|---|---|
| 1 | `cfp.ingest.accounts_download` | Companies House listing pages, remote zips | `data/interim/accounts_index/*.parquet` (index of all 24.9M filings), `data/raw/accounts_sample/*.zip`, `manifest.json` | HTTP range requests; 0.5% company hash sample. Flags `--from`, `--to`, `--rate`, `--max-gb`, `--dry-run`. **5 GB budget, 4.06 GB used: ask before raising it.** About 30 minutes. |
| 2 | `cfp.ingest.accounts_parse` | sample zips | `data/interim/accounts.parquet` (one row per filing) | About 1 hour on 8 cores. Skips archives already parsed, so **use `--force` after changing the parser.** |
| 3 | `cfp.ingest.company_data` (optional) | bulk company CSV | `data/interim/company_snapshot.parquet` | Today's state of 5.7M companies. Used for checks and `--skip-active` only, **never as features**. |
| 4 | `cfp.ingest.fetch_companies --keep-awake` | `accounts.parquet` | `data/raw/ch_api_cache/responses.sqlite` | About 2,100 companies an hour (rate limit). Resumable; companies are fetched in random (hash) order, so a partial fetch is a fair sample. `--skip-active` fetches only what labels need. |
| 5 | `cfp.labels.build_labels` | accounts + API cache | `data/processed/labels.parquet` | Cache only: never calls the network. |
| 6 | `cfp.features.build_features` | labels + API cache | `data/processed/features.parquet` | 73 features. Cache only. |
| 7 | `cfp.train.train` | features + labels | `reports/*`, `models/lightgbm.txt`, `models/model_meta.json`, an MLflow run | About 1-2 minutes. |
| 8 | `cfp.train.drift` | features | `reports/drift.md` | PSI per feature, training years vs recent years. |
| 9 | `docker compose up --build` | `models/`, `.env` | serves on port 8000 | Or `uv run uvicorn cfp.serve.api:app --reload` without Docker. |

`scripts/refresh.sh YYYY-MM` runs steps 1-8 for a new month of accounts.

## 5. Code map

```
src/cfp/
  config.py            Settings (env/.env), paths
  company_number.py    normalise_company_number(): "566433" -> "00566433"; the join key everywhere
  ingest/
    accounts_download  remote-zip range reads, hash sampling, byte budget, manifest
    accounts_parse     iXBRL/XBRL -> one row per filing (CONCEPTS list, dimension rules);
                       parse_live_document() for serving
    ch_api             CompaniesHouseClient: token bucket, retries, 429 handling, SQLite cache,
                       cache_only mode, Document API download
    company_data       bulk company snapshot loader
    fetch_companies    per-company fetch orchestration
  labels/
    build_labels       insolvency cases and dates -> label_insolvency_12m, label_strike_off_12m,
                       excluded_reason
  features/
    events.py          as_of(): the single time gate for every API-based feature
    financial.py       sign repair, then amounts, ratios, year-on-year, missingness (accounts only)
    filings.py         late filing, gazette notices, office moves, director changes,
                       registered-office postcode as of the prediction date
    officers.py        directors in office on the prediction date
    charges.py         outstanding / registered / satisfied charges
    profile.py         allow-listed static fields: age, legal form, jurisdiction, SIC section
    build_features.py  FEATURES registry (single source of truth) and table assembly
  train/
    data.py            SPLITS (2018-22 / 2023 / 2024-25), categories, to_matrix()
    train.py           rule + logistic + LightGBM, rolling-validation tuning, Platt calibration,
                       reports, MLflow
    evaluate.py        metrics, bootstrap intervals, plots
    calibrate.py       Platt scaling
    explain.py         SHAP and plain-English reasons
    drift.py           PSI drift check
  serve/
    live.py            LiveScorer: profile -> latest accounts -> Document API iXBRL ->
                       the same features as training -> model
    model.py           RiskModel: loads models/, checks the feature list, scores + SHAP
    api.py             FastAPI app, metrics middleware, static web page
    schemas.py         request and response models
app/                   index.html, app.js, styles.css (served by FastAPI)
scripts/refresh.sh     monthly refresh
tests/                 118 tests; fixtures are real Companies House responses
```

## 6. Rules you must not break

1. **The prediction date is the filing date** of the accounts (when they became public), not
   the balance sheet date. See `labels.attach_filing_dates`.
2. **Every API-based feature goes through `features/events.as_of`**, which keeps only events
   dated on or before the prediction date. The one exception is `office_postcode_area`, which
   deliberately reconstructs the address in force on the prediction date.
   `tests/test_leakage.py` rebuilds each row's features from a register snapshot taken on its
   prediction date and requires identical output. It checks **every** column in `FEATURES`, so
   new features are covered automatically, provided the test data exercises the event type
   you use.
3. **Outcome fields stay in the label step.** `company_status`, `date_of_cessation`,
   accounts/confirmation due dates and insolvency links are used only in `build_labels`, which
   drops them from its output. `features/profile.py` reads only `ALLOWED_FIELDS`.
4. **`FEATURES` in `build_features.py` is the single source of truth.** Output columns, the
   docs table and tests all derive from it. If `models/model_meta.json` lists different
   features from the code, the API starts in **degraded** mode instead of scoring wrongly.
5. **No random splits, and never choose using the test years.** Tuning and model choice use
   `TUNING_YEARS` (2021-2023) only.
6. **Never invent API fields.** Save real responses first
   (`uv run python -m cfp.ingest.ch_api save-fixtures <company_number>...`), then write code
   against `tests/fixtures/ch_api/`.
7. **Secrets.** The key lives only in `.env`, which both git and Docker ignore. When the
   Document API redirects to S3, a **separate client without the key** downloads the file;
   a test checks this.
8. **The accounts table keeps values exactly as tagged.** All repairs (sign fixes, taxonomy
   mapping) live in `features/financial.py`, where they are tested.
9. **Ask before downloading more than 5 GB in total**, and respect the API rate limit (the
   client does, as long as you use it).

## 7. Common tasks

**Add a feature**

1. Write it in the right `features/*.py` module, joining API events through `as_of`.
2. Add it to `FEATURES` with its source and a one-line meaning.
3. If it uses a new kind of event, add that event to the synthetic company in
   `tests/test_leakage.py` (before and after the prediction dates).
4. Regenerate the feature table in `docs/DATA.md` from `feature_docs_markdown()`.
5. Rebuild features, retrain, and **rebuild the Docker image** (`docker compose up -d --build`).

**Retrain only:** `uv run python -m cfp.train.train`, then `docker compose restart`. The model
is mounted into the container, not built into the image.

**Change serving code or the web page:** `docker compose up -d --build`. Browsers cache the
static files, so hard-refresh the page.

**Get more data:** re-run `uv run python -m cfp.ingest.fetch_companies --keep-awake`. It resumes
where it stopped (company 3,557). `--skip-active` gets labels for the rest in about 2 hours
instead of about 13. Then rebuild labels, features and the model.

**Browse experiments:** `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`.

**Run the checks CI runs:**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
uv run mypy --platform linux   # CI runs on Linux; this catches platform-only type errors
```

## 8. Tests

118 tests, about 73% coverage. All run offline.

| File | Covers |
|---|---|
| `test_ch_api.py` | Rate limit (proved with a sliding window), 429 / `Retry-After` / `X-Ratelimit-Reset`, retries, cache, cache-only mode |
| `test_accounts_download.py` | Range reads against a mock S3, sampling, download budget |
| `test_accounts_parse.py` | A synthetic filing and two real ones; both dimension conventions; sign and scale formats |
| `test_build_labels.py` | Real Carillion and Tesco responses; window edges; solvent liquidations; ended cases |
| `test_company_data.py` | Bulk snapshot loader, using the real CSV header |
| `test_fetch_companies.py` | Which endpoints are called, and when |
| `test_financial_features.py` | Sign repair in both directions |
| `test_leakage.py` | The snapshot test (Carillion and synthetic), outcome-column guard, profile allow-list |
| `test_train.py`, `test_drift.py` | Training, calibration, explanations and drift on synthetic data with a known signal |
| `test_serve.py` | The full API against a mocked Companies House; API key never sent to S3; no-score messages |

A `network` marker exists for tests that call the real API (`uv run pytest -m network`), but none
are written yet.

## 9. Serving

**Endpoints**

| Endpoint | Does |
|---|---|
| `POST /predict` `{"company_number": "..."}` | Score, calibrated probability, percentile, risk band, top-5 reasons, Companies House links, warnings |
| `GET /search?q=` | Companies House search |
| `GET /health` | `ok`, or `degraded` if the model is missing or mismatched or the key is not set |
| `GET /metrics` | Request counts, 4xx/5xx counts, latency mean/p50/p95 (in-process; resets on restart) |
| `GET /docs` | Interactive API documentation |

**Scoring flow** (`serve/live.py`):

1. Profile. If the company is not active (liquidation, administration, dissolved...), return
   a message instead of a score.
2. One filtered request for the latest accounts filing.
3. The iXBRL of that filing from the Document API. PDF-only accounts (common for large groups)
   get a message.
4. Full filing history, officers and charges.
5. The same parser and feature code as training, with the prediction date set to that filing
   date, then the model and the top-5 SHAP reasons.

The score therefore means "risk of insolvency within 12 months of the latest accounts filing",
with a warning when those accounts are more than about 13 months old.

**Caches:** API responses in the SQLite cache; downloaded iXBRL files in
`data/raw/ch_api_cache/documents/`. In Docker both live in the named volume `api-cache`.

**Concurrency:** one lock serialises all Companies House calls, which suits a rate limit of
about 2 requests a second.

**Docker:** slim Python 3.12 image (1.44 GB, mostly the scientific Python stack), non-root user,
health check. Compose mounts `./models` read-only and reads `.env` at run time; neither the key
nor any data is in the image.

## 10. Known gotchas

* **PowerShell 5.1 `-Encoding utf8` writes a byte-order mark.** It silently broke
  `pyproject.toml` for mypy once. Edit config files with Python or an editor.
* **The Windows console (cp1252) crashes on box-drawing characters and emoji.** The code prints
  ASCII tables for this reason.
* **Git Bash stops responding when the parser uses every core.** PowerShell still works.
* **Keep-awake only prevents idle sleep.** A closed lid or low battery still stopped long jobs
  twice. Plug in and keep the lid open for long runs.
* **The SQLite cache has one writer.** Short reads alongside a running fetch are fine; for heavy
  analysis, work on a copy made with `sqlite3.backup`.
* **Polars evaluates all expressions in one `with_columns` call against the input frame.** If
  one expression needs another's result (for example a filled null), put them in separate calls.
* **MLflow 3.18 refuses the old `./mlruns` folder store**, which is why runs go to SQLite.
* **`gh` 2.36 cannot run `gh pr edit`** (it asks GitHub for a retired "Projects (classic)" field).
  Use `gh api -X PATCH repos/<owner>/<repo>/pulls/<n>` or upgrade `gh`.
* **CRLF warnings on commit are normal.** `.gitattributes` stores text files with LF.
* **Containers:** rebuild the image when code or `app/` changes; restart it when only the model
  changes.

## 11. Limits and technical debt

In priority order:

1. **Only 54 failures.** The API fetch covers 12% of the sample. More data is the biggest lever
   on model quality.
2. **The API has no authentication or rate limiting of its own.** Anyone who can reach port 8000
   spends *your* Companies House quota. Keep it local, or add auth and per-client limits before
   deploying anywhere public.
3. **Postcode area ranks 5th in SHAP:** likely partly overfitting, and a possible source of
   regional bias. Consider broader regions.
4. **Approximations:** accounts are treated as due 9 months after the balance sheet date (plcs
   have 6, and first accounts differ); SIC code and legal form are today's values.
5. **The test set has been looked at after data fixes**, so treat its numbers as somewhat
   optimistic. A fresh year of outcomes is the clean check.
6. **Operations:** no Docker build in CI; metrics are in-process only (no Prometheus); the
   document cache grows without limit; the split years are hard-coded in `SPLITS`.
7. **The strike-off label is built but not modelled.**

## 12. Where to read more

| Document | Read it for |
|---|---|
| [DECISIONS.md](DECISIONS.md) | 74 decisions, each with its reason. Read before changing a design choice. |
| [DATA.md](DATA.md) | Sample, label definition, class balance, the full feature table, data quality issues |
| [MODEL_CARD.md](MODEL_CARD.md) | Results, calibration, limits, biases, ethics |
| [../reports/](../reports/) | Metrics, PR curve, reliability plot, SHAP summary, drift report |
| [../CLAUDE.md](../CLAUDE.md) | Project rules, which also apply to AI assistants working on the code |
| [../PROJECT_SPEC.md](../PROJECT_SPEC.md) | The original brief |
