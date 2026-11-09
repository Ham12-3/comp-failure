# CLAUDE.md

Project rules for Claude Code. The full brief is in [PROJECT_SPEC.md](PROJECT_SPEC.md); read it before starting a phase.

## What this is

A portfolio ML app that predicts the probability a UK company enters insolvency (liquidation,
administration, receivership) within 12 months, from free Companies House data. Code quality,
honest evaluation and clear docs matter more than the last 1% of accuracy.

## Commands

```bash
uv sync                      # install everything (runtime + dev group)
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy                  # type check (strict, src + tests)
uv run pytest                # tests, excluding `network`-marked ones
uv run pytest -m network     # tests that call the real Companies House API
```

CI (`.github/workflows/ci.yml`) runs: `uv sync --locked`, `ruff check`, `ruff format --check`,
`mypy`, `pytest`. All four must pass before a phase is done.

Pipeline steps (each is resumable and skips work already done):

```bash
uv run python -m cfp.ingest.accounts_download --dry-run   # size estimate only
uv run python -m cfp.ingest.accounts_download             # sample accounts -> data/raw/accounts_sample/
uv run python -m cfp.ingest.accounts_parse                # -> data/interim/accounts.parquet
uv run python -m cfp.ingest.ch_api save-fixtures <company_number>...   # real API examples
uv run python -m cfp.ingest.company_data                  # -> data/interim/company_snapshot.parquet
uv run python -m cfp.ingest.fetch_companies [--limit N]   # API data for sampled companies (~14.5 h)
uv run python -m cfp.labels.build_labels                  # cache only -> data/processed/labels.parquet
uv run python -m cfp.features.build_features              # cache only -> data/processed/features.parquet
```

The download keeps a running byte total in `data/raw/accounts_sample/manifest.json` and refuses
to go past `--max-gb` (default 5). Do not raise it without asking.

## How to work

- **One phase at a time** (see PROJECT_SPEC.md section 10). At the end of each phase: run all
  checks, write a short summary of what was built and the decisions made, append the decisions
  to `docs/DECISIONS.md`, then **stop and wait for the user's go-ahead**.
- **Sample first.** Build every step on a small sample (100k-200k companies). Scale up only once
  the pipeline works end to end.
- **CPU only.** No GPU-only libraries or settings.
- Add a runtime dependency only in the phase whose code needs it (`uv add <pkg>`), and commit
  `uv.lock` with it.

## Hard rules

### Data and APIs
- **Ask before downloading more than 5 GB** of data in total.
- **Never invent API fields.** Check real Companies House responses and save one example of each
  endpoint in `tests/fixtures/`. Code and tests use field names from those fixtures.
- Respect the API rate limit with a token bucket, back off on HTTP 429, and cache every response
  on disk (`data/raw/ch_api_cache/`) so nothing is fetched twice.
- **Never hard-code or commit secrets.** The API key comes from `COMPANIES_HOUSE_API_KEY` via
  `cfp.config.Settings` (a `SecretStr`). `.env` is gitignored; `.env.example` holds placeholders.
- Downloaded files are untrusted data: write them under `data/raw/` only, never execute them.

### Leakage (the most important rules)
- **Prediction date = the date accounts were filed** (when the information became public), not the
  balance sheet date.
- **Never use `company_status`, dissolution dates, or any field that reflects the outcome** as a
  feature.
- **Every non-financial feature filters `event_date <= prediction_date`.** A pytest test must fail
  if any feature uses an event after the prediction date.
- **No random train/test splits.** Split by prediction date only.
- Drop rows whose 12-month outcome window is not yet fully observable.
- **If a result looks too good (e.g. PR-AUC > 0.9), assume leakage** and investigate before moving on.

### Code style
- Simple, readable code over clever code. Type hints everywhere (`mypy --strict` must pass).
- Use `pathlib`, not `os.path`. Use timezone-aware or plain `date` objects deliberately; ruff's
  `DTZ` rules are on because sloppy dates cause leakage.
- Prefer DuckDB / Polars on Parquet for large data; avoid loading everything into pandas.
- Tests must not hit the network unless marked `@pytest.mark.network`. Use fixtures and mocked
  transports (e.g. `httpx.MockTransport`) for API client tests.

## Layout

```
src/cfp/ingest/    download + parse accounts, Companies House API client
src/cfp/labels/    insolvency labels per (company, prediction date)
src/cfp/features/  point-in-time features
src/cfp/train/     training, evaluation, calibration
src/cfp/serve/     FastAPI app + schemas
app/               front end
data/              raw/ interim/ processed/  (gitignored, never commit data)
reports/           plots and metrics committed as evaluation evidence
docs/              DATA.md, MODEL_CARD.md, DECISIONS.md
```

The dev machine is Windows; keep paths and scripts cross-platform (CI runs on Linux).
