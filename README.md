# UK Company Failure Predictor

Predicts the probability that a UK company enters insolvency (liquidation, administration or
receivership) within 12 months, using free Companies House data, and explains each score with
its top contributing factors.

> **Status:** Phase 3 (features and leakage tests); the full API fetch for Phase 2 labels is
> running. See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the plan, [docs/DATA.md](docs/DATA.md)
> for the data and [docs/DECISIONS.md](docs/DECISIONS.md) for decisions made so far.
>
> Not financial or credit advice.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then add your Companies House API key
uv run pytest
```

Build the accounts sample (about 3.6 GB of downloads, then about an hour of parsing on 8 cores):

```bash
uv run python -m cfp.ingest.accounts_download --dry-run   # size estimate
uv run python -m cfp.ingest.accounts_download
uv run python -m cfp.ingest.accounts_parse
```

Lint, format and type-check:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
```
