# UK Company Failure Predictor

Predicts the probability that a UK company enters insolvency (liquidation, administration or
receivership) within 12 months, using free Companies House data, and explains each score with
its top contributing factors.

> **Status:** Phases 0-5 done (data, labels, features, leakage tests, model, API and web page). Results and
> limits: [docs/MODEL_CARD.md](docs/MODEL_CARD.md), [reports/metrics.md](reports/metrics.md).
> See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the plan, [docs/DATA.md](docs/DATA.md) for the data
> and [docs/DECISIONS.md](docs/DECISIONS.md) for decisions made so far.
>
> Not financial or credit advice.

## Run the app

```bash
cp .env.example .env               # add a Live Companies House API key
uv run python -m cfp.train.train   # writes models/ (needs labels + features, see below)
docker compose up --build          # then open http://localhost:8000
```

Type a company name or number. The page shows the calibrated probability of insolvency within
12 months of the latest accounts, how that compares with other companies, the top 5 reasons,
and links to the Companies House records. The same is available as JSON:

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"company_number": "10851932"}'
curl "localhost:8000/search?q=carillion"
curl localhost:8000/health
curl localhost:8000/metrics
```

Interactive API docs: http://localhost:8000/docs

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

Labels, features and model (only the fetch calls the API; the rest read the cache):

```bash
uv run python -m cfp.ingest.fetch_companies --keep-awake   # API data (resumable, rate-limited)
uv run python -m cfp.labels.build_labels
uv run python -m cfp.features.build_features
uv run python -m cfp.train.train
```

Lint, format and type-check:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
```
