# UK Company Failure Predictor

Predicts the probability that a UK company enters insolvency (liquidation, administration or
receivership) within 12 months, using free Companies House data, and explains each score with
its top contributing factors.

> **Status:** Phase 0 (project setup). See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the plan and
> [docs/DECISIONS.md](docs/DECISIONS.md) for decisions made so far.
>
> Not financial or credit advice.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then add your Companies House API key
uv run pytest
```

Lint, format and type-check:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
```
