# Decisions log

One entry per decision: what was chosen, and why. Newest phase at the bottom.

## Phase 0: Setup (2026-11-07)

| # | Decision | Why |
|---|----------|-----|
| 0.1 | **`uv` with the `uv_build` backend, `src/` layout, package name `cfp`.** | `src/` layout stops tests accidentally importing from the working tree instead of the installed package. `uv_build` is uv's native backend, so no extra build tool. |
| 0.2 | **Python pinned to 3.12** (`.python-version`, `requires-python = ">=3.12,<3.13"`). | Matches the spec and the Docker base image we will use in Phase 5; one version keeps the lockfile and CI reproducible. |
| 0.3 | **Runtime dependencies added phase by phase**, not all up front. Phase 0 only needs `pydantic` + `pydantic-settings`. | Keeps CI fast and the lockfile honest: every dependency is there because some code uses it. |
| 0.4 | **Config via `pydantic-settings`** (`cfp.config.Settings`), API key held as `SecretStr` and optional. | One place for paths and secrets; the key is masked in reprs/logs (tested). Optional so CI and offline tests run without a key; code that needs it calls `require_api_key()`, which fails with instructions. |
| 0.5 | **ruff** for lint + format (line length 100), rules `E,W,F,I,B,UP,SIM,N,PTH,DTZ,RUF`. | `DTZ` flags naive `datetime` use; date handling is where leakage bugs hide in this project, so we want to be forced to be explicit. `PTH` keeps paths cross-platform (Windows dev, Linux CI). |
| 0.6 | **mypy `--strict`** on `src` and `tests`, with the pydantic plugin. | The spec asks for type hints everywhere; strict mode enforces it rather than hoping. |
| 0.7 | **`network` pytest marker, excluded by default.** | Unit tests must be offline and deterministic. Tests that call the real API run only with `pytest -m network`. |
| 0.8 | **CI on GitHub Actions, `ubuntu-latest`:** `uv sync --locked`, `ruff check`, `ruff format --check`, `mypy`, `pytest --cov`. | `--locked` fails CI if `uv.lock` is out of date with `pyproject.toml`. Single Python version (see 0.2). |
| 0.9 | **All of `data/` gitignored** except the `raw/`, `interim/`, `processed/` placeholders; `mlruns/` and model files gitignored. `reports/` *is* committed. | Data is large and re-downloadable; plots and metrics in `reports/` are evaluation evidence that belongs in the repo. |
| 0.10 | **`.gitattributes` forces LF line endings.** | The dev machine is Windows; avoids CRLF diffs and formatter disagreements with Linux CI. |
| 0.11 | **Dockerfile / docker-compose deferred to Phase 5.** | Nothing to serve yet; writing them now would mean rewriting them later. |
