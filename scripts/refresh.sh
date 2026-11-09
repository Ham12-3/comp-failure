#!/usr/bin/env bash
# Monthly refresh: add one new month of accounts, update API data, rebuild, retrain, check drift.
#
#   scripts/refresh.sh 2025-11
#
# Each step is resumable and skips work already done. New filings only gain labels once their
# 12-month outcome window has closed, so a monthly run mostly adds outcomes that have matured.
# The train/validation/test years are fixed in cfp/train/data.py (SPLITS): move them forward
# when a new year matures. Review reports/metrics.md and reports/drift.md before restarting the
# API (docker compose restart) to serve the new model in models/.
set -euo pipefail

MONTH="${1:?usage: scripts/refresh.sh YYYY-MM (the filing month to add)}"

uv run python -m cfp.ingest.accounts_download --from "$MONTH" --to "$MONTH"
uv run python -m cfp.ingest.accounts_parse
uv run python -m cfp.ingest.fetch_companies --keep-awake
uv run python -m cfp.labels.build_labels
uv run python -m cfp.features.build_features
uv run python -m cfp.train.train
uv run python -m cfp.train.drift

echo "Done: check reports/metrics.md and reports/drift.md, then: docker compose restart"
