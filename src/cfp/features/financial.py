"""Financial features from the accounts filing itself.

Everything here comes from one filing (current year and the prior-year comparatives printed
in it), all of which became public on the prediction date. No other data is used.

Steps:
1. **Sign repair**, in two directions (see docs/DATA.md):
   * Balances that cannot be negative (assets, cash, debtors, creditors, provisions, employees
     ...) are sometimes tagged with a minus sign because they are *shown* in brackets as
     deductions (about 3% of filings). They are made positive.
   * Net figures are sometimes shown in brackets but tagged positive. Accounting identities say
     which sign is right:
   net current assets = current assets - creditors due within one year;
   total assets less current liabilities = fixed assets + net current assets;
   net assets = total assets less current liabilities - creditors after one year - provisions.
   A tagged value is flipped only if its negative fits the identity better than it does.
2. **Derived amounts** (current liabilities, total assets, total liabilities, debt).
3. **Ratios, year-on-year changes, size, missingness flags and report metadata.**
"""

from __future__ import annotations

import polars as pl

PERIODS = ("", "_prev")

# Balances that are non-negative by definition in UK accounts. (Turnover is left alone: a
# negative turnover is rare but possible.)
NEVER_NEGATIVE = (
    "current_assets",
    "fixed_assets",
    "cash",
    "debtors",
    "inventories",
    "creditors_within_one_year",
    "creditors_after_one_year",
    "creditors_no_maturity",
    "provisions",
    "bank_borrowings_within_one_year",
    "bank_borrowings_after_one_year",
    "employees",
)


def _flip_if_closer(value: str, expected: pl.Expr) -> pl.Expr:
    """Return `value` with its sign flipped when -value is closer to `expected`."""
    v = pl.col(value)
    flip = (v > 0) & ((expected + v).abs() < (expected - v).abs())
    return pl.when(flip).then(-v).otherwise(v)


def repair_signs(frame: pl.DataFrame) -> pl.DataFrame:
    """Fix sign-tagging errors in net current assets, total assets less current liabilities,
    net assets and equity, for both periods. Adds `sign_repaired` (any fix in either period).

    Each identity uses values repaired by the step before it, so the order matters.
    """
    never_negative = [f"{c}{p}" for c in NEVER_NEGATIVE for p in PERIODS]
    flags: list[pl.Expr] = [(pl.col(c) < 0).fill_null(False) for c in never_negative]
    flagged = frame.with_columns(_any_negative=pl.any_horizontal(flags))
    out = flagged.with_columns(pl.col(never_negative).abs())
    flags = [pl.col("_any_negative")]
    for p in PERIODS:
        nca = f"net_current_assets{p}"
        talcl = f"total_assets_less_current_liabilities{p}"
        repaired = (nca, talcl, f"net_assets{p}", f"equity{p}")
        out = out.with_columns(pl.col(c).alias(f"_orig_{c}") for c in repaired)

        out = out.with_columns(
            _flip_if_closer(
                nca, pl.col(f"current_assets{p}") - pl.col(f"creditors_within_one_year{p}")
            )
        )
        out = out.with_columns(
            _flip_if_closer(talcl, pl.col(f"fixed_assets{p}").fill_null(0) + pl.col(nca))
        )
        expected_net_assets = (
            pl.coalesce(talcl, pl.col(f"fixed_assets{p}").fill_null(0) + pl.col(nca))
            - pl.col(f"creditors_after_one_year{p}").fill_null(0)
            - pl.col(f"provisions{p}").fill_null(0)
        )
        out = out.with_columns(
            _flip_if_closer(f"net_assets{p}", expected_net_assets),
            _flip_if_closer(f"equity{p}", expected_net_assets),  # equity = net assets
        )
        flags += [(pl.col(c) != pl.col(f"_orig_{c}")).fill_null(False) for c in repaired]
    originals = [
        f"_orig_{c}"
        for p in PERIODS
        for c in (
            f"net_current_assets{p}",
            f"total_assets_less_current_liabilities{p}",
            f"net_assets{p}",
            f"equity{p}",
        )
    ]
    return out.with_columns(sign_repaired=pl.any_horizontal(flags)).drop(
        [*originals, "_any_negative"]
    )


def _safe_ratio(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den.abs() > 0).then(num / den).otherwise(None)


def _pct_change(col: str) -> pl.Expr:
    return _safe_ratio(pl.col(col) - pl.col(f"{col}_prev"), pl.col(f"{col}_prev").abs())


RAW_PRESENCE = (
    "current_assets",
    "cash",
    "debtors",
    "fixed_assets",
    "creditors_within_one_year",
    "creditors_after_one_year",
    "turnover",
    "profit_loss",
    "employees",
)


def financial_features(accounts: pl.DataFrame) -> pl.DataFrame:
    """Feature columns (plus `row_id`) from parsed accounts rows (which must carry `row_id`)."""
    f = repair_signs(accounts)
    f = f.with_columns(
        net_assets_best=pl.coalesce("net_assets", "equity"),
        net_assets_best_prev=pl.coalesce("net_assets_prev", "equity_prev"),
        current_liabilities=pl.coalesce(
            "creditors_within_one_year",
            pl.col("current_assets") - pl.col("net_current_assets"),
            "creditors_no_maturity",
        ),
        total_assets=pl.col("fixed_assets").fill_null(0) + pl.col("current_assets"),
        total_assets_prev=pl.col("fixed_assets_prev").fill_null(0) + pl.col("current_assets_prev"),
        bank_debt=pl.col("bank_borrowings_within_one_year").fill_null(0)
        + pl.col("bank_borrowings_after_one_year").fill_null(0),
    )
    f = f.with_columns(
        total_liabilities=pl.col("total_assets") - pl.col("net_assets_best"),
        total_creditors=pl.col("current_liabilities")
        + pl.col("creditors_after_one_year").fill_null(0),
    )
    return f.select(
        "row_id",
        # amounts
        pl.col("net_assets_best").alias("net_assets"),
        "total_assets",
        "current_assets",
        "current_liabilities",
        "cash",
        "total_liabilities",
        pl.col("total_creditors").alias("total_debt"),
        "bank_debt",
        "turnover",
        "profit_loss",
        "employees",
        # ratios
        _safe_ratio(pl.col("current_assets"), pl.col("current_liabilities")).alias("current_ratio"),
        _safe_ratio(pl.col("cash"), pl.col("current_liabilities")).alias(
            "cash_to_current_liabilities"
        ),
        _safe_ratio(pl.col("total_liabilities"), pl.col("total_assets")).alias("debt_to_assets"),
        _safe_ratio(pl.col("net_assets_best"), pl.col("total_assets")).alias("equity_ratio"),
        (pl.col("net_assets_best") < 0).alias("net_assets_negative"),
        (pl.col("net_current_assets") < 0).alias("net_current_liabilities"),
        # size
        pl.col("total_assets").abs().log1p().alias("log_total_assets"),
        # year on year (comparatives from the same filing)
        (pl.col("net_assets_best") - pl.col("net_assets_best_prev")).alias("net_assets_change"),
        _pct_change("total_assets").alias("total_assets_pct_change"),
        _pct_change("current_assets").alias("current_assets_pct_change"),
        _pct_change("cash").alias("cash_pct_change"),
        _pct_change("turnover").alias("turnover_pct_change"),
        ((pl.col("net_assets_best") < 0) & (pl.col("net_assets_best_prev") >= 0)).alias(
            "net_assets_turned_negative"
        ),
        pl.col("net_assets_best_prev").is_not_null().alias("has_prior_year"),
        # missingness (a signal in its own right)
        *[pl.col(c).is_null().alias(f"missing_{c}") for c in RAW_PRESENCE],
        "n_numeric_facts",
        "sign_repaired",
        # report metadata tagged in the filing
        "accounting_standard",
        # The 2023 taxonomy added a `FilletedAccounts` member; before that, the same accounts
        # were tagged `FullAccounts`. Map the new member back so the category means the same
        # thing in every year (found by the drift check, docs/DECISIONS.md 6.2).
        pl.col("accounts_type").replace({"FilletedAccounts": "FullAccounts"}),
        "accounts_status",
        "is_dormant",
        # Months by which the reporting period differs from a normal year. Measured in months,
        # not days, so leap years do not register as a change (also found by the drift check).
        ((pl.col("period_end").sub(pl.col("prior_period_end")).dt.total_days() - 365.25) / 30.44)
        .round(0)
        .alias("period_length_change_months"),
    )
