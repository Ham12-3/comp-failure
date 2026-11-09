# Feature drift

Population Stability Index, reference = filings 2018-2022, current = filings 2024-2025.
PSI < 0.1 stable, 0.1-0.25 watch, > 0.25 shift.

Shift: 4 features, watch: 2, stable: 67.

| Feature | PSI | Status |
|---|---|---|
| `employees` | 0.396 | shift |
| `missing_employees` | 0.376 | shift |
| `accounts_status` | 0.281 | shift |
| `accounts_type` | 0.274 | shift |
| `accounting_standard` | 0.230 | watch |
| `is_dormant` | 0.185 | watch |
| `postcode_area` | 0.033 | stable |
| `sic_section` | 0.027 | stable |
| `current_ratio` | 0.025 | stable |
| `missing_creditors_after_one_year` | 0.025 | stable |
| `days_since_previous_accounts` | 0.023 | stable |
| `cash_to_current_liabilities` | 0.020 | stable |
| `net_assets` | 0.019 | stable |
| `total_assets` | 0.019 | stable |
| `log_total_assets` | 0.019 | stable |
