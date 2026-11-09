# Data

What data the project uses, how the sample was drawn, and what is known to be wrong with it.
Labels (Phase 2) and features (Phase 3) will be added below.

## 1. Sources

| Source | What we take | Access |
|---|---|---|
| [Accounts Data Product](https://download.companieshouse.gov.uk/en_monthlyaccountsdata.html) ([historic](https://download.companieshouse.gov.uk/historicmonthlyaccountsdata.html)) | Financial statements of every company that filed accounts electronically (about 75% of all filings), one iXBRL/XBRL file per filing, grouped into one zip per filing month. | Free bulk download, no key. Served from S3, so byte ranges work. |
| [Companies House Public Data API](https://developer-specs.company-information.service.gov.uk/) | Company profile, filing history, officers, charges, insolvency, PSC. Used for labels and non-financial features. | Free key; 600 requests per 5 minutes. |

## 2. Accounts sample (Phase 1)

**Archives:** all 94 monthly archives from **January 2018 to October 2025**. Each archive
holds the accounts *filed* in that month (checked: balance sheet dates cluster about 9 months
before the archive month, the private-company filing deadline).

**Sampling:** a company is in the sample if `sha256("cfp-v1:" + company_number)` falls in the
lowest **0.5%** of the hash range. Every filing of a sampled company in any of the 94 months is
kept, so the sample is a panel: about one row per company per year. Members are fetched by
HTTP range request from the remote zips (see `docs/DECISIONS.md` 1.1-1.3).

**Download cost:** 3.57 GB in total, against archives that add up to about 190 GB:

| | |
|---|---|
| Archives read | 94 (Jan 2018 - Oct 2025) |
| Filings indexed (all companies) | 24,869,089 |
| Filings sampled | 125,032 (0.50%) |
| Zip directories downloaded | 2.61 GB |
| Accounts files downloaded | 0.96 GB |

780 nested `.zip` members (0.003% of all files; mostly May-October 2025) were skipped.

**Files:**

| Path | Content |
|---|---|
| `data/interim/accounts_index/<archive>.parquet` | Index of *every* filing in each archive (company number, balance sheet date, byte offset). About 25 million rows in total; lets us resample without downloading directories again. |
| `data/raw/accounts_sample/<archive>.zip` | The sampled accounts files, byte-for-byte as published. |
| `data/raw/accounts_sample/manifest.json` | Per archive: files in it, files sampled, bytes downloaded. Feeds the 5 GB budget check. |
| `data/interim/accounts_parsed/<archive>.parquet` | Parsed rows per archive. |
| `data/interim/accounts.parquet` | All parsed rows combined: **one row per filing**. |

## 3. Parsed accounts table (`data/interim/accounts.parquet`)

| Column(s) | Meaning |
|---|---|
| `company_number`, `balance_sheet_date` | From the file name. Company numbers are 8 characters. |
| `filing_month` | First day of the archive month = month the accounts were filed. A month-level stand-in for the filing date until Phase 2 adds the exact date from filing history. |
| `source_archive`, `source_member`, `file_type` | Provenance (`html` = iXBRL, `xml` = older XBRL). |
| `parse_error` | Null unless the file could not be parsed. |
| `period_end`, `prior_period_end` | Balance sheet dates of the current and comparative columns. |
| `accounting_standard`, `accounts_type`, `accounts_status`, `trading_status`, `legal_form` | Report metadata tagged in the filing (e.g. `Micro-entities`, `FilletedAccounts`, `AuditExempt-NoAccountantsReport`). |
| `is_dormant`, `date_authorised`, `date_received` | Dormant flag; date the board approved the accounts; date Companies House received them (only in older XBRL files). |
| `n_numeric_facts` | Number of numeric facts tagged: a rough measure of how much the filing discloses. |
| Financial values, each with a `_prev` comparative | `equity`, `net_assets`, `net_current_assets`, `total_assets_less_current_liabilities`, `fixed_assets`, `current_assets`, `cash`, `debtors`, `inventories`, `creditors_within_one_year`, `creditors_after_one_year`, `creditors_no_maturity`, `provisions`, `bank_borrowings_within_one_year`, `bank_borrowings_after_one_year`, `turnover`, `gross_profit`, `operating_profit`, `profit_before_tax`, `profit_loss`, `employees`. |

Values are exactly as tagged: null when not tagged (never filled with 0), and **not
sign-corrected** (see section 5).

**Parse results** (all 94 archives):

| | |
|---|---|
| Filings parsed | 125,032 (0 parse errors) |
| Companies | 29,584 (4.2 filings each on average) |
| Filing months | Jan 2018 - Oct 2025 (13-18k filings per year) |
| Format | 99.7% iXBRL, 423 older XBRL `.xml` files |
| Accounting standard tagged | Micro-entities 45%, FRS 102 29%, small entities 23%, not tagged 2% |
| Dormant | 16.6% |
| Prior-year comparatives present | 77.4% |
| No core balance-sheet value found | 0.5% in every year |

Share of filings with each value present (current period):

| Value | Present | Value | Present |
|---|---|---|---|
| equity | 91.4% | creditors_within_one_year | 73.7% |
| net_assets | 83.4% | creditors_after_one_year | 28.6% |
| net_current_assets | 81.9% | creditors_no_maturity | 4.4% |
| total_assets_less_current_liabilities | 77.6% | provisions | 14.1% |
| current_assets | 76.1% | bank_borrowings (within / after one year) | 6.3% / 4.6% |
| employees | 71.0% | turnover | 2.4% |
| cash | 42.5% | profit_loss | 3.5% |
| fixed_assets | 41.9% | profit_before_tax | 1.3% |
| debtors | 29.1% | gross_profit, operating_profit | 1.0% |
| inventories | 10.7% | | |

Equity is missing slightly more often in recent filings (4.9% of non-dormant filings in 2018,
11.6% in 2025) while `net_assets` is present; the two are equal by definition, so Phase 3 can
fill one from the other.

## 4. Labels (Phase 2, in progress)

Built by `cfp.labels.build_labels` into `data/processed/labels.parquet`, one row per filing.

| Column | Definition |
|---|---|
| `prediction_date` | Filing date of the accounts, from filing history (`filing_date_source` says how it was found). |
| `label_insolvency_12m` | 1 if a compulsory liquidation, creditors' voluntary liquidation, administration or receivership starts in (prediction date, +12 months]. Members' voluntary liquidations (solvent) and CVAs do not count. |
| `insolvency_case_type`, `insolvency_start_date` | The first such case in the window. |
| `solvent_liquidation_12m` | A members' voluntary liquidation started in the window (recorded, not a failure). |
| `label_strike_off_12m` | Dissolved in the window with no insolvency case: the second target. |
| `excluded_reason` | Null for usable rows; otherwise `already_insolvent`, `outcome_not_observable` or `no_profile`. |

Full definitions and reasons: `docs/DECISIONS.md` 2.1-2.6. Row counts, date range and class
balance will be filled in once the API fetch for all 29,584 sampled companies has run.

**Early indication** (Free Company Data Product, 1 November 2026): of the 29,584 sampled
companies, 17,949 are still on the register, 376 of them in liquidation and 18 in
administration; the other 11,635 have been dissolved (by strike-off or after liquidation; the
API tells which).

## 5. Features (Phase 3)

Built by `cfp.features.build_features` into `data/processed/features.parquet`: one row per
filing, the keys `company_number`, `balance_sheet_date`, `source_member`, `prediction_date`
and 73 features. Join to `labels.parquet` on `company_number` + `source_member`.

**Point-in-time rules** (enforced by `tests/test_leakage.py`):

* Accounts features use only the filing itself (current year and its prior-year comparatives),
  public on the prediction date. Signs are repaired with accounting identities first.
* Every API-based feature goes through one gate, `features/events.as_of`, which keeps events
  dated on or before the prediction date. Filing-history events are dated by when they were
  filed (public), charges by `delivered_on`.
* "In office" / "outstanding" features take things that started by the prediction date and
  check whether they had ended by then.
* The registered office postcode is the address in force on the prediction date: the last
  move filed on or before it, else the address the first later move left, else today's address
  if the company never moved since. Today's address alone would leak: liquidators move the
  office to their own (Carillion: Wolverhampton to Leeds).
* Never read: `company_status`, `date_of_cessation`, accounts/confirmation due dates, insolvency
  flags or links, or any label column.

The leakage test compares, row by row, features built from the full API history with features
built from a register snapshot taken on the prediction date, on real Carillion data and on a
synthetic company. Breaking the gate on purpose (removing the date filter, or using today's
address) makes it fail.

**Approximations, documented rather than hidden:**

* Accounts are treated as due 9 months after the balance sheet date (private companies). Public
  companies have 6 months and first accounts have different rules, so `days_late_this_filing`
  is rough for those (`first_accounts` flags the latter).
* SIC code and legal form come from today's profile; they rarely change.
* `directors_in_office` uses effective appointment/resignation dates, which can be up to 14
  days earlier than when they were filed.

| Feature | Source | Meaning |
|---|---|---|
| `net_assets` | accounts | Net assets (falls back to equity). |
| `total_assets` | accounts | Fixed assets + current assets. |
| `current_assets` | accounts | Current assets. |
| `current_liabilities` | accounts | Creditors due within one year (or current assets - net current assets). |
| `cash` | accounts | Cash at bank and in hand. |
| `total_liabilities` | accounts | Total assets - net assets. |
| `total_debt` | accounts | Creditors due within + after one year. |
| `bank_debt` | accounts | Bank loans and overdrafts, within + after one year (0 if not tagged). |
| `turnover` | accounts | Turnover, if filed (rare for small companies). |
| `profit_loss` | accounts | Profit or loss for the year, if filed. |
| `employees` | accounts | Average number of employees. |
| `current_ratio` | accounts | Current assets / current liabilities. |
| `cash_to_current_liabilities` | accounts | Cash / current liabilities. |
| `debt_to_assets` | accounts | Total liabilities / total assets. |
| `equity_ratio` | accounts | Net assets / total assets. |
| `net_assets_negative` | accounts | Net assets below zero. |
| `net_current_liabilities` | accounts | Net current assets below zero. |
| `log_total_assets` | accounts | log(1 + |total assets|): company size. |
| `net_assets_change` | accounts | Net assets minus prior-year net assets. |
| `total_assets_pct_change` | accounts | Relative change in total assets. |
| `current_assets_pct_change` | accounts | Relative change in current assets. |
| `cash_pct_change` | accounts | Relative change in cash. |
| `turnover_pct_change` | accounts | Relative change in turnover. |
| `net_assets_turned_negative` | accounts | Net assets negative now, not negative last year. |
| `has_prior_year` | accounts | Prior-year comparatives present. |
| `missing_current_assets` | accounts | Current assets not tagged. |
| `missing_cash` | accounts | Cash not tagged. |
| `missing_debtors` | accounts | Debtors not tagged. |
| `missing_fixed_assets` | accounts | Fixed assets not tagged. |
| `missing_creditors_within_one_year` | accounts | Short-term creditors not tagged. |
| `missing_creditors_after_one_year` | accounts | Long-term creditors not tagged. |
| `missing_turnover` | accounts | Turnover not filed. |
| `missing_profit_loss` | accounts | Profit or loss not filed. |
| `missing_employees` | accounts | Employee count not tagged. |
| `n_numeric_facts` | accounts | Number of numeric facts tagged: how much was disclosed. |
| `sign_repaired` | accounts | A sign-tagging error was corrected in this filing. |
| `accounting_standard` | accounts | Micro-entities / SmallEntities / FRS102 / ... |
| `accounts_type` | accounts | Full / filleted / abridged accounts (tagged). |
| `accounts_status` | accounts | Audited or audit-exempt (tagged). |
| `is_dormant` | accounts | Company declares itself dormant. |
| `prior_period_gap_days` | accounts | Days between this and the prior balance sheet date. |
| `days_late_this_filing` | filing history | Filing date - (balance sheet date + 9 months); negative = early. |
| `first_accounts` | filing history | No earlier accounts filing on record. |
| `days_since_previous_accounts` | filing history | Days since the previous accounts filing. |
| `previous_accounts_late_3y` | filing history | Earlier accounts filed late in the last 3 years. |
| `previous_accounts_max_days_late_3y` | filing history | Worst lateness of earlier accounts, last 3 years. |
| `confirmations_late_3y` | filing history | Confirmation statements / annual returns filed late, last 3 years. |
| `confirmation_max_days_late_3y` | filing history | Worst confirmation-statement lateness, last 3 years. |
| `days_since_confirmation_made_up` | filing history | Days since the last confirmation statement's made-up date. |
| `filings_12m` | filing history | Filings of any kind in the last 12 months. |
| `office_moves_12m` | filing history | Registered office changes in the last 12 months. |
| `default_address_applied_ever` | filing history | Companies House replaced an invalid registered office (RP05). |
| `strike_off_notices_12m` | filing history | Compulsory strike-off gazette notices (GAZ1), last 12 months. |
| `strike_off_notices_ever` | filing history | Compulsory strike-off gazette notices, ever. |
| `strike_off_discontinued_12m` | filing history | Strike-off action discontinued (DISS40), last 12 months. |
| `strike_off_applications_ever` | filing history | Voluntary strike-off applications (DS01), ever. |
| `reference_date_changes_12m` | filing history | Accounting reference date changes (AA01), last 12 months. |
| `psc_changes_12m` | filing history | Person-with-significant-control filings, last 12 months. |
| `director_appointments_12m` | filing history | Director appointments filed in the last 12 months. |
| `director_terminations_12m` | filing history | Director terminations filed in the last 12 months. |
| `accounts_type_filed` | filing history | Accounts type Companies House recorded for this filing. |
| `postcode_area` | filing history + profile | Postcode area of the registered office on the prediction date. |
| `directors_in_office` | officers | Directors in office on the prediction date. |
| `director_median_tenure_days` | officers | Median tenure of those directors, in days. |
| `charges_outstanding` | charges | Charges registered and not satisfied by the prediction date. |
| `charges_registered_12m` | charges | Charges registered in the last 12 months. |
| `charges_satisfied_12m` | charges | Charges satisfied in the last 12 months. |
| `charges_ever` | charges | Charges ever registered. |
| `company_age_years` | profile | Years since incorporation. |
| `company_type` | profile | Legal form (ltd, plc, llp, ...). |
| `jurisdiction` | profile | England and Wales / Scotland / Northern Ireland. |
| `sic_division` | profile | First two digits of the first SIC code (today's code). |
| `n_sic_codes` | profile | Number of SIC codes. |

## 6. Known data quality issues

1. **Sign errors.** Some filings show a negative value in brackets but tag it as positive
   (missing `sign="-"`). Example: `tests/fixtures/accounts/Prod224_2510_10851932_20250731.html`
   shows net assets of (7,385) and tags 7,385. Phase 3 must repair signs using accounting
   identities before computing "negative net assets" and similar features.
2. **Little profit and loss data.** Small and micro companies may omit the profit and loss
   account, and most do, so turnover and profit are missing for over 95% of filings.
3. **Two tagging conventions for maturity** (`WithinOneYear` vs `CurrentFinancialInstruments`).
   Both are handled; a few filers use neither, and their creditors go to `creditors_no_maturity`.
4. **Only electronic filings.** Companies House says about a quarter of accounts are filed on
   paper; those are not in the Accounts Data Product, so paper filers are missing from the
   sample. Whether distressed companies file on paper more or less often is unknown.
5. **Filing month, not date.** Until Phase 2, `filing_month` is the only filing-time field.
