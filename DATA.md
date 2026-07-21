# Data Guide

This project requires FDIC call-report CSV files, obtained separately (they are freely
downloadable — see below). The full dataset is ~80 MB and is not bundled.

## Required Files

Place these CSV files in `data/raw/fdic/`. The six core files are **required**; the
Performance & Condition Ratios file is **optional but recommended** — including it unlocks
the full 15-ratio CAMELS panel (see Feature Engineering).

| File | Key Columns Used | Required |
|------|------------------|:--------:|
| Total Liabilities and Capital | `RBCT1J`, `CERT` | ✅ |
| Total Assets | `ASSET`, `CERT` | ✅ |
| Total Interest Income | `INTINC`, `CERT` | ✅ |
| Total Interest Expense | `EINTEXP`, `CERT` | ✅ |
| Past Due and Nonaccrual Assets | `NAASSET`, `P9ASSET`, `CERT` | ✅ |
| Net Charge-Offs | `NTLS`, `CERT` | ✅ |
| Performance & Condition Ratios | `RBCRWAJ`, `RBC1AAJ`, `IDT1CER`, `ROA`, `ROE`, `NIMY`, `EEFFR`, `NPERFV`, `NCLNLSR`, `ELNATRY`, `LNLSDEPR`, `CERT` | ⭐ optional |

All files must share the same `CERT` (institution certificate) key for merging.

## Feature Engineering

The pipeline builds a **CAMELS-style panel**. Four ratios are always computed from raw
columns; eleven pre-computed FDIC ratios are added when the Performance & Condition
Ratios file is present (`features.py` is schema-aware, so the 6-file setup still runs
with just the four computed ratios).

**Computed (always):**

```
Capital_Ratio       = RBCT1J / ASSET
Net_Interest_Margin = (INTINC - EINTEXP) / ASSET
Texas_Ratio         = (NAASSET + P9ASSET) / RBCT1J
Charge_Off_Ratio    = NTLS / ASSET
```

**Passed through from the Performance & Condition Ratios report (when available):**

| Feature | Source | CAMELS dimension |
|---------|--------|------------------|
| Total_Capital_Ratio | `RBCRWAJ` | Capital |
| Tier1_Leverage_Ratio | `RBC1AAJ` | Capital |
| CET1_Ratio | `IDT1CER` | Capital |
| Nonperforming_Assets_Pct | `NPERFV` | Asset quality |
| NCO_to_Loans | `NCLNLSR` | Asset quality |
| Reserves_to_Loans | `ELNATRY` | Asset quality |
| Efficiency_Ratio | `EEFFR` | Management |
| Return_on_Assets | `ROA` | Earnings |
| Return_on_Equity | `ROE` | Earnings |
| Net_Interest_Margin_Pct | `NIMY` | Earnings |
| Loans_to_Deposits | `LNLSDEPR` | Liquidity |

## Time-Series Data (FDIC API)

`fdic-ml --generate-trends` pulls the same ratios for multiple past quarters directly
from the public **FDIC BankFind Suite API** (`https://banks.data.fdic.gov`). No local
history files are needed; each quarterly pull is cached under
`backend/artifacts/trends_cache/` for offline re-runs.

## Obtaining FDIC Data

1. **FDIC BankFind API** — https://banks.data.fdic.gov/docs/
2. **FDIC Statistics at a Glance** — https://www.fdic.gov/analysis/quarterly-banking-profile/
3. **Bulk downloads** — quarterly call report data available via FDIC data tools

Export or download the six report categories above for a single reporting date.

## Local Setup

```bash
mkdir -p data/raw/fdic
# Copy your 6 CSV files into data/raw/fdic/
```

Then run with the full-data config:

```bash
fdic-ml --config backend/configs/local.example.json --mode both
```

## Expected Row Counts

| Dataset | Institutions |
|---------|-------------|
| Full (2025-09-30 snapshot) | 4,388 |
| After 1%/99% outlier filter (4-ratio panel) | 4,144 (~5.6% removed) |
| After 1%/99% outlier filter (15-ratio CAMELS panel) | 3,941 (~10% removed) |

## Column Reference

See [docs/reference/FDIC_Column_Names.txt](docs/reference/FDIC_Column_Names.txt) for the full column dictionary from the source dataset.

## Data Quality Notes

- All source files should have unique `CERT` values (no duplicate institution keys)
- Merge deduplicates repeated column names across files — conflicting values are silently dropped
- Outlier filtering removes rows outside 1st/99th percentile on any engineered feature
